from typing import Any, Dict, Tuple, Optional
from pathlib import Path
import yaml
import pickle

import pandas as pd
import numpy as np
import dotenv

from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import category_encoders as ce

from mykaggle.model.gbdt import GBDT
from mykaggle.trainer.cv_strategy import Stratified
from mykaggle.feature.feature_factory import FeatureFactory
from mykaggle.feature.base import Feature
# from mykaggle.trainer.metric.macro_f1 import macro_f1_lgb
from mykaggle.lib.lgbm_util import compute_importances, save_importances
from mykaggle.lib.plot import plot_confusion_matrix
from mykaggle.lib.routine import timer, fix_seed, get_logger, save_config
from mykaggle.lib.ml_logger import MLLogger


#
# Settings
#


IS_DEBUG = False
S = yaml.safe_load('''
name: 'nlp_lgbm_binary_classification'
competition: sample
mode: training
seed: 1019
device: cuda
is_full_training: true
training:
    train_file: train.csv
    test_file: test.csv
    do_cv: true
    cv: stratified
    num_folds: 5
feature:
    features:
        - tfidf_code
        - tfidf_line_c
        - ext_te
    params:
        tfidf_code:
            ngram_range: [1, 2]
            max_features: 5000
        tfidf_line_c:
            ngram_range: [1, 2]
            max_features: 5000
model:
    model_type: lightgbm
    objective: binary
    learning_rate: 0.1
    max_depth: -1
    num_leaves: 10
    colsample_bytree: .7
    metric: 'auc' # 'None'
    num_boost_round: 10000
    early_stopping_rounds: 100
    verbose_eval: 100
    num_classes: 1
ensemble:
    models:
''')
ST = S['training']
SM = S['model']

#
# Prepare
#

fix_seed()
LOGGER = get_logger(__name__)
DATADIR = Path('./data/dubai/')
CKPTDIR = Path('./ckpt/') / S['name']
if not CKPTDIR.exists():
    CKPTDIR.mkdir()


#
# Load Data
#


DF_TRAIN = pd.read_csv(DATADIR / ST['train_file'])
if IS_DEBUG:
    DF_TRAIN = DF_TRAIN.iloc[:1000]
DF_TEST = pd.read_csv(DATADIR / ST['test_file'])
DF_SUB = pd.read_csv(DATADIR / 'sample_submission.csv')

BASE_COLUMN = 'id'
TARGET_COLUMN = 'label'
FOLD_COLUMN = 'fold'


class MyStratified(Stratified):
    def preprocess(self, df: pd.DataFrame, *args, **kwargs) -> pd.DataFrame:
        df['ext_label'] = df['file_extension'] + "_" + df['label'].astype('str')
        return df


if FOLD_COLUMN not in DF_TRAIN.columns or ST['do_fold']:
    cv = MyStratified(ST['num_folds'])
    DF_TRAIN = cv.split_and_set(DF_TRAIN, y_column='ext_label')

LOGGER.info(f'Training data: {len(DF_TRAIN)}, Test data: {len(DF_TEST)}')


#
# Feature
#


def preprocess(df_train: pd.DataFrame, df_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train[['A', 'B', 'C', 'D', 'E']] = df_train['code'].str.split('\n', expand=True)
    df_test[['A', 'B', 'C', 'D', 'E']] = df_test['code'].str.split('\n', expand=True)
    return df_train, df_test


class TfidfEncoding(Feature):
    '''
    テキストの TF-IDF エンコーディング
    '''

    def __init__(
        self,
        name: str,
        input_column: str,
        ngram_range: Tuple[int],
        max_features: Optional[int] = None,
        train: bool = True
    ) -> None:
        super().__init__(name=name, train=train, base_column=BASE_COLUMN)
        self.input_column = input_column
        self.ngram_range = ngram_range
        self.max_features = max_features

    def create(
        self,
        base: pd.DataFrame,
        others: Dict[str, pd.DataFrame],
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        df_another = others['another'].copy()
        if self.train:
            df_whole = pd.concat([df_main, df_another])
        else:
            df_whole = pd.concat([df_another, df_main])

        processor = TfidfVectorizer(ngram_range=self.ngram_range, max_features=self.max_features)
        tfidf = processor.fit_transform(df_whole[self.input_column])
        column_names = [f'tfidf_{self.input_column}_{str(i)}' for i in range(tfidf.shape[1])]
        if self.train:
            df_main.loc[:, column_names] = tfidf[:len(df_main), :].toarray()
        else:
            df_main.loc[:, column_names] = tfidf[len(df_another):, :].toarray()

        return df_main.loc[:, column_names]


@Feature.register('tfidf_code')
class TfidfCodeEncoding(TfidfEncoding):
    def __init__(
        self,
        ngram_range: Tuple[int],
        max_features: Optional[int] = None,
        train: bool = True
    ) -> None:
        super().__init__(
            name='tfidf_code',
            input_column='code',
            ngram_range=ngram_range,
            max_features=max_features,
            train=train
        )


@Feature.register('tfidf_line_c')
class TfidfLineCEncoding(TfidfEncoding):
    def __init__(
        self,
        ngram_range: Tuple[int],
        max_features: Optional[int] = None,
        train: bool = True
    ) -> None:
        super().__init__(
            name='tfidf_line_c',
            input_column='C',
            ngram_range=ngram_range,
            max_features=max_features,
            train=train
        )


@Feature.register('ext_te')
class ExtTargetEncoding(Feature):
    '''
    file_extention による target encoding
    '''

    def __init__(
        self,
        train: bool = True
    ) -> None:
        super().__init__(name='ext_te', train=train, base_column=BASE_COLUMN)

    def create(
        self,
        base: pd.DataFrame,
        others: Dict[str, pd.DataFrame],
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        df_another = others['another'].copy()

        col = 'file_extension'
        te = ce.TargetEncoder(cols=col)
        if self.train:
            te_train = te.fit_transform(df_main[col], df_main['label'])
            df_main['te_ext'] = te_train
        else:
            te_train = te.fit_transform(df_another[col], df_another['label'])
            te_test = te.transform(df_main[col])
            df_main['te_ext'] = te_test

        return df_main.loc[:, ['te_ext']]


def get_features(
    settings: Dict[str, Any], df_train: pd.DataFrame, df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_train, df_test = preprocess(df_train, df_test)
    df_base_train = df_train[[BASE_COLUMN, TARGET_COLUMN, FOLD_COLUMN]].copy()
    df_base_test = df_test[[BASE_COLUMN]].copy()
    df_others_train = {
        'main': df_train.copy(),
        'another': df_test.copy(),
    }
    df_others_test = {
        'main': df_test.copy(),
        'another': df_train.copy(),
    }

    feature_names = settings['feature']['features']
    LOGGER.info(f'Using features: {feature_names}')
    feature_params = settings['feature']['params'] or {}
    LOGGER.info(f'With features params: {feature_params}')
    df_f_train = FeatureFactory.run(
        feature_names, feature_params,
        df_base_train.copy(), df_others_train, train=True, use_cache=False, save_cache=False
    )
    df_f_test = FeatureFactory.run(
        feature_names, feature_params,
        df_base_test.copy(), df_others_test, train=False, use_cache=False, save_cache=False
    )
    LOGGER.info(f'train feature shape: {df_f_train.shape}, test feature shape: {df_f_test.shape}')
    return df_f_train, df_f_test


def train(
    s: Dict[str, Any],
    logger: MLLogger,
    df: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Tuple[np.ndarray, ...]:
    st = s['training']
    sm = s['model']
    logger.log_params(st)
    logger.log_params(s['feature'])
    logger.log_params(sm)

    df_f, df_f_test = get_features(s, df, df_test)
    y = df[TARGET_COLUMN].values
    df_f_test = df_f_test.drop(BASE_COLUMN, axis=1)
    LOGGER.info(df_f)
    models = []
    if sm['num_classes'] == 1:
        oof_preds = np.zeros((len(y)))
        test_preds = np.zeros((len(df_f_test)))
    else:
        oof_preds = np.zeros((len(y), sm['num_classes']))
        test_preds = np.zeros((len(df_f_test), sm['num_classes']))
    importances = pd.DataFrame()

    for fold in range(st['num_folds']):
        df_train = df_f[df_f[FOLD_COLUMN] != fold]
        y_train = df_train[TARGET_COLUMN].values
        df_valid = df_f[df_f[FOLD_COLUMN] == fold]
        y_valid = df_valid[TARGET_COLUMN].values

        x_train = df_train.drop([FOLD_COLUMN, BASE_COLUMN, TARGET_COLUMN], axis=1)
        x_valid = df_valid.drop([FOLD_COLUMN, BASE_COLUMN, TARGET_COLUMN], axis=1)

        with timer(prefix=f'train fold={fold + 1} '):
            model = GBDT('lightgbm', sm)
            model.train(x_train, y_train, x_valid, y_valid)
        val_preds = model.predict(x_valid)
        oof_preds[df_valid.index] = val_preds
        test_pred = model.predict(df_f_test)
        test_preds += test_pred / st['num_folds']
        models.append(model)
        importances = compute_importances(importances, x_train.columns, models[fold], fold=fold)
        fold_score = roc_auc_score(y_valid, val_preds)
        logger.log_metric(f'f1_fold{fold}', fold_score)
        LOGGER.info(f'Fold {fold} Macro-F1: {fold_score:.4f}')
        model.save(str(CKPTDIR / f'model_{fold}.txt'))
        logger.log_artifact(str(CKPTDIR / f'model_{fold}.txt'))

    score = roc_auc_score(y, oof_preds)
    logger.log_metric('auc', score)
    LOGGER.info('FINISHED; whole score: {:.4f}'.format(score))
    save_importances(importances, CKPTDIR)
    plot_confusion_matrix(oof_preds > 0.5, y, CKPTDIR)
    pickle.dump(oof_preds, open(CKPTDIR / 'oof_preds.pkl', 'wb'))
    pickle.dump(test_preds, open(CKPTDIR / 'test_preds.pkl', 'wb'))
    return oof_preds, test_preds


def do_training(settings: Dict[str, Any], df_train: pd.DataFrame, df_test: pd.DataFrame):
    ml_logger = MLLogger('cfiken', CKPTDIR)
    with ml_logger.start(experiment_name=settings['competition'], run_name=settings['name']):
        save_config(settings, CKPTDIR, ml_logger)
        _, test_preds = train(settings, ml_logger, df_train, df_test)
    return test_preds


def do_inference(settings: Dict[str, Any], df_train: pd.DataFrame, df_test: pd.DataFrame):
    whole_preds = np.zeros((len(df_test)))
    return whole_preds


def submit(s: Dict[str, Any], df_sub: pd.DataFrame, preds: np.ndarray) -> None:
    df_sub[TARGET_COLUMN] = (preds > 0.5).astype(int)
    df_sub.to_csv(CKPTDIR / 'submission_cfiken.csv', index=False)
    LOGGER.info('submit finished.')
    LOGGER.info(df_sub)


def run(settings: Dict[str, Any]):
    mode = settings.get('mode', 'training')
    LOGGER.info(f'start {S["name"]} by mode {mode}')
    if mode == 'ensemble':
        # preds = do_ensemble(settings, df_train, df_test)
        preds = np.zeros(len(DF_TEST))
        submit(settings, DF_SUB, preds)
    elif mode == 'inference':
        preds = do_inference(settings, DF_TRAIN, DF_TEST)
        submit(settings, DF_SUB, preds)
    else:
        preds = do_training(settings, DF_TRAIN, DF_TEST)
        # preds = do_inference(settings, DF_TRAIN, DF_TEST)
        submit(settings, DF_SUB, preds)


if __name__ == '__main__':
    dotenv.load_dotenv()
    fix_seed(S['seed'])
    run(S)
