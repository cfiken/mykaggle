from typing import Any, Dict, Tuple
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
import yaml
import pickle

import pandas as pd
import numpy as np
import dotenv

from sklearn.metrics import roc_auc_score

from mykaggle.model.gbdt import GBDT
from mykaggle.feature.feature_factory import FeatureFactory
from mykaggle.feature.base import Feature
# from mykaggle.trainer.metric.macro_f1 import macro_f1_lgb
from mykaggle.lib.lgbm_util import compute_importances, save_importances
from mykaggle.lib.plot import plot_confusion_matrix
from mykaggle.lib.routine import timer, fix_seed, get_logger
from mykaggle.lib.logger.logger import Logger
from mykaggle.lib.logger.factory import LoggerFactory, LoggerType


#
# Settings
#


IS_DEBUG = False
IS_KAGGLE = False
S = yaml.safe_load('''
name: 'tabular_lgbm_binary_classification'
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
        - simple
    params:
model:
    model_type: lightgbm
    objective: binary
    learning_rate: 0.1
    max_depth: -1
    num_leaves: 10
    colsample_bytree: .7
    metric: 'auc' # 'None'
    num_boost_round: 10000
    early_stopping_rounds: 1000
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
DATADIR = Path('./data/titanic/')
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

BASE_COLUMN = 'PassengerId'
TARGET_COLUMN = 'Survived'
FOLD_COLUMN = 'fold'


if FOLD_COLUMN not in DF_TRAIN.columns or ST['do_fold']:
    from mykaggle.trainer.cv_strategy import CVStrategy
    cv = CVStrategy.create(ST['cv'], ST['num_folds'])
    DF_TRAIN = cv.split_and_set(DF_TRAIN, y_column=TARGET_COLUMN)

LOGGER.info(f'Training data: {len(DF_TRAIN)}, Test data: {len(DF_TEST)}')


#
# Feature
#

@Feature.register('simple')
class SimpleEncoding(Feature):
    '''
    基本的な特徴のエンコーディング
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='simple', train=train, base_column=BASE_COLUMN)

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

        le = LabelEncoder()
        le.fit(df_whole['Sex'])
        df_main['le_Sex'] = le.transform(df_main['Sex'])

        target_columns = ['Pclass', 'le_Sex', 'Age', 'SibSp', 'Parch', 'Fare']

        return df_main.loc[:, target_columns]


def get_features(
    settings: Dict[str, Any], df_train: pd.DataFrame, df_test: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    feature_params = settings['feature']['params'] or {}
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
    logger: Logger,
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
        logger.upload_model(str(CKPTDIR / f'model_{fold}.txt'))

    score = roc_auc_score(y, oof_preds)
    logger.log_metric('auc', score)
    LOGGER.info('FINISHED; whole score: {:.4f}'.format(score))
    save_importances(importances, CKPTDIR)
    plot_confusion_matrix(oof_preds > 0.5, y, CKPTDIR)
    pickle.dump(oof_preds, open(CKPTDIR / 'oof_preds.pkl', 'wb'))
    pickle.dump(test_preds, open(CKPTDIR / 'test_preds.pkl', 'wb'))
    return oof_preds, test_preds


def do_training(settings: Dict[str, Any], df_train: pd.DataFrame, df_test: pd.DataFrame):
    logger_type = LoggerType.STD if IS_KAGGLE else LoggerType.MLFLOW
    ml_logger = LoggerFactory.create(logger_type, 'cfiken', CKPTDIR)
    with ml_logger.start(experiment_name=settings['competition'], run_name=settings['name']):
        ml_logger.save_config(settings)
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
