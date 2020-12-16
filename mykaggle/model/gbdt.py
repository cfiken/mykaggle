from typing import Any, Optional, Union, Dict, List
from enum import Enum

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as cat

TypeAnyModel = Union[lgb.Booster, xgb.Booster, cat.CatBoost]


class ModelType(Enum):
    LightGBM = 'lightgbm'
    XGBoost = 'xgboost'
    CatBoost = 'catboost'


class GBDT:
    '''
    LightGBM, XGBoost, CatBoost をラップするモデル
    '''

    def __init__(self, type: Union[str, ModelType], params: Dict[str, Any]) -> None:
        '''
        Args:
          type: lightgbm/xgboost/catboost の str か ModelType.
          params: gbdt モデルに入れるパラメータの dict.
        '''
        self._type = ModelType(type) if isinstance(type, str) else type
        self._params = params
        self._model = None

    def train(
        self,
        x_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.DataFrame, np.ndarray],
        x_valid: Union[pd.DataFrame, np.ndarray],
        y_valid: Union[pd.DataFrame, np.ndarray],
        cat_features: Optional[List[Union[str, int]]] = None,
        feature_names: Optional[List[str]] = None
    ) -> None:
        if not isinstance(x_train, pd.DataFrame) and feature_names is None:
            raise ValueError('Feature names are not specified. Use pd.DataFrame for inputs or pass feature_names')
        if self._is_lightgbm:
            train_data = lgb.Dataset(x_train, label=y_train)
            valid_data = lgb.Dataset(x_valid, label=y_valid)
            self._model = lgb.train(
                self._params_as_params(self._params),
                train_data,
                valid_names=['train', 'valid'],
                valid_sets=[train_data, valid_data],
                **self._params_as_kwargs(self._params)
            )
        elif self._is_xgboost:
            feature_names = feature_names or x_train.columns
            train_data = xgb.DMatrix(x_train, label=y_train, feature_names=feature_names)
            valid_data = xgb.DMatrix(x_valid, label=y_valid, feature_names=feature_names)
            self._model = xgb.train(
                self._params_as_params(self._params),
                train_data,
                evals=[(train_data, 'train'), (valid_data, 'valid')],
                **self._params_as_kwargs(self._params)
            )
        elif self._is_catboost:
            train_data = cat.Pool(x_train, y_train, cat_features=cat_features)
            valid_data = cat.Pool(x_valid, y_valid, cat_features=cat_features)
            self._model = cat.CatBoost(
                self._params_as_params(self._params),
            )
            self._model.fit(
                train_data,
                eval_set=valid_data,
                use_best_model=True,
                **self._params_as_kwargs(self._params)
            )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        if self._is_lightgbm:
            pred = self._model.predict(inputs)
        elif self._is_xgboost:
            inputs = xgb.DMatrix(inputs)
            pred = self._model.predict(inputs, ntree_limit=self._model.best_ntree_limit)
        elif self._is_catboost:
            pred = self._model.predict(inputs)
        return pred

    def feature_importance(self, type: str):
        '''
        モデルの feature importance を返します。
        return:
          key が feature name, value が importance の Dict
        '''
        if self._is_lightgbm:
            columns = self._model.feature_name()
            f_imp = self._model.feature_importance(type)
            return dict(zip(columns, f_imp))
        elif self._is_xgboost:
            return self._model.get_score(importance_type=type)
        elif self._is_catboost:
            columns = self._model.feature_names_
            f_imp = self._model.feature_importances_
            return dict(zip(columns, f_imp))

    def save(self, filename: str) -> None:
        self._model.save_model(filename)

    def _params_as_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        gbdt_params = {}
        if self._is_lightgbm:
            gbdt_params['objective'] = params.get('objective')
            gbdt_params['learning_rate'] = params.get('learning_rate')
            gbdt_params['max_depth'] = params.get('max_depth')
            gbdt_params['num_leaves'] = params.get('num_leaves')
            gbdt_params['colsample_bytree'] = params.get('colsample_bytree')
            gbdt_params['metric'] = params.get('metric')
        elif self._is_xgboost:
            gbdt_params['objective'] = params.get('objective')
            gbdt_params['learning_rate'] = params.get('learning_rate')
            gbdt_params['max_depth'] = params.get('max_depth')
            gbdt_params['colsample_bytree'] = params.get('colsample_bytree')
        elif self._is_catboost:
            gbdt_params['objective'] = params.get('objective')
            gbdt_params['learning_rate'] = params.get('learning_rate')
            gbdt_params['use_best_model'] = params.get('use_best_model', True)
            gbdt_params['num_boost_round'] = self._params.get('num_boost_round')
        return gbdt_params

    def _params_as_kwargs(self, params: Dict[str, Any]) -> Dict[str, Any]:
        kwargs = {}
        if self._is_lightgbm:
            kwargs['num_boost_round'] = self._params.get('num_boost_round')
            kwargs['early_stopping_rounds'] = self._params.get('early_stopping_rounds')
            kwargs['verbose_eval'] = self._params.get('verbose_eval')
            kwargs['feval'] = self._params.get('feval')
        elif self._is_xgboost:
            kwargs['num_boost_round'] = self._params.get('num_boost_round')
            kwargs['early_stopping_rounds'] = self._params.get('early_stopping_rounds')
            kwargs['verbose_eval'] = self._params.get('verbose_eval')
            kwargs['feval'] = self._params.get('feval')
        elif self._is_catboost:
            kwargs['early_stopping_rounds'] = self._params.get('early_stopping_rounds')
            kwargs['verbose_eval'] = self._params.get('verbose_eval')
        return kwargs

    @property
    def _is_lightgbm(self) -> bool:
        return self._type == ModelType.LightGBM

    @property
    def _is_xgboost(self) -> bool:
        return self._type == ModelType.XGBoost

    @property
    def _is_catboost(self) -> bool:
        return self._type == ModelType.CatBoost
