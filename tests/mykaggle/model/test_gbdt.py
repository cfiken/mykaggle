from typing import Any, Dict
import numpy as np
import pandas as pd

from mykaggle.model.gbdt import GBDT, ModelType

PARAMS: Dict[str, Any] = {
    'objective': 'RMSE',
    'learning_rate': 0.01,
    'max_depth': -1,
    'num_leaves': 31,
    'colsample_bytree': .7,
    'metric': "None",
    'num_boost_round': 10,
    'verbose_eval': False,
}

lgb_params = PARAMS.copy()
xgb_params = PARAMS.copy()
xgb_params['objective'] = 'reg:squarederror'
xgb_params['max_depth'] = 6
cat_params = PARAMS.copy()


class TestGBDT:

    def test_init(self):
        model1 = GBDT('lightgbm', lgb_params)
        assert model1._type == ModelType.LightGBM
        assert model1._is_lightgbm
        model2 = GBDT('xgboost', xgb_params)
        assert model2._type == ModelType.XGBoost
        assert model2._is_xgboost
        model3 = GBDT('catboost', cat_params)
        assert model3._type == ModelType.CatBoost
        assert model3._is_catboost

    def test_lgb_train_predict_fimp(self):
        model = GBDT('lightgbm', lgb_params)

        x_train = pd.DataFrame(np.random.rand(10, 10))
        x_train.columns = [str(c) for c in x_train.columns]
        y_train = np.random.randint(0, 2, (10))
        x_valid = pd.DataFrame(np.random.rand(10, 10))
        y_valid = np.random.randint(0, 2, (10))
        model.train(x_train, y_train, x_valid, y_valid)

        dummy_inputs = np.random.rand(10, 10)
        outputs = model.predict(dummy_inputs)
        assert outputs.shape == (10,)

        f_imp = model.feature_importance('gain')
        assert isinstance(f_imp, dict)
        assert len(f_imp.keys()) == 10

    def test_xgb_train_predict_fimp(self):
        model = GBDT('xgboost', xgb_params)

        x_train = pd.DataFrame(np.random.rand(100, 10))
        x_train.columns = [str(c) for c in x_train.columns]
        y_train = np.random.randint(0, 2, (100))
        x_valid = pd.DataFrame(np.random.rand(100, 10))
        y_valid = np.random.randint(0, 2, (100))
        model.train(x_train, y_train, x_valid, y_valid)

        dummy_inputs = pd.DataFrame(np.random.rand(10, 10))
        outputs = model.predict(dummy_inputs)
        assert outputs.shape == (10,)

        f_imp = model.feature_importance('gain')
        assert isinstance(f_imp, dict)
        assert len(f_imp.keys()) == 10

    def test_cat_train_predict_fimp(self):
        model = GBDT('catboost', cat_params)

        x_train = pd.DataFrame(np.random.rand(100, 10))
        x_train.columns = [str(c) for c in x_train.columns]
        y_train = np.random.randint(0, 2, (100))
        x_valid = pd.DataFrame(np.random.rand(100, 10))
        y_valid = np.random.randint(0, 2, (100))
        model.train(x_train, y_train, x_valid, y_valid)

        dummy_inputs = np.random.rand(10, 10)
        outputs = model.predict(dummy_inputs)
        assert outputs.shape == (10,)

        f_imp = model.feature_importance('gain')
        assert isinstance(f_imp, dict)
        assert len(f_imp.keys()) == 10
