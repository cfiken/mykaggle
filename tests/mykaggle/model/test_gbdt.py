import numpy as np
import pandas as pd

from mykaggle.model.gbdt import GBDT, ModelType


class TestGBDT:

    def test_init(self):
        model1 = GBDT('lightgbm', {})
        assert model1._type == ModelType.LightGBM
        assert model1._is_lightgbm
        model2 = GBDT('xgboost', {})
        assert model2._type == ModelType.XGBoost
        assert model2._is_xgboost
        model3 = GBDT('catboost', {})
        assert model3._type == ModelType.CatBoost
        assert model3._is_catboost

    def test_train(self):
        model1 = GBDT('lightgbm', {})
        model2 = GBDT('xgboost', {})
        model3 = GBDT('catboost', {})

        x_train = pd.DataFrame(np.random.rand(10, 10))
        y_train = np.random.randint(0, 2, (10))
        x_valid = pd.DataFrame(np.random.rand(10, 10))
        y_valid = np.random.randint(0, 2, (10))
        model1.train(x_train, y_train, x_valid, y_valid)
        model2.train(x_train, y_train, x_valid, y_valid)
        model3.train(x_train, y_train, x_valid, y_valid)

    def test_params_as_params(self):
        pass

    def test_params_as_kwargs(self):
        pass

    def test_predict(self):
        pass

    def test_feature_importance(self):
        pass
