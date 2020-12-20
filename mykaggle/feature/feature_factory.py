from typing import Any, List, Dict
import pandas as pd
from mykaggle.feature.base import Feature


class FeatureFactory:

    @classmethod
    def run(
        cls,
        names: List[str],
        params: Dict[str, Dict[str, Any]],
        base: pd.DataFrame,
        others: pd.DataFrame,
        train: bool,
        use_cache: bool = False,
        save_cache: bool = False
    ) -> pd.DataFrame:
        '''
        与えられた特徴の名前とパラメータのリストから train/test の特徴を作って返す
        '''
        features = cls.get_features(names, params, train=train)
        df = cls.create_feature(features, base, others, use_cache, save_cache)
        return df

    @classmethod
    def get_features(cls, names: List[str], params: Dict[str, Dict[str, Any]], train: bool) -> List[Feature]:
        '''
        Args:
          names: 特徴の名前のリスト
          params: 特徴の constructor に渡すパラメータ
          train: 学習用かテスト用か
        '''
        features = []
        for name in names:
            constructor = Feature.by_name(name)
            param = params.get(name, {})
            f = constructor(train=train, **param)
            features.append(f)
        return features

    @classmethod
    def create_feature(
        cls,
        features: List[Feature],
        base: pd.DataFrame,
        others: Dict[str, pd.DataFrame],
        use_cache: bool = False,
        save_cache: bool = False
    ) -> pd.DataFrame:
        '''
        与えられた特徴クラスのリストから特徴の DataFrame を作って返す
        '''
        df = base.copy()
        for f in features:
            df = f(df, others=others, use_cache=use_cache, save_cache=save_cache)
        return df
