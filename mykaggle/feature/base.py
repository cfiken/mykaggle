from abc import ABCMeta, abstractmethod
from typing import Dict, Optional
import pandas as pd
from pathlib import Path

from mykaggle.util.logger import get_logger

FEATURE_DIR = Path('../data/feature/')
logger = get_logger(__name__)


class Feature(metaclass=ABCMeta):
    '''
    ひとまとまりの特徴を表すベースクラス。
    特徴を作成する create を実装することで、キャッシュ付き特徴作成クラスとして使える。
    '''

    @abstractmethod
    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        '''
        base となる DataFrame とその他 DataFrame を組み合わせて特徴を作るメソッド。
        :params base: 最終的に merge する index を含む DataFrame
        :params others: 特徴を作るための他の Base 以外の DataFrame, dict の形で渡す
        '''
        pass

    def __init__(self, name: str, train: bool = True, category: Optional[str] = None) -> None:
        '''
        :params name: 特徴の名前 e.g.) gender, age
        :params train: train 用の特徴であれば True, test 用であれば False
        :params category: 特徴をまとめる dir を作る場合指定する e.g.) 特定コンペの名前など
        '''
        self.name = name
        self.train = train
        self.name_prefix = 'train' if train else 'test'
        self.category = category

    def __call__(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        use_cache: bool = False,
        save_cache: bool = False,
        merge: bool = True,
        *args, **kwargs
    ) -> pd.DataFrame:
        '''
        特徴を実際に使うときに呼ぶメソッド。
        前後にキャッシュとして特徴を保存する/キャッシュされた特徴をロードするようにしている。
        :params base: 最終的に merge する index を含む DataFrame
        :params others: 特徴を作るための他の Base 以外の DataFrame, dict の形で渡す
        :params use_cache: キャッシュを使うかどうか
        :params save_cache: 作成した特徴を保存するかどうか
        '''
        if use_cache:
            if not self._path.exists():
                logger.info(f'Creating {self.name_prefix}_{self.name} since it has not been created yet.')
            else:
                feature = self._load()
        feature = self.create(base, others, *args, **kwargs)
        if save_cache:
            self._save(feature)
        if merge:
            output = pd.merge(base, feature, how=self._merge_how, on=self._merge_on)
        else:
            output = pd.concat([base, feature], axis=1)
        return output

    def _load(self) -> pd.DataFrame:
        return pd.read_feather(self._path)

    def _save(self, df: pd.DataFrame) -> None:
        '''
        作った特徴を保存する。特徴保存先がない場合は作成する。
        :params df: 保存する特徴の DataFrame
        '''
        if not self._path.parent.exists():
            self._path.parent.mkdir(parents=True, exist_ok=True)
        df.to_feather(self._path)

    @property
    def _path(self) -> Path:
        name = self.name + '.ftr'
        if self.category is not None:
            return FEATURE_DIR / self.category / self.name_prefix / name
        return FEATURE_DIR / self.name_prefix / name

    @property
    def _merge_how(self) -> str:
        return 'left'

    @property
    def _merge_on(self) -> str:
        return 'id'
