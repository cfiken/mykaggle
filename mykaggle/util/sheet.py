import os
from typing import NamedTuple, Any, Mapping, Optional
import requests
import json
from logging import getLogger


ENV_GS_WEBHOOK_URL = 'GS_WEBHOOK_URL'


class SheetContents(NamedTuple):  # Python 3.7 になったら dataclass に変更したい

    sheet: str
    title: str = ''
    cv: float = 0
    description: str = ''
    parameters: str = ''

    def to_dict(self):
        params = {}
        params['sheet'] = self.sheet
        params['title'] = self.title
        params['cv'] = self.cv
        params['description'] = self.description
        params['parameters'] = self.parameters
        return params

    @classmethod
    def parameter_dict_to_str(cls, parameters: Mapping[str, Any]):
        params_text = ""
        for i, (k, v) in enumerate(list(parameters.items())):
            if i != 0:
                params_text += '\n'
            params_text += f'{k}: {v}'
        return params_text


class Sheet:

    __slots__ = ['_url', '_logger']

    def __init__(self, url: Optional[str] = None):
        if url is None:
            url = os.environ[ENV_GS_WEBHOOK_URL]
        if url is None:
            raise ValueError('webhook url is needed')
        self._url = url
        self._logger = getLogger(__name__)

    def post(self, sheet: str, title: str, cv: float, description: str, parameters: Mapping[str, Any]):
        params_text = SheetContents.parameter_dict_to_str(parameters)
        contents = SheetContents(sheet, title, cv, description, params_text)
        self._post(contents)

    def post_tf2nq(self,
                   title: str,
                   cv: float,
                   description: str,
                   parameters: Mapping[str, Any]):
        self.post('TF2NQ', title, cv, description, parameters)

    def _post(self, contents: SheetContents):
        payload = contents.to_dict()
        response = requests.post(self._url, data=json.dumps(payload))
        self._logger.info(response.reason)
        return response

    @classmethod
    def create(cls, url: str) -> 'Sheet':
        return cls(url)


if __name__ == '__main__':
    # test run
    from toguro.lib.logger import get_logger
    logger = get_logger('common.lib.sheet')
    url = os.environ[ENV_GS_WEBHOOK_URL]
    sheet = Sheet(url)
    sheet.post_severstal('モデルA', 0.528, 'Aを使ったモデルだよ', {'batch_size': 32, 'num_epochs': 100})
