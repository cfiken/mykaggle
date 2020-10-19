import os
from typing import Optional, Union
import requests
from requests import Response
import json


class Slack:

    channel_name = '#kaggle-log'
    env_webhook_url = 'MY_SLACK_WEBHOOK_URL'

    __slots__ = ['_url', '_logger']

    def __init__(self, web_hook_url: Optional[str] = None) -> None:
        if web_hook_url is None and self.env_webhook_url in os.environ:
            web_hook_url = os.environ[self.env_webhook_url]
        if web_hook_url is None:
            raise ValueError('Set web_hook_url by args or environment variable.')
        self._url = web_hook_url

    def notify_success(self, job_name: str) -> None:
        message = f'{job_name} が終わったようだな'
        self._notify(message)

    def notify_failed(self, job_name: str, error: Optional[Union[Exception, str]] = None) -> None:
        message = f'{job_name} failed. なぜだぁ なぜ死んだァ~~~~\n'
        if error:
            message += f'ERROR: {error}'
        self._notify(message)

    def _notify(self, message: str) -> Response:
        payload = {}
        payload['text'] = message
        response = requests.post(self._url, data=json.dumps(payload))
        self._logger.debug(response)
        return response
