from typing import Any, Generator, Optional
import yaml
from pathlib import Path
import torch
from torch import nn
import contextlib

from mykaggle.lib.slack import Slack
from mykaggle.lib.routine import get_logger
from mykaggle.lib.logger.logger import Logger, StatusType

LOGGER = get_logger(__name__)


NECESSARY_ENV = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'MLFLOW_TRACKING_URI', 'MY_SLACK_WEBHOOK_URL']


class StdLogger(Logger):
    '''
    機械学習用途の Logger で、出力先を意識せず使えるようにします。
    現在は TensorBoard と mlflow しかないので雑に分岐で作っています。
    将来増えたりでちゃんとやる場合は各モジュール向けロガークラスとまとめて使うモデルクラスを作ります。
    mlflow で使用する場合はリクエストに必要な情報を .env に記載しておく必要があります。
    '''

    __slots__ = ['_user', '_logdir', '_slack']

    def __init__(
        self,
        user: str,
        logdir: Path,
    ) -> None:
        '''
        :param user: ユーザの名前
        :param logdir: TensorBoard のログやモデルを保存するディレクトリ
        :param output_types: どこに出力するかのリスト
        '''
        super().__init__(user, logdir)
        self._slack = Slack()

    @contextlib.contextmanager
    def start(self, experiment_name: str, run_name: str) -> Generator:
        LOGGER.info(f'START experiment for {experiment_name}')
        LOGGER.info(f'Run {run_name} by {self._user}')
        try:
            yield
            self._log_success()
            self._slack.notify_success(run_name)
        except (Exception, KeyboardInterrupt) as e:
            self._log_failure()
            self._slack.notify_failure(run_name, e)
            raise e
        LOGGER.info(f'FINISHED experiment for {experiment_name} with name {run_name}')

    def log_params(self, params: dict[str, Any]) -> None:
        '''
        モデル学習の際のハイパーパラメータを保存します。
        :param params: ハイパーパラメータの Dict
        '''
        params = dict(params)
        LOGGER.info(f'Parameters: {params}')

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        '''
        計算した複数の metric のログを取りたいときに使用します。
        学習後などに validation / test set などで計算した metric を保存することを想定しています。
        学習途中の tf.metrics.Metric を使った metric は log_tf_metric を使用してください。
        :param metrics: key が名前、metric の値が value の Dict
        :param step: 現在の step, 学習中出ない場合は 0 で入れれば良い
        '''
        LOGGER.info(f'{metrics} at {step}')

    def log_metric(self, name: str, metric: float, step: int = 0) -> None:
        '''
        計算した単体の metric のログを取りたいときに使用します。
        学習後などに validation / test set などで計算した metric を保存することを想定しています。
        学習途中の tf.metrics.Metric を使った metric は log_tf_metric を使用してください。
        :param name: metric の名前
        :param metric: metric の値
        :param step: 現在の step, 学習中出ない場合は 0 で入れれば良い
        '''
        LOGGER.info(f'{name}:{metric} at {step}')

    def save_config(self, settings: dict[str, Any]) -> None:
        filepath = self._logdir / 'config.yml'

        def serialize(obj: dict[str, Any]) -> dict[str, Any]:
            new_obj: dict[str, Any] = {}
            for k, v in obj.items():
                if isinstance(v, dict):
                    new_obj[k] = serialize(v)
                elif isinstance(v, (str, int, float, bool, list, tuple)):
                    new_obj[k] = v
                else:
                    new_obj[k] = str(v)
            return new_obj

        with open(filepath, 'w') as f:
            yaml.dump(serialize(settings), f)

    def save_model(
        self,
        model,
        filename: str = 'model',
        artifact_path: Optional[str] = None,
        upload: bool = False
    ):
        if isinstance(model, nn.Module):
            self._save_torch_model(model, filename, artifact_path, upload)
        else:
            self._save_file(model, filename, artifact_path, upload)

    def upload_model(self, filename: str, artifact_path: Optional[str] = None) -> None:
        LOGGER.warn('LoggerType: STD has no function for upload.')

    def _save_torch_model(
        self,
        model: nn.Module,
        filename: str = 'model',
        artifact_path: Optional[str] = None,
        upload: bool = False
    ) -> None:
        '''
        モデルをローカルに保存しつつ、artifacts として mlflow に保存します。
        :param model: モデル
        :param filename: モデルを保存する場合につける名前
        '''
        torch.save(model.state_dict(), str(self._logdir / filename))
        if upload:
            self.upload_model(filename, artifact_path)

    def _save_file(
        self,
        file,
        filename: str,
        artifact_path: Optional[str] = None,
        upload: bool = False
    ) -> None:
        pass

    def _log_running(self) -> None:
        '''
        実験が最後まで回りきったことを記録します。
        '''
        self._log_status(StatusType.RUNNING)

    def _log_success(self) -> None:
        '''
        実験が最後まで回りきったことを記録します。
        '''
        self._log_status(StatusType.SUCCESS)

    def _log_failure(self) -> None:
        '''
        実験が失敗したことを記録します。
        '''
        self._log_status(StatusType.FAILURE)

    def _log_status(self, status: StatusType) -> None:
        LOGGER.info(f'status: {status}')
