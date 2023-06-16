import os
import yaml
from typing import Any, Generator, Optional
from pathlib import Path
import mlflow
import torch
from torch import nn
import dotenv
import contextlib

from mykaggle.lib.slack import Slack
from mykaggle.lib.routine import get_logger
from mykaggle.lib.logger.logger import Logger, StatusType

logger = get_logger(__name__)


NECESSARY_ENV = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'MLFLOW_TRACKING_URI', 'MY_SLACK_WEBHOOK_URL']


class MLLogger(Logger):
    '''
    機械学習用途の mlflow 先に出力する Logger です。
    リクエストに必要な情報を .env に記載しておく必要があります。
    '''

    __slots__ = ['_user', '_logdir', '_slack']

    def __init__(
        self,
        user: str,
        logdir: Path,
    ) -> None:
        '''
        :param user: ユーザの名前
        :param logdir: ログやモデルを保存するディレクトリ
        '''
        super().__init__(user, logdir)

        # Log に必要な環境変数が入っているかどうかを先に検証
        dotenv.load_dotenv()
        self._assert_env()
        self._slack = Slack()

    @contextlib.contextmanager
    def start(self, experiment_name: str, run_name: str) -> Generator:
        mlflow.set_experiment(experiment_name)  # type: ignore
        with mlflow.start_run(run_name=run_name):  # type: ignore
            self._log_running()
            self._log_user(self._user)
            try:
                yield
                # 実験が途中で失敗することもあるため、実験前後で notebook は保存する
                self._log_success()
                self._slack.notify_success(run_name)
            except (Exception, KeyboardInterrupt) as e:
                self._log_failure()
                self._slack.notify_failure(run_name, e)
                raise

    def log_params(self, params: dict[str, Any]) -> None:
        '''
        モデル学習の際のハイパーパラメータを保存します。
        :param params: ハイパーパラメータの Dict
        '''
        params = dict(params)
        mlflow.log_params(params)  # type: ignore

    def log_metrics(self, metrics: dict[str, float], step: int = 0) -> None:
        '''
        計算した複数の metric のログを取りたいときに使用します。
        学習後などに validation / test set などで計算した metric を保存することを想定しています。
        :param metrics: key が名前、metric の値が value の Dict
        :param step: 現在の step, 学習中出ない場合は 0 で入れれば良い
        '''
        mlflow.log_metrics(metrics, step=step)  # type: ignore

    def log_metric(self, name: str, metric: float, step: int = 0) -> None:
        '''
        計算した単体の metric のログを取りたいときに使用します。
        学習後などに validation / test set などで計算した metric を保存することを想定しています。
        :param name: metric の名前
        :param metric: metric の値
        :param step: 現在の step, 学習中出ない場合は 0 で入れれば良い
        '''
        mlflow.log_metric(name, metric, step=step)  # type: ignore

    def save_file(
        self,
        model,
        filename: str = 'model',
        artifact_path: Optional[str] = None,
        upload: bool = False
    ):
        pass

    def upload_model(self, filename: str, artifact_path: Optional[str] = None) -> None:
        self._log_artifact(str(self._logdir / filename), artifact_path=artifact_path)

    def save_model(
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
            self._log_artifacts(str(self._logdir), artifact_path=artifact_path)

    def _log_user(self, user: str) -> None:
        '''
        ユーザ名を記録します。
        :param user: ユーザ名
        '''
        mlflow.set_tag('user', user)  # type: ignore

    def _log_artifact(self, file: str, artifact_path: Optional[str] = None) -> None:
        '''
        file で与えられたパスにあるファイルを mlflow に artifact として保存します。
        :param file: 保存したいファイルパス
        :artifact_path: 保存先のあるディレクトリ以下保存したい場合に指定する。fold ごとにディレクトリを分けたい場合など。
        '''
        try:
            mlflow.log_artifact(file, artifact_path=artifact_path)  # type: ignore
        except Exception as e:
            logger.error(f'ERROR: log_artifacts failed. error: {e}')

    def _log_artifacts(self, dir: str, artifact_path: Optional[str] = None) -> None:
        '''
        dir で与えられたディレクトリ以下にあるファイルを mlflow に artifact として保存します。
        :param file: 保存したいファイルパス
        :artifact_path: 保存先のあるディレクトリ以下保存したい場合に指定する。fold ごとにディレクトリを分けたい場合など。
        '''
        try:
            mlflow.log_artifacts(dir, artifact_path=artifact_path)  # type: ignore
        except Exception as e:
            logger.error(f'ERROR: log_artifacts failed. error: {e}')

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
        self._log_artifact(str(filepath))

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
        mlflow.set_tag('status', status.value)  # type: ignore

    def _assert_env(self) -> None:
        for env in NECESSARY_ENV:
            if env not in os.environ:
                raise EnvironmentError(f'MLLogger need environment variables: {env}.')
