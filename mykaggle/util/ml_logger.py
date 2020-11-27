import os
from typing import Any, List, Dict, Union, Optional, Generator
from pathlib import Path
import contextlib
from enum import Enum
import json
import tensorflow as tf
import numpy as np
import mlflow
import dotenv

from mykaggle.util.slack import Slack


class OutputType(Enum):
    TENSORBOARD = 'tensorboard'
    MLFLOW = 'mlflow'
    STD = 'std'  # for debug


class StatusType(Enum):
    RUNNING = 'RUNNING'
    SUCCESS = 'SUCCESS'
    FAILURE = 'FAILURE'


DEFAULT_OUTPUT_TYPES = [OutputType.MLFLOW]
NECESSARY_ENV = ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'MLFLOW_TRACKING_URI', 'MY_SLACK_WEBHOOK_URL']


class MLLogger:
    '''
    機械学習用途の Logger で、出力先を意識せず使えるようにします。
    現在は TensorBoard と mlflow しかないので雑に分岐で作っています。
    将来増えたりでちゃんとやる場合は各モジュール向けロガークラスとまとめて使うモデルクラスを作ります。
    mlflow で使用する場合はリクエストに必要な情報を .env に記載しておく必要があります。
    '''

    __slots__ = ['_user', '_logdir', '_summary_writer', '_output_types', '_slack']

    def __init__(
        self,
        user: str,
        logdir: Path,
        output_types: List[OutputType] = DEFAULT_OUTPUT_TYPES
    ) -> None:
        '''
        :param user: ユーザの名前
        :param logdir: TensorBoard のログやモデルを保存するディレクトリ
        :param output_types: どこに出力するかのリスト
        '''
        self._user = user
        self._logdir = logdir
        self._output_types = output_types

        # Log に必要な環境変数が入っているかどうかを先に検証
        dotenv.load_dotenv()
        self._assert_env()

        self._slack = Slack()
        if OutputType.TENSORBOARD in output_types:
            self._summary_writer = tf.summary.create_file_writer(str(logdir))

    @contextlib.contextmanager
    def start(self, experiment_name: str, run_name: str) -> Generator:
        if OutputType.MLFLOW in self._output_types:
            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name):
                self.log_running()
                self.log_user(self._user)
                try:
                    yield
                    # 実験が途中で失敗することもあるため、実験前後で notebook は保存する
                    self.log_success()
                    self._slack.notify_success(run_name)
                except (Exception, KeyboardInterrupt) as e:
                    self.log_failure()
                    self._slack.notify_failure(run_name, e)
                    raise
        else:
            yield
        if OutputType.TENSORBOARD in self._output_types:
            self._summary_writer.flush()
            self._summary_writer.close()

    def log_params(self, params: Dict[str, Any]) -> None:
        '''
        モデル学習の際のハイパーパラメータを保存します。
        :param params: ハイパーパラメータの Dict
        '''
        params = dict(params)
        if 'compute_dtype' in params:
            params['compute_dtype'] = 'tf.float16' if params['compute_dtype'] == tf.float16 else 'tf.float32'

        if OutputType.TENSORBOARD in self._output_types:
            json_text = json.dumps(params, indent=4).replace('    ', '&nbsp;&nbsp;&nbsp;&nbsp;').replace('\n', '  \n')
            with self._summary_writer.as_default():
                tf.summary.text('Hyperparameter', json_text, 0)

        if OutputType.MLFLOW in self._output_types:
            mlflow.log_params(params)

        if OutputType.STD in self._output_types:
            print(params)

    def log_tf_metrics(self, metrics: Dict[str, Union[tf.metrics.Metric, tf.Tensor]], step: np.int32) -> None:
        '''
        TensorFlow で学習中に複数の metric のログを取るのに使用します。
        :param metrics: 名前を key に、tf.metrics.Metric か tf.Tensor で与えられる値を持った metric の Dict
        :param step: 現在の step
        '''
        metrics = {k: v if isinstance(v, tf.Tensor) else v.result() for k, v in metrics.items()}
        if OutputType.TENSORBOARD in self._output_types:
            with self._summary_writer.as_default():
                for k, v in metrics.items():
                    tf.summary.scalar(k, v, step=step)

        metrics = {k: v.numpy() for k, v in metrics.items()}
        if OutputType.MLFLOW in self._output_types:
            mlflow.log_metrics(metrics, step=step)

        if OutputType.STD in self._output_types:
            for k, v in metrics.items():
                print(f'{k}: {v}')

    def log_tf_metric(self, name: str, metric: Union[tf.metrics.Metric, tf.Tensor], step: np.int32) -> None:
        '''
        TensorFlow で学習中に単体の metric のログを取るのに使用します。
        :param name: metric の名前
        :param metric: tf.metrics.Metric の型そのままか、tf.Tensor で与えられる値
        :param step: 現在の step
        '''
        value = metric if isinstance(metric, tf.Tensor) else metric.result()
        if OutputType.TENSORBOARD in self._output_types:
            with self._summary_writer.as_default():
                tf.summary.scalar(name, value, step=step)
        if OutputType.MLFLOW in self._output_types:
            mlflow.log_metric(name, value.numpy(), step=step)
        if OutputType.STD in self._output_types:
            print(f'{name}: {value}')

    def log_metrics(self, metrics: Dict[str, Union[np.float32, np.int32]], step: int = 0) -> None:
        '''
        計算した複数の metric のログを取りたいときに使用します。
        学習後などに validation / test set などで計算した metric を保存することを想定しています。
        学習途中の tf.metrics.Metric を使った metric は log_tf_metric を使用してください。
        :param metrics: key が名前、metric の値が value の Dict
        :param step: 現在の step, 学習中出ない場合は 0 で入れれば良い
        '''
        if OutputType.MLFLOW in self._output_types:
            mlflow.log_metrics(metrics, step=step)

    def log_metric(self, name: str, metric: Union[np.float32, np.int32], step: int = 0) -> None:
        '''
        計算した単体の metric のログを取りたいときに使用します。
        学習後などに validation / test set などで計算した metric を保存することを想定しています。
        学習途中の tf.metrics.Metric を使った metric は log_tf_metric を使用してください。
        :param name: metric の名前
        :param metric: metric の値
        :param step: 現在の step, 学習中出ない場合は 0 で入れれば良い
        '''
        if OutputType.MLFLOW in self._output_types:
            mlflow.log_metric(name, metric, step=step)

    def save_model(self, model: tf.keras.Model, filename: str = 'model', artifact_path: Optional[str] = None) -> None:
        '''
        モデルをローカルに保存しつつ、artifacts として mlflow に保存します。
        :param model: モデル
        :param filename: モデルを保存する場合につける名前
        '''
        model.save_weights(str(self._logdir / filename))
        if OutputType.MLFLOW in self._output_types:
            self.log_artifacts(str(self._logdir), artifact_path=artifact_path)

    def log_user(self, user: str) -> None:
        '''
        ユーザ名を記録します。
        :param user: ユーザ名
        '''
        mlflow.set_tag('user', user)

    def log_artifact(self, file: str, artifact_path: Optional[str] = None) -> None:
        '''
        file で与えられたパスにあるファイルを mlflow に artifact として保存します。
        :param file: 保存したいファイルパス
        :artifact_path: 保存先のあるディレクトリ以下保存したい場合に指定する。fold ごとにディレクトリを分けたい場合など。
        '''
        if OutputType.MLFLOW in self._output_types:
            mlflow.log_artifact(file)

    def log_artifacts(self, dir: str, artifact_path: Optional[str] = None) -> None:
        '''
        dir で与えられたディレクトリ以下にあるファイルを mlflow に artifact として保存します。
        :param file: 保存したいファイルパス
        :artifact_path: 保存先のあるディレクトリ以下保存したい場合に指定する。fold ごとにディレクトリを分けたい場合など。
        '''
        if OutputType.MLFLOW in self._output_types:
            mlflow.log_artifacts(dir, artifact_path=artifact_path)

    def log_running(self) -> None:
        '''
        実験が最後まで回りきったことを記録します。
        '''
        self._log_status(StatusType.RUNNING)

    def log_success(self) -> None:
        '''
        実験が最後まで回りきったことを記録します。
        '''
        self._log_status(StatusType.SUCCESS)

    def log_failure(self) -> None:
        '''
        実験が失敗したことを記録します。
        '''
        self._log_status(StatusType.FAILURE)

    def _log_status(self, status: StatusType) -> None:
        if OutputType.MLFLOW in self._output_types:
            mlflow.set_tag('status', status.value)

    def _assert_env(self) -> None:
        if OutputType.MLFLOW not in self._output_types:
            return
        for env in NECESSARY_ENV:
            if env not in os.environ:
                raise EnvironmentError(f'MLLogger need environment variables: {env}.')
