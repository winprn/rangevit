import json
import os
import shutil
import tempfile
from typing import Any, Dict, Optional

import utils.tools as tools


class MLflowManager:
    """Utility wrapper to handle optional MLflow logging."""

    def __init__(self, settings, is_master: Optional[bool] = None):
        self.settings = settings
        self.is_master = tools.is_main_process() if is_master is None else is_master
        self.enabled = bool(getattr(settings, 'mlflow_enable', False)) and self.is_master
        self._mlflow = None
        self._active_run = None

    def start_run(self):
        if not self.enabled or self._active_run is not None:
            return
        try:
            import mlflow
        except ModuleNotFoundError:
            print('WARNING: MLflow is not installed but `mlflow.enable` is True. '
                  'Proceeding without MLflow logging.')
            self.enabled = False
            return

        self._mlflow = mlflow

        tracking_uri = getattr(self.settings, 'mlflow_tracking_uri', None)
        if tracking_uri:
            self._safe_mlflow_call('set tracking URI', mlflow.set_tracking_uri, tracking_uri)
            if not self.enabled:
                return

        experiment_name = getattr(self.settings, 'mlflow_experiment_name', 'RangeViT')
        self._safe_mlflow_call('set experiment', mlflow.set_experiment, experiment_name)
        if not self.enabled:
            return

        tags = dict(getattr(self.settings, 'mlflow_tags', {}))
        description = getattr(self.settings, 'mlflow_description', None)
        if description:
            tags.setdefault('mlflow.note.content', description)

        run = self._safe_mlflow_call(
            'start run',
            mlflow.start_run,
            run_name=getattr(self.settings, 'mlflow_run_name', None),
            nested=bool(getattr(self.settings, 'mlflow_nested', False)),
            tags=tags if tags else None,
        )
        if run is not None and self.enabled:
            self._active_run = run

    def log_settings(self):
        if not self.enabled or self._mlflow is None:
            return
        params = {}
        for key, value in self.settings.__dict__.items():
            if key in ('config', 'mlflow_tags'):
                continue
            serialized = self._serialize_param(value)
            if serialized is not None:
                params[key] = serialized
        if params:
            # MLflow recommends logging at most 100 params.
            for chunk in self._chunk_dict(params, chunk_size=100):
                if not self.enabled:
                    break
                self._safe_mlflow_call('log params', self._mlflow.log_params, chunk)

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        if not self.enabled or self._mlflow is None or not metrics:
            return
        sanitized = {}
        for key, value in metrics.items():
            sanitized_value = self._to_float(value)
            if sanitized_value is not None:
                sanitized[key] = sanitized_value
        if sanitized:
            self._safe_mlflow_call('log metrics', self._mlflow.log_metrics, sanitized, step=step)

    def log_artifact(self, path: str, artifact_path: Optional[str] = None):
        if not self.enabled or self._mlflow is None:
            return
        if path and os.path.exists(path):
            if os.path.isdir(path):
                self._safe_mlflow_call('log artifacts', self._mlflow.log_artifacts, path, artifact_path=artifact_path)
            else:
                self._safe_mlflow_call('log artifact', self._mlflow.log_artifact, path, artifact_path=artifact_path)

    def log_config(self, config: Optional[Dict[str, Any]], artifact_path: str = 'config', filename: str = 'config_used.yaml'):
        if not self.enabled or self._mlflow is None or config is None:
            return
        temp_dir = tempfile.mkdtemp()
        config_filename = filename or 'config_used.yaml'
        config_path = os.path.join(temp_dir, config_filename)
        try:
            try:
                import yaml
            except ModuleNotFoundError:
                yaml = None

            with open(config_path, mode='w') as cfg_file:
                if yaml is not None:
                    yaml.safe_dump(config, cfg_file, sort_keys=False)
                else:
                    json.dump(config, cfg_file, indent=2)

            self.log_artifact(config_path, artifact_path=artifact_path)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def end_run(self, status: str = 'FINISHED'):
        if not self.enabled or self._mlflow is None:
            return
        if self._active_run is not None:
            self._safe_mlflow_call('end run', self._mlflow.end_run, status=status)
            self._active_run = None

    def _safe_mlflow_call(self, action: str, func, *args, **kwargs):
        """Execute MLflow API calls and gracefully handle failures."""
        try:
            return func(*args, **kwargs)
        except Exception as exc:  # noqa: broad-except - best effort logging only
            print(f'WARNING: Failed to {action} via MLflow: {exc}. Disabling MLflow logging.')
            return None

    @staticmethod
    def _serialize_param(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, (str, int, float, bool)):
            return value if isinstance(value, str) else str(value)
        if isinstance(value, (list, tuple)):
            return ','.join([str(v) for v in value])
        if isinstance(value, dict):
            return json.dumps(value)
        return str(value)

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if hasattr(value, 'item'):
            try:
                return float(value.item())
            except (ValueError, TypeError):
                return None
        return None

    @staticmethod
    def _chunk_dict(data: Dict[str, Any], chunk_size: int):
        if chunk_size <= 0:
            yield data
            return
        chunk = {}
        for idx, (key, value) in enumerate(data.items(), start=1):
            chunk[key] = value
            if idx % chunk_size == 0:
                yield chunk
                chunk = {}
        if chunk:
            yield chunk
