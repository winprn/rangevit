import os
import platform
from typing import Any, Dict, Optional


def _import_mlflow():
    try:
        import mlflow  # type: ignore
        return mlflow
    except Exception:
        return None


def is_enabled(tracking_uri: Optional[str] = None) -> bool:
    """MLflow is enabled if a tracking URI is provided or configured via env vars and the package is importable."""
    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    return bool(uri) and _import_mlflow() is not None


def setup(tracking_uri: Optional[str] = None, experiment: Optional[str] = None) -> bool:
    """Configure MLflow tracking URI and experiment.

    Returns True if MLflow is available and configured; False otherwise.
    """
    mlflow = _import_mlflow()
    if mlflow is None:
        return False

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return False

    mlflow.set_tracking_uri(uri)
    exp_name = experiment or os.getenv("MLFLOW_EXPERIMENT", "rangevit")
    mlflow.set_experiment(exp_name)
    return True


def start_run(run_name: Optional[str] = None):
    mlflow = _import_mlflow()
    if mlflow is None:
        return _NullContext()
    return mlflow.start_run(run_name=run_name)


def log_params(params: Dict[str, Any]) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        mlflow.log_params({k: _to_primitive(v) for k, v in params.items()})
    except Exception:
        pass


def set_tags(tags: Dict[str, Any]) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        mlflow.set_tags({k: _to_primitive(v) for k, v in tags.items()})
    except Exception:
        pass


def log_metric(key: str, value: float, step: Optional[int] = None) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        if step is None:
            mlflow.log_metric(key, float(value))
        else:
            mlflow.log_metric(key, float(value), step=step)
    except Exception:
        pass


def log_artifact(path: str, artifact_path: Optional[str] = None) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        if artifact_path:
            mlflow.log_artifact(path, artifact_path=artifact_path)
        else:
            mlflow.log_artifact(path)
    except Exception:
        pass


def log_input_dataset(name: Optional[str], context: str = "training") -> None:
    """Log a dataset input to the current MLflow run.

    Uses mlflow.data.dataset.Dataset to construct a lightweight dataset object
    and logs it via mlflow.log_input. Silently no-ops if MLflow is unavailable
    or if the provided name is empty/None.
    """
    if not name:
        return
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        from mlflow.data.dataset import Dataset  # type: ignore
        ds = Dataset(name=str(name))
        mlflow.log_input(ds, context=context)
    except Exception:
        # Best-effort logging; ignore any MLflow errors
        pass


def log_pytorch_model(model, artifact_path: str = "model", registered_model_name: Optional[str] = None) -> None:
    mlflow = _import_mlflow()
    if mlflow is None:
        return
    try:
        import mlflow.pytorch  # type: ignore

        # Handle DDP-wrapped models
        try:
            to_save = model.module
        except Exception:
            to_save = model

        if registered_model_name:
            mlflow.pytorch.log_model(to_save, artifact_path=artifact_path, registered_model_name=registered_model_name)
        else:
            mlflow.pytorch.log_model(to_save, artifact_path=artifact_path)
    except Exception:
        pass


def default_run_name(model_type: str, run_id: Optional[str]) -> str:
    if run_id:
        return str(run_id)
    host = platform.node() or "host"
    return f"{model_type}-{host}"


def collect_params_from_settings(settings) -> Dict[str, Any]:
    return {
        "model_type": settings.config.get("model_type", "rangevit"),
        "dataset": getattr(settings, "dataset", None),
        "n_classes": getattr(settings, "n_classes", None),
        "image_size": tuple(getattr(settings, "image_size", [])) if hasattr(settings, "image_size") else None,
        "batch_size": getattr(settings, "batch_size", None),
        "batch_size_val": getattr(settings, "batch_size_val", None),
        "lr": getattr(settings, "lr", None),
        "epochs": getattr(settings, "n_epochs", None),
        "optimizer": "AdamW",
        "use_fp16": getattr(settings, "use_fp16", False),
        "val_frequency": getattr(settings, "val_frequency", None),
        "seed": getattr(settings, "seed", None),
    }


def collect_tags_from_settings(settings) -> Dict[str, Any]:
    return {
        "id": getattr(settings, "id", None),
        "host": platform.node(),
        "distributed": getattr(settings, "distributed", False),
        "git_sha": _git_sha(),
    }


def _git_sha() -> Optional[str]:
    try:
        import subprocess
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        return sha
    except Exception:
        return None


def _to_primitive(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    try:
        return float(v)
    except Exception:
        try:
            return str(v)
        except Exception:
            return None


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False
