# mypy: disable-error-code="import"
"""TabICL adapter: foundation is a single ``.ckpt`` bundled and re-bound via ``model_path``."""

import os
import shutil
import tempfile
from typing import Any

from snowflake.ml.model._packager.model_handlers.tabular_foundation_model_adapters import (
    base,
)
from snowflake.ml.model._packager.model_meta import model_meta as model_meta_api


class TabICLAdapter(base.ModelAdapter):
    """TabICL: foundation is a single ``.ckpt`` bundled and re-bound via ``model_path``."""

    ADAPTER_ID = "tabicl"
    MODULE = "tabicl"
    CLASSIFIER_CLASS = "TabICLClassifier"
    REGRESSOR_CLASS = "TabICLRegressor"
    # ``model_path`` / ``allow_auto_download`` / ``device`` are rebound at load to point at
    # the bundled checkpoint and disable network access; the saved values are irrelevant.
    EXCLUDED_INIT_PARAMS = ("model_path", "allow_auto_download", "device")

    _HF_REPO_ID = "jingang/TabICL"
    _CHECKPOINT_FILENAME = "checkpoint.ckpt"

    @classmethod
    def save_foundation(cls, model: Any, foundation_dir: str) -> None:
        # Prefer an explicit local checkpoint; otherwise resolve the model's checkpoint
        # from the Hugging Face hub so the bundle is fully self-contained (offline serving).
        local_path = getattr(model, "model_path", None)
        if local_path and os.path.isfile(local_path):
            checkpoint_source = local_path
        else:
            from huggingface_hub import hf_hub_download

            checkpoint_version = getattr(model, "checkpoint_version", None)
            if not checkpoint_version:
                raise ValueError(
                    "TabICL model has neither a local model_path nor a checkpoint_version to resolve; "
                    "set model_path to a downloaded .ckpt before logging."
                )
            checkpoint_source = hf_hub_download(repo_id=cls._HF_REPO_ID, filename=checkpoint_version)
        shutil.copyfile(checkpoint_source, os.path.join(foundation_dir, cls._CHECKPOINT_FILENAME))

    @classmethod
    def load_estimator(
        cls,
        *,
        estimator_type: str,
        model_type: str,
        foundation_dir: str,
        init_params: dict[str, Any],
        device: str,
    ) -> base.TabularModel:
        # tabicl loads numba code compiled with @njit(cache=True); numba's default cache location sits
        # next to the installed package, which is read-only in the serving sandbox, so the import fails
        # with "no locator available". Point numba at a writable dir before it is first imported here.
        os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join(tempfile.gettempdir(), "numba_cache"))

        from tabicl import TabICLClassifier, TabICLRegressor

        checkpoint_path = base.assert_path_within_directory(
            os.path.join(foundation_dir, cls._CHECKPOINT_FILENAME), foundation_dir
        )
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"TabICL checkpoint not found at {checkpoint_path}")

        kwargs = {key: value for key, value in init_params.items() if key not in cls.EXCLUDED_INIT_PARAMS}
        common = dict(model_path=checkpoint_path, allow_auto_download=False, device=device, **kwargs)
        if estimator_type == cls.CLASSIFIER_CLASS:
            return TabICLClassifier(**common)
        if estimator_type == cls.REGRESSOR_CLASS:
            return TabICLRegressor(**common)
        raise ValueError(f"Unsupported TabICL estimator type: {estimator_type}")

    @classmethod
    def add_dependencies(cls, model_meta: model_meta_api.ModelMetadata) -> None:
        # tabicl is PyPI-only and pulls torch + huggingface_hub transitively.
        model_meta.env._include_if_absent_pip(["tabicl"], check_local_version=False)
        model_meta.env._include_if_absent_pip(["scikit-learn>=1.6,<1.9"], check_local_version=False)
