# Copyright (c) 2024 Snowflake Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Regression net for the streaming-FV authored-cadence contract.

Pinned contract (post `batch_schedule` → `refresh_freq` rename, and post
tiled-streaming carve-out):

* A *non-tiled* `StreamingFeatureView` rejects authored `refresh_freq`
  (the new declarative key) with a validator error pointing at the kind.
  A non-tiled streaming FV compiles to a zero-lag VIEW over the
  `$UDF_TRANSFORMED` pipeline, so an authored refresh cadence would have
  no runtime effect — the validator surfaces this loudly instead of
  silently dropping it at deploy time.
* A *tiled* `StreamingFeatureView` (with `feature_granularity` +
  `features`) ACCEPTS `refresh_freq`: its aggregate materialises as an
  offline Dynamic Table whose refresh cadence is `refresh_freq`, exactly
  as the imperative `FeatureView(refresh_freq=...)` constructor accepts
  it.  (The OFT `target_lag_sec` is a *separate* field that the runtime
  forces to 0 for all streaming kinds — it is not the DT cadence.)
* `StreamingFeatureView` also rejects authored `batch_schedule` (the
  legacy key) with the migration error pointing at `refresh_freq`,
  even though the field is not valid on the kind at all — the rename
  pointer is the most actionable thing to surface (it tells operators
  the field exists under a new name on `BatchFeatureView`, and they
  can decide whether to drop the line entirely or move the cadence to
  a batch FV).

Both errors must fire on:

* the Python authoring form (`StreamingFeatureView.model_validate`),
* the wire-format / YAML loader path (`_dict_to_spec`).

This module replaces the previous legacy regression net
(`test_streaming_fv_replan_drift_with_batch_schedule.py`), whose
fixture authored `batch_schedule: 5 minutes` on a streaming FV — a
shape that is no longer constructable post-rename.

References:

* `decl/spec_models.py::FeatureView._validate_streaming_refresh_freq` —
  validator that allows `refresh_freq` on tiled streaming FVs and rejects
  it on non-tiled streaming / realtime FVs.
* `decl/spec_models.py::FeatureView._reject_legacy_authoring_keys` —
  validator that translates legacy `batch_schedule` into the
  migration error pointing at `refresh_freq`.
* `docs/LIMITATIONS.md` § "StreamingFeatureView / RealtimeFeatureView
  — `target_lag` and `refresh_freq` are not authoring-accepted".
* `declarative_feature_store/RELEASE_NOTES.md` — operator-facing
  migration guide.
"""

from __future__ import annotations

from typing import Any

import pytest

from snowflake.ml.feature_store.decl.loader import _dict_to_spec
from snowflake.ml.feature_store.decl.spec_models import FeatureView
from snowflake.ml.test_utils import pytest_driver


def _streaming_fv_payload(extra: dict[str, Any]) -> dict[str, Any]:
    """Build a minimal authoring payload for a `StreamingFeatureView`.

    Args:
        extra: Extra keys to merge into the payload.  The caller uses
            this to inject the field under test (`refresh_freq` or
            `batch_schedule`) so the surface-name in the error is
            unambiguously the operator's chosen authoring key.

    Returns:
        Streaming FV payload ready for `model_validate` /
        `_dict_to_spec`.
    """
    base = {
        "kind": "StreamingFeatureView",
        "name": "MY_STREAM_FV",
        "entities": ["USER_ID"],
        "sources": [{"name": "src", "source_type": "Stream"}],
    }
    base.update(extra)
    return base


class TestStreamingFvRejectsAuthoredRefreshFreq:
    """`StreamingFeatureView` rejects the new `refresh_freq` key."""

    def test_python_form_rejects_refresh_freq(self) -> None:
        """`FeatureView.model_validate` raises with a streaming-kind pointer."""
        with pytest.raises(ValueError) as exc:
            FeatureView.model_validate(_streaming_fv_payload({"refresh_freq": "5 minutes"}))
        msg = str(exc.value)
        assert "refresh_freq" in msg
        assert "StreamingFeatureView" in msg or "streaming" in msg.lower()

    def test_yaml_loader_rejects_refresh_freq(self) -> None:
        """The YAML / JSON authoring path via `_dict_to_spec` raises too."""
        with pytest.raises(ValueError) as exc:
            _dict_to_spec(_streaming_fv_payload({"refresh_freq": "5 minutes"}))
        msg = str(exc.value)
        assert "refresh_freq" in msg


class TestStreamingFvRejectsLegacyBatchSchedule:
    """`StreamingFeatureView` translates legacy `batch_schedule` into the
    rename migration error.

    The migration error fires before the kind-specific
    `refresh_freq` rejector — operators see the actionable pointer to
    the new key first, then can decide whether to drop the line
    entirely (since `refresh_freq` is itself rejected on streaming FVs).
    """

    def test_python_form_rejects_legacy_batch_schedule(self) -> None:
        """`FeatureView.model_validate` raises with a migration message."""
        with pytest.raises(ValueError) as exc:
            FeatureView.model_validate(_streaming_fv_payload({"batch_schedule": "5 minutes"}))
        msg = str(exc.value)
        assert "batch_schedule" in msg
        assert "has been renamed to" in msg
        assert "refresh_freq" in msg

    def test_yaml_loader_rejects_legacy_batch_schedule(self) -> None:
        """The YAML / JSON authoring path via `_dict_to_spec` raises too.

        `_dict_to_spec` greps the migration-error message for the
        substring "has been renamed to" to decide that a legacy-key
        error must propagate to the user instead of being silently
        degraded to a bare `SpecBase`, so this assertion also pins the
        substring contract that loader path depends on.
        """
        with pytest.raises(ValueError) as exc:
            _dict_to_spec(_streaming_fv_payload({"batch_schedule": "5 minutes"}))
        msg = str(exc.value)
        assert "batch_schedule" in msg
        assert "has been renamed to" in msg
        assert "refresh_freq" in msg


class TestTiledStreamingFvAcceptsRefreshFreq:
    """A tiled `StreamingFeatureView` accepts authored `refresh_freq`."""

    @staticmethod
    def _tiled_payload(extra: dict[str, Any]) -> dict[str, Any]:
        base: dict[str, Any] = {
            "kind": "StreamingFeatureView",
            "name": "MY_TILED_STREAM_FV",
            "entities": ["USER_ID"],
            "timestamp_col": "TIMESTAMP",
            "sources": [{"name": "src", "source_type": "Stream"}],
            "feature_granularity": "1 hour",
            "feature_aggregation_method": "tiles",
            "features": [{"function": "sum", "window": "1h"}],
        }
        base.update(extra)
        return base

    def test_python_form_accepts_refresh_freq_when_tiled(self) -> None:
        """`FeatureView.model_validate` keeps `refresh_freq` for a tiled FV."""
        fv = FeatureView.model_validate(self._tiled_payload({"refresh_freq": "5 minutes"}))
        assert fv.refresh_freq == "5 minutes"

    def test_yaml_loader_accepts_refresh_freq_when_tiled(self) -> None:
        """The YAML / JSON authoring path via `_dict_to_spec` keeps it too."""
        spec = _dict_to_spec(self._tiled_payload({"refresh_freq": "5 minutes"}))
        assert isinstance(spec, FeatureView)
        assert spec.refresh_freq == "5 minutes"


if __name__ == "__main__":
    pytest_driver.main()
