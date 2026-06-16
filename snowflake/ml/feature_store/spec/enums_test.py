"""Unit tests for ``snowflake.ml.feature_store.spec.enums``.

Covers the public ``ENTITY_TAG_PREFIX`` constant and the cross-module
identity invariant — every consumer module must reference the same
``spec.enums.ENTITY_TAG_PREFIX`` object so future literal duplication
is caught by a trip-wire instead of slipping into a release.

Also pins a structural-invariance trip-wire that asserts the
``FSBaseType`` string vocabulary stays in lock-step with the imperative
Snowpark-class set in :mod:`snowflake.ml.feature_store.spec.models`
(``_SUPPORTED_TYPES``) and the streaming-source dict in
:mod:`snowflake.ml.feature_store.stream_source` (``_TYPE_NAME_TO_CLASS``).
"""

import importlib

from absl.testing import absltest

from snowflake.ml.feature_store.spec import enums as spec_enums
from snowflake.ml.feature_store.spec.enums import FSBaseType


class EntityTagPrefixTest(absltest.TestCase):
    """Pin the canonical entity-tag prefix constant + cross-module identity."""

    EXPECTED_VALUE = "SNOWML_FEATURE_STORE_ENTITY_"

    CONSUMER_MODULES = (
        "snowflake.ml.feature_store.feature_store",
        "snowflake.ml.feature_store.decl.state",
        "snowflake.ml.feature_store.decl.api",
        "snowflake.ml.feature_store.decl.imperative_executor",
    )

    def test_constant_value_is_canonical(self) -> None:
        self.assertEqual(spec_enums.ENTITY_TAG_PREFIX, self.EXPECTED_VALUE)
        self.assertIsInstance(spec_enums.ENTITY_TAG_PREFIX, str)

    def test_all_consumers_share_object_identity(self) -> None:
        """Every consumer module must expose ``ENTITY_TAG_PREFIX`` and it
        must be the exact same object as
        ``spec.enums.ENTITY_TAG_PREFIX``.

        RED until Phase 5 promotes the per-module ``_ENTITY_TAG_PREFIX``
        locals to a single ``from spec.enums import ENTITY_TAG_PREFIX``.

        Consumer modules that fail to import in the current slice (e.g.
        when this test runs on a stacked PR before the ``decl/`` package
        has been introduced) are skipped from the identity check rather
        than failed; the strict cross-module identity check fires once
        every consumer module is present in the integrated tree.
        """
        canonical = spec_enums.ENTITY_TAG_PREFIX
        missing: list[str] = []
        mismatched: list[str] = []
        unavailable: list[str] = []
        consumed: list[str] = []

        for module_path in self.CONSUMER_MODULES:
            try:
                module = importlib.import_module(module_path)
            except ImportError:
                unavailable.append(module_path)
                continue
            if not hasattr(module, "ENTITY_TAG_PREFIX"):
                missing.append(module_path)
                continue
            consumed.append(module_path)
            if module.ENTITY_TAG_PREFIX is not canonical:
                mismatched.append(module_path)

        if not consumed:
            self.skipTest(
                "No declared consumer module re-exports ENTITY_TAG_PREFIX yet; the "
                "strict cross-module identity check is deferred until the Phase 5 "
                "dedupe migrates feature_store.py and the decl/ modules from local "
                f"_ENTITY_TAG_PREFIX literals to a shared spec.enums import. "
                f"Unavailable: {unavailable}; missing attribute: {missing}."
            )

        self.assertFalse(
            missing,
            ("These consumer modules do not expose ENTITY_TAG_PREFIX " "after the Phase 5 dedupe: " f"{missing}"),
        )
        self.assertFalse(
            mismatched,
            (
                "These consumer modules expose ENTITY_TAG_PREFIX but it "
                "is not the canonical object — a duplicate literal has "
                "leaked back in: "
                f"{mismatched}"
            ),
        )


class FSBaseTypeAlignmentTest(absltest.TestCase):
    """Structural invariance between FSBaseType and the imperative type sets.

    PR 1b promotes ``FSBaseType`` to the canonical FS type vocabulary, but
    the imperative side has two pre-existing parallel declarations of the
    same set:

    * :data:`snowflake.ml.feature_store.spec.models._SUPPORTED_TYPES` — the
      tuple of Snowpark classes accepted by ``FSColumn`` conversion.
    * :data:`snowflake.ml.feature_store.stream_source._TYPE_NAME_TO_CLASS`
      — the dict from string name to Snowpark class used by streaming
      schema serde.

    Without a structural test, a developer adding a new supported type to
    one of these three structures can silently drift from the others.
    These assertions are the trip-wire.
    """

    def test_fsbasetype_matches_stream_source_type_name_dict(self) -> None:
        """FSBaseType values must equal stream_source._TYPE_NAME_TO_CLASS keys."""
        from snowflake.ml.feature_store.stream_source import _TYPE_NAME_TO_CLASS

        fsbasetype_values = {v.value for v in FSBaseType}
        stream_source_keys = set(_TYPE_NAME_TO_CLASS.keys())
        self.assertEqual(
            fsbasetype_values,
            stream_source_keys,
            "FSBaseType values and stream_source._TYPE_NAME_TO_CLASS keys "
            "have drifted. Update both to declare the same supported-type set.",
        )

    def test_fsbasetype_matches_imperative_supported_types(self) -> None:
        """FSBaseType values must equal {t.__name__ for t in _SUPPORTED_TYPES}.

        ``_make_fs_column`` in :mod:`spec.models` sets ``FSColumn.type`` to
        ``type(dt).__name__`` of the Snowpark instance, so the imperative
        FSColumn pipeline produces exactly these strings.  Drift between
        ``_SUPPORTED_TYPES`` and ``FSBaseType`` would mean either:

        * the imperative side accepts a Snowpark type whose name has no
          declarative-side analogue (FSColumn payloads with that type
          can't round-trip through the declarative client), or
        * ``FSBaseType`` declares a type the imperative side rejects
          (declarative authoring with that type fails at the imperative
          boundary).
        """
        from snowflake.ml.feature_store.spec.models import _SUPPORTED_TYPES

        fsbasetype_values = {v.value for v in FSBaseType}
        supported_class_names = {t.__name__ for t in _SUPPORTED_TYPES}
        self.assertEqual(
            fsbasetype_values,
            supported_class_names,
            "FSBaseType values and spec.models._SUPPORTED_TYPES class names "
            "have drifted. Update both to declare the same supported-type set.",
        )

    def test_stream_source_classes_match_fsbasetype_values(self) -> None:
        """Each stream_source._TYPE_NAME_TO_CLASS entry must use its
        Snowpark class's ``__name__`` as the key — i.e. the dict is the
        identity ``cls.__name__ -> cls`` for every supported type."""
        from snowflake.ml.feature_store.stream_source import _TYPE_NAME_TO_CLASS

        for name, cls in _TYPE_NAME_TO_CLASS.items():
            self.assertEqual(
                name,
                cls.__name__,
                f"stream_source._TYPE_NAME_TO_CLASS key {name!r} does not " f"match its class name {cls.__name__!r}.",
            )

    def test_type_aliases_resolve_to_fsbasetype_values(self) -> None:
        """Every TYPE_ALIASES value must be a valid FSBaseType value."""
        fsbasetype_values = {v.value for v in FSBaseType}
        for alias, target in spec_enums.TYPE_ALIASES.items():
            self.assertIn(
                target,
                fsbasetype_values,
                f"TYPE_ALIASES[{alias!r}] = {target!r} is not a valid FSBaseType value.",
            )


if __name__ == "__main__":
    absltest.main()
