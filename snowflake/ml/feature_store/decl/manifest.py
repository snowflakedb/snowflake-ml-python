"""Feature-store ``manifest.yml`` model and parser (Phase 1A).

This module mirrors the structure of the Snowflake CLI DCM reference
model but is self-contained — it imports only the Python standard
library and ``pyyaml``. Per the project's architecture boundaries
(see ``docs/ARCHITECTURE.md``), the declarative library at this layer
must not pull in Snowpark, the Snowflake connector, the rest of the
feature-store package, or any CLI plugin code.

The locked design decisions enforced here are recorded in
``plans/MANIFEST_YML_LAYOUT_DECISIONS.md``:

* **D2 (with-role).** Per-target required fields are
  ``account_identifier``, ``database``, ``schema``. ``role`` and
  ``templating_config`` are optional. ``warehouse`` is NOT a manifest
  field — it is sourced from the active connection. A target that
  declares ``warehouse:`` is rejected with
  :class:`ManifestConfigurationError`.
* **D5 (drop-config).** Template configurations are case-insensitive
  on read and canonicalized to UPPERCASE (mirrors DCM behavior).
* **Manifest constants.** ``MANIFEST_TYPE = "feature_store"`` and
  ``SUPPORTED_MANIFEST_VERSION = 1``.

Exception classes (:class:`ManifestNotFoundError`,
:class:`InvalidManifestError`, :class:`ManifestConfigurationError`)
are defined inline in this module by design. ``decl/errors.py`` is
intentionally left untouched in this phase.
"""

from __future__ import annotations

import logging
import pathlib
from dataclasses import dataclass, field
from typing import Any, Optional

import yaml

MANIFEST_FILE_NAME = "manifest.yml"
MANIFEST_TYPE = "feature_store"
SOURCES_FOLDER = "sources"
SUPPORTED_MANIFEST_VERSION = 1
PLANS_SUBPATH = ("out", "plan")

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Exceptions (intentionally inline; do NOT move to decl/errors.py)
# ---------------------------------------------------------------------------


class ManifestNotFoundError(Exception):
    """Raised when ``manifest.yml`` is absent from the expected directory."""


class InvalidManifestError(Exception):
    """Raised when ``manifest.yml`` is malformed, unparsable, or has an
    unsupported manifest_version or project type."""


class ManifestConfigurationError(Exception):
    """Raised when a manifest's structure parses but its content is
    semantically invalid — for example a target missing a required
    field, a target declaring the forbidden ``warehouse`` field, a
    ``default_target`` pointing at an undefined target, or a target
    whose ``templating_config`` does not match any defined
    configuration."""


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class FSTemplating:
    """Templating block for the feature-store manifest (D5).

    Shape: ``{defaults: {...}, configurations: {<NAME>: {...}}}``.
    Configuration names are uppercased on load to match DCM behavior.
    """

    defaults: dict[str, Any] = field(default_factory=dict)
    configurations: dict[str, dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Optional[dict[str, Any]]) -> FSTemplating:
        if not data:
            return cls()
        configurations = data.get("configurations") or {}
        return cls(
            defaults=data.get("defaults") or {},
            configurations={str(k).upper(): v for k, v in configurations.items()},
        )


@dataclass
class FSTarget:
    """A single feature-store target (per D2: with-role).

    Required fields: ``account_identifier``, ``database``, ``schema``.
    Optional fields: ``role`` (identifier-normalized via ``upper()``),
    ``templating_config`` (uppercased on load).
    """

    name: str
    account_identifier: str
    database: str
    schema: str
    role: str = ""
    templating_config: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FSTarget:
        name = str(data.get("name", "")).upper()

        if "warehouse" in data:
            raise ManifestConfigurationError(
                f"Target '{name}' has unsupported field 'warehouse'. "
                "Warehouse comes from the active connection (per D2)."
            )

        account_identifier = data.get("account_identifier")
        database = data.get("database")
        schema = data.get("schema")

        missing = []
        if not account_identifier:
            missing.append("account_identifier")
        if not database:
            missing.append("database")
        if not schema:
            missing.append("schema")
        if missing:
            raise ManifestConfigurationError(f"Target '{name}' is missing required field(s): {', '.join(missing)}.")

        role = data.get("role") or ""
        templating_config = data.get("templating_config")

        return cls(
            name=name,
            account_identifier=str(account_identifier),
            database=str(database),
            schema=str(schema),
            role=str(role).upper() if role else "",
            templating_config=str(templating_config).upper() if templating_config else None,
        )


@dataclass
class FSManifest:
    """Parsed ``manifest.yml`` for a feature-store project."""

    manifest_version: int
    project_type: str
    default_target: Optional[str] = None
    targets: dict[str, FSTarget] = field(default_factory=dict)
    templating: FSTemplating = field(default_factory=FSTemplating)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FSManifest:
        targets_data = data.get("targets") or {}
        targets: dict[str, FSTarget] = {}
        for raw_name, target_data in targets_data.items():
            upper_name = str(raw_name).upper()
            merged = dict(target_data or {})
            merged["name"] = upper_name
            targets[upper_name] = FSTarget.from_dict(merged)

        default_target = data.get("default_target")
        if default_target is None and len(targets) == 1:
            default_target = next(iter(targets.keys()))
            log.info(
                "Derived default target from single-target manifest (default_target=%s).",
                default_target,
            )

        manifest_version_raw = data.get("manifest_version")
        if manifest_version_raw is None:
            raise InvalidManifestError(
                "Manifest version is undefined. " f"Expected manifest_version={SUPPORTED_MANIFEST_VERSION}."
            )
        try:
            manifest_version = int(manifest_version_raw)
        except (ValueError, TypeError):
            raise InvalidManifestError(
                f"Manifest version '{manifest_version_raw}' is not valid. " "Expected an integer."
            )

        project_type_raw = data.get("type", "")
        project_type = str(project_type_raw).lower() if project_type_raw else ""

        manifest = cls(
            manifest_version=manifest_version,
            project_type=project_type,
            default_target=str(default_target).upper() if isinstance(default_target, str) else None,
            targets=targets,
            templating=FSTemplating.from_dict(data.get("templating")),
        )
        manifest.validate()
        return manifest

    @classmethod
    def load(cls, source_path: pathlib.Path) -> FSManifest:
        """Load and validate a ``manifest.yml`` from ``source_path``.

        Args:
            source_path: Directory containing ``manifest.yml``.

        Returns:
            FSManifest parsed and validated from the on-disk file.

        Raises:
            ManifestNotFoundError: If ``manifest.yml`` is not present in
                ``source_path``.
            InvalidManifestError: If the file is empty, cannot be parsed
                as YAML, or fails structural validation (unsupported
                ``manifest_version``, wrong ``type``).
        """
        manifest_file = pathlib.Path(source_path) / MANIFEST_FILE_NAME
        log.info("Loading feature-store manifest from %s.", manifest_file)
        if not manifest_file.exists():
            log.info("Feature-store manifest file not found at %s.", manifest_file)
            raise ManifestNotFoundError(f"{MANIFEST_FILE_NAME} was not found in directory {source_path}.")

        try:
            raw = manifest_file.read_text()
            data = yaml.safe_load(raw)
        except yaml.YAMLError as exc:
            log.info("Feature-store manifest YAML parse error at %s: %s", manifest_file, exc)
            raise InvalidManifestError(f"Manifest file at {manifest_file} could not be parsed as YAML: {exc}") from exc

        if not data:
            log.info("Feature-store manifest file is empty or invalid at %s.", manifest_file)
            raise InvalidManifestError(f"Manifest file at {manifest_file} is empty or invalid.")

        return cls.from_dict(data)

    def validate(self) -> None:
        """Validate manifest-level invariants.

        Per-target invariants (required fields, ``warehouse`` rejection,
        ``templating_config`` resolution) live on :meth:`FSTarget.from_dict`
        and :meth:`get_target` respectively.

        Raises:
            InvalidManifestError: If ``type`` is missing/wrong or
                ``manifest_version`` is not the supported version.
            ManifestConfigurationError: If ``default_target`` references
                a target that is not defined.
        """
        if not self.project_type:
            raise InvalidManifestError(f"Manifest file type is undefined. Expected {MANIFEST_TYPE}.")

        if self.project_type != MANIFEST_TYPE:
            raise InvalidManifestError(
                f"Manifest file is defined for type '{self.project_type}'. " f"Expected {MANIFEST_TYPE}."
            )

        if self.manifest_version != SUPPORTED_MANIFEST_VERSION:
            raise InvalidManifestError(
                f"Manifest version {self.manifest_version} is not supported. "
                f"Expected version {SUPPORTED_MANIFEST_VERSION}."
            )

        if self.default_target is not None and self.default_target not in self.targets:
            raise ManifestConfigurationError(
                f"default_target '{self.default_target}' is not defined in targets " f"({sorted(self.targets.keys())})."
            )

    def _validate_target_configuration_exists(self, target: FSTarget) -> None:
        if target.templating_config and target.templating_config not in self.templating.configurations:
            log.info(
                "Feature-store target references unknown templating configuration " "(target=%s, configuration=%s).",
                target.name,
                target.templating_config,
            )
            raise ManifestConfigurationError(
                f"Target '{target.name}' references unknown templating configuration " f"'{target.templating_config}'."
            )

    def get_target(self, target_name: str) -> FSTarget:
        """Resolve a target by name (case-insensitive).

        Args:
            target_name: Target name as supplied (case-insensitive).

        Returns:
            FSTarget matching ``target_name`` (resolved upper-case).

        Raises:
            ManifestConfigurationError: If no target with the given name
                is defined, or the target's ``templating_config``
                references an undefined configuration.
        """
        upper = str(target_name).upper()
        log.info("Resolving feature-store target '%s'.", upper)
        if upper not in self.targets:
            log.info("Requested feature-store target '%s' was not found in manifest.", upper)
            raise ManifestConfigurationError(
                f"Target '{upper}' not found in manifest " f"(available targets: {sorted(self.targets.keys())})."
            )
        target = self.targets[upper]
        self._validate_target_configuration_exists(target)
        return target

    def get_effective_target(self, target_name: Optional[str] = None) -> FSTarget:
        """Resolve an explicit target or fall back to ``default_target``.

        Args:
            target_name: Optional explicit target name. When ``None`` the
                manifest's ``default_target`` is used (auto-derived for
                single-target manifests).

        Returns:
            FSTarget resolved from ``target_name`` or the manifest default.

        Raises:
            ManifestConfigurationError: If ``target_name`` is ``None``
                and no ``default_target`` is configured, or if the
                resolved name does not match a defined target.
        """
        if target_name:
            return self.get_target(target_name)
        if self.default_target:
            return self.get_target(self.default_target)
        log.info("No feature-store target specified and no default_target in manifest.")
        raise ManifestConfigurationError("No target specified and no default_target defined in manifest.")


@dataclass
class TargetContext:
    """Resolved per-target context (post-template-resolution).

    This is what callers receive after the manifest target has been
    merged with templating defaults / configurations / runtime
    overrides. Phase 1B (resolver) populates ``template_vars``; in this
    phase the dataclass is only declared so downstream agents can
    import it.
    """

    target_name: str
    account_identifier: str
    database: str
    schema: str
    role: str
    template_vars: dict[str, Any] = field(default_factory=dict)
