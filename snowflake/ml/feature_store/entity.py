from typing import Optional

from snowflake.ml._internal.utils.sql_identifier import (
    SqlIdentifier,
    to_sql_identifiers,
)

_ENTITY_NAME_LENGTH_LIMIT = 32
_FEATURE_VIEW_ENTITY_TAG_DELIMITER = ","
_ENTITY_JOIN_KEY_DELIMITER = ","
# join key length limit is the length limit of TAG value
_ENTITY_JOIN_KEY_LENGTH_LIMIT = 256
# The maximum number of join keys:
#   https://docs.snowflake.com/en/user-guide/object-tagging#specify-tag-values
_ENTITY_MAX_NUM_JOIN_KEYS = 300


class Entity:
    """
    Entity encapsulates additional metadata for feature definition.
    Entity is typically used together with FeatureView to define join_keys and associate relevant FeatureViews.
    It can also be used for FeatureView search and lineage tracking.
    """

    def __init__(self, name: str, join_keys: list[str], *, desc: str = "") -> None:
        """
        Creates an Entity instance.

        Args:
            name: name of the Entity.
            join_keys: join keys associated with a FeatureView, used for feature retrieval.
            desc: description of the Entity.

        Example::

            >>> fs = FeatureStore(...)
            >>> e_1 = Entity(
            ...     name="my_entity",
            ...     join_keys=['col_1'],
            ...     desc='My first entity.'
            ... )
            >>> fs.register_entity(e_1)
            >>> fs.list_entities().show()
            -----------------------------------------------------------
            |"NAME"     |"JOIN_KEYS"  |"DESC"            |"OWNER"     |
            -----------------------------------------------------------
            |MY_ENTITY  |["COL_1"]    |My first entity.  |REGTEST_RL  |
            -----------------------------------------------------------

        """
        self._validate(name, join_keys)

        self.name: SqlIdentifier = SqlIdentifier(name)
        self.join_keys: list[SqlIdentifier] = to_sql_identifiers(join_keys)
        self.owner: Optional[str] = None
        self.desc: str = desc

    def _validate(self, name: str, join_keys: list[str]) -> None:
        if len(name) > _ENTITY_NAME_LENGTH_LIMIT:
            raise ValueError(f"Entity name `{name}` exceeds maximum length: {_ENTITY_NAME_LENGTH_LIMIT}")
        if _FEATURE_VIEW_ENTITY_TAG_DELIMITER in name:
            raise ValueError(f"Entity name contains invalid char: `{_FEATURE_VIEW_ENTITY_TAG_DELIMITER}`")
        if len(join_keys) > _ENTITY_MAX_NUM_JOIN_KEYS:
            raise ValueError(
                f"Maximum number of join keys are {_ENTITY_MAX_NUM_JOIN_KEYS}, " "but {len(join_keys)} is provided."
            )
        if len(set(join_keys)) != len(join_keys):
            raise ValueError(f"Duplicate join keys detected in: {join_keys}")
        for k in join_keys:
            # TODO(wezhou) move this logic into SqlIdentifier.
            if _ENTITY_JOIN_KEY_DELIMITER in k:
                raise ValueError(f"Invalid char `{_ENTITY_JOIN_KEY_DELIMITER}` detected in join key {k}")
            if len(k) > _ENTITY_JOIN_KEY_LENGTH_LIMIT:
                raise ValueError(f"Join key: {k} exceeds length limit {_ENTITY_JOIN_KEY_LENGTH_LIMIT}.")

    def _to_dict(self) -> dict[str, str]:
        entity_dict = self.__dict__.copy()
        for k, v in entity_dict.items():
            if isinstance(v, SqlIdentifier):
                entity_dict[k] = str(v)
        return entity_dict

    @staticmethod
    def _construct_entity(name: str, join_keys: list[str], desc: str, owner: str) -> "Entity":
        e = Entity(name, join_keys, desc=desc)
        e.owner = owner
        return e

    def __repr__(self) -> str:
        states = (f"{k}={v}" for k, v in vars(self).items())
        return f"{type(self).__name__}({', '.join(states)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return False

        return (
            self.name == other.name
            and self.desc == other.desc
            and self.join_keys == other.join_keys
            and self.owner == other.owner
        )
