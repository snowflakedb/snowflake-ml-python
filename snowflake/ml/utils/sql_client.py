from enum import Enum


class CreationOption(Enum):
    FAIL_IF_NOT_EXIST = 1
    CREATE_IF_NOT_EXIST = 2
    OR_REPLACE = 3


class CreationMode:
    def __init__(self, *, if_not_exists: bool = False, or_replace: bool = False) -> None:
        self.if_not_exists = if_not_exists
        self.or_replace = or_replace

    def get_ddl_phrases(self) -> dict[CreationOption, str]:
        if_not_exists_sql = " IF NOT EXISTS" if self.if_not_exists else ""
        or_replace_sql = " OR REPLACE" if self.or_replace else ""
        return {
            CreationOption.CREATE_IF_NOT_EXIST: if_not_exists_sql,
            CreationOption.OR_REPLACE: or_replace_sql,
        }
