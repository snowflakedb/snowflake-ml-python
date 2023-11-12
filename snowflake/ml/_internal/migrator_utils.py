class UnableToUpgradeError(Exception):
    def __init__(self, last_supported_version: str) -> None:
        self.last_supported_version = last_supported_version
