from enum import Enum


class ModelMonitorAggregationWindow(Enum):
    WINDOW_1_HOUR = 60
    WINDOW_1_DAY = 24 * 60

    def __init__(self, minutes: int) -> None:
        super().__init__()
        self.minutes = minutes


class ModelMonitorRefreshInterval:
    EVERY_30_MINUTES = "30 minutes"
    HOURLY = "1 hours"
    EVERY_6_HOURS = "6 hours"
    EVERY_12_HOURS = "12 hours"
    DAILY = "1 days"
    WEEKLY = "7 days"
    BIWEEKLY = "14 days"
    MONTHLY = "30 days"

    _ALLOWED_TIME_UNITS = {"minutes": 1, "hours": 60, "days": 24 * 60}

    def __init__(self, raw_time_str: str) -> None:
        try:
            num_units_raw, time_units = raw_time_str.strip().split(" ")
            num_units = int(num_units_raw)  # try to cast
        except Exception as e:
            raise ValueError(
                f"""Failed to parse refresh interval with exception {e}.
                Provide '<num> <minutes | hours | days>'.
See https://docs.snowflake.com/en/sql-reference/sql/create-dynamic-table#required-parameters for more info."""
            )
        if time_units.lower() not in self._ALLOWED_TIME_UNITS:
            raise ValueError(
                """Invalid time unit in refresh interval. Provide '<num> <minutes | hours | days>'.
See https://docs.snowflake.com/en/sql-reference/sql/create-dynamic-table#required-parameters for more info."""
            )
        minutes_multiplier = self._ALLOWED_TIME_UNITS[time_units.lower()]
        self.minutes = num_units * minutes_multiplier

    def __eq__(self, value: object) -> bool:
        if not isinstance(value, ModelMonitorRefreshInterval):
            return False
        return self.minutes == value.minutes
