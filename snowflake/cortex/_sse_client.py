import json
from typing import Any, Iterator, Optional

_FIELD_SEPARATOR = ":"


class Event:
    """Representation of an event from the event stream."""

    def __init__(
        self,
        id: Optional[str] = None,
        event: str = "message",
        data: str = "",
        comment: Optional[str] = None,
        retry: Optional[int] = None,
    ) -> None:
        self.id = id
        self.event = event
        self.data = data
        self.comment = comment
        self.retry = retry

    def __str__(self) -> str:
        s = f"{self.event} event"
        if self.id:
            s += f" #{self.id}"
        if self.data:
            s += ", {} byte{}".format(len(self.data), "s" if len(self.data) else "")
        else:
            s += ", no data"
        if self.comment:
            s += f", comment: {self.comment}"
        if self.retry:
            s += f", retry in {self.retry}ms"
        return s


# This is copied from the snowpy library:
# https://github.com/snowflakedb/snowpy/blob/main/libs/snowflake.core/src/snowflake/core/rest.py#L39
# TODO(SNOW-1750723) - Current there’s code duplication across snowflake-ml-python
# and snowpy library for Cortex REST API which was done to meet our GA timelines
# Once snowpy has a release with https://github.com/snowflakedb/snowpy/pull/679, we should
# remove the class here and directly refer from the snowflake.core package directly
class SSEClient:
    def __init__(self, event_source: Any, char_enc: str = "utf-8") -> None:
        self._event_source = event_source
        self._char_enc = char_enc

    def _read(self) -> Iterator[bytes]:
        data = b""
        for chunk in self._event_source:
            for line in chunk.splitlines(True):
                data += line
                if data.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    yield data
                    data = b""
        if data:
            yield data

    def events(self) -> Iterator[Event]:
        content_type = self._event_source.headers.get("Content-Type")
        # The check for empty content-type is present because it's being populated after
        # the change in https://github.com/snowflakedb/snowflake/pull/217654.
        # This can be removed once the above change makes it to prod or we move to snowpy
        # for SSEClient implementation.
        if content_type == "text/event-stream" or not content_type:
            return self._handle_sse()
        elif content_type == "application/json":
            return self._handle_json()
        else:
            raise ValueError(f"Unknown Content-Type: {content_type}")

    def _handle_sse(self) -> Iterator[Event]:
        for chunk in self._read():
            event = Event()
            # Split before decoding so splitlines() only uses \r and \n
            for line_bytes in chunk.splitlines():
                # Decode the line.
                line = line_bytes.decode(self._char_enc)

                # Lines starting with a separator are comments and are to be
                # ignored.
                if not line.strip() or line.startswith(_FIELD_SEPARATOR):
                    continue

                data = line.split(_FIELD_SEPARATOR, 1)
                field = data[0]

                # Ignore unknown fields.
                if not hasattr(event, field):
                    continue

                if len(data) > 1:
                    # From the spec:
                    # "If value starts with a single U+0020 SPACE character,
                    # remove it from value."
                    if data[1].startswith(" "):
                        value = data[1][1:]
                    else:
                        value = data[1]
                else:
                    # If no value is present after the separator,
                    # assume an empty value.
                    value = ""

                # The data field may come over multiple lines and their values
                # are concatenated with each other.
                current_value = getattr(event, field, "")
                if field == "data":
                    new_value = current_value + value + "\n"
                else:
                    new_value = value
                setattr(event, field, new_value)

            # Events with no data are not dispatched.
            if not event.data:
                continue

            # If the data field ends with a newline, remove it.
            if event.data.endswith("\n"):
                event.data = event.data[0:-1]

            # Empty event names default to 'message'
            event.event = event.event or "message"

            if event.event != "message":  # ignore anything but “message” or default event
                continue

            yield event

    def _handle_json(self) -> Iterator[Event]:
        data_list = json.loads(self._event_source.data.decode(self._char_enc))
        for data in data_list:
            yield Event(
                id=data.get("id"),
                event=data.get("event"),
                data=data.get("data"),
                comment=data.get("comment"),
                retry=data.get("retry"),
            )

    def close(self) -> None:
        self._event_source.close()
