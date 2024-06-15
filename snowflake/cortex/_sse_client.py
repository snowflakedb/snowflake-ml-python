from typing import Iterator, cast

import requests


class Event:
    def __init__(self, event: str = "message", data: str = "") -> None:
        self.event = event
        self.data = data

    def __str__(self) -> str:
        s = f"{self.event} event"
        if self.data:
            s += f", {len(self.data)} bytes"
        else:
            s += ", no data"
        return s


class SSEClient:
    def __init__(self, response: requests.Response) -> None:

        self.response = response

    def _read(self) -> Iterator[str]:

        lines = b""
        for chunk in self.response:
            for line in chunk.splitlines(True):
                lines += line
                if lines.endswith((b"\r\r", b"\n\n", b"\r\n\r\n")):
                    yield cast(str, lines)
                    lines = b""
        if lines:
            yield cast(str, lines)

    def events(self) -> Iterator[Event]:
        for raw_event in self._read():
            event = Event()
            # splitlines() only uses \r and \n
            for line in raw_event.splitlines():

                line = cast(bytes, line).decode("utf-8")

                data = line.split(":", 1)
                field = data[0]

                if len(data) > 1:
                    # "If value starts with a single U+0020 SPACE character,
                    # remove it from value. .strip() would remove all white spaces"
                    if data[1].startswith(" "):
                        value = data[1][1:]
                    else:
                        value = data[1]
                else:
                    value = ""

                # The data field may come over multiple lines and their values
                # are concatenated with each other.
                if field == "data":
                    event.data += value + "\n"
                elif field == "event":
                    event.event = value

            if not event.data:
                continue

            # If the data field ends with a newline, remove it.
            if event.data.endswith("\n"):
                event.data = event.data[0:-1]  # Replace trailing newline - rstrip would remove multiple.

            # Empty event names default to 'message'
            event.event = event.event or "message"

            if event.event != "message":  # ignore anything but “message” or default event
                continue

            yield event

    def close(self) -> None:
        self.response.close()
