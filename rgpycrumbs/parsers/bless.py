import datetime
import re

BLESS_LOG = re.compile(
    r"""
\[(?P<timestamp>.*?)\]  # Capture timestamp in a named group
\s                      # Match a whitespace character
(?P<logdata>.*)         # Capture the rest of the log data in a named group
""",
    re.X,
)


# Usage: mktime('2024-10-28T18:58:24Z') - mktime('2024-10-28T18:58:21Z')
def BLESS_TIME(x: str) -> datetime.datetime:  # noqa: N802
    return datetime.datetime.strptime(x, "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=datetime.UTC
    )
