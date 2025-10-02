import re
from datetime import datetime
from typing import Match, Tuple

BLESS_LOG = re.compile(
    r"""
\[(?P<timestamp>.*?)\]  # Capture timestamp in a named group
\s                      # Match a whitespace character
(?P<logdata>.*)         # Capture the rest of the log data in a named group
""",
    re.X,
)


def parse_bless_time(timestamp_str: str) -> datetime:
    """
    Parse a BLESS timestamp string in ISO format to a datetime object.

    Args:
        timestamp_str: Timestamp string like '2024-10-28T18:58:24Z'.

    Returns:
        Parsed datetime object.

    Raises:
        ValueError: If the timestamp string is invalid.
    """
    return datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%SZ")


def parse_bless_log_line(log_line: str) -> Tuple[str, str] | None:
    """
    Parse a BLESS log line using the regex pattern.

    Args:
        log_line: The full log line string.

    Returns:
        Tuple of (timestamp_str, logdata) if matched, else None.
    """
    match: Match | None = BLESS_LOG.match(log_line)
    if match:
        return match.group("timestamp"), match.group("logdata")
    return None
