import gzip
from collections import deque
from itertools import islice
from pathlib import Path


def _open_text(file_path: Path):
    opener = gzip.open if file_path.suffix == ".gz" else open
    return opener(file_path, "rt", encoding="utf-8")


def tail(filename: str | Path, n: int) -> list[str]:
    """Return the last *n* lines of a file (supports gzip).

    Line endings are stripped, and a file shorter than *n* lines is returned
    whole.

    .. versionadded:: 0.0.2

    .. versionchanged:: 1.14.0
       Plain files no longer raise ``io.UnsupportedOperation``, and both kinds
       return lines without line endings. Gzip files return all *n* lines even
       when those lines average more than 1024 bytes.
    """
    if n < 0:
        msg = "n must be non-negative"
        raise ValueError(msg)
    if n == 0:
        return []
    with _open_text(Path(filename)) as stream:
        return [line.rstrip("\r\n") for line in deque(stream, maxlen=n)]


def head_search(filename: Path, sstr: str, n=60):
    """
    Checks if the search string is found in the first n lines of a file.

    .. versionadded:: 0.0.2

    .. versionchanged:: 1.14.0
       A file shorter than *n* lines returns ``False`` instead of raising
       ``StopIteration``.

    Args:
        filename (Path): Path to the file.
        sstr (str): The search string.
        n (int, optional): Number of lines to read. Defaults to 60.

    Returns:
        bool: True if the search string is found, False otherwise.
    """

    try:
        with _open_text(Path(filename)) as stream:
            return any(sstr in line.strip() for line in islice(stream, n))
    except (OSError, EOFError) as e:
        print(f"Error reading file: {e}")
        return False
