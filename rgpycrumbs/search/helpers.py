import gzip
import os
from pathlib import Path


# Kanged from here and there
# e.g. https://stackoverflow.com/a/76733254/1895378
# XXX: Not good at all, doesn't even handle exponential search
def tail(filename, n):
    file_path = Path(filename)

    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rb") as f:
            try:
                f.seek(-n * 1024, os.SEEK_END)
                lines = f.readlines()[-n:]
                return [line.decode("utf-8").rstrip("\n") for line in lines]
            except EOFError as e:
                return ""
    else:
        with open(file_path, "rt") as f:
            f.seek(-n * 1024, os.SEEK_END)
            lines = f.readlines()[-n:]
            return lines


# XXX: Another hack
def head_search(filename: Path, sstr: str, n=60):
    """
    Checks if the search string is found in the first n lines of a file.

    Args:
        filename (Path): Path to the file.
        sstr (str): The search string.
        n (int, optional): Number of lines to read. Defaults to 60.

    Returns:
        bool: True if the search string is found, False otherwise.
    """

    try:
        with (
            gzip.open(filename, "rt")
            if filename.suffix == ".gz"
            else open(filename, "rt")
        ) as f:
            for _ in range(n):
                line = next(f).strip()
                if sstr in line:
                    return True
            return False
    except (IOError, EOFError) as e:
        print(f"Error reading file: {e}")
        return False
