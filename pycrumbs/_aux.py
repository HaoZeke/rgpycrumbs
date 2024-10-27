from pathlib import Path
import contextlib
import subprocess


def getstrform(pathobj):
    return str(pathobj.absolute())


def get_gitroot():
    gitroot = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            cwd=Path.cwd(),
        )
        .stdout.decode("utf-8")
        .strip()
    )
    return gitroot


@contextlib.contextmanager
def switchdir(path):
    curpath = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(curpath)
