from pathlib import Path
import subprocess


def getstrform(pathobj):
    return str(pathobj.absolute())


gitroot = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True
    )
    .stdout.decode("utf-8")
    .strip()
)
