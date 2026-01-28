# Usage:
# https://atomistic-cookbook.org/examples/eon-pet-neb/eon-pet-neb.html
import shutil
import subprocess
import sys


def _run_command_live(
    cmd: str | list[str],
    *,
    check: bool = True,
    timeout: float | None = None,
    capture: bool = False,
    encoding: str = "utf-8",
) -> subprocess.CompletedProcess:
    """
    Internal: run command and stream stdout/stderr live to current stdout.
    If capture=True, also collect combined output and return it
    in CompletedProcess.stdout.
    """
    shell = isinstance(cmd, str)
    cmd_str = cmd if shell else cmd[0]

    # If list form, ensure program exists before trying to run
    if not shell and shutil.which(cmd_str) is None:
        raise FileNotFoundError(f"{cmd_str!r} is not on PATH")

    # Start the process
    # We combine stderr into stdout so we only have one stream to read
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=encoding,
        shell=shell,
        bufsize=1,  # Line buffered
    )

    collected = [] if capture else None

    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            # Stream into notebook or terminal live
            print(line, end="")
            sys.stdout.flush()
            if capture:
                collected.append(line)

        # Wait for the process to actually exit after stream closes
        returncode = proc.wait(timeout=timeout)

    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise
    except KeyboardInterrupt:
        proc.kill()
        proc.wait()
        raise
    finally:
        if proc.stdout:
            proc.stdout.close()

    if check and returncode != 0:
        output_str = "".join(collected) if capture else ""
        raise subprocess.CalledProcessError(returncode, cmd, output=output_str)

    return subprocess.CompletedProcess(
        cmd, returncode, stdout="".join(collected) if capture else None
    )


def run_command_or_exit(
    cmd: str | list[str], capture: bool = False, timeout: float | None = 300
) -> subprocess.CompletedProcess:
    """
    Helper wrapper to run commands, stream output, and exit script/notebook
    cleanly on failure so sphinx-gallery sees the errors appropriately.
    """
    try:
        return _run_command_live(cmd, check=True, capture=capture, timeout=timeout)
    except FileNotFoundError as e:
        print(f"Executable not found: {e}", file=sys.stderr)
        sys.exit(2)
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}", file=sys.stderr)
        sys.exit(e.returncode)
    except subprocess.TimeoutExpired:
        print("Command timed out", file=sys.stderr)
        sys.exit(124)
