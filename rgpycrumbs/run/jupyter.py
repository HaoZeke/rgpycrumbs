# Usage:
# https://atomistic-cookbook.org/examples/eon-pet-neb/eon-pet-neb.html
import contextlib
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from typing import Final

# Seconds the output reader gets to reach the end of the output after a kill.
# Killing the process group ended the output within 0.71 ms idle and within
# 20 ms with four busy processes per CPU (300 trials each, 8 CPUs), so one
# second is 50 times the slowest measured case.
KILLED_OUTPUT_DRAIN_SECONDS: Final = 1.0


def _kill_process_group(proc: subprocess.Popen) -> None:
    """Kill *proc* with the processes it started, then reap *proc*."""
    if os.name == "posix":
        # The group outlives its leader while any process it started runs.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
    else:
        proc.kill()
    proc.wait()


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

    *timeout* bounds the whole run, including stretches where the child
    writes nothing and output that processes it started keep open.
    """
    shell = isinstance(cmd, str)
    cmd_str = cmd if shell else cmd[0]

    # If list form, ensure program exists before trying to run
    if not shell and shutil.which(cmd_str) is None:
        raise FileNotFoundError(f"{cmd_str!r} is not on PATH")

    # Start the process
    # We combine stderr into stdout so we only have one stream to read
    proc = subprocess.Popen(  # noqa: S603
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding=encoding,
        shell=shell,
        bufsize=1,  # Line buffered
        # On POSIX the session's process group holds everything the command
        # starts, so a kill reaches the children of a shell command too.
        start_new_session=True,
    )

    collected = [] if capture else None
    reader_errors: list[BaseException] = []

    def stream_output() -> None:
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                # Stream into notebook or terminal live
                print(line, end="")
                sys.stdout.flush()
                if collected is not None:
                    collected.append(line)
        except BaseException as exc:  # re-raised on the calling thread
            reader_errors.append(exc)

    # Reading on this thread would block until the child closes its output,
    # so the timeout would never start while the child stayed silent.
    reader = threading.Thread(target=stream_output, daemon=True)
    reader.start()
    start = time.monotonic()
    try:
        returncode = proc.wait(timeout=timeout)
        if timeout is None:
            reader.join()
        else:
            # A process the command started can keep the output open after
            # the command exits.
            reader.join(max(timeout - (time.monotonic() - start), 0.0))
            if reader.is_alive():
                raise subprocess.TimeoutExpired(cmd, timeout)
    except (subprocess.TimeoutExpired, KeyboardInterrupt):
        _kill_process_group(proc)
        raise
    finally:
        reader.join(KILLED_OUTPUT_DRAIN_SECONDS)
        # Closing the pipe under a reader that is still blocked on it would
        # wait for that read, so a reader still running keeps the pipe.
        if not reader.is_alive() and proc.stdout:
            proc.stdout.close()

    if reader_errors:
        raise reader_errors[0]
    if check and returncode != 0:
        output_str = "".join(collected) if capture else ""
        raise subprocess.CalledProcessError(returncode, cmd, output=output_str)

    return subprocess.CompletedProcess(
        cmd, returncode, stdout="".join(collected) if capture else None
    )


def run_command_or_exit(
    cmd: str | list[str], capture: bool = False, timeout: float | None = None
) -> subprocess.CompletedProcess:
    """
    Helper wrapper to run commands, stream output, and exit script/notebook
    cleanly on failure so sphinx-gallery sees the errors appropriately.

    *timeout* is in seconds and bounds the whole command, including output
    that processes it started keep open. At the deadline the command and the
    processes it started are killed, and the script exits with status 124.
    The default, ``None``, waits without a bound.

    On POSIX the command runs in its own session, so Ctrl-C in a terminal
    interrupts only the helper, which then kills the command and the
    processes it started.

    .. versionadded:: 0.1.0
    .. versionchanged:: 1.14.0
       *timeout* stops a running command and the processes it started. It
       used to start only after the command closed its output, so it never
       stopped a command that kept its output open. The default is ``None``
       instead of 300, so commands that ran past 300 seconds still finish.
       The command runs in its own session.
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
