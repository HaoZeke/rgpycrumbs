``rgpycrumbs.run.jupyter.run_command_or_exit`` now stops a command still running at its ``timeout``.
It also stops the processes that the command started.
The timeout used to start only after the command closed its output.
So it never stopped a command that kept its output open.
The default is now ``timeout=None``, so commands that ran past the old 300 seconds still finish.
On POSIX the command runs in its own session.
Ctrl-C in a terminal then reaches only the helper, which kills the command.
