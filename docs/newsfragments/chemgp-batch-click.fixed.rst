``rgpycrumbs chemgp plot-gp batch`` now works with Click 8.2 and later, including the 8.3.1 in the pixi lock.
Every entry used to fail with ``unexpected keyword argument 'mix_stderr'``.
Entries now run through the ``main`` of their subcommand instead of Click's test runner.
The test runner rebinds ``sys.stdout`` and ``sys.stderr`` for the whole process.
Two ``--parallel`` workers could then leave ``sys.stdout`` bound to a closed stream.
A failed entry now reports the error message of its subcommand.
