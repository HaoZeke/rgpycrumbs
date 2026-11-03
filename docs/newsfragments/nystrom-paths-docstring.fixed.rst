The ``nystrom_paths_needed`` docstring now describes its result as a count of complete input paths.
The old text named optimization steps and gave the floor division that ``NystromGradientIMQ`` uses.
The helper returns that division rounded up, plus one buffer path.
