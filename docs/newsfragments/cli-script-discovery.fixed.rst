The ``rgpycrumbs`` dispatcher lists a module as a command only when the module has an inline ``# /// script`` metadata block.
It used to list 16 library modules as commands that did nothing.
Two of them were ``rgpycrumbs run jupyter`` and ``rgpycrumbs surfaces standard``.
The ``func``, ``parsers``, ``run`` and ``surfaces`` groups held no scripts, so they are gone from ``rgpycrumbs --help``.
