Surface models without a ``length_scale`` now start it at the median distance between distinct inputs.
The inverse multiquadric models start ``epsilon`` the same way.
Each model used to scale the mean extent of the inputs by its own factor.
``FastMatern`` used 0.2, ``GradientSE`` 0.4 and ``GradientMatern`` 0.5.
``FastIMQ`` and ``GradientIMQ`` used 0.8.
``GradientRQ`` started at 1.5 and ``NystromGradientIMQ`` at 0.5, whatever the data.
No source gave those values.
Garreau, Jitkrittum and Kanagawa analyse the median heuristic.
Riabiz et al. use it for the inverse multiquadric kernel.
Repeated inputs are left out of the median, and fewer than two distinct inputs raise ``ValueError``.
