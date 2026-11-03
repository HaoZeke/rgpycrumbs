Surface models now keep a caller's ``smoothing`` as the noise variance.
Six models raised any value below ``1e-4`` to ``1e-4``.
Take ``FastMatern`` with ``smoothing=1e-6`` on the five-point bowl in the tests.
Its training residual was ``2.9e-4`` and is now ``3.6e-6``.
``GradientRQ`` ignored ``smoothing`` and used ``1e-2``.
Optimized ``GradientIMQ`` fits capped the noise at ``1e-2`` or the caller's value, whichever was larger.
They now cap it at the caller's value.
A ``smoothing`` of zero marks exact data and gets the roundoff jitter, and a negative value raises ``ValueError``.
Optimized fits of exact data start the noise at the largest prior variance.
