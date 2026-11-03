Surface models now treat the data as exact when the caller gives no ``smoothing``.
The default had been ``1e-3`` for ``FastTPS``, ``FastMatern``, ``FastIMQ`` and ``NystromGradientIMQ``.
The other gradient models used ``1e-4``.
Fixed fits now take the roundoff jitter as the noise, and optimized fits start the noise at the largest prior variance.
Pass the old value as ``smoothing`` to keep the earlier fits.
