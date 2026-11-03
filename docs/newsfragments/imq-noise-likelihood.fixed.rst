Optimized ``GradientIMQ`` fits no longer pull the noise toward ``1e-2``.
The objective had a quadratic penalty on the log noise, and no source gave its target or width.
On 40 Muller-Brown points the fitted noise drops from ``1.5e-3`` to ``1.2e-6``.
The negative log likelihood at the fit falls from ``38.9`` to ``24.6``.
The gamma prior on ``epsilon`` keeps shape 2, which Chung et al. propose so that the estimate stays off zero.
Its rate now puts the prior mode at the starting ``epsilon``, without the old ``1e-6`` offset.
