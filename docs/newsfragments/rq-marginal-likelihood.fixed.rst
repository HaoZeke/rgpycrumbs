Optimized ``GradientRQ`` fits now maximize the marginal likelihood alone.
The objective also had penalties toward a length scale of ``1.5``, an ``alpha`` of ``0.8`` and a noise of ``1e-2``.
No source gave those targets, and the other surface models fit the likelihood without them.
On 40 Muller-Brown points the old fit stopped at a length scale of ``0.349``.
The likelihood alone reaches ``0.297``, and the negative log likelihood there is ``116.8`` instead of ``120.6``.
