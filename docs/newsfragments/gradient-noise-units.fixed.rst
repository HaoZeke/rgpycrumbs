``GradientIMQ`` with ``noise_per_obs`` gave every gradient component a tenth of the energy noise at its point.
That ratio drops the inverse squared length that a gradient variance carries.
Each gradient component now keeps the energy's ratio of noise to prior variance.
For the inverse multiquadric kernel, the gradient noise is the energy noise over ``epsilon**2``.
