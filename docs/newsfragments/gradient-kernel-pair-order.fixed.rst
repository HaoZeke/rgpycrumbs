The gradient-enhanced rational quadratic surface now builds a positive semidefinite covariance matrix.
Its gradient-gradient blocks were transposed.
Row ``i`` and column ``j`` held the second derivative in ``x2[i]`` and ``x1[j]``.
The mirrored kernel makes that block asymmetric, so the transposed matrix had negative eigenvalues.
The inverse multiquadric and squared-exponential blocks are symmetric, so the same change leaves their values unchanged.
