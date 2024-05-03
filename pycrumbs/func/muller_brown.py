import numpy as np

# Usage:
# x = np.linspace(-1.5, 1.2, 400)
# y = np.linspace(-0.2, 2.0, 400)
# X, Y = np.meshgrid(x, y)
# Z = muller_brown([X, Y])

def muller_brown(x):
    A = [-200, -100, -170, 15]
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    x0 = [1, 0, -0.5, -1]
    y0 = [0, 0.5, 1.5, 1]

    value = 0
    for i in range(4):
        value += A[i] * np.exp(
            a[i] * (x[0] - x0[i]) ** 2
            + b[i] * (x[0] - x0[i]) * (x[1] - y0[i])
            + c[i] * (x[1] - y0[i]) ** 2
        )
    return value


def muller_brown_gradient(x):
    A = [-200, -100, -170, 15]
    a = [-1, -1, -6.5, 0.7]
    b = [0, 0, 11, 0.6]
    c = [-10, -10, -6.5, 0.7]
    x0 = [1, 0, -0.5, -1]
    y0 = [0, 0.5, 1.5, 1]

    dfdx = 0
    dfdy = 0
    for i in range(4):
        dfdx += (
            A[i]
            * (2 * a[i] * (x[0] - x0[i]) + b[i] * (x[1] - y0[i]))
            * np.exp(
                a[i] * (x[0] - x0[i]) ** 2
                + b[i] * (x[0] - x0[i]) * (x[1] - y0[i])
                + c[i] * (x[1] - y0[i]) ** 2
            )
        )
        dfdy += (
            A[i]
            * (b[i] * (x[0] - x0[i]) + 2 * c[i] * (x[1] - y0[i]))
            * np.exp(
                a[i] * (x[0] - x0[i]) ** 2
                + b[i] * (x[0] - x0[i]) * (x[1] - y0[i])
                + c[i] * (x[1] - y0[i]) ** 2
            )
        )
    return np.array([dfdx, dfdy])
