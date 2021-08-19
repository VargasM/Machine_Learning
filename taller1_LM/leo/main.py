import sys
import numpy as np
import matplotlib.pyplot as plt

import LMOptimizer
import Regression

plt.ion()
figure, axes = plt.subplots(1, 3, sharey=False)
data_axes = axes[0]
J_axes = axes[1]
dJ_axes = axes[2]

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python ' + sys.argv[0] + ' wi=?? r=?? x0=?? x1=?? m=?? alpha=??')
        sys.exit(1)
    # end if
    args = {'w': {}, 'b': 0, 'alpha': 1.0, 'r': 0.0, 'x0': -1.0, 'x1': 1.0, 'm': 20}
    eps = 1e-8
    for a in sys.argv:
        v = a.split('=')
        if len(v) == 2:
            if v[0][0] == 'w':
                if v[0] == 'w0':
                    args['b'] = v[1]
                else:
                    args['w'][int(v[0][1:])] = v[1]
                # end if
            else:
                args[v[0]] = v[1]
            # end if
        # end if
    # end for

    # -- Build input objects from arguments
    args['w'] = sorted(args['w'].items(), reverse=True)
    w = np.zeros((1, args['w'][0][0]))
    for e in args['w']:
        w[0, e[0] - 1] = float(e[1])
    # end for
    b = float(args['b'])
    r = float(args['r'])
    m = int(args['m'])
    alpha = float(args['alpha'])
    x0 = float(args['x0'])
    x1 = float(args['x1'])
    eps = 1e-8
    n = w.shape[1]

    # -- Create data
    X = np.matrix(
        [((x1 - x0) * float(i) / float(m - 1)) + x0 for i in range(m)]
    ).T
    for i in range(n - 1):
        X = np.append(X, np.power(X[:, 0], i + 2), axis=1)
    # end for
    Y = (X @ w.T) + b
    X += np.random.randn(m, n) * r
    Y += np.random.randn(m, 1) * r

    data_axes.scatter([X[:, 0]], [Y], color='red', marker='+')

    # Solve regression
    cost_function = Regression.MSECost(X, Y)
    lm_regression, iterations = LMOptimizer.levenber_marquardt(
        cost_function,
        alpha=alpha)

    print('=================================================================')
    print('Levenberg Marquardt descent : ' + str(lm_regression))
    print('Number of iterations : ' + str(iterations))
    print('=================================================================')

    plt.ioff()

    vX = np.ones((m, 1))
    vX = np.append(
        vX,
        np.matrix(
            [((x1 - x0) * float(i) / float(m - 1)) + x0 for i in range(m)]
        ).T,
        axis=1
    )
    for i in range(n - 1):
        vX = np.append(vX, np.power(vX[:, 1], i + 2), axis=1)
    # end for

    g_vY = vX @ lm_regression.T
    data_axes.plot(vX[:, 1], g_vY, color='green')

    plt.show()