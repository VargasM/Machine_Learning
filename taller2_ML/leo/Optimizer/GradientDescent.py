## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import math
import numpy


## -------------------------------------------------------------------------
def GradientDescent(cost, **kwargs):
    a = 1e-1
    l = 0.0
    lt = 'ridge'
    I = 1e10
    e = 1e-8
    ds = 100
    df = None
    n = cost.VectorSize()
    t = numpy.random.rand(1, n) * 1e-1

    if 'learning_rate' in kwargs: a = float(kwargs['learning_rate'])
    if 'regularization' in kwargs: l = float(kwargs['regularization'])
    if 'reg_type' in kwargs: lt = kwargs['reg_type']
    if 'max_iter' in kwargs: I = int(kwargs['max_iter'])
    if 'epsilon' in kwargs: e = float(kwargs['epsilon'])
    if 'debug_step' in kwargs: ds = int(kwargs['debug_step'])
    if 'debug_function' in kwargs: df = kwargs['debug_function']
    if 'init_theta' in kwargs:
        t0 = kwargs['init_theta']
        if isinstance(t0, (int, float)):
            t = numpy.ones((1, n)) * float(t0)
        elif isinstance(t0, list):
            t = numpy.matrix(t0)
        elif isinstance(t0, numpy.matrix):
            t = t0
        # end if
    # end if

    # Init loop
    [J, gt] = cost.CostAndGradient(t)
    if l > 0:
        if lt == 'ridge':
            J += l * (t @ t.T)
            gt += 2.0 * l * t
        elif lt == 'lasso':
            J += l * numpy.abs(t).sum()
            gt += l * (t > 0).astype(gt.dtype).sum()
            gt -= l * (t < 0).astype(gt.dtype).sum()
        # end if
    # end if
    dJ = math.inf
    i = 0
    while dJ > e and i < I:

        # Step forward
        t -= gt * a
        [Jn, gt] = cost.CostAndGradient(t)
        if l > 0:
            if lt == 'ridge':
                Jn += l * (t @ t.T)[0, 0]
                gt += 2.0 * l * t
            elif lt == 'lasso':
                Jn += l * numpy.abs(t).sum()
                gt += l * (t > 0).astype(gt.dtype).sum()
                gt -= l * (t < 0).astype(gt.dtype).sum()
            # end if
        # end if
        dJ = J - Jn
        J = Jn

        # Debug
        if i % ds == 0:
            print("Cost {}. Iteration Number: {}".format(J, i))
        # end if
        i += 1

    # end while
    print("Cost {}. Iteration Number: {}".format(J, i))
    # end if

    return (t, i)
# end def

## eof - $RCSfile$
