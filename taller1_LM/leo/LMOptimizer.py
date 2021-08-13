import math
import numpy as np


def next_w_estimation(w_estimation, jacobian, alpha, identity):
    jt_t = jacobian.T @ jacobian
    alpha_i = alpha * identity
    inverse = np.linalg.inv(jt_t + alpha_i)
    return w_estimation - (inverse @ jt_t)


def levenber_marquardt(cost_function, **kwargs):
    # Levenberg - Marquardt parameter
    alpha = 10
    max_error = 1e-8
    debug_step = 100
    max_iteration = 1e10
    debug_function = None
    independent_variables = cost_function.vector_size()
    identity_matrix = np.identity(independent_variables)
    w_estimation = np.random.rand(1, independent_variables) * 1e-1

    if 'alpha' in kwargs:
        alpha = float(kwargs['alpha'])
    if 'max_error' in kwargs:
        max_error = float(kwargs['max_error'])
    if 'debug_step' in kwargs:
        debug_step = int(kwargs['debug_step'])
    if 'max_iterations' in kwargs:
        max_error = int(kwargs['max_iterations'])
    if 'debug_function' in kwargs:
        debug_function = kwargs['debug_function']

    # Initial error and jacobian matrix computation
    m = 1
    num_iteration = 0
    error = math.inf

    while error > max_error and num_iteration < max_iteration:

        [error, jacobian] = cost_function.cost_and_derivatives(w_estimation)

        # step forward
        next_w = next_w_estimation(w_estimation, jacobian, alpha, identity_matrix)

        # new error
        next_error = cost_function.cost_and_derivatives(next_w, only_cost=True)

        if next_error > error:
            while next_error > error and m <= 5:
                alpha = alpha * 10
                m += 1

                # Recalculate w_estimation
                next_w = next_w_estimation(w_estimation, jacobian, alpha, identity_matrix)
                next_error = cost_function.cost_and_derivatives(next_w, only_cost=True)

        if next_error < error:
            alpha = alpha / 10

        m = 1
        w_estimation = next_w

        num_iteration += 1

    return w_estimation, num_iteration
