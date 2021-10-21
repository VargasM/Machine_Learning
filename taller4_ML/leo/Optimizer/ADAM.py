import math

import numpy as np


def ADAM(cost, **kwargs):
    b1 = 0.9
    b2 = 0.999
    alpha = 0.05
    epsilon = 1e-8
    debug_step = 10
    learning_rate = 1e-2
    num_iterations = 1e10
    debug_function = None

    if 'learning_rate' in kwargs:
        learning_rate = float(kwargs['learning_rate'])

    if 'regularization' in kwargs:
        alpha = float(kwargs['regularization'])

    if 'debug_step' in kwargs:
        debug_step = int(kwargs['debug_step'])

    if 'debug_function' in kwargs:
        debug_function = kwargs['debug_function']

    if 'max_iter' in kwargs:
        num_iterations = int(kwargs['max_iter'])

    if 'epsilon' in kwargs:
        epsilon = float(kwargs['epsilon'])

    if 'b1' in kwargs:
        b1 = float(kwargs['b1'])

    if 'b2' in kwargs:
        b2 = float(kwargs['b2'])

    t = 1
    stop = False
    current_cost = math.inf
    delta_cost = math.inf
    args = cost.GetInitialParameters()
    m_0 = np.zeros(args.shape)
    v_0 = np.zeros(args.shape)

    # Main loop
    while delta_cost > epsilon and t < num_iterations and not stop:

        # Batch loop
        for batch in range(cost.GetNumberOfBatches()):
            # Compute gradients vector
            it_cost, gradient = cost.CostAndGradient(args, batch, alpha)

            # Update biased first moment estimate
            m_t = (b1 * m_0) + (1 - b1) * gradient

            # Update biased second raw moment estimate
            v_t = (b2 * v_0) + (1 - b2) * np.multiply(gradient, gradient)

            # Compute bias-corrected first moment estimate
            m_t_corrected = m_t / (1 - b1 ** t)

            # Compute bias-corrected second raw moment estimate
            v_t_corrected = v_t / (1 - b2 ** t)

            # Update parameters
            args = args - ((learning_rate * m_t_corrected) / (np.sqrt(v_t_corrected) + epsilon))
            m_0 = m_t
            v_0 = v_t

            # Execute debug function
            if debug_function is not None:
                stop = debug_function(cost.getModel(), it_cost, delta_cost, t, t % debug_step == 0)

