import sys
import numpy as np

from Data import Normalize
from Debug.Cost import Cost
from Helper.Parser.ArgParse import ArgParse
from Optimizer.GradientDescent import GradientDescent
from Model.NeuralNetwork.FeedForward import FeedForward


def debug_function(model, j, dj, i, show):
    if show:
        print("Iteration: ", i, ". Cost: ", j)


if __name__ == '__main__':
    parser = ArgParse()
    parser.add_argument('network_descriptor', type=str)
    parser.add_argument('neural_network_parameters', type=str)
    parser.add_argument('nn_output', type=str)
    args = parser.parse_args()

    database = np.load('./_0_input_data/mnist/mnist.npz')
    x_train = database['x_train']
    x_test = database['x_test']
    y_train = database['y_train']
    y_test = database['y_test']

    # 784
    input_layer_size = x_test.shape[1] * x_test.shape[2]

    x_train = np.reshape(x_train, (x_train.shape[0], input_layer_size))
    y_train = np.reshape(y_train, (y_train.shape[0], 1))
    x_test = np.reshape(x_test, (x_test.shape[0], input_layer_size))
    y_test = np.reshape(y_test, (y_test.shape[0], 1))

    x_train, x_off, x_div = Normalize.Center(x_train)
    x_test = x_test - x_off

    neural_network = FeedForward()

    if args.neural_network_parameters == 'None':
        # Initialize neural network
        neural_network.LoadParameters(args.network_descriptor)

        # Configure neural network cost function
        cost = FeedForward.Cost(x_train, y_train, neural_network, batch_size=args.batch_size)
        cost.SetPropagationTypeToCategoricalCrossEntropy()

        GradientDescent(
            cost,
            learning_rate=args.learning_rate,
            max_iter=args.max_iterations,
            epsilon=args.epsilon,
            regularization=args.regularization,
            reg_type=args.reg_type,
            debug_step=args.debug_step,
            debug_function=debug_function
        )

        if args.nn_output != 'None':
            neural_network.SaveParameters(args.nn_output)

    else:
        neural_network.LoadParameters(args.neural_network_parameters)

    # To test the neural network:

    # -- Categorize examples
    m = y_test.shape[0]
    p = neural_network.GetLayerOutputSize(-1)
    eP = np.eye(p)
    y_test_cat = eP[y_test, ].reshape(m, p)

    # -- Estimate all examples
    y_estimation = neural_network(x_test)
    y_estimation_cat = eP[np.argmax(y_estimation, axis=1), ].reshape(m, p)

    # -- Confusion matrix and accuracy
    K = y_test_cat.T @ y_estimation_cat
    acc = np.diag(K).sum() / K.sum()
    print('=====================================')
    np.savetxt(sys.stdout, K, fmt='% 5d')
    print('=====================================')
    print('Accuracy = ' + str(acc * 100) + '%')
    print('=====================================')
