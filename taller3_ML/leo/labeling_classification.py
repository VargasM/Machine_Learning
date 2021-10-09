import numpy as np
from Data import Normalize
from Data import Algorithms
from Debug.Labeling import Labeling
from Model.NeuralNetwork.FeedForward import FeedForward
from Helper.Parser.ArgParse import ArgParse
from Optimizer.GradientDescent import GradientDescent


if __name__ == "__main__":
    parser = ArgParse()
    parser.add_argument('network_descriptor', type=str)
    parser.add_argument('datafile', type=str)
    parser.add_argument('neural_network_parameters', type=str)
    parser.add_argument('nn_output', type=str)
    args = parser.parse_args()

    # Read data from file specified in command line
    input_data = np.loadtxt(args.datafile, delimiter=',')
    np.random.shuffle(input_data)

    X_tra, y_tra, X_tst, y_tst = None, None, None, None
    neural_network = FeedForward()

    # Need to train neural network and test it
    if args.neural_network_parameters == 'None':

        # Split data in X and Y
        X_tra, y_tra, X_tst, y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size=0.7, test_size=0.3)
        X_tra, X_off, X_div = Normalize.Center(X_tra)

        # Initialize neural network
        neural_network.LoadParameters(args.network_descriptor)

        # Configure neural network cost function
        cost = FeedForward.Cost(X_tra, y_tra, neural_network, batch_size=args.batch_size)
        cost.SetPropagationTypeToBinaryCrossEntropy()

        # Configure debug function
        debug = Labeling(X_tra, y_tra, threshold=0.5)

        # Configure iterative solution based on gradient descent
        GradientDescent(
            cost,
            learning_rate=args.learning_rate,
            max_iter=args.max_iterations,
            epsilon=args.epsilon,
            regularization=args.regularization,
            reg_type=args.reg_type,
            debug_step=args.debug_step,
            debug_function=debug
        )

        debug.KeepFigures()

        if args.nn_output != 'None':
            neural_network.SaveParameters(args.nn_output)

    # Create neural network with input parameters and test it
    else:

        X_tra, y_tra, X_tst, y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size=0.0, test_size=1.0)

        neural_network.LoadParameters(args.neural_network_parameters)

    y_est = neural_network(X_tst)
    y_est = np.round(y_est)
    # Compute network accuracy

    res = np.where(y_tst == y_est, 1, 0)
    network_accuracy = res.sum() / res.shape[0]

    print(neural_network)
    print('Network accuracy: ', network_accuracy)
