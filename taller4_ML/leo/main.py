import numpy as np

from Model.SVM import SVM
from Debug import Labeling
from Optimizer.ADAM import ADAM
from Data import Algorithms, Normalize
from Helper.Parser.ArgParse import ArgParse
from Optimizer.GradientDescent import GradientDescent


if __name__ == '__main__':
    parser = ArgParse()
    parser.add_argument('datafile', type=str)
    parser.add_argument('optimization_method', type=str)
    args = parser.parse_args()

    input_data = np.loadtxt(args.datafile, delimiter=',')
    np.random.shuffle(input_data)

    X_tra, y_tra, X_tst, y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size=0.7, test_size=0.3)
    X_tra, X_off, X_div = Normalize.Standardize(X_tra)
    X_tst = (X_tst - X_off) / X_div

    y_est = None

    # Define debug function
    debug_function = Labeling(X_tra, y_tra, threshold=0.5)

    # Create SVM model
    svm = SVM(in_x=X_tra, in_y=y_tra)

    # Create cost function for SVM
    cost = SVM.Cost(svm)

    if args.optimization_method == 'ADAM':

        # Call ADAM optimizer
        ADAM(
            cost=cost,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            debug_step=args.debug_step,
            debug_function=debug_function,
            max_iter=args.max_iterations
        )

    else:

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

    y_est = svm(X_tst)

    debug_function.KeepFigures()

