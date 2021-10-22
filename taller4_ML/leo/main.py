import time
import numpy as np

from Model.SVM import SVM
from Debug import Labeling
from Optimizer.ADAM import ADAM
from Data import Algorithms, Normalize
from Helper.Parser.ArgParse import ArgParse
from Optimizer.GradientDescent import GradientDescent


def compute_accuracy(estimation, expected):
    processed_est = np.where(estimation < 0, -1, 1)
    acc = np.where(processed_est == expected, 1, 0).mean()
    return acc


def print_results(model: str, accuracy: float, exec_time: float, cost: float):
    print('================= %s ==================' % model)
    print('Accuracy = ' + str(accuracy * 100) + '%')
    print('Minimum cost reached = ' + str(cost))
    print('Execution time = ' + str(exec_time) + ' seconds.')


if __name__ == '__main__':
    parser = ArgParse()
    parser.add_argument('datafile', type=str)
    args = parser.parse_args()

    input_data = np.loadtxt(args.datafile, delimiter=',')
    np.random.shuffle(input_data)

    X_tra, y_tra, X_tst, y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size=0.7, test_size=0.3)
    X_tra, X_off, X_div = Normalize.Standardize(X_tra)
    X_tst = (X_tst - X_off) / X_div

    y_est = None

    # Define debug function
    debug_function_svm = Labeling(X_tra, y_tra, threshold=0.5)
    debug_function_gd = Labeling(X_tra, y_tra, threshold=0.5)
    debug_function_gd_mini = Labeling(X_tra, y_tra, threshold=0.5)
    debug_function_gd_est = Labeling(X_tra, y_tra, threshold=0.5)

    # Create SVM model
    svm_adam = SVM(in_x=X_tra, in_y=y_tra)
    svm_gd = SVM(in_x=X_tra, in_y=y_tra)
    svm_gd_mini = SVM(in_x=X_tra, in_y=y_tra)
    svm_gd_est = SVM(in_x=X_tra, in_y=y_tra)

    # Create cost function for SVM
    cost_adam = SVM.Cost(svm_adam)
    cost_gd = SVM.Cost(svm_gd)
    cost_gd_mini = SVM.Cost(svm_gd_mini, batch_size=8)
    cost_gd_est = SVM.Cost(svm_gd_est, batch_size=1)

    # Call ADAM optimizer ---------------------------------------------------------------------------------------------
    adam_start_time = time.time()
    adam_cost = ADAM(
        cost=cost_adam,
        learning_rate=args.learning_rate,
        regularization=args.regularization,
        debug_step=args.debug_step,
        debug_function=debug_function_svm,
        max_iter=args.max_iterations
    )

    adam_exec_time = time.time() - adam_start_time

    # Print results for ADAM
    y_est_adam = svm_adam(X_tst)
    print_results('ADAM', compute_accuracy(y_est_adam, y_tst), adam_exec_time, adam_cost)

    # Call gradient descent with bath_size = 0 ------------------------------------------------------------------------
    gd_start = time.time()
    gd_cost = GradientDescent(
        cost_gd,
        learning_rate=args.learning_rate,
        max_iter=args.max_iterations,
        epsilon=args.epsilon,
        regularization=args.regularization,
        reg_type=args.reg_type,
        debug_step=args.debug_step,
        debug_function=debug_function_gd
    )

    gd_exec_time = time.time() - gd_start

    # Print results for Batch gradient descent
    y_est_gd = svm_gd(X_tst)
    print_results('GRADIENT DESCENT BATCH', compute_accuracy(y_est_gd, y_tst), gd_exec_time, gd_cost)

    gd_mini_start = time.time()

    # Call gradient descent with bath_size = 8 ------------------------------------------------------------------------
    gd_mini_cost = GradientDescent(
        cost_gd_mini,
        learning_rate=args.learning_rate,
        max_iter=args.max_iterations,
        epsilon=args.epsilon,
        regularization=args.regularization,
        reg_type=args.reg_type,
        debug_step=args.debug_step,
        debug_function=debug_function_gd_mini
    )

    # Print results for Mini-Batch gradient descent
    gd_mini_exec = time.time() - gd_mini_start
    y_est_gd_mini = svm_gd_mini(X_tst)
    print_results('GRADIENT DESCENT MINI-BATCH', compute_accuracy(y_est_gd_mini, y_tst), gd_mini_exec, gd_mini_cost)

    # Call gradient descent with bath_size = 1 ------------------------------------------------------------------------
    gd_stochastic_start = time.time()
    gd_est_cost = GradientDescent(
        cost_gd_est,
        learning_rate=args.learning_rate,
        max_iter=round(args.max_iterations / 2),
        epsilon=args.epsilon,
        regularization=args.regularization,
        reg_type=args.reg_type,
        debug_step=args.debug_step,
        debug_function=debug_function_gd_est
    )

    # Print results for Stochastic Gradient Descent
    gd_stochastic_exe = time.time() - gd_stochastic_start
    y_est_gd_est = svm_gd_est(X_tst)
    print_results('STOCHASTIC GRADIENT DESCENT', compute_accuracy(y_est_gd_est, y_tst), gd_stochastic_exe, gd_est_cost)

    debug_function_svm.KeepFigures()
    debug_function_gd.KeepFigures()
    debug_function_gd_mini.KeepFigures()
    debug_function_gd_est.KeepFigures()
