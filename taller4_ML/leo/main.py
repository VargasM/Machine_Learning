import numpy as np

from Model.SVM import SVM
from Data import Algorithms
from Helper.Parser.ArgParse import ArgParse


if __name__ == '__main__':
    parser = ArgParse()
    parser.add_argument('datafile', type=str)
    args = parser.parse_args()

    input_data = np.loadtxt(args.datafile, delimiter=',')
    np.random.shuffle(input_data)

    X_tra, y_tra, X_tst, y_tst, *_ = Algorithms.SplitData(input_data, 1, train_size=0.7, test_size=0.3)

    svm = SVM(in_x=X_tra, in_y=y_tra)

    cost = SVM.Cost(svm)
    print(cost.CostAndGradient(theta=0.5, batch=0, alpha=0))
