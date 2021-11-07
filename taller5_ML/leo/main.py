import math
import cv2
import numpy as np
import pandas as pd
from Data import Normalize
from Optimizer.ADAM import ADAM
from Data.Algorithms import SplitData
from Helper.Parser.ArgParse import ArgParse
from Model.NeuralNetwork.FeedForward import FeedForward


def debug_function(model, j, dj, i, show):
    if show:
        print("Iteration: ", i, ". Cost: ", j)


if __name__ == '__main__':
    parser = ArgParse()
    parser.add_argument('network_descriptors', type=str)
    parser.add_argument('input_data', type=str)
    args = parser.parse_args()

    data = None

    if args.input_data == "None":
        df = pd.read_csv('./input_data/age_gender.csv')
        df = df.drop(['age', 'ethnicity', 'img_name'], axis=1)

        data = []
        input_data = df.to_numpy()
        for r in range(input_data.shape[0]):
            data.append(np.append(np.array(input_data[r, 1].split(" ")), [input_data[r, 0]]))

        data = np.array(data, dtype=np.uint8)
        np.random.shuffle(data)

        np.savetxt('./input_data/input_data.csv', data, delimiter=',', newline='\n', fmt='%u')

    else:
        data = np.loadtxt(args.input_data, delimiter=',')

    # First, split data in training and testing
    x_train, y_train, x_test, y_test, *_ = SplitData(data, 1, train_size=0.8, test_size=0.2)

    # Normalize the data
    x_train, x_off, x_div = Normalize.Center(x_train)

    models = []

    # Retrieve descriptors for each network
    network_descriptors = args.network_descriptors.split("-")

    # Then, split the training data in bags
    # Compute the length of each bag
    bag_size = math.floor(x_train.shape[0] / 4)
    bag_begin = 0
    for i in range(4):
        bag_end = bag_begin + bag_size

        # create a bag for x and y training
        if bag_end < data.shape[0]:
            x_t = x_train[bag_begin:bag_end, :]
            y_t = y_train[bag_begin:bag_end, :]
        else:
            x_t = x_train[bag_begin:-1, :]
            y_t = y_train[bag_begin:-1, :]

        # Then, create a model using the bag created
        nn_model = FeedForward()
        if len(network_descriptors) < 4:
            nn_model.LoadParameters('./input_data/neural_descriptor/' + network_descriptors[0])
        else:
            nn_model.LoadParameters('./input_data/neural_descriptor/' + network_descriptors[i])

        # Define the cost function for the current model
        nn_cost = FeedForward.Cost(x_t, y_t, nn_model, batch_size=args.batch_size)
        nn_cost.SetPropagationTypeToBinaryCrossEntropy()

        # Execute ADAM optimizer
        ADAM(
            cost=nn_cost,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            debug_step=args.debug_step,
            debug_function=debug_function,
            max_iter=args.max_iterations
        )

        # Once the model is trained, store it in the models array
        models.append(nn_model)
        print("Model ", i, " finished training process.")
        bag_begin = bag_end

    # Once all the models are trained, compute the estimations for each one
    estimations = []

    for model in models:
        estimations.append(np.round(model(x_test)))

    estimations = np.array(estimations)

    average_est = []
    for j in range(estimations.shape[1]):
        models_estimations = estimations[:, j, 0]
        average_est.append(np.histogram(models_estimations, bins=[0, 1])[0].argmax())

    y_test = y_test.flatten()
    average_est = np.array(average_est)
    res = np.where(y_test == average_est, 1, 0)
    bagging_accuracy = res.sum() / res.shape[0]

    print('Bagging accuracy: ', str(bagging_accuracy * 100), "%")
    for model in models:
        print(model)
