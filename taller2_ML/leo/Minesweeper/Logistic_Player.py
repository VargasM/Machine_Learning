## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import sys
import numpy as np
from MineSweeperBoard import *
from Model.Logistic import Logistic
from Debug.Polynomial import Polynomial
from Optimizer.GradientDescent import GradientDescent
from Regression.MaximumLikelihood import MaximumLikelihood

# -------------------------------------------------------------------------
if len(sys.argv) < 5:
    print("Usage: python", sys.argv[0], "width height mines training_file_name")
    sys.exit(1)
# end if
w = int(sys.argv[1])
h = int(sys.argv[2])
m = int(sys.argv[3])
training_filename = sys.argv[4]
board = MineSweeperBoard(w, h, m)

neighbours = [
    [-1, -1],
    [-1, 0],
    [-1, 1],
    [0, 1],
    [1, 1],
    [1, 0],
    [1, -1],
    [0, -1]
]

learning_rate = 1e-3
maximum_iterations = 10000
epsilon = 1e-8
debug_step = 10000
init_theta = np.zeros((1, 8))

# First board is going to be used to train the model
training_board = board.m_Mines
training_data = np.loadtxt(fname=training_filename, delimiter=',')

x0 = training_data[:, :-1]
y0 = training_data[:, -1]

cost = MaximumLikelihood(x0, y0)

weights, iterations = GradientDescent(
    cost=cost,
    learning_rate=learning_rate,
    init_theta=init_theta,
    maximum_iterations=maximum_iterations,
    epsilon=epsilon,
    debug_step=debug_step
)

model = Logistic(weights[0, :-1], weights[-1, -1])
patches = [[9 for j in range(h)] for i in range(w)]

# Creating a new board to test model
# board = MineSweeperBoard(w, h, m)

while not board.have_won() and not board.have_lose():
    print(board)

    max_c = -1
    max_i = -1
    max_j = -1
    for i in range(w):
        for j in range(h):
            if patches[i][j] == 9:
                x = []

                for n in neighbours:
                    neighbour_i = i + n[0]
                    neighbour_j = j + n[1]

                    if neighbour_i < 0 or neighbour_j < 0 or neighbour_i >= w or neighbour_j >= h:
                        # TODO: It can be 0 or 9, Try it!
                        x += [9]
                    else:
                        x += [patches[neighbour_i][neighbour_j]]

                c = model(x, threshold=False)
                if c > max_c:
                    max_c = c
                    max_i = i
                    max_j = j
                # end if
            # end if
        # end for
    # end for
    patches[max_i][max_j] = board.click(max_j, max_i)
# end while

print(board)
if board.have_won():
    print("You won!")
elif board.have_lose():
    print("You lose :-(")
# end if

## eof - $RCSfile$
