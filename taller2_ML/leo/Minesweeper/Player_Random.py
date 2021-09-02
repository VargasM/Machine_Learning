## =========================================================================
## @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
## =========================================================================

import numpy as np
import sys
from MineSweeperBoard import *

## -------------------------------------------------------------------------
if len(sys.argv) < 5:
    print("Usage: python", sys.argv[0], "width height mines neighbours_size(8, 24)")
    sys.exit(1)
# end if
w = int(sys.argv[1])
h = int(sys.argv[2])
m = int(sys.argv[3])
neighbour_size = int(sys.argv[4])

neighbours = None
patches = [[9 for j in range(h)] for i in range(w)]

if neighbour_size == 8:
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
else:
    neighbours = [
        [-2, -2],
        [-2, -1],
        [-2, 0],
        [-2, 1],
        [-2, 2],
        [-1, -2],
        [-1, -1],
        [-1, 0],
        [-1, 1],
        [-1, 2],
        [0, -2],
        [0, -1],
        [0, 1],
        [0, 2],
        [1, -2],
        [1, -1],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, -2],
        [2, -1],
        [2, 0],
        [2, 1],
        [2, 2]
    ]

data = []

for i in range(0, 3):
    board = MineSweeperBoard(w, h, m)
    while not board.have_won() and not board.have_lose():
        print(board)

        data_row = []

        i = int(input("Specify row"))
        j = int(input("Specify column"))
        value = board.click(j, i)

        # update patches
        patches[i][j] = value

        for neighbor in neighbours:
            neighbour_i = i + neighbor[0]
            neighbour_j = j + neighbor[1]

            if neighbour_i < 0 or neighbour_j < 0 or neighbour_i >= w or neighbour_j >= h:
                # TODO: It can be 0 or 9, Try it!
                data_row.append(0)
            else:
                data_row.append(patches[neighbour_i][neighbour_j])

        data_row.append(1 if value != 9 else 0)

        data.append(data_row)

# end while

np_data = np.array(data, dtype=np.int)
np.savetxt('neighbours_24_2.csv', np_data, delimiter=',')

print(board)
if board.have_won():
    print("You won!")
elif board.have_lose():
    print("You lose :-(")
# end if

## eof - $RCSfile$
