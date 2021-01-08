import numpy as np

I = np.array([[1, 0, 0],
              [0, 1, 0],
              [0, 0, 1]])
M = np.array([[3, 2, 1],
              [2, -1, 5],
              [1, 0, 2]])



print(np.einsum("ij,ji->", I, M))
