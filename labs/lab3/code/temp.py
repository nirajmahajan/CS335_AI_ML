import numpy as np


image = np.array([[1,1,0],[0,1,1],[0,0,1]])
tot_pixels = 2500
# thresholding
image[image > 0.4] = 1
image[image <= 0.4] = 0
indices = image == 1
grid = np.indices((3,3))
# get corresponding x, y corrdinates
grid_x = grid[0][indices]
grid_y = grid[1][indices]