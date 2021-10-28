import numpy as np 

array = np.fromfile("scores.raw",dtype=np.float32).reshape(3000,21)
print(array.shape)
