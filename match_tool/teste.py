import numpy as np

data = np.load('a000420.npy')

res = np.where(np.all(data == data[147, 157], axis=2))
res = list(zip(res[0], res[1]))

