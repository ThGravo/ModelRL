import numpy as np

a = np.zeros((4,4))

file = "file1.npy"
np.save(file,a)

file.seek(0)


np.load(file)
