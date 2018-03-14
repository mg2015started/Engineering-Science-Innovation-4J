import numpy.random as nr
import numpy as np

a = nr.uniform(-1,3,size=(500,2))
b = nr.uniform(-5,-1,size=(500,2))

np.savez('array_archive.npz', a=a, b=b)