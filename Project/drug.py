import scipy.io
import numpy as np
import pickle

FILE="drug.pickle"

mat=scipy.io.loadmat('DtiData.mat')
# print(type(mat))

print(np.asarray(mat['DTIMatrix']).shape)
print(np.unique(mat['DTIMatrix']))

pickle.dump(mat['DTIMatrix'], open(FILE, 'wb'))