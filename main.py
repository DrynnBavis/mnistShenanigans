import numpy as np
import matplotlib.pyplot as plt

x = np.array([[0,0,1],
              [0,1,1],
              [1,0,1],
              [1,1,1]])

y = np.array([[0,0,1,1]]).T #Just transposing to make it 4x1

#plt.matshow(np.hstack((x,y)), fignum=10, cmap=plt.cm.gray)
#plt.show()

#sigmoid
def nonLin(x, deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

np.random.seed()

#initialize synapse
syn0 = 2*np.random.random((3,1)) - 1

print(x)

for iter in range(10000):
    # forward prop
    layer0 = x
    layer1 = nonLin(np.dot(layer0, syn0)) #simply multiplying input x weight

    # calculate error
    layer1_error = y - layer1

    # backward prop
    layer1_delta = layer1_error * nonLin(layer1, True);

    # update weights
    syn0 += np.dot(layer0.T, layer1_delta)

print("Output: ")
print(layer1)
