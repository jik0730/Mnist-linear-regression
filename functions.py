import autograd.numpy as np
import random

"""
Description: Compute softmax funciton.
Input_1 (np.array(10,?)): Linear equation(Wx + b).
Output_1 (np.array(10,?)): Output of softmax(y).
"""
def softmax(x):
    max = np.amax(x, axis=0)
    z = x - max
    return np.exp(z) / np.sum(np.exp(z), axis=0)


"""
Description: Loss function. (NOT USED)
Input_1 (np.array(?,10)): Actual distribution of y.
Input_2 (np.array(?,10)): Expected distribution of y.
Output_1 (float): Cross-entropy scalar value.
"""
def cross_entropy(y_, y):
    #return np.sum(np.sum(y_*np.log(y), axis=1), axis=0) / y_.shape[0]
    return np.mean(np.sum(y_*np.log(y), axis=1))


"""
Description: Regression function.
Input_1 (np.array(?,785)): Batch input.
Input_2 (np.array(785,10)): Parameter W.
Output_1 (np.array(?,10)): Expected distribution of y.
"""
def regression(x, W):
    # return np.transpose(softmax(np.transpose(np.matmul(x, W) + b)))
    return np.transpose(softmax(np.transpose(np.matmul(x, W))))


"""
Description: Output batch of size 100.
Input_1 (np.array(?,784)): Input images data set.
Input_1 (np.array(?,10)): Input labels data set.
Output_1 (np.array(100,784), np.array(100, 10)): Batch of size 100.

Suggestion: Maybe need to prevent replicates of random rows.
"""
def batch(img, lab):
    toReturn1 = np.empty((0,784))
    toReturn2 = np.empty((0,10))
    for i in range(200):
        r = random.randint(1, img.shape[0]-1)
        toReturn1 = np.append(toReturn1, [img[r]], axis=0)
        toReturn2 = np.append(toReturn2, [lab[r]], axis=0)
    return toReturn1, toReturn2




