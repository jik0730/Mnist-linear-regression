import sys
import autograd.numpy as np
import matplotlib.pyplot as plt
from autograd import grad
from read_data import mnist
from functions import softmax
from functions import batch
from functions import regression


def main(tr_i, tr_l, te_i, te_l):
    if len(sys.argv) != 5:
        print("The number of input arguments should be 5.")
        return

    # Initialize data set
    training_images, training_labels, test_images, test_labels = mnist(tr_i, tr_l, te_i, te_l)

    # Initialize parameters
    W = np.zeros((785,10)) # W appended by bias(b)
    ones = np.ones((200,1))
    ones2 = np.ones((10000,1))
    plt_y = []
    
    # Cross entropy function
    def cross_entropy(W):
        y = np.transpose(softmax(np.transpose(np.matmul(batch_xs, W))))
        y += sys.float_info.min
        toReturn = np.mean(-np.sum(batch_ys*np.log(y), axis=1))
        plt_y.append(toReturn.value)
        return toReturn
    training_gradient = grad(cross_entropy)
    
    for i in range(1000):
        batch_xs, batch_ys = batch(training_images, training_labels)
        batch_xs = np.append(batch_xs, ones, axis=1)
        W -= training_gradient(W) * 0.0001

    # Plotting for testing
    #plt.plot(plt_y)
    #plt.show()

    y_ = np.argmax(np.transpose(test_labels), axis=0)
    test_images = np.append(test_images, ones2, axis=1)
    yy = np.argmax(np.transpose(regression(test_images, W)), axis=0)
    correct = np.mean(np.equal(y_, yy))
    print("Correctness: ", correct)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
