
from NeuralNet import *

def xor():
    inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = array([[0, 1], [1, 0], [1, 0], [0, 1]])
    net = NeuralNet([2, 2, 2], 0.1, 0.99)
    for i in xrange(10000):
        E = net.learn(inputs, targets)
        if(i % 100 == 0):
            print "Error =", E
    for inp in inputs:
        print inp, '->', net.test(inp)


if __name__ == "__main__":
    xor()