
from NeuralNet import *

def xor():
    inputs = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    targets = array([[0], [1], [1], [0]])
    net = NeuralNet([2, 2, 1], 0.1, 0.95, 2)
    Em1 = 1
    for i in xrange(20000):
        E = net.learn(inputs, targets)
        if(i % 100 == 0):
            print "Error =", E
        Em1 = E
    for inp in inputs:
        print inp, '->', net.test(inp)


if __name__ == "__main__":
    xor()