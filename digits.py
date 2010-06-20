import mnist
from numpy import *
from numpy.random import *
from NeuralNet import *


our_digits = [3, 7, 1]
nbDigit = len(our_digits)
our_digits_bin = dict([(our_digits[i], mnist.flaggedArrayByIndex(i, nbDigit)) for i in range(nbDigit)])

# these are array types !
dataset = mnist.makeMnistDataSet('train-images.idx3-ubyte','train-labels.idx1-ubyte')
testset = mnist.makeMnistDataSet('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')

def isOneOfOurDigit(data):
    return data[1] in our_digits

def reverseFlaggedArray(arr):
    i = argmax(arr)
    if arr[i] == 0:
        return -1
    else:
        return our_digits[i]

def digits():
    our_dataset = filter(isOneOfOurDigit, dataset)
    our_testset = filter(isOneOfOurDigit, dataset)
    imagelen = len(our_dataset[0][0])
    net = NeuralNet([imagelen, imagelen, nbDigit])
    for i in xrange(len(our_dataset)):
        E = net.backProp(our_dataset[i][0], our_digits_bin[our_dataset[i][1]])
        if(i % 100 == 0):
            print "Error =", E
    nbError = 0
    for i in xrange(len(our_testset)):
        r = net.test(our_testset[i][0])
        if reverseFlaggedArray(r) != our_testset[i][1]:
            nbError += 1
    print "error rate : ", (nbError+0.0)/len(our_testset)*100, "%"
if __name__ == "__main__":
    digits()
