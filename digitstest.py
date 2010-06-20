import mnist
from numpy import *
from pylab import *
from NeuralNet import *

our_digits = [3, 7, 1]
sep = 0.5
nbDigit = len(our_digits)
our_digits_bin = dict([(our_digits[i], mnist.flaggedArrayByIndex(i, nbDigit)) for i in range(nbDigit)])

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

def plotError(errors, t):
    xlabel('epoch')
    ylabel('MSE')
    title(t)
    plot(errors,'.')
    savefig(t+".png")
    #show()

def normalize(dataset):
    D = array([data[0] for data in dataset])
    u = mean(D, 0)
    v = var(D, 0)
    v = map(lambda vd: 1*(vd == 0) or vd, v)
    D = (D-u)/v

def test(net, testset):
    nbError = 0
    for i in xrange(len(testset)):
        r = net.test(testset[i][0])
        if reverseFlaggedArray(r) != testset[i][1]:
            nbError += 1
    Egen = (nbError+0.0)/len(testset)
    return Egen

def digitstest():
    our_dataset = filter(isOneOfOurDigit, dataset)
    our_testset = filter(isOneOfOurDigit, testset)
    normalize(our_dataset)
    normalize(our_testset)
    imagelen = len(our_dataset[0][0])
    eta = 0.01
    m = 0.9
    hiddenN = 600
    limit = int(len(our_dataset)*sep)
    print "original training set: ", len(our_dataset)
    print "training set: ", limit+1, "validation set: ", len(our_dataset) - limit
    print "test set", len(our_testset)
    net = NeuralNet([imagelen, 500, nbDigit], eta, m, 0.1)
    errors = []
    meanError = 1
    meanWidth = 2000
    # training
    for i in xrange(1, limit):
        E = net.backProp(our_dataset[i][0], our_digits_bin[our_dataset[i][1]])
        meanErrorBak = meanError
        meanError = ((meanWidth-1)*meanError + E)/meanWidth
        errors.append(meanError)
    plotError(errors, "error training")
    errors = []
    meanError = 1
    meanErrorBak = 1
    h = 50
    # validation
    for i in xrange(limit, len(our_dataset)):
        E = net.backProp(our_dataset[i][0], our_digits_bin[our_dataset[i][1]])
        meanError = ((meanWidth-1)*meanError + E)/meanWidth
        errors.append(meanError)
        if i % h == 0:
            slope = (meanError - meanErrorBak)/h
            meanErrorBak = meanError
            if slope > -0.0001:
                print "early-stopping %d/%d" %(i+1-limit, len(our_dataset)-limit)
                break
    plotError(errors, "Error training+validation (eta=%1.2f, m=%1.2f, n=%d)" %(eta, m, hiddenN))
    Egen = test(net, our_testset)
    print "%f;%f;%d;%f" %(eta, m, hiddenN, Egen*100)

if __name__ == "__main__":
    digitstest()

