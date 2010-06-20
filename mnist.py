import os
from scipy import *
import struct
from pylab import *
from random import choice


def getLabels(filename):
    fp = file(filename)
    magicnumber, length = struct.unpack('>ii', fp.read(8))
    assert magicnumber in (2049, 2051), ("Not an MNIST file: %i" % magicnumber)
    for _ in xrange(length):
        label, = struct.unpack('B', fp.read(1))
        yield label


def getImages(filename):
    fp = file(filename)
    chunk = fp.read(16)
    magicnumber, length, numrows, numcols = struct.unpack('>iiii', chunk)
    assert magicnumber in (2049, 2051), ("Not an MNIST file: %i" % magicnumber)
    imagesize = numrows * numcols
    for _ in xrange(length):
        imagestring = fp.read(imagesize)
        image = struct.unpack('B' * imagesize, imagestring)
        yield array(image)/255.0


def flaggedArrayByIndex(idx, length):
    arr = zeros(length)
    arr[idx] = 1.
    return arr


def makeMnistDataSet(image_file, label_file):
    """Return an mnist dataset."""
    images = getImages(image_file)
    #labels = (flaggedArrayByIndex(l, 10) for l in getLabels(label_file))
    labels = getLabels(label_file)
    return zip(images, labels)

def displaySomeSamples(dataset):
    figure(1)
    clf()
    for index in range(1,26):
        ax = subplot(5,5,index)
        image = choice(dataset)[0]
        imshow(image.reshape((28,28)), cmap=cm.gray, )
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())


#     this function loads the training sets and displays 25 random images from it
def run():
    dataset = makeMnistDataSet('train-images.idx3-ubyte','train-labels.idx1-ubyte')
    testset = makeMnistDataSet('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')
    displaySomeSamples(dataset)

