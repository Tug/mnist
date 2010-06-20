import mnist
import numpy as np
import numpy.random as random
import pylab
import math

dataset = mnist.makeMnistDataSet('train-images.idx3-ubyte','train-labels.idx1-ubyte')
testset = mnist.makeMnistDataSet('t10k-images.idx3-ubyte','t10k-labels.idx1-ubyte')

def filterDataSet(dataset, label):
    return filter(lambda data: data[1] == label, dataset)

def run():
    our_dataset = filterDataSet(dataset, 1)
    images = np.array([data[0] for data in our_dataset])
    ws, mus, sigs2 = ExpectationMaximization(images)
    our_dataset = filterDataSet(testset, 1)
    N = np.alen(dataset[0])
    images = np.array([data[0] for data in our_dataset])
    Elikelihood = np.zeros(mus, N)
    for i in range(mus):
        x_m_mu2 = our_dataset - mus[i]
        Elikelihood[i,:] = np.log(np.sum(ws[i] * gaussian2(x_m_mu2, sigs2[i], N)))
    gids = np.argmax(Elikelihood, 0)

def gaussian(dataset, mu, sigma2, N):
    detS = np.prod(sigma2)
    invS = 1/sigma2
    x_m_mu2s = (dataset - mu) ** 2
    a = 1/sqrt( (2*math.pi)**N * detS )
    return np.array([ a * 0.5 * np.dot(invS, x_m_mu2) for x_m_mu2 in x_m_mu2s])

def gaussian2(dataset, mu, sigma2, N):
    detS = np.prod(sigma2)
    invS = 1/sigma2
    x_m_mu2s = (dataset - mu) ** 2
    loga = (2.0/N) * np.log(2*math.pi) + 2 * np.log(detS)
    logval = []
    for x_m_mu2 in x_m_mu2s:
        logNs = invS * x_m_mu2
        logNs -= np.max(logNs)
        logval.append(loga - 0.5 * np.sum(logNs))
    return np.exp(np.array(logval))

def initParameters(m, N):
    mu = 255*random.rand(m, N)
    sigma2 = random.rand(m, N)+1
    w = random.rand(m)/N
    return (mu, sigma2, w)

def ExpectationMaximization(dataset):
    # dimension of the space
    N = np.alen(dataset[0])
    m = 10
    minw = 0.01
    minsigma = 0.01
    # mu: esperance
    # sigma2: variance
    # w: mixing weight
    mu, sigma2, w = initParameters(m, N)
    epsi = 0.1
    conv = False
    while not conv:
        Elikelihood = 0
        # for each mixture component
        for j in range(m):
            # Expectation
            # gamma: responsibility values
            gamma = w[j] * gaussian2(dataset, mu[j], sigma2[j], N)
            Nwj = np.sum(gamma)
            gamma = gamma/Nwj
            # Maximization (of the likelihood)
            gammat = np.array([gamma]).T
            mu[j]     = np.sum( gammat * dataset, 0 ) / Nwj
            sigma2[j] = np.sum( gammat * ((dataset - mu[j]) ** 2), 0 ) / Nwj
            w[j]      = Nwj/N
            # prevent variances from reaching 0
            sigma2[j] = map(lambda sig2: sig2 * (sig2 >= minsigma) or minsigma, sigma2[j])
            # prevent mixin coefficient from reaching 0
            if w[j] < minw:
                w[j] = minw
            Elikelihood -= np.log(Nwj)
        print Elikelihood
        conv = np.abs(Elikelihood) < epsi
    return (w, mu, sigma2)

if __name__ == "__main__":
    run()

