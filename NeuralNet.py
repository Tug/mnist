import copy
import math
from numpy import *
from numpy.random import *


def gout(x):
    return array(1.)/(1+exp(-x))

def dgout(x):
    return exp(-x)/(1+exp(-x))**2;

def ghidden(x):
    return array(1.)/(1+exp(-x+5)) + array(1.)/(1+exp(-x-5)) - 1

def dghidden(x):
    return exp(-x+5)/(1+exp(-x+5))**2 + exp(-x-5)/(1+exp(-x-5))**2

#def retroError(w, d):
#    if len(x.shape) == 1:
#        x = array([x])
#    if len(y.shape) == 1:
#        y = array([y]).T
#    if len(y) == 1:
#        return x * y[0]
#    else:
#        return dot(x, y)

class NeuralNet:

    def __init__(self, S, eta=0.1, m=0.9, wrange=2):
        self.S = S
        self.eta = eta
        self.m = m
        self.nbLayers = len(S)
        self.w = []
        self.Dw = []

        # Add a bias neuron for each layer except the last one
        for i in range(self.nbLayers-1):
            self.w.append( wrange*(2*rand(S[i+1], S[i]+1)-1) )
            self.Dw.append( zeros( (S[i+1],S[i]+1) ) )
            #self.S[i] = S[i]+1

        #self.w[0] = array([[0.13776874061,  -0.0317713676677, 0.00450988854744],[0.103181761176, -0.0964332998828, -0.0380263450198]])
        #self.w[1] = array([[1.13519435614, -0.786749095684, 1.48263390523]])

    def backProp(self, xin, tout):
        # xin and tout are one dimensional list
        #

        x = range(self.nbLayers)
        h = range(self.nbLayers-1)
        out = self.nbLayers-1
        x[0] = append(xin, 1)

        for n in range(out-1):
            h[n] = dot(self.w[n], transpose(x[n]))
            x[n+1] = append(ghidden(h[n]), 1)
        h[out-1] = dot(self.w[out-1], transpose(x[out-1]))
        x[out] = gout(h[out-1])

        d = array([ dgout(h[out-1]) * (tout - x[out]) ]).T
        self.Dw[out-1] = self.eta * (d * x[out-1])

        for n in range(out-1,0,-1):
            d = array([ dghidden(h[n-1]) ]).T * dot(self.w[n][:,:self.S[n]].T, d)
            self.Dw[n-1] = self.eta * (d * x[n-1]) + self.m * self.Dw[n-1];

        for i in range(len(self.w)):
            self.w[i] += self.Dw[i]

        E = 0.5 * sum( (tout - x[out])**2 )
        return E

    def learn(self, xins, touts):
        E = 0
        for i in range(len(xins)):
            E += self.backProp(xins[i], touts[i])
        return E

    def test(self, xin):
        x = append(xin, 1)
        out = self.nbLayers-1
        for n in range(out-1):
            x = append(ghidden(dot(self.w[n], transpose(x))), 1)
        xout = gout(dot(self.w[out-1], transpose(x)))
        return 1*(xout > 0.5)

