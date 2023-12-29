from numpy import random
import numpy as np
import scipy

def sim_beta(m, la, sigma): 
    beta = random.normal(loc=0.0, scale=np.sqrt(sigma[0]), size=[m,1]) # scale = standard deviation
    beta *= random.binomial(1, la, size=[m,1])
    return beta

def sim_r(R, beta): # assumes variance of phenotype is 1
    # r = R * beta + e; e=N(0,1/gamw)
    # beta is mx1 vector 
    # mu is nx1 vector 
    m = len(beta)
    g = np.matmul(R, beta)
    sigmaG = np.var(g)
    e = random.normal(loc=0.0, scale=np.sqrt(1-sigma[0]), size=[m,1])
    r = g + e
    return r

def sim_linear(m, R, h2, CV):
    sigma = h2/CV
    la = CV/m
    beta = sim_beta(m, la, sigma)
    r = sim_r(R, beta) 
    return r,beta
