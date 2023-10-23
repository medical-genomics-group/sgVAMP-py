from sgvamp import VAMP
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import r2_score
import time
import argparse

# Test run for sgvamp
print("...Test run of VAMP for summary statistics\n")

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-N", "--N", help = "Number of samples")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=10)
parser.add_argument("-h2", "--h2", help = "Heritability used in simulations", default=0.5)
parser.add_argument("-lam", "--lam", help = "Sparsity (lambda) used in simulations", default=0.5)
args = parser.parse_args()

# Input parameters
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
iterations = int(args.iterations)
h2 = float(args.h2) # heritability for simulations
lam = float(args.lam) # Sparsity for simulations

# Simmulations
print("...Simulating data\n")
X = np.random.binomial(2, p=0.4, size=[N,M])
X = (X - np.mean(X,axis=0)) / np.std(X, axis=0) # Standardization
X /= np.sqrt(N)
beta = np.random.normal(loc=0.0, scale=1.0, size=[M,1]) # scale = standard deviation
beta *= np.random.binomial(1, lam, size=[M,1])
g = X @ beta
print("Var(g) =", np.var(g))
w = np.random.normal(loc=0.0, scale=np.sqrt(1/h2 - 1), size=[N,1])
y = g + w
print("Var(y) =", np.var(y))
print("h2 =", np.var(g) / np.var(y))
print("\n")

y = (y - np.mean(y)) / np.std(y) # y standardization

# TODO Here we should load LD matrix
R = X.T @ X
r = X.T @ y

# sgVAMP init
sgvamp = VAMP(lam=lam, rho=0.5, gam1=100, gamw=1/h2)

# Inference
print("...Running sgVAMP\n")
ts = time.time()
xhat1, gamw = sgvamp.infer(R, r, M, N, iterations, est=False)
print("\n")
te = time.time()

# Print running time
print("sgVAMP total running time: %0.4fs \n" % (te - ts))

# Print metrics
R2s = []
corrs = []
for it in range(iterations):
    yhat = X @ xhat1[it]
    R2 = r2_score(y, yhat) # R squared metric
    R2s.append(R2)

    corr = np.corrcoef(xhat1[it].squeeze(), beta.squeeze()) # Pearson correlation coefficient of xhat1 and true signal beta
    corrs.append(corr[0,-1])
print("R2 over iterations: \n", R2s)
print("Corr(x1hat,beta) over iterations: \n", corrs)
print("gamw over iterations: \n", gamw)