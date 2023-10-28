import numpy as np
import argparse

# TODO This script is not tested

# Test run for sgvamp
print("...Simulating data for sgVAMP\n")

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-out", "--out", help = "Output path")
parser.add_argument("-N", "--N", help = "Number of samples")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-h2", "--h2", help = "Heritability used in simulations", default=0.8)
parser.add_argument("-lam", "--lam", help = "Sparsity (lambda) used in simulations", default=0.5)
args = parser.parse_args()

# Input parameters
outfpath = args.out
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
h2 = float(args.h2) # heritability for simulations
lam = float(args.lam) # Sparsity for simulations

# Simmulations
print("...Simulating data\n")
X = np.random.binomial(2, p=0.4, size=[N,M])
X = (X - np.mean(X,axis=0)) / np.std(X, axis=0) # Standardization
X /= np.sqrt(N)

# Simulating phenotype
print("...Simulating phenotype\n")
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

r = X.T @ y

np.save(outfpath + "_phen.npy", y) # save phenotype y
np.save(outfpath + "_bet.npy", beta) # save true signal
np.save(outfpath + "_r.npy", r) # save XTy vector
np.save(outfpath + "_R.npy", X.T @ X) # save simulated ld matrix