import numpy as np
import argparse
import random

# This script generates genotype and phenotype for multiple cohorts

# Test run for sgvamp
print("...Simulating data for sgVAMP\n")

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-out", "--out", help = "Output path")
parser.add_argument("-N", "--N", help = "Number of samples")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-h2", "--h2", help = "Heritability used in simulations", default=0.8)
parser.add_argument("-lam", "--lam", help = "Sparsity (lambda) used in simulations", default=0.5)
parser.add_argument("-K", "--K", help = "Number of cohorts", default=2)
args = parser.parse_args()

# Input parameters
outfpath = args.out
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
h2 = float(args.h2) # heritability for simulations
lam = float(args.lam) # Sparsity for simulations
K = int(args.K) # Number of cohorts

cm = int(M * lam) # Number of causal markers
bvar = h2 / cm # beta variance
idx = random.sample(range(M), cm) # indices of causal markers
beta = np.zeros((M,1)) # true signals beta
beta[idx,0] = np.random.normal(0,np.sqrt(bvar),cm)
np.save("%s_bet.npy" % (outfpath), beta) # save true signal

# Simmulations
for i in range(K):
    print("...Cohort %d\n" % i, flush=True)
    print("...Simulating data\n")
    X = np.random.binomial(2, p=0.4, size=[N,M])
    X = (X - np.mean(X,axis=0)) / np.std(X, axis=0) # Standardization

    # Simulating phenotype
    print("...Simulating phenotype\n", flush=True)
    g = X @ beta
    print("Var(g) =", np.var(g), flush=True)
    w = np.random.normal(loc=0.0, scale=np.sqrt(1 - h2), size=[N,1])
    y = g + w
    print("Var(y) =", np.var(y), flush=True)
    print("h2 =", np.var(g) / np.var(y), flush=True)

    #y = (y - np.mean(y)) / np.std(y) # y standardization

    X /= np.sqrt(N) # normalization
    r = X.T @ y # marginal estimate vector
    R = X.T @ X # full LD matrix

    print("...Saving data to files\n", flush=True)
    np.save("%s_%d_phen.npy" % (outfpath, i), y) # save phenotype y
    np.save("%s_%d_r.npy" % (outfpath, i), r) # save XTy vector
    np.save("%s_%d_R.npy" % (outfpath, i), R) # save simulated ld matrix
    print("\n", flush=True)