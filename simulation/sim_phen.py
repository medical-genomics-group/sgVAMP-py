import numpy as np
import argparse
import struct
import random
from bed_reader import open_bed, sample_file

# Phenotype simulation on the top of real data
print("...Phenotype simulation for sgVAMP\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-bed", "--bed", help = "Path to bed file")
parser.add_argument("-out", "--out", help = "Output path")
parser.add_argument("-N", "--N", help = "Number of samples")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-h2", "--h2", help = "Heritability used in simulations", default=0.8)
parser.add_argument("-lam", "--lam", help = "Sparsity (lambda) used in simulations", default=0.5)
args = parser.parse_args()

# Input parameters
bed_fpath = args.bed
outpath = args.out
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
h2 = float(args.h2) # heritability for simulations
lam = float(args.lam) # Sparsity for simulations

# Reading data from bed file
print("...Reading data from bed file %s\n" % bed_fpath, flush=True)

# Load X matrix from bed file
# use bed_reader librabry
bed = open_bed(bed_fpath)
X = bed.read()

# Standardization
X = (X - np.mean(X,axis=0)) / np.std(X, axis=0) 

# Simulating phenotype
print("...Simulating phenotype\n", flush=True)
CV = int(M * lam) # Number of causal markers
print("Causal variants = %d" % CV)

sigma2 = h2 / CV # beta variance
idx = random.sample(range(M), CV) # indices of causal markers
beta = np.zeros((M,1)) # true signals beta
beta[idx,0] = np.random.normal(0, np.sqrt(sigma2), CV)
g = X @ beta
print("Var(g) =", np.var(g))
gamw = 1 / (1 - h2)
sigma2w = 1 / gamw
w = np.random.normal(loc=0.0, scale=np.sqrt(sigma2w), size=[N,1])
y = g + w
print("Var(y) =", np.var(y))
print("h2 =", np.var(g) / np.var(y))
print("\n", flush=True)

X /= np.sqrt(N) # normalization
r = X.T @ y # marginal estimate vector

np.save(outpath + "_phen.npy", y) # save phenotype y
np.save(outpath + "_bet.npy", beta) # save true signal
np.save(outpath + "_r.npy", r) # save XTy vector