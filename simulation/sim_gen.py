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
args = parser.parse_args()

# Input parameters
outfpath = args.out
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples

# Simmulations
print("...Simulating data\n")
X = np.random.binomial(2, p=0.4, size=[N,M])
X = (X - np.mean(X,axis=0)) / np.std(X, axis=0) # Standardization
X /= np.sqrt(N)

np.save(outpath + "_gen.npy", X) # save simulated genotype matrix X