import numpy as np
import argparse
import struct

# Phenotype simulation on the top of real data
print("...Phenotype simulation for sgVAMP\n")

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
print("...Reading data from bed file %s\n" % bed_fpath)

# Load X matrix from bed file
X = np.zeros((N,M))
f = open(bed_fpath, "rb")
raw_data = np.fromfile(f, dtype=np.uint8)

#skip magic number
raw_data = raw_data[3:]

col_len_byte = N / 4 # One byte contatins genotype information of 4 individuals
if N % 4:
    col_len_byte = N / 4 + 1 

for j in range(M):
    for i in range(N):
        c = int(i / 4)
        c = int((j * col_len_byte) + c); # Position of byte containing i-th individual's genotype
        shift = (i % 4) * 2 # Bit shift within the byte
        d = raw_data[c] >> shift
        bit0 = bool(d & 0b00000001)
        bit1 = bool(d & 0b00000010)

        if bit0 and bit1: # 11 Homozygous for second allele
            X[i,j] = 2
        elif not bit0 and not bit1: # 00 Homozygous for first allele
            X[i,j] = 0
        elif not bit0 and bit1: # 10 Heterozygous
            X[i,j] = 1
        else: # 01 Missing genotype
            X[i,j] = 0 # TODO aproximate missing values
f.close()

# Standardization
X = (X - np.mean(X,axis=0)) / np.std(X, axis=0) 
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

np.save(outpath + "_phen.npy", y) # save phenotype y
np.save(outpath + "_bet.npy", beta) # save true signal
np.save(outpath + "_r.npy", r) # save XTy vector