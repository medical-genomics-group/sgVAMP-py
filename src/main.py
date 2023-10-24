from sgvamp import VAMP
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import r2_score
import time
import argparse
import scipy

# Test run for sgvamp
print("...Test run of VAMP for summary statistics\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-ld_file", "--ld-file", help = "Path to LD matrix")
parser.add_argument("-r_file", "--r-file", help = "Path to XTy file")
parser.add_argument("-N", "--N", help = "Number of samples")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=10)
parser.add_argument("-h2", "--h2", help = "Heritability used in simulations", default=0.5)
parser.add_argument("-lam", "--lam", help = "Sparsity (lambda) used in simulations", default=0.5)
args = parser.parse_args()

# Input parameters
ld_fpath = args.ld_file
r_fpath = args.r_file
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
iterations = int(args.iterations)
h2 = float(args.h2) # heritability for simulations
lam = float(args.lam) # Sparsity for simulations


# Loading LD matrix and XTy vector
print("...loading LD matrix and XTy vector", flush=True)
R = scipy.sparse.load_npz(ld_fpath).toarray()
r = np.load(r_fpath)
print("Data loaded. Shapes: ", R.shape, r.shape, flush=True)

# sgVAMP init
sgvamp = VAMP(lam=lam, rho=0.5, gam1=100, gamw=1/h2)

# Inference
print("...Running sgVAMP\n", flush=True)
ts = time.time()
xhat1, gamw = sgvamp.infer(R, r, M, N, iterations, est=True)
print("\n")
te = time.time()

# Print running time
print("sgVAMP total running time: %0.4fs \n" % (te - ts), flush=True)

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
print("gamw over iterations: \n", gamw, flush=True)