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
parser.add_argument("-true_signal_file", "--true-signal-file", help = "Path to true signal file")
parser.add_argument("-N", "--N", help = "Number of samples")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=10)
parser.add_argument("-gamw", "--gamw", help = "Initial noise precision", default=5)
parser.add_argument("-gam1", "--gam1", help = "Initial signal precision", default=100)
parser.add_argument("-lam", "--lam", help = "Initial sparsity", default=0.5)
parser.add_argument("-lmmse_damp", "--lmmse-damp", help = "Use LMMSE damping", default=True)
parser.add_argument("-learn_gamw", "--learn-gamw", help = "Learn or fix gamw", default=True)
parser.add_argument("-rho", "--rho", help = "Damping factor rho", default=0.5)
parser.add_argument("-ld_format", "--ld-format", help = "LD matrix format (npy or npz)", default='npz')
args = parser.parse_args()

# Input parameters
ld_fpath = args.ld_file
r_fpath = args.r_file
true_signal_fpath = args.true_signal_file
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
iterations = int(args.iterations)
gamw = float(args.gamw) # Initial noise precision
gam1 = float(args.gam1) # initial signal precision
lam = float(args.lam) # Sparsity for simulations
ld_format = args.ld_format # npy or npz
rho = float(args.rho) # damping
lmmse_damp = bool(int(args.lmmse_damp)) # damping
learn_gamw = bool(int(args.learn_gamw)) # wheter to learn or not gamw

print("--ld-file", ld_fpath)
print("--lmmse-damp", lmmse_damp)
print("--learn-gamw", learn_gamw)
print("\n", flush=True)

rho = float(args.rho) # damping
# Loading LD matrix and XTy vector
print("...loading LD matrix and XTy vector", flush=True)
if ld_format == 'npz':
    R = scipy.sparse.load_npz(ld_fpath).toarray()
elif ld_format == 'npy':
    R = np.load(ld_fpath)["arr_0"]
else:
    raise Exception("Unsupported LD format!")
r = np.load(r_fpath)
print("LD matrix and XTy loaded. Shapes: ", R.shape, r.shape, flush=True)

# Loading true signals
beta = np.load(true_signal_fpath)
print("True signals loaded. Shape: ", beta.shape, flush=True)

# sgVAMP init
sgvamp = VAMP(lam=lam, rho=rho, gam1=gam1, gamw=gamw)

# Inference
print("...Running sgVAMP\n", flush=True)
ts = time.time()
xhat1, gamw = sgvamp.infer(R, r, M, N, iterations, est=True, cg=True, learn_gamw=learn_gamw, lmmse_damp=lmmse_damp)
print("\n")
te = time.time()

# Print running time
print("sgVAMP total running time: %0.4fs \n" % (te - ts), flush=True)

# Print metrics
corrs = []
for it in range(iterations):
    corr = np.corrcoef(xhat1[it].squeeze(), beta.squeeze()) # Pearson correlation coefficient of xhat1 and true signal beta
    corrs.append(corr[0,-1])
print("Corr(x1hat,beta) over iterations: \n", corrs)
print("gamw over iterations: \n", gamw, flush=True)