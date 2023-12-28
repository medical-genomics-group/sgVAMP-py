from sgvamp import VAMP
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import r2_score
import time
import argparse
import scipy
import struct
from mpi4py import MPI

# Test run for sgvamp
print("### VAMP for summary statistics ###\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-ld_files", "--ld-files", help = "Path to LD matrices in the form of .npz file, separated by commas")
parser.add_argument("-r_files", "--r-files", help = "Path to XTy .npy files, separated by commas")
parser.add_argument("-true_signal_file", "--true-signal-file", help = "Path to true signal .npy file")
parser.add_argument("-out_dir", "--out-dir", help = "Output directory")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-N", "--N", help = "Number of samples")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=10)
parser.add_argument("-gamw", "--gamw", help = "Initial noise precision", default=5)
parser.add_argument("-gam1", "--gam1", help = "Initial signal precision", default=100)
parser.add_argument("-lam", "--lam", help = "Initial sparsity", default=0.5)
parser.add_argument("-lmmse_damp", "--lmmse-damp", help = "Use LMMSE damping", default=True)
parser.add_argument("-learn_gamw", "--learn-gamw", help = "Learn or fix gamw", default=True)
parser.add_argument("-rho", "--rho", help = "Damping factor rho", default=0.5)
parser.add_argument("-cg_maxit", "--cg-maxit", help = "CG max iterations", default=500)
parser.add_argument("-s", "--s",  help = "Rused = (1-s) * R + s * Id", default=0.1)
parser.add_argument("-sigmas", "--sigmas",  help = "variance groups for the slab mixtures, separated by commas", default="1")
parser.add_argument("-p_weights", "--p_weights",  help = "probability weights for the slab mixtures, separated by commas", default="1")
args = parser.parse_args()

# Input parameters
ld_fpaths = args.ld_files.split(",")
r_fpaths = args.r_files.split(",")
true_signal_fpath = args.true_signal_file
out_dir = args.out_dir
out_name = args.out_name
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
iterations = int(args.iterations)
gamw = float(args.gamw) # Initial noise precision
gam1 = float(args.gam1) # initial signal precision
lam = float(args.lam) # Sparsity for simulations
rho = float(args.rho) # damping
lmmse_damp = bool(int(args.lmmse_damp)) # damping
learn_gamw = bool(int(args.learn_gamw)) # whether to learn gamw or not 
cg_maxit = int(args.cg_maxit) # CG max iterations
rho = float(args.rho) # damping
s = float(args.s) # regularization parameter for the correlation matrix
sigmas = np.array([float(x) for x in args.sigmas.split(",")]) # variance groups for the prior distribution
p_weights = np.array([float(x) for x in args.p_weights.split(",")]) # variance groups for the prior distribution
L = len(sigmas) # number of mixture components 
K = len(ld_fpaths) # numer of GWAS studies

# initializing MPI processes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

print("--ld-file", ld_fpaths[rank])
print("--lmmse-damp", lmmse_damp)
print("--learn-gamw", learn_gamw)
print("--cg-maxit", cg_maxit)
print("\n", flush=True)

# Loading LD matrix and XTy vector
print("...loading LD matrix and XTy vector", flush=True)
R = scipy.sparse.load_npz(ld_fpaths[rank])
R = (1-s) * R + s * scipy.sparse.identity(M)
r = np.loadtxt(r_fpaths[rank]).reshape((M,1))
print("LD matrix and XTy loaded. Shapes: ", R.shape, r.shape, flush=True)

# Loading true signals
f = open(true_signal_fpath, "rb")
buffer = f.read(M * 8)
beta = struct.unpack(str(M)+'d', buffer)
beta = np.array(beta).reshape((M,1))
beta *= np.sqrt(N)
print("True signals loaded. Shape: ", beta.shape, flush=True)

comm.Barrier()

# sgVAMP init
sgvamp = VAMP(lam=lam, rho=rho, gam1=gam1, sigmas=sigmas, p_weights=p_weights, gamw=gamw, out_dir=out_dir, out_name=out_name)

# Inference
print("...Running sgVAMP\n", flush=True)
ts = time.time()

xhat1 = sgvamp.infer(R, r, M, N, K, iterations, cg_maxit=cg_maxit, learn_gamw=learn_gamw, lmmse_damp=lmmse_damp, Comm=comm)

print("\n")
te = time.time()

comm.Barrier()

# Print running time and metrics
if rank==0:
    print("sgVAMP total running time: %0.4fs \n" % (te - ts), flush=True)

    # Print metrics
    corrs = []
    l2s = []

    for it in range(iterations):
        corr = np.corrcoef(xhat1[it].squeeze(), beta.squeeze()) # Pearson correlation coefficient of xhat1 and true signal beta
        corrs.append(corr[0,-1])
        l2 = np.linalg.norm(xhat1[it].squeeze() - beta.squeeze()) / np.linalg.norm(beta.squeeze()) # L2 norm error
        l2s.append(l2)

    print("Corr(x1hat,beta) over iterations: \n", corrs)
    print("L2 error (x1hat, beta) over iterations: \n", l2s, flush=True)