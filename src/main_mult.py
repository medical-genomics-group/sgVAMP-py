from sgvamp import VAMP, multiVAMP
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import r2_score
import time
import argparse
import scipy
import struct

# Test run for sgvamp
print("...Test run of VAMP for summary statistics\n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-ld_files", "--ld-files", help = "Path to LD matrices in .npz files, separated by comma ")
parser.add_argument("-r_files", "--r-files", help = "Path to XTy .npy files separated by comma")
parser.add_argument("-true_signal_file", "--true-signal-file", help = "Path to true signal .npy file")
parser.add_argument("-out_dir", "--out-dir", help = "Output directory")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-N", "--N", help = "Number of samples")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=10)
parser.add_argument("-K", "--K", help = "Number of cohorts", default=1)
parser.add_argument("-sigmas", "--sigmas", help = "Variances of different cohorts", default="")
parser.add_argument("-p_weights", "--p-weights", help = "Prior weights of different cohorts", default="")
parser.add_argument("-gamw", "--gamw", help = "Initial noise precision", default=5)
parser.add_argument("-gam1", "--gam1", help = "Initial signal precision", default=100)
parser.add_argument("-lam", "--lam", help = "Initial sparsity", default=0.5)
parser.add_argument("-lmmse_damp", "--lmmse-damp", help = "Use LMMSE damping", default=True)
parser.add_argument("-learn_gamw", "--learn-gamw", help = "Learn or fix gamw", default=True)
parser.add_argument("-rho", "--rho", help = "Damping factor rho", default=0.5)
parser.add_argument("-cg_maxit", "--cg-maxit", help = "CG max iterations", default=500)
parser.add_argument("-s", "--s",  help = "Rused = (1-s) * R + s * Id", default=0.1)
args = parser.parse_args()

# Input parameters
ld_fpaths = args.ld_files
r_fpaths = args.r_files
true_signal_fpath = args.true_signal_file
out_dir = args.out_dir
out_name = args.out_name
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
iterations = int(args.iterations)
K = int(args.K)
sigmas = args.sigmas
p_weights = args.p_weights
gamw = float(args.gamw) # Initial noise precision
gam1 = float(args.gam1) # initial signal precision
lam = float(args.lam) # Sparsity for simulations
rho = float(args.rho) # damping
lmmse_damp = bool(int(args.lmmse_damp)) # damping
learn_gamw = bool(int(args.learn_gamw)) # wheter to learn or not gamw
cg_maxit = int(args.cg_maxit) # CG max iterations
rho = float(args.rho) # damping
s = float(args.s) # regularization parameter for the correlation matrix

print("--ld-file", ld_fpaths)
print("--lmmse-damp", lmmse_damp)
print("--learn-gamw", learn_gamw)
print("--cg-maxit", cg_maxit)
print("\n", flush=True)

# Loading LD matrix and XTy vector
print("...loading LD matrix and XTy vector", flush=True)
R_list = []
r_list = []
sigma_list = []
p_weight_list = []

ld_fpath_list = ld_fpaths.split(",")
r_fpath_list = r_fpaths.split(",")
sigma_list = [float(x) for x in sigmas.split(",")] # variance groups for the prior distribution
p_weight_list = [float(x) for x in p_weights.split(",")] # variance groups for the prior distribution

if len(ld_fpath_list) != K:
    raise Exception("Specified number of cohorts is not equal to number of LD matrices provided!")
if len(r_fpath_list) != K:
    raise Exception("Specified number of cohorts is not equal to number of marginal estimates provided!")

for i in range(K):
    ld_fpath = ld_fpath_list[i]
    r_fpath = r_fpath_list[i]

    #R = scipy.sparse.load_npz(ld_fpath)
    R = np.load(ld_fpath)
    R = (1-s) * R + s * scipy.sparse.identity(M)
    R_list.append(R)
    r = np.load(r_fpath)
    #r = np.loadtxt(r_fpath).reshape((M,1))
    r_list.append(r)

print("LD matrix and XTy loaded. Shapes: ", R.shape, r.shape, flush=True)

# multi-cohort sgVAMP init
sgvamp = multiVAMP(lam=lam, rho=rho, gam1=gam1, gamw=gamw, sigmas=sigma_list, p_weights=p_weight_list, out_dir=out_dir, out_name=out_name)

# Inference
print("...Running multi-cohort sgVAMP\n", flush=True)
ts = time.time()

xhat1 = sgvamp.infer(R_list, r_list, M, N, iterations, K, cg_maxit=cg_maxit, learn_gamw=learn_gamw, lmmse_damp=lmmse_damp)

print("\n")
te = time.time()

# Print running time
print("multi-cohort sgVAMP total running time: %0.4fs \n" % (te - ts), flush=True)