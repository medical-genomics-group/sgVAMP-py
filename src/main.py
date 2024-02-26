from sgvamp import VAMP
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import r2_score
import time
import argparse
import scipy
import struct
import logging
from mpi4py import MPI

# Initializing MPI processes
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size_MPI = comm.Get_size()

# Configuring logging options
logging.basicConfig(format='%(message)s', level=logging.DEBUG)

if rank == 0:
    logging.info(" ### VAMP for summary statistics ###\n")

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-ld_files", "--ld-files", help = "Path to LD matrices in .npz files, separated by comma ")
parser.add_argument("-r_files", "--r-files", help = "Path to XTy .npy files separated by comma")
parser.add_argument("-true_signal_file", "--true-signal-file", help = "Path to true signal .npy file", default=None)
parser.add_argument("-out_dir", "--out-dir", help = "Output directory")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
parser.add_argument("-N", "--N", help = "Number of samples in each cohort, saparated by comma")
parser.add_argument("-M", "--M", help = "Number of markers")
parser.add_argument("-K", "--K", help = "Number of cohorts", default=1)
parser.add_argument("-L", "--L", help = "Number of prior mixture components", default=2)
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=10)
parser.add_argument("-prior_vars", "--prior-vars", help = "Prior mixture variances of different cohorts", default="0,1")
parser.add_argument("-prior_probs", "--prior-probs", help = "Prior mixture probabilites of different cohorts", default="0.99,0.01")
parser.add_argument("-gamw", "--gamw", help = "Initial noise precision", default=5)
parser.add_argument("-gam1", "--gam1", help = "Initial signal precision", default=100)
parser.add_argument("-lmmse_damp", "--lmmse-damp", help = "Use LMMSE damping", default=True)
parser.add_argument("-learn_gamw", "--learn-gamw", help = "Learn or fix gamw", default=True)
parser.add_argument("-rho", "--rho", help = "Damping factor rho", default=0.5)
parser.add_argument("-cg_maxit", "--cg-maxit", help = "CG max iterations", default=500)
parser.add_argument("-s", "--s",  help = "Rused = (1-s) * R + s * Id", default=0.1)
parser.add_argument("-prior_update", "--prior-update",  help = "Learning prior probabilites", default=None)
parser.add_argument("-em_prior_maxit", "--em-prior-maxit",  help = "Maximal number of iterations that prior-learning EM is allowed to perform", default=100)
args = parser.parse_args()

# Input parameters
ld_fpaths = args.ld_files
r_fpaths = args.r_files
true_signal_fpath = args.true_signal_file
out_dir = args.out_dir
out_name = args.out_name
M = int(args.M) # Number of markers
Ns = args.N # Number of samples
iterations = int(args.iterations)
K = int(args.K)
L = int(args.L)
prior_vars = args.prior_vars
prior_probs = args.prior_probs
gamw = float(args.gamw) # Initial noise precision
gam1 = float(args.gam1) # initial signal precision
rho = float(args.rho) # damping
lmmse_damp = bool(int(args.lmmse_damp)) # damping
learn_gamw = bool(int(args.learn_gamw)) # wheter to learn or not gamw
cg_maxit = int(args.cg_maxit) # CG max iterations
rho = float(args.rho) # damping
s = float(args.s) # regularization parameter for the correlation matrix
prior_update = args.prior_update # whether or not to update prior probabilities
em_prior_maxit = int(args.em_prior_maxit) # prior-learning EM max iterations

ld_fpaths_list = ld_fpaths.split(",")
r_fpaths_list = r_fpaths.split(",")
N_list = [int(n) for n in Ns.split(",")]
N = N_list[rank]
prior_vars_list = [float(x) for x in prior_vars.split(",")] # variance groups for the prior distribution
prior_probs_list = [float(x) for x in prior_probs.split(",")] # probability groups for the prior distribution


if len(ld_fpaths_list) != K:
    raise Exception("Specified number of cohorts is not equal to number of LD matrices provided!")
if len(r_fpaths_list) != K:
    raise Exception("Specified number of cohorts is not equal to number of marginal estimates provided!")
if len(prior_vars_list) != L:
    raise Exception("Number of prior variances must be L!")
if len(prior_probs_list) != L:
    raise Exception("Number of prior mixture probabilites must be L!")

if rank == 0:
    logging.info("Input arguments:")
    logging.info(f"--ld-files {ld_fpaths}")
    logging.info(f"--r-files {r_fpaths}")
    logging.info(f"--out-name {out_name}")
    logging.info(f"--out-dir {out_dir}")
    logging.info(f"--true-signal-file {true_signal_fpath}")
    logging.info(f"--N {Ns}")
    logging.info(f"--M {M}")
    logging.info(f"--K {K}")
    logging.info(f"--L {L}")
    logging.info(f"--iterations {iterations}")
    logging.info(f"--prior-vars {prior_vars}")
    logging.info(f"--prior-probs {prior_probs}")
    logging.info(f"--gam1 {gam1}")
    logging.info(f"--gamw {gamw}")
    logging.info(f"--lmmse-damp {lmmse_damp}")
    logging.info(f"--learn-gamw {learn_gamw}")
    logging.info(f"--rho {rho}")
    logging.info(f"--cg-maxit {cg_maxit}")
    logging.info(f"--s {s}")
    logging.info(f"--prior-update {prior_update}")
    if prior_update == "em":
        logging.info(f"--em_prior_maxit {em_prior_maxit}\n")

# Loading LD matrix and XTy vector
if rank == 0:
    logging.info(f"...loading LD matrix and XTy vector\n")

ld_fpath = ld_fpaths_list[rank]
r_fpath = r_fpaths_list[rank]

if ld_fpath.endswith('.npz'):
    R = scipy.sparse.load_npz(ld_fpath)
elif ld_fpath.endswith('.npy'):
    R = np.load(ld_fpath)
else: 
    raise Exception("Unsupported LD matrix format!")

logging.info(f"Rank {rank} loaded LD matrix with shape {R.shape}\n")

R = (1-s) * R + s * scipy.sparse.identity(M) # R regularization

if r_fpath.endswith('.txt'):
    r = np.loadtxt(r_fpath).reshape((M,1))
elif r_fpath.endswith('.npy'):
    r = np.load(r_fpath)
else:
    raise Exception("Unsupported XTy vector format!")

logging.info(f"Rank {rank} loaded XTy vector with shape {r.shape}\n")

# Loading true signals
x0 = np.zeros(M)
if true_signal_fpath != None:
    if true_signal_fpath.endswith(".bin"):
        f = open(true_signal_fpath, "rb")
        buffer = f.read(M * 8)
        x0 = struct.unpack(str(M)+'d', buffer)
        x0 = np.array(x0).reshape((M,1))
        x0 *= np.sqrt(N)
    elif true_signal_fpath.endswith(".npy"):
        x0 = np.load(true_signal_fpath)
        x0 *= np.sqrt(N) 
    else:
        raise Exception("Unsupported true signal format!")
    if rank == 0:
        logging.info(f"True signals loaded. Shape: {x0.shape}\n")
else:
    x0 = None

a = np.array(N_list) / sum(N_list) # scaling factor for group

# multi-cohort sgVAMP init
sgvamp = VAMP(  N=N,
                M=M,
                K=K,
                rho=rho, 
                gam1=gam1, 
                gamw=gamw,
                a=a, 
                prior_vars=prior_vars_list, 
                prior_probs=prior_probs_list, 
                out_dir=out_dir, 
                out_name=out_name,
                comm=comm)

# Inference
if rank == 0:
    logging.info("...Running sgVAMP\n")

ts = time.time()

xhat1 = sgvamp.infer(   R, 
                        r,
                        iterations,
                        x0=x0,
                        cg_maxit=cg_maxit, 
                        em_prior_maxit = em_prior_maxit,
                        learn_gamw=learn_gamw, 
                        lmmse_damp=lmmse_damp,
                        prior_update=prior_update)
te = time.time()

# Print running time
if rank == 0:
    logging.info(f"sgVAMP total running time: {(te - ts):0.4f}s\n")

if x0 is not None:
    # Print metrics
    alignments = []
    l2s = []

    for it in range(iterations):
        alignment = np.inner(xhat1[it].squeeze(), x0.squeeze()) / np.linalg.norm(xhat1[it].squeeze()) / np.linalg.norm(x0.squeeze())
        alignments.append(alignment)
        l2 = np.linalg.norm(xhat1[it].squeeze() - x0.squeeze()) / np.linalg.norm(x0.squeeze()) # L2 norm error
        l2s.append(l2)
    if rank == 0:
        logging.info(f"Alignment(x1hat, x0) over iterations: \n {alignments}\n")
        logging.info(f"L2 error(x1hat, x0) over iterations: \n {l2s}\n")

MPI.Finalize()