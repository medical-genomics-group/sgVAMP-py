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
import pandas as pd
#import dask.dataframe as dd
import os

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
parser.add_argument("-M", "--M", help = "Number of markers in each cohort, separated by comma")
parser.add_argument("-K", "--K", help = "Number of cohorts", default=1)
parser.add_argument("-L", "--L", help = "Number of prior mixture components", default=2)
parser.add_argument("-iterations", "--iterations", help = "Number of iterations", default=10)
parser.add_argument("-prior_vars", "--prior-vars", help = "Prior mixture variances of different cohorts", default="0,1")
parser.add_argument("-prior_probs", "--prior-probs", help = "Prior mixture probabilites of different cohorts", default="0.99,0.01")
parser.add_argument("-gamw", "--gamw", help = "Initial noise precision", default=5)
parser.add_argument("-gam1", "--gam1", help = "Initial signal precision", default=0.000001)
parser.add_argument("-lmmse_damp", "--lmmse-damp", help = "Use LMMSE damping", default=False)
parser.add_argument("-learn_gamw", "--learn-gamw", help = "Learn or fix gamw", default=True)
parser.add_argument("-rho", "--rho", help = "Damping factor rho", default=0.5)
parser.add_argument("-cg_maxit", "--cg-maxit", help = "CG max iterations", default=500)
parser.add_argument("-s", "--s",  help = "Rused = (1-s) * R + s * Id", default=0.0)
parser.add_argument("-prior_update", "--prior-update",  help = "Learning prior probabilites", default="em")
parser.add_argument("-em_prior_maxit", "--em-prior-maxit",  help = "Maximal number of iterations that prior-learning EM is allowed to perform", default=100)
parser.add_argument("-bim_files", "--bim-files",  help = "Path to files containing list of snps", default=None)
args = parser.parse_args()

# Input parameters
ld_fpaths = args.ld_files
r_fpaths = args.r_files
true_signal_fpath = args.true_signal_file
out_dir = args.out_dir
out_name = args.out_name
Ms = args.M # Number of markers
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
bim_fpaths = args.bim_files

ld_fpaths_list = ld_fpaths.split(",")
r_fpaths_list = r_fpaths.split(",")
bim_fpaths_list = bim_fpaths.split(',')

N_list = [int(n) for n in Ns.split(",")]
M_list = [int(m) for m in Ms.split(",")]
N = N_list[rank]
Nt = sum(N_list)
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
    logging.info(f"--M {Ms}")
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
        logging.info(f"--em_prior_maxit {em_prior_maxit}")
    logging.info(f"--bim-files {bim_fpaths}\n")

# Loading .bim files
if rank == 0:
    logging.info(f"...loading .bim files\n")
ts = time.time()
bim_ref = []
bim_list = []
for k in range(K):
    bim_df = pd.read_table(bim_fpaths_list[k], sep='\s+', header=None, names=['Chromosome','Variant','Position','Coordinate','Allele1','Allele2'])
    bim_list.append(list(bim_df['Variant']))
    if k == 0:
        bim_ref_df = bim_df
    else:
        #bim_ref_df = pd.merge(bim_ref_df, bim_df, on=['Chromosome','Variant','Position','Coordinate'], how='outer')
        bim_ref_df = pd.merge(bim_ref_df, bim_df, on=['Variant'], how='outer', suffixes=('', '_y'))

bim_ref_df = bim_ref_df.sort_values(by=['Coordinate'])
bim_ref = list(bim_ref_df['Variant'])
M = len(bim_ref)

if rank == 0:
    logging.info(f"Total number of markers in reference is {M} \n")

if rank == 0:
    logging.info(f"...Saving refenrence .bim file \n")
    bim_ref_df.iloc[:,:6].to_csv(os.path.join(out_dir, out_name + ".bim"), header=None, sep='\t', index=False)

rs_miss = list(set(bim_ref) - set(bim_list[rank]))
idx = {rs:i for i,rs in enumerate(bim_ref)} # for storing SNP reference index
i_map = [idx[rs] for i,rs in enumerate(bim_list[rank])] # for maping SNP indices oiginal - reference

source = np.ones(M) * rank # Vector of rank indices indicating where to ask for missing data
for rs in rs_miss:
    idx_rs = []
    for k in range(K):
        if k != rank:
            if rs in bim_list[k]:
                idx_rs.append(k)
    kx = np.argmax(np.array(N_list)[idx_rs])
    source[idx[rs]] = kx
logging.debug(f"Rank {rank}: Handling .bim file took {time.time() - ts} seconds \n")

# Loading R matrix and r vector
if rank == 0:
    logging.info(f"...loading R matrix and r vector\n")

ts = time.time()

ld_fpath = ld_fpaths_list[rank]
r_fpath = r_fpaths_list[rank]

r = np.zeros(M)
if r_fpath.endswith('.txt'):
    r_k = np.loadtxt(r_fpath).reshape((M_list[rank]))
elif r_fpath.endswith('.npy'):
    r_k = np.load(r_fpath).reshape((M_list[rank]))
elif r_fpath.endswith('.linear'):
    df_r_k = pd.read_table(r_fpath, sep='\s+')
    r_k = np.array(df_r_k['BETA']).reshape((M_list[rank]))
    r_k[np.isnan(r_k)] = 0
    r_k *= np.sqrt(N)
else:
    raise Exception("Unsupported r vector format!")

# Reorder r vector based on reference
for j in range(len(r_k)):
    r[i_map[j]] = r_k[j]

logging.info(f"Rank {rank} loaded r vector with shape {r.shape}\n")
logging.debug(f"Rank {rank}: Loading r vector took {time.time() - ts} seconds \n")

# Loading R matrix
ts = time.time()

if ld_fpath.endswith('.npz'):
    R = scipy.sparse.load_npz(ld_fpath)
elif ld_fpath.endswith('.npy'):
    R = np.load(ld_fpath)
elif ld_fpath.endswith('.ld'):

    R_df = pd.read_table(ld_fpath, sep='\s+') # Load .ld file
    indA = [idx[rs] for rs in list(R_df['SNP_A'])] # Index SNPs by reference
    indB = [idx[rs] for rs in list(R_df['SNP_B'])]
    R_col = list(R_df['R']) # Correlation values

    # Send requests
    for k in range(K):
        if k != rank:
            i_list = [i for i in range(len(source)) if source[i] == k]
            comm.send(i_list, k, tag=0)
    
    # receive requests
    req_set = {}
    for k in range(K):
        if k != rank:
            req_set[k] = comm.recv(source=k, tag=0)
    
    # Send data
    for k in range(K):
        if k != rank:
            data = []
            for ind in req_set[k]:
                for i,corr in enumerate(R_col):
                    if indA[i] == ind or indB[i] == ind:
                        data.append([indA[i], indB[i], corr])
            comm.send(np.array(data), k, tag=1) # sending R data
            #logging.debug(f"Rank {rank} sended R data {data}\n")
            data = r[req_set[k]] 
            comm.send(data, k, tag=2) # sending r vector
            #logging.debug(f"Rank {rank} sended r data {data}\n")

    # receive data
    for k in range(K):
        if k != rank:
            if k in source:
                data = comm.recv(source=k, tag=1)
                #logging.debug(f"Rank {rank} recived data {data}\n")
                for i in range(data.shape[0]):
                    indA.append(data[i,0])
                    indB.append(data[i,1])
                    R_col.append(data[i,2])

                data = comm.recv(source=k, tag=2)
                #logging.debug(f"Rank {rank} recieved r data {data}\n")
                r[source == k] = data

    ind_r = list(range(M)) + indA + indB
    ind_c = list(range(M)) + indB + indA
    del indA
    del indB
    v = np.array(list(np.ones(M)) + R_col + R_col)
    del R_col
    R = scipy.sparse.csr_matrix((v, (ind_r, ind_c)), shape=(M, M))
    
else: 
    raise Exception("Unsupported R matrix format!")

logging.info(f"Rank {rank} loaded R matrix with shape {R.shape}\n")
logging.debug(f"Rank {rank}: Loading R matrix took {time.time() - ts} seconds \n")

R = (1-s) * R + s * scipy.sparse.identity(M) # R regularization
r = r.reshape((M,1))

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
                Nt=Nt,
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
    logging.info(f"sgVAMP inference running time: {(te - ts):0.4f}s\n")

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