import numpy as np
import argparse
import pandas as pd
import os
import scipy

# Initialize parser
parser = argparse.ArgumentParser()

parser.add_argument("-ld_file", "--ld-file", help = "Path to .ld file", default=None)
parser.add_argument("-r_file", "--r-file", help = "Path to .linear file", default=None)
args = parser.parse_args()

ld_file = args.ld_file
r_file = args.r_file

print("Input arguments:")
print("--ld-file", ld_file)
print("--r-file", r_file)
print("\n", flush=True)

out_fpath_r = r_file.split(".assoc.linear")[0] + ".npy"
out_fpath_R = ld_file.split(".ld")[0] + ".npz"
print(out_fpath_r)
print(out_fpath_R)

df_r = pd.read_table(r_file, sep='\s+')
print(f".linear file loaded. Shape: {df_r.shape}", flush=True)
print(f"storing r vector to {out_fpath_r}")
np.save(out_fpath_r, df_r['BETA'].values)
M = len(df_r)

R_df = pd.read_table(ld_file, sep='\s+')#, nrows=10000) # Load .ld file
print(f".ld file loaded. Shape: {R_df.shape}", flush=True)
rs_ref = list(df_r['SNP'])
idx = {rs:i for i,rs in enumerate(rs_ref)}
indA = [idx[rs] for rs in list(R_df['SNP_A'])] # Index SNPs by reference
indB = [idx[rs] for rs in list(R_df['SNP_B'])]
R_col = list(R_df['R']) # Correlation values
del R_df
ind_r = list(range(M)) + indA + indB
ind_c = list(range(M)) + indB + indA
del indA
del indB
v = np.array(list(np.ones(M)) + R_col + R_col)
del R_col
R = scipy.sparse.csr_matrix((v, (ind_r, ind_c)), shape=(M, M))
print(f"storing R matrix to {out_fpath_R}")
scipy.sparse.save_npz(out_fpath_R, R, compressed=True)
del R