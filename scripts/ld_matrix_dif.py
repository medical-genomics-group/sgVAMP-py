import os
import sys

import numpy as np
import pandas as pd

import scipy.sparse
import scipy.sparse.linalg

R1 = scipy.sparse.tril(scipy.sparse.load_npz("LDmatrix_20K.npz"))
R2 = scipy.sparse.tril(scipy.sparse.load_npz("LDmatrix_20K_plink.npz"))


## code to create the matrix
# full_LD = np.load("full_LDmatrix.npz") #this is the file with just the XTX matrix
# full_LD = full_LD["arr_0"]

# snps = pd.read_csv("selected_snps_20K.txt", sep=":", header=None)
# prev_i = 0
# blocks = []
# for i in snps.groupby([0])[1].count().to_numpy():
#    blocks.append(full_LD[prev_i:prev_i+i, prev_i:prev_i+i])
#    prev_i +=i
    
# del full_LD

# full_LD_matrix = scipy.sparse.tril(scipy.sparse.block_diag(blocks))

# del blocks
# scipy.sparse.save_npz("LDmatrix_20K_full.npz", full_LD_matrix)

full_LD_matrix = scipy.sparse.tril(scipy.sparse.load_npz("LDmatrix_20K_full.npz"))
R1_dif = scipy.sparse.linalg.norm(R1 - full_LD_matrix)
R2_dif = scipy.sparse.linalg.norm(R2 - full_LD_matrix)
R1_R2_dif = scipy.sparse.linalg.norm(R2 - R1)
print(f"xarray LD: {R1_dif}")
print(f"plink LD: {R2_dif}")
print(f"plink xarray: {R1_R2_dif}")