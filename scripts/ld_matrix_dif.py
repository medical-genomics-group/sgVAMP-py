import os
import sys

import numpy as np
import pandas as pd

import scipy.sparse
import scipy.sparse.linalg

## load LD matrices from plink and xarray
R1 = scipy.sparse.load_npz("/nfs/scistore17/robingrp/avillanu/sgVAMP/LDmatrix_20K.npz")
R2 = scipy.sparse.load_npz("/nfs/scistore17/robingrp/avillanu/sgVAMP/LDmatrix_20K_plink.npz")

'''
1. Create full LD matrix:
first we load the genotype matrix. For creating this matrix i use the plink command
$ plink --bfile 1000g_20K --recodeA --tab --out 1000g_20K --noweb
and this converts the bed file into a regular 0,1,2 matrix. I removed the headers and identifiers with the command
$ cat 1000g_20K.raw | tail -n +2 | cut  -f7- > 1000g_20K.txt
and save it in a txt file

now we load this file and standardize it 
'''
X = np.loadtxt("/nfs/scistore17/robingrp/avillanu/sgVAMP/1000g_20K.txt")
print("1000g_20K.txt loaded! \n")

# XTX = X.T.dot(X)
# np.savez("full_LDmatrix.npz")

'''
2. Standardize and normalize the XTX matrix
The file full_LDmatrix.npz is just this, computing the matrix XTX and save it as a npz file.
We can standardize each marker, then compute XTX and divide by the number of markers.
'''

X = (X - X.mean(axis=0))/X.std(axis=0)
XTX = X.T.dot(X)
N_arr = np.ones(XTX.shape[1])*XTX.shape[0]
XTX_norm = XTX/N_arr
del XTX

#np.savetxt("full_LD_20K_norm.txt", XTX_norm)
#np.savez("full_LD_20K_norm.npz", XTX_norm)

'''
3. Get the blocks from the markers, so that we obtain a sparse matrix.
We first create small block matrices for each chromosome, 
then combine the in a sparse block diagonal matrix and take the lower triangle.
'''

snps = pd.read_csv("/nfs/scistore17/robingrp/avillanu/sgVAMP/selected_snps_20K.txt", sep=":", header=None)
prev_i = 0
blocks = []
for i in snps.groupby([0])[1].count().to_numpy():
    blocks.append(XTX_norm[prev_i:prev_i+i, prev_i:prev_i+i])
    prev_i +=i
    
del XTX_norm

XTX_norm_sp = scipy.sparse.tril(scipy.sparse.block_diag(blocks))

#scipy.sparse.save_npz("LDmatrix_20K_blocks_norm_norm.npz", XTX_norm_sp)
'''
4. Calculate the differences and print the results.
'''
R1_dif = scipy.sparse.linalg.norm(R1 - XTX_norm_sp)
R2_dif = scipy.sparse.linalg.norm(R2 - XTX_norm_sp)
R1_R2_dif = scipy.sparse.linalg.norm(R2 - R1)
print("Matrix standardized and norm: ")
print(f"xarray LD: {R1_dif} (max {np.abs(R1 - XTX_norm_sp).max()})")
print(f"plink LD: {R2_dif} (max {np.abs(R2 - XTX_norm_sp).max()})")
print(f"plink xarray: {R1_R2_dif} (max {np.abs(R1 - R2).max()})")

########### 
# this part of the script is for the matrices that I had created before and are not standardized
###########
'''
Now we do calculate the differences using non-standardized matrices
LDmatrix_20K_full.npz is a block diagonal matrix of the XTX matrix 
'''

full_LD_sp = scipy.sparse.tril(scipy.sparse.load_npz("LDmatrix_20K_full.npz")) 
#actually this file shouldn't be named like this bc is not full but blocked

'''
Normalization on the triangular blocked matrix: We divide by the number of markers
and then print the differences.
'''
N_arr = np.ones(full_LD_sp.shape[1])*full_LD_sp.shape[0]
full_LD_sp_norm = full_LD_sp.multiply(1/N_arr)
del full_LD_sp


R1_dif = scipy.sparse.linalg.norm(R1 - full_LD_sp_norm)
R2_dif = scipy.sparse.linalg.norm(R2 - full_LD_sp_norm)
R1_R2_dif = scipy.sparse.linalg.norm(R2 - R1)
print("Sparse blocked matrix normalized: ")
print(f"xarray LD: {R1_dif} (max {np.abs(R1 - XTX_norm_sp).max()})")
print(f"plink LD: {R2_dif} (max {np.abs(R2 - XTX_norm_sp).max()})")
print(f"plink xarray: {R1_R2_dif} (max {np.abs(R1 - R2).max()})")

'''
Now for the final part of the script I wanted to try if we obtain the same result if 
we get the full matrix get the blocks and compare to the previous result (it should be the
same its just a check up if I had done something wrong with the previous block code)
basically ignore this part
'''
full_LD = np.load("full_LDmatrix.npz")
full_LD = full_LD["arr_0"]

N_arr = np.ones(full_LD.shape[1])*full_LD.shape[0]
full_LD_norm = full_LD/N_arr
del full_LD

snps = pd.read_csv("selected_snps_20K.txt", sep=":", header=None)
prev_i = 0
blocks = []
for i in snps.groupby([0])[1].count().to_numpy():
    blocks.append(full_LD_norm[prev_i:prev_i+i, prev_i:prev_i+i])
    prev_i +=i
    
del full_LD_norm

new_full_LD_norm_sp = scipy.sparse.tril(scipy.sparse.block_diag(blocks))

del blocks
#scipy.sparse.save_npz("LDmatrix_20K_sp_norm.npz", new_full_LD_norm_sp)

R1_dif = scipy.sparse.linalg.norm(R1 - new_full_LD_norm_sp)
R2_dif = scipy.sparse.linalg.norm(R2 - new_full_LD_norm_sp)
R1_R2_dif = scipy.sparse.linalg.norm(R2 - R1)
print("Matrix first normed then blocked: ")
print(f"xarray LD: {R1_dif} (max {np.abs(R1 - XTX_norm_sp).max()})")
print(f"plink LD: {R2_dif} (max {np.abs(R2 - XTX_norm_sp).max()})")
print(f"plink xarray: {R1_R2_dif} (max {np.abs(R1 - R2).max()})")

