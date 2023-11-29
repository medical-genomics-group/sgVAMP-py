import os
import sys

import numpy as np
import argparse
import scipy.sparse
import scipy.sparse.linalg

def parse_args():
    parser = argparse.ArgumentParser(description='matrix_difference')

    parser.add_argument('--matrix1_file', type=str, help='path to matrix 1', required = True)
    parser.add_argument('--matrix1_sparse', type=bool, help='is matrix 1 sparse? True|False', required = True)
    parser.add_argument('--matrix2_file', type=str, help='path to matrix 2', required = True)
    parser.add_argument('--matrix2_sparse', type=bool, help='is matrix 2 sparse? True|False', required = True)
    args = parser.parse_args()
    return args
    
## load LD matrices from plink and xarray
def get_matrix(file, sparse):
    print(file)
    print(sparse)
    m_format = file.split(".")[-1]
    if m_format == "npz" and sparse == True:  
        R = scipy.sparse.load_npz(file)
    elif (m_format == "npy" or m_format == "npz") and sparse == False:
        R = np.load(file)
    elif m_format == "txt":
        R = np.loadtxt(file)
    else:    
        raise TypeError("Wrong matrix format. only npz, npy or txt allowed.")
 
    return R
    
def main():
    args = parse_args()
    print(args)
    sparse_1 = bool(args.matrix1_sparse)
    sparse_2 = bool(args.matrix2_sparse)
    
    
    R1 = get_matrix(args.matrix1_file, sparse=sparse_1)
    R2 = get_matrix(args.matrix2_file, sparse=sparse_2)

    if args.matrix1_sparse and args.matrix2_sparse:
        dif = scipy.sparse.linalg.norm(R1 - R2)
    else:
        dif = np.linalg.norm(R1np - R2np)
        '''
    elif args.matrix1_sparse:
        R1np = R1.toarray()
        dif = np.linalg.norm(R1np - R2np)
    elif 
    
    ## Calculate the differences and print the results.
    
    if args.matrix1_format == "txt" and args.matrix2_format == "txt":
        R1np = R1.toarray()
        R2np = R2.toarray()
        dif = np.linalg.norm(R1np - R2np)
    elif args.matrix1_format == "txt":
        R2np = R2.toarray()
        dif = np.linalg.norm(R1 - R2np)
    elif args.matrix2_format == "txt":
        R1np = R1.toarray()
        dif = np.linalg.norm(R1np - R2)
    else:
    
        dif = scipy.sparse.linalg.norm(R1 - R2)
        '''
    print(f"Matrix 1: {args.matrix1_file} ")
    print(f"Matrix 2: {args.matrix2_file} ")
    print(f"L2norm difference: {dif} (max {np.abs(R1 - R2).max()})")

if __name__ == "__main__":
    main()
