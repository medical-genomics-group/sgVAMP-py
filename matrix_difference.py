import os
import sys

import numpy as np
import argparse
import scipy.sparse
import scipy.sparse.linalg

def parse_args():
    parser = argparse.ArgumentParser(description='BayesW')

    parser.add_argument('matrix1_file', type=str, help='path to matrix 1', required = True)
    parser.add_argument('matrix1_format', type=str, help='format of matrix 1 (txt, or npz)', required = True)
    parser.add_argument('matrix2_file', type=str, help='path to matrix 2', required = True)
    parser.add_argument('matrix2_format', type=str, help='format of matrix 2 (txt, or npz)', required = True)
    args = parser.parse_args()
    return args
    
## load LD matrices from plink and xarray

def main():
    if args.matrix1_format == "npz":  
        R1 = scipy.sparse.load_npz(args.matrix1_file)
    else:
        R1 = np.loadtxt(args.matrix1_file)

    if args.matrix2_format == "npz":  
        R2 = scipy.sparse.load_npz(args.matrix2_file)
    else:
        R2 = np.loadtxt(args.matrix2_file)
   

    ## Calculate the differences and print the results.

    if args.matrix1_format == "txt" & args.matrix2_format == "txt":
        R1np = R1.toarray()
        R2np = R2.toarray()
        dif = nplinalg.norm(R1np - R2np)
    elif args.matrix1_format == "txt":
        R2np = R2.toarray()
        dif = nplinalg.norm(R1 - R2np)
    elif args.matrix2_format == "txt":
        R1np = R1.toarray()
        dif = nplinalg.norm(R1np - R2)
    else:
        dif = scipy.sparse.linalg.norm(R1 - R2)
        
    print(f"Matrix 1: {args.matrix1_file} ")
    print(f"Matrix 2: {args.matrix1_file} ")
    print(f"L2norm difference: {dif} (max {np.abs(R1 - R2).max()})")


