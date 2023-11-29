import os
import sys

import numpy as np
import scipy
import argparse
import magenpy as mgp

def parse_args():
    parser = argparse.ArgumentParser(description='magenpy_LD_matrix')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)

    parser.add_argument('--outdir', type=str, help='path to directory where the results are stored', required = True)
    parser.add_argument('--plink2_path', type=str, help='path to plink2', required = True)
    parser.add_argument('--plink19_path', type=str, help='path to plink1.9', required = True)
    parser.add_argument('--geno', type=str, help='input genotype file', required = True)
    parser.add_argument('--name', type=str, default="", help='output name', required = False)
    
    parser.add_argument('--kb_window', type=int, default=1000, help='window size in kb', required = False)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    fname = args.geno
    outdir = args.outdir
    n = args.n
    p = args.p
    outname = args.name
    kb_window = args.kb_window
    
    ## indicate where the plink executables are
    mgp.set_option("plink1.9_path", args.plink19_path)
    mgp.set_option("plink2_path", args.plink2_path)
    gdl = mgp.GWADataLoader(fname, backend='plink')
    gdl.compute_ld(estimator='windowed',
                   output_dir=f"{outdir}/output/ld",
                    kb_window_size=int(kb_window))
    matrices = [m.to_csr_matrix() for m in gdl.ld.values()]
    LD_matrix = scipy.sparse.block_diag(matrices)
    scipy.sparse.save_npz(f"{os.path.join(outdir, outname)}.npz", LD_matrix)
    
    print(f"Block matrix saved: {os.path.join(outdir, outname)}.npz")

if __name__ == "__main__":
    main()
    
    
    


