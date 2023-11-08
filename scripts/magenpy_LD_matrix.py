import os
import sys

import numpy as np
import scipy
import argparse
import magenpy as mgp

def parse_args():
    parser = argparse.ArgumentParser(description='BayesW')
    parser.add_argument('--n', type=int, help='number of individuals', required = True)
    parser.add_argument('--p', type=int, help='number of markers', required = True)

    parser.add_argument('--outdir', type=str, help='path to directory where the results are stored', required = True)
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
    mgp.set_option("plink1.9_path", "/nfs/scistore07/clustersw/shared/plink/1.90/plink")
    mgp.set_option("plink2_path", "/mnt/nfs/clustersw/shared/plink/220415/plink2")
    gdl = mgp.GWADataLoader(fname, backend='plink')
    gdl.compute_ld(estimator='windowed',
                   output_dir=f"{outdir}/output/ld",
                    kb_window_size=int(kb_window))
    matrices = [m.to_csr_matrix() for m in gdl.ld.values()]
    LD_matrix = scipy.sparse.block_diag(matrices)
    scipy.sparse.save_npz(f"{os.path.join(outdir, outname)}.npz", LD_matrix)

if __name__ == "__main__":
    main()
    
    
    


