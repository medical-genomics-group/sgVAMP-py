import numpy as np

from sklearn.linear_model import Lasso, Ridge
import argparse
import scipy

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("-ld_file", "--ld-file", type=str, help = "Path to LD matrix .npz file")
    parser.add_argument("-r_file", "--r-file",type=str, help = "Path to XTy .npy file")
    parser.add_argument("-N", "--N", type=str, help = "Number of samples")
    parser.add_argument("-M", "--M", type=str, help = "Number of markers")
    parser.add_argument("-alpha", "--alpha",type=float, default=0.1, help = "Regularization parameter")
    parser.add_argument("-out", "--out", type=str, help = "Path to out file for betas")
    parser.add_argument("-true_signal_file", "--true-signal-file", help = "Path to true signal .npy file")
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    print(args)
    ld_fpath = args.ld_file
    r_fpath = args.r_file
    true_signal_fpath = args.true_signal_file
    M = int(args.M) # Number of markers
    N = int(args.N) # Number of samples
    alpha = args.alpha
    out = args.out


    # Loading LD matrix and XTy vector

    print("...loading LD matrix and XTy vector", flush=True)
    R = scipy.sparse.load_npz(ld_fpath)
    r = np.loadtxt(r_fpath)
    print("LD matrix and XTy loaded. Shapes: ", R.shape, r.shape, flush=True)
    beta = np.fromfile(true_signal_fpath)
    print("True signals loaded. Shape: ", beta.shape, flush=True)

    print("--r-file", r_fpath)
    print("--ld-file", ld_fpath)

    

    sparse_lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
    sparse_lasso.fit(R, r)

    beta_lasso = sparse_lasso.coef_
    np.savetxt( out +"_lasso.txt", beta_lasso)

    corr = np.corrcoef(beta_lasso.squeeze(), beta.squeeze())
    print("Coorelation Lasso: ", corr)

    sparse_ridge = Ridge(alpha=alpha, fit_intercept=True, max_iter=1000)
    sparse_ridge.fit(R, r)

    beta_ridge = sparse_ridge.coef_
    np.savetxt( out +"_ridge.txt", beta_ridge)

    corr = np.corrcoef(beta_lasso.squeeze(), beta.squeeze())
    print("Coorelation Ridge: ", corr)

    beta_ls = scipy.sparse.linalg.cg(R + np.identity(R), r)
    np.savetxt( out +"_ls.txt", beta_ls)
    
    
    corr = np.corrcoef(beta_ls.squeeze(), beta.squeeze()) # Pearson correlation coefficient of xhat1 and true signal beta
    print("Coorelation Least Squares: ", corr)

    print("Done!")



if __name__ == "__main__":
    main()



