import numpy as np

from sklearn.linear_model import Lasso
import argparse
import scipy

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-ld_file", "--ld-file", type=str, help = "Path to LD matrix .npz file")
parser.add_argument("-r_file", "--r-file",type=str, help = "Path to XTy .npy file")
parser.add_argument("-N", "--N", type=str, help = "Number of samples")
parser.add_argument("-M", "--M", type=str, help = "Number of markers")
parser.add_argument("-alpha", "--alpha",type=float, default=0.1, help = "Regularization parameter")
parser.add_argument("-out", "--out", type=str, help = "Path to out file for betas")
args = parser.parse_args()

ld_fpath = args.ld_file
r_fpath = args.r_file
M = int(args.M) # Number of markers
N = int(args.N) # Number of samples
alpha = args.alpha
out = args.out

# Loading LD matrix and XTy vector

print("...loading LD matrix and XTy vector", flush=True)
R = scipy.sparse.load_npz(ld_fpath)
r = np.load(r_fpath)
print("LD matrix and XTy loaded. Shapes: ", R.shape, r.shape, flush=True)

print("--r-file", r_fpath)
print("--ld-file", ld_fpath)

beta_ls = scipy.sparse.linalg.cg(R, r)
np.savetxt(beta_ls, out +"_ls.txt")

sparse_lasso = Lasso(alpha=alpha, fit_intercept=True, max_iter=1000)
sparse_lasso.fit(R, r)

beta_lasso = sparse_lasso.coef_
np.savetxt(beta_lasso, out +"_lasso.txt")





