import numpy as np
import argparse
import scipy
import matplotlib.pyplot as plt
import os

# Script for LD matrix visualization
print("...Visualize LD matrix \n", flush=True)

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-ld_file", "--ld-file", help = "Path to LD matrix")
parser.add_argument("-ld_format", "--ld-format", help = "LD matrix format (npy or npz)", default='npz')
parser.add_argument("-out_dir", "--out-dir", help = "Directory, where to store the image.")
args = parser.parse_args()

# Input parameters
ld_fpath = args.ld_file
ld_format = args.ld_format # npy or npz
out_dir = args.out_dir

print("--ld-file", ld_fpath)
print("--ld-format", ld_format)
print("--out-dir", out_dir)
print("\n", flush=True)

# Get basename of file from input LD file
ld_base = os.path.basename(ld_fpath)
ld_base = ld_base.split('.')[0]
ld_dir = os.path.dirname(ld_fpath)

# Loading LD matrix
print("...loading LD matrix", flush=True)

if ld_format == 'npz':
    R = scipy.sparse.load_npz(ld_fpath).toarray()
    
elif ld_format == 'npy':
    R = np.load(ld_fpath)["arr_0"]
else:
    raise Exception("Unsupported LD format!")

M = R.shape[0]

fig, ax = plt.subplots(1)
t = ax.imshow(np.abs(R))
fig.colorbar(t)

out_fpath = os.path.join(out_dir, ld_base+'.png')
print("...Saving LD matrix figure to file", out_fpath)
fig.savefig(out_fpath)