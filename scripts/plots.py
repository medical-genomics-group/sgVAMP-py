import matplotlib.pyplot as plt
import numpy as np
import argparse
import csv
import os

# Initialize parser
parser = argparse.ArgumentParser()
parser.add_argument("-csv_params", "--csv-params", help = "Path to sgVAMP CSV file")
parser.add_argument("-csv_metrics", "--csv-metrics", help="Path to sgVAMP metrics CSV file")
parser.add_argument("-out_name", "--out-name", help = "Output file name")
args = parser.parse_args()

csv_params_fpath = args.csv_params
csv_metrics_fpath = args.csv_metrics
out_name = args.out_name

print("Input arguments:")
print("--csv-params", csv_params_fpath)
print("--csv-metrics", csv_metrics_fpath)
print("--out-name", out_name)
print("\n", flush=True)

# Get output dir from input file
out_dir = os.path.dirname(csv_metrics_fpath)

# Load hyperparameters
gamws = []
gam1s = []
gam2s = []
alpha1s = []
alpha2s = []
lams = []
its = []
csv_file = open(csv_params_fpath, mode='r')
csv_reader = csv.reader(csv_file, delimiter='\t')
next(csv_reader, None) # skip header
for row in csv_reader:
    its.append(int(row[0]))
    gamws.append(float(row[1]))
    gam1s.append(float(row[2]))
    gam2s.append(float(row[3]))
    alpha1s.append(float(row[4]))
    alpha2s.append(float(row[5]))
    lams.append(float(row[6]))

iterations = max(its) + 1

# Load metrics
l2s = []
alignments = []

csv_file = open(csv_metrics_fpath, mode='r')
csv_reader = csv.reader(csv_file, delimiter='\t')
next(csv_reader, None) # skip header
for row in csv_reader:
    alignments.append(float(row[1]))
    l2s.append(float(row[2]))

# Plot metrics and parameters
print("L2 error (xhat1, x0) over iterations: \n", l2s, flush=True)
print("Alignment (xhat1, x0) over iterations: \n", alignments, flush=True)
print("gam2 over iterations: \n", gam2s, flush=True)

plt.rcParams.update({'font.size': 15})
fig, ax = plt.subplots(5,figsize=(12, 10), dpi=300)

ax[0].plot(l2s, "-o")
ax[1].plot(alignments, "-o")
ax[2].plot(gam1s, "-o", label="gam1")
ax[3].plot(gam2s, "-o", label="gam2")
ax[4].plot(lams, "-o", label="lam")
ax[0].set_ylabel("L2_err(xhat1,x0)")
ax[1].set_ylabel("Align(xhat1,x0)")
ax[2].set_ylabel("gam1")
ax[3].set_ylabel("gam2")
ax[4].set_ylabel("lam")
ax[3].set_xlabel("iteration")

fig.tight_layout()

out_fpath = os.path.join(out_dir, out_name + ".png")
print("...saving to file", out_fpath)
fig.savefig(out_fpath)