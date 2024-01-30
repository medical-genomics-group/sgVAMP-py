# sgVAMP-py
Python implementation of gVAMP for summary statistics.

# Example

```
module load python3

sloc={Path to sgVAMP/src folder}

python3 ${sloc}/main.py [Input options]
```

# Input options

| Option | Description |
| --- | --- |
| `--ld-files` | Path to LD matrices files, separated by comma |
| `--r-files`| Path to XTy files separated by comma |
| `--true-signal-file ` | Path to true signal file |
| `--out-dir` | Output directory |
| `--out-name` | Output file name |
| `--N` | Number of samples |
| `--M` | Number of markers |
| `--K` | Number of cohorts |
| `--L` | Number of prior mixture components (Including spike) |
| `--iterations` | Number of iterations |
| `--prior-vars` | Prior mixture variances of different cohorts, separated by comma (e.g. `0,1`). First must be 0. |
| `--prior-probs` | Prior mixture probabilites of different cohorts, separated by comma (e.g. `0.99,0.01`). Must sum up to 1. |
| `--gamw` | Initial noise precision |
| `--gam1` | Initial signal precision |
| `--lmmse-damp` | Use LMMSE damping |
| `--learn-gamw` | Learn or fix gamw |
| `--rho` | Damping factor rho |
| `--cg-maxit` | CG max iterations |
| `--s` | Rused = (1-s) * R + s * Id |
| `--mle-prior-update` | Updating prior using MLE |

# Output files
Signal estimates over iterations are stored in binary files: ``{out_dir}/{out_name}__xhat_it_{it}.bin``

Separate CSV output files are created for each cohort: ``{out_dir}/{out_name}_cohort_{id}.csv``

### CSV file structure:
| Iteration | gamw | gam1 | gam2 | alpha1 | alpha2 |
| --- | --- | --- | --- | --- | --- |
...

