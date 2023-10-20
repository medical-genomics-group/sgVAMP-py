from sgvamp import VAMP
import numpy as np
from numpy.linalg import norm
from sklearn.metrics import r2_score

# Test run for sgvamp
print("...Test run of VAMP for summary statistics\n")

# Initialization
N = 1000 # Number of samples
M = 2000 # Number of markers
iterations = 10
h2 = 0.8 # heritability for simulations
lam = 0.5 # Sparsity for simulations

# Simmulations
print("...Simulating data\n")
X = np.random.binomial(2, p=0.4, size=[N,M])
X = (X - np.mean(X,axis=0)) / np.std(X, axis=0) # Standardization
X /= np.sqrt(N)
beta = np.random.normal(loc=0.0, scale=1.0, size=[M,1]) # scale = standard deviation
beta *= np.random.binomial(1, lam, size=[M,1])
g = X @ beta
print("Var(g) =", np.var(g))
w = np.random.normal(loc=0.0, scale=np.sqrt(1/h2 - 1), size=[N,1])
y = g + w
print("Var(y) =", np.var(y))
print("h2 =", np.var(g) / np.var(y))
print("\n")

# TODO Here we should load LD matrix
R = X.T @ X
r = X.T @ y

# sgVAMP init
sgvamp = VAMP(lam=lam, rho=0.5, gam1=100, gamw=1/h2)

# Inference
print("...Running sgVAMP\n")
xhat1 = sgvamp.infer(R, r, iterations)
print("\n")

# Print metrics
R2s = []
corrs = []
for it in range(iterations):
    yhat = X @ xhat1[it]
    R2 = r2_score(y, yhat) # R squared metric
    R2s.append(R2)

    corr = np.corrcoef(xhat1[it].squeeze(), beta.squeeze()) # Pearson correlation coefficient of xhat1 and true signal beta
    corrs.append(corr[0,-1])
print("R2 over iterations: \n", R2s)
print("Corr(x1hat,beta) over iterations: \n", corrs)
