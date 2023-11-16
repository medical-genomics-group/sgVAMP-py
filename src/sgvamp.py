# gVAMP for summary statistics

from scipy.stats import norm
import numpy as np
from numpy.random import binomial
from numpy.linalg import inv
from scipy.sparse.linalg import cg as con_grad

class VAMP:
    def __init__(self, rho, lam, gamw, gam1):
        self.eps = 1e-32
        self.lam = lam
        self.rho = rho
        self.gamw = gamw
        self.gam1 = gam1

    def denoiser(self, r, gam1):
        A = (1-self.lam) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
        B = self.lam * norm.pdf(r, loc=0, scale=np.sqrt(1.0 + 1.0/gam1))
        ratio = gam1 * r / (gam1 + 1.0) * B / (A + B + self.eps)
        return ratio

    def der_denoiser(self, r, gam1):
        A = (1-self.lam) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
        B = self.lam * norm.pdf(r, loc=0, scale=np.sqrt(1.0 + 1.0/gam1))
        Ader = A * (-r*gam1)
        Bder = B * (-r) / (1.0 + 1.0/gam1 + self.eps)
        BoverAplusBder = ( Bder * A - Ader * B ) / (A+B + self.eps) / (A+B + self.eps)
        ratio = gam1 / (gam1 + 1.0) * B / (A + B + self.eps) + BoverAplusBder * r * gam1 / (gam1 + 1.0)
        return ratio
    
    def infer(self,R,r,M,N,iterations,est=True,cg=True,cg_maxit=500,learn_gamw=True, lmmse_damp=True):

        # initialization
        r1 = np.zeros((M,1))
        xhat1 = np.zeros((M,1)) # signal estimates in Denoising step
        xhat2 = np.zeros((M,1)) # signal estimates in LMMSE step
        gam1 = self.gam1
        rho = self.rho # Damping factor
        gamw = self.gamw # Precision of noise
        xhat1s = []
        I = np.identity(M) # Identity matrix
        gamws = []
        alpha1 = 0
        alpha2 = 0

        for it in range(iterations):
            print("-----ITERATION %d -----"%(it), flush=True)
            # Denoising
            print("...Denoising", flush=True)
            xhat1_prev = xhat1
            alpha1_prev = alpha1
            vect_den_beta = lambda x: self.denoiser(x, gam1)
            xhat1 = vect_den_beta(r1)
            xhat1 = rho * xhat1 + (1 - rho) * xhat1_prev # apply damping on xhat1
            xhat1s.append(xhat1)
            alpha1 = np.mean( self.der_denoiser(r1, gam1) )
            alpha1 = rho * alpha1 + (1 - rho) * alpha1_prev # apply damping on alpha1
            gam2 = gam1 * (1 - alpha1) / alpha1
            r2 = (xhat1 - alpha1 * r1) / (1 - alpha1)

            # LMMSE
            print("...LMMSE", flush=True)
            xhat2_prev = xhat2
            alpha2_prev = alpha2
            A = (gamw * R + gam2 * I) # Sigma2 = A^(-1)
            mu2 = (gamw * r + gam2 * r2)
            
            if not cg or not est: 
                Sigma2 = inv(A) # Precompute A^(-1) if needed

            if cg:
                # Conjugate gradient for solving linear system A^(-1) @ mu2 = Sigma2 @ mu2
                xhat2, ret = con_grad(A, mu2, maxiter=cg_maxit)
                if ret > 0: print("WARNING: CG 1 convergence after %d iterations not achieved!" % ret)
                xhat2.resize((M,1))
            else:
                # True inverse
                xhat2 = Sigma2 @ mu2

            if lmmse_damp:
                xhat2 = rho * xhat2 + (1 - rho) * xhat2_prev # damping on xhat2

            # Generate iid random vector [-1,1] of size M
            u = binomial(p=1/2, n=1, size=M) * 2 - 1

            if est:
                # Hutchinson trace estimator
                # Sigma2 = (gamw * R + gam2 * I)^(-1)
                if cg:
                    # Conjugate gradient for solving linear system (gamw * R + gam2 * I)^(-1) @ u
                    Sigma2_u, ret = con_grad(A,u, maxiter=cg_maxit)
                    if ret > 0: print("WARNING: CG 2 convergence after %d iterations not achieved!" % ret)
                else:
                    # True inverse
                    Sigma2_u = Sigma2 @ u
                TrSigma2 = u.T @ Sigma2_u # Tr[Sigma2] = u^T @ Sigma2 @ u 
            else:
                # True trace computation
                TrSigma2 = np.trace(Sigma2)

            alpha2 = gam2 * TrSigma2 / M
            if lmmse_damp:
                alpha2 = rho * alpha2 + (1 - rho) * alpha2_prev # damping on alpha2
            gam1 = gam2 * (1 - alpha2) / alpha2
            r1 = (xhat2 - alpha2 * r2) / (1-alpha2)
            z = N - (2 * xhat2.T @ r) + (xhat2.T @ R @ xhat2)

            if learn_gamw:
                if est:
                    # Hutchinson trace estimator
                    TrRSigma2 = u.T @ R @ Sigma2_u # u^T @ R @ [(gamw * R + gam2 * I)^(-1) @ u]
                else:
                    # True trace computation
                    TrRSigma2 = np.trace(R @ Sigma2) 

                # Update noise precison
                gamw_prev = gamw
                gamw = 1 / (z / N + TrRSigma2 / N)
                gamw = float(gamw.squeeze())
                gamw = rho * gamw + (1 - rho) * gamw_prev # damping on gamw
            gamws.append(gamw)

        return xhat1s, gamws