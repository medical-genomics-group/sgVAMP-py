# gVAMP for summary statistics

from scipy.stats import norm
import numpy as np
from numpy.random import binomial
from numpy.linalg import inv
from scipy.sparse.linalg import cg as con_grad
import scipy
import os
import csv
import struct
import logging

class VAMP:
    def __init__(self, K, rho, gamw, gam1, prior_vars, prior_probs, out_dir, out_name, comm):
        self.eps = 1e-32
        self.K = K
        self.rho = rho
        self.gamw = gamw
        self.gam1 = gam1 #np.full(K, gam1)
        self.lam = 1 - prior_probs[0]
        self.sigmas = np.array(prior_vars[1:]) # a vector containing variances of different groups except the first one, length = L-1
        self.omegas = np.array([ p / sum(prior_probs[1:]) for p in prior_probs[1:]])
        self.setup_io(out_dir, out_name)
        self.comm = comm

    def setup_io(self, out_dir, out_name):
        self.out_dir = out_dir
        self.out_name = out_name

        # Setup output CSV file for hyperparameters
        for i in range(self.K):
            csv_file = open(os.path.join(self.out_dir, "%s_cohort_%d.csv" % (self.out_name, i+1)), 'w', newline="")
            csv_writer = csv.writer(csv_file, delimiter='\t')
            header = ["it", "gamw", "gam1", "gam2", "alpha1", "alpha2"]
            csv_writer.writerow(header)
            csv_file.close()

    def write_params_to_file(self, params, cohort_idx):
        # Setup output CSV file for hyperparameters
        csv_file = open(os.path.join(self.out_dir, "%s_cohort_%d.csv" % (self.out_name, cohort_idx+1)), 'a', newline="")
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(params)
        csv_file.close()
    
    def write_xhat_to_file(self, it, xhat):
        # Setup output binary file for xhat estimates
        fname = "%s_xhat_it_%d.bin" % (self.out_name, it)
        f = open(os.path.join(self.out_dir, fname), "wb")
        f.write(struct.pack(str(len(xhat))+'d', *xhat.squeeze()))
        f.close()

    def denoiser(self, r, gam1): # not numerically stable! AD
        A = (1-self.lam) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
        B = self.lam * norm.pdf(r, loc=0, scale=np.sqrt(self.sigmas + 1.0/gam1))
        ratio = gam1 * r / (gam1 + 1.0/self.sigmas) * B / (A + B + self.eps)
        return ratio

    def der_denoiser(self, r, gam1): # not numerically stable! AD
        A = (1-self.lam) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
        B = self.lam * norm.pdf(r, loc=0, scale=np.sqrt(self.sigmas + 1.0/gam1))
        Ader = A * (-r*gam1)
        Bder = B * (-r) / (self.sigmas + 1.0/gam1 + self.eps)
        BoverAplusBder = ( Bder * A - Ader * B ) / (A+B + self.eps) / (A+B + self.eps)
        ratio = gam1 / (gam1 + 1.0/self.sigmas) * B / (A + B + self.eps) + BoverAplusBder * r * gam1 / (gam1 + 1.0/self.sigmas)
        return ratio

    def denoiser_meta(self, rs, gam1s):
        # gam1s = a vector of gam1 values over different GWAS studies
        sigma2_meta = 1.0 / (sum(gam1s) + 1.0/self.sigmas)  # a vector of dimension L - 1
        mu_meta = np.inner(rs, gam1s) * sigma2_meta
        max_ind = (np.array( mu_meta * mu_meta / sigma2_meta)).argmax()
        EXP = np.exp(0.5 * (mu_meta * mu_meta * sigma2_meta[max_ind] - mu_meta[max_ind] * mu_meta[max_ind] * sigma2_meta) / ( sigma2_meta * sigma2_meta[max_ind]) )
        Num = self.lam * sum(self.omegas * EXP * mu_meta * np.sqrt(sigma2_meta / self.sigmas))
        EXP2 = np.exp(- 0.5 * ((mu_meta[max_ind])**2 / sigma2_meta[max_ind]))
        Den = (1-self.lam) * EXP2 + self.lam * sum(self.omegas * EXP * np.sqrt(sigma2_meta / self.sigmas))
        return Num/Den

    def der_denoiser_meta(self, rs, gam1s):
        sigma2_meta = 1.0 / (sum(gam1s) + 1.0/ self.sigmas) # a numpy vector of dimension L - 1
        mu_meta = np.inner(rs, gam1s) * sigma2_meta # a numpy vector of dimension L - 1
        max_ind = (np.array( mu_meta * mu_meta / sigma2_meta)).argmax()
        EXP = np.exp(0.5 * (mu_meta * mu_meta * sigma2_meta[max_ind] - mu_meta[max_ind] * mu_meta[max_ind] * sigma2_meta) / ( sigma2_meta * sigma2_meta[max_ind]) )
        Num = self.lam * sum( self.omegas * EXP * mu_meta * np.sqrt(sigma2_meta / self.sigmas))
        EXP2 = np.exp(- 0.5 * ((mu_meta[max_ind])**2 / sigma2_meta[max_ind]))
        Den = (1- self.lam) * EXP2 + self.lam * sum( self.omegas * EXP * np.sqrt(sigma2_meta / self.sigmas))
        DerNum = self.lam * sum( self.omegas * EXP * (mu_meta * mu_meta + sigma2_meta) * gam1s * np.sqrt(sigma2_meta / self.sigmas) )
        DerDen = self.lam * sum( self.omegas * mu_meta * EXP * gam1s * np.sqrt(sigma2_meta / self.sigmas) )
        return (DerNum * Den - DerDen * Num) / (Den * Den)

    def infer(self,R,r,M,N,iterations,cg_maxit=500,learn_gamw=True, lmmse_damp=True):

        # Initialization
        rank = self.comm.Get_rank()
        r = r.reshape((M,1))
        r1 = r.reshape((M,1))
        r1s = np.zeros((self.K, M))
        xhat1 = np.zeros((M,1)) # signal estimates in Denoising step
        xhat2 = np.zeros((M,1)) # signal estimates in LMMSE step
        Sigma2_u_prev = np.zeros((M,1))
        rho = self.rho # Damping factor
        gam1 = self.gam1
        gam1s = np.zeros(self.K)
        gamw = self.gamw # Precision of noise
        xhat1s = []
        I = scipy.sparse.identity(M) # Identity matrix
        gamws = []
        alpha1 = 0
        alpha2 = 0

        for it in range(iterations):
            if rank == 0:
                logging.info(f"\n -----ITERATION {it} -----")
            
            # Collect gam1s and r1s
            #logging.info(f"rank {rank}: gam1={gam1}")
            gam1s[rank] = gam1
            r1s[rank,:] = r1.squeeze()
            for i in range(self.K):
                #if i != rank:
                gam1s[i] = self.comm.bcast(gam1s[i], root=i)
                r1s[i,:] = self.comm.bcast(r1s[i,:], root=i)
            
            if rank == 0:
                logging.debug(f"gam1s={gam1s}")

            if rank == 0:
                logging.info(f"...Data from all ranks collected")

            # Denoising
            if rank == 0:
                logging.info(f"...Denoising")
            
            xhat1_prev = xhat1
            alpha1_prev = alpha1

            xhat1 = np.array([self.denoiser_meta(r1s[:,j], gam1s) for j in range(M)]).reshape((M,1))

            if it > 0:
                xhat1 = rho * xhat1 + (1 - rho) * xhat1_prev # apply damping on xhat1

            xhat1s.append(xhat1)
            
            if rank == 0:
                self.write_xhat_to_file(it, xhat1)

            alpha1 = np.mean(np.array([self.der_denoiser_meta(r1s[:,j], gam1s) for j in range(M)]))

            if it > 0:
                alpha1 = rho * alpha1 + (1 - rho) * alpha1_prev # apply damping on alpha1
            
            np.clip(alpha1, 1e-5, (1 - 1e-5))

            if rank==0:
                logging.debug(f"[rank = {rank}] alpha1 = {alpha1}")

            # LMMSE for multiple cohorts
            logging.info(f"...LMMSE cohort {rank}")

            xhat2_prev = xhat2
            alpha2_prev = alpha2
            r1 = r1.reshape((M,1))
            xhat1 = xhat1.reshape((M,1))
            gam2 = gam1 * (1 - alpha1) / alpha1

            if rank==0:
                logging.debug(f"[rank = {rank}] gam2 = {gam2}")

            r2 = (xhat1 - alpha1 * r1) / (1 - alpha1)

            A = (gamw * R + gam2 * I) # Sigma2 = A^(-1)
            mu2 = (gamw * r + gam2 * r2)

            # Conjugate gradient for solving linear system A^(-1) @ mu2 = Sigma2 @ mu2
            xhat2, ret = con_grad(A, mu2, maxiter=cg_maxit, x0=xhat2_prev)
            
            if ret > 0: 
                logging.info(f"Rank {rank} WARNING: CG 1 convergence after {ret} iterations not achieved!")
            xhat2.resize((M,1))

            if lmmse_damp:
                xhat2 = rho * xhat2 + (1 - rho) * xhat2_prev # damping on xhat2

            # Generate iid random vector [-1,1] of size M
            u = binomial(p=1/2, n=1, size=M) * 2 - 1

            # Hutchinson trace estimator
            # Sigma2 = (gamw * R + gam2 * I)^(-1)
            # Conjugate gradient for solving linear system (gamw * R + gam2 * I)^(-1) @ u

            Sigma2_u, ret = con_grad(A,u, maxiter=cg_maxit, x0=Sigma2_u_prev)
            Sigma2_u_prev = Sigma2_u

            if ret > 0: 
                logging.info(f"Rank {rank} WARNING: CG 2 convergence after {ret} iterations not achieved!")

            TrSigma2 = u.T @ Sigma2_u # Tr[Sigma2] = u^T @ Sigma2 @ u 

            alpha2 = gam2 * TrSigma2 / M

            if rank==0:
                logging.debug(f"[rank = {rank}] alpha2 = {alpha2}")

            if lmmse_damp:
                alpha2 = rho * alpha2 + (1 - rho) * alpha2_prev # damping on alpha2
            gam1 = gam2 * (1 - alpha2) / alpha2
            r1 = np.squeeze((xhat2 - alpha2 * r2) / (1-alpha2))

            if learn_gamw:

                z = N - (2 * xhat2.T @ r) + (xhat2.T @ R @ xhat2)

                # Hutchinson trace estimator
                TrRSigma2 = u.T @ R @ Sigma2_u # u^T @ R @ [(gamw * R + gam2 * I)^(-1) @ u]

                # Update noise precison
                gamw_prev = gamw
                gamw = 1 / (z / N + TrRSigma2 / N)
                gamw = float(gamw.squeeze())
                # gamw = rho * gamw + (1 - rho) * gamw_prev # damping on gamw
                gamws.append(gamw)

            if rank==0:
                logging.debug(f"gamw = {gamw:0.9f} \n")

            # Write parameters for current cohort to csv file
            self.write_params_to_file([it, gamw, gam1, gam2, alpha1, alpha2], rank)

        return xhat1s