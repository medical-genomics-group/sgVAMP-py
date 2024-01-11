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
from mpi4py import MPI
import logging

class VAMP:
    def __init__(self, rho, lam, sigmas, p_weights, gamw, gam1, out_dir, out_name):
        self.eps = 1e-32
        self.lam = lam
        self.sigmas = sigmas # a vector containing variances of different groups
        self.p_weights = p_weights # vector containing probability weights for the slab component 
        self.rho = rho
        self.gamw = gamw
        self.gam1 = gam1
        self.setup_io(out_dir, out_name)

    def setup_io(self, out_dir, out_name):
        self.out_dir = out_dir
        self.out_name = out_name

        # Setup output CSV file for hyperparameters
        csv_file = open(os.path.join(self.out_dir, self.out_name + ".csv"), 'w', newline="")
        csv_writer = csv.writer(csv_file)
        header = ["it", "gamw", "gam1", "gam2", "alpha1", "alpha2"]
        csv_writer.writerow(header)
        csv_file.close()

    def write_params_to_file(self, params):
        # Setup output CSV file for hyperparameters
        csv_file = open(os.path.join(self.out_dir, self.out_name + ".csv"), 'a', newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(params)
        csv_file.close()
    
    def write_xhat_to_file(self, it, xhat):
        # Setup output binary file for xhat estimates
        fname = "%s_xhat_it_%d.bin" % (self.out_name, it)
        f = open(os.path.join(self.out_dir, fname), "wb")
        f.write(struct.pack(str(len(xhat))+'d', *xhat.squeeze()))
        f.close()

    # gam1s = a vector of gam1 values over different GWAS studies
    def denoiser_meta(self, rs, gam1s):
        sigma2_meta = 1.0 / (sum(gam1s) + 1.0/self.sigmas)  # a vector of dimension L
        mu_meta = np.inner(rs, gam1s) * sigma2_meta
        max_ind = (np.array( mu_meta * mu_meta / sigma2_meta)).argmax()
        EXP = np.exp(0.5 * (mu_meta * mu_meta * sigma2_meta[max_ind] - mu_meta[max_ind] * mu_meta[max_ind] * sigma2_meta) / ( sigma2_meta * sigma2_meta[max_ind] ) )
        Num = self.lam * sum(self.p_weights * EXP * mu_meta * np.sqrt(sigma2_meta / self.sigmas))
        EXP2 = np.exp(- 0.5 * ((mu_meta[max_ind])**2 / sigma2_meta[max_ind]))
        Den = (1-self.lam) + self.lam * sum(self.p_weights * EXP * np.sqrt(sigma2_meta / self.sigmas))
        return Num/Den

    def denoiser(self, r, gam1): # not numerically stable! AD
        A = (1-self.lam) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
        B = self.lam * norm.pdf(r, loc=0, scale=np.sqrt(self.sigmas + 1.0/gam1))
        ratio = gam1 * r / (gam1 + 1.0) * B / (A + B + self.eps)
        return ratio

    def der_denoiser_meta(self, rs, gam1s):
        sigma2_meta = 1.0 / (sum(gam1s) + 1.0/self.sigmas)  # a vector of dimension L
        mu_meta = np.inner(rs, gam1s) * sigma2_meta
        max_ind = (np.array( mu_meta * mu_meta / sigma2_meta)).argmax()
        EXP = np.exp(0.5 * (mu_meta * mu_meta * sigma2_meta[max_ind] - mu_meta[max_ind] * mu_meta[max_ind] * sigma2_meta) / (sigma2_meta * sigma2_meta[max_ind]))
        Num = self.lam * sum(self.p_weights * EXP * mu_meta * np.sqrt(sigma2_meta / self.sigmas))
        
        EXP2 = np.exp(- 0.5 * ((mu_meta[max_ind])**2 / sigma2_meta[max_ind]))
        Den = (1-self.lam) * EXP2 + self.lam * sum(self.p_weights * EXP * np.sqrt(sigma2_meta / self.sigmas))
        
        DerNum = self.lam * sum(self.p_weights * EXP * (sigma2_meta * mu_meta * mu_meta + 1) * sigma2_meta * gam1s * np.sqrt(sigma2_meta / self.sigmas))
        DerDen = self.lam * sum(self.p_weights * mu_meta * mu_meta * EXP * gam1s * sigma2_meta * sigma2_meta * np.sqrt(sigma2_meta / self.sigmas))
        return (DerNum * Den - DerDen * Num) / (Den * Den)
    
    def der_denoiser(self, r, gam1): # not numerically stable! AD
        A = (1-self.lam) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
        B = self.lam * norm.pdf(r, loc=0, scale=np.sqrt(self.sigmas + 1.0/gam1))
        Ader = A * (-r*gam1)
        Bder = B * (-r) / (1.0 + 1.0/gam1 + self.eps)
        BoverAplusBder = ( Bder * A - Ader * B ) / (A+B + self.eps) / (A+B + self.eps)
        ratio = gam1 / (gam1 + 1.0) * B / (A + B + self.eps) + BoverAplusBder * r * gam1 / (gam1 + 1.0)
        return ratio
    
    def infer(self,R,r,M,N,K,iterations,cg_maxit=500,learn_gamw=True, lmmse_damp=True, Comm=MPI.COMM_WORLD):

        # initialization
        rank = Comm.Get_rank()
        size_MPI = Comm.Get_size()

        r1 = r #np.zeros((M,1))
        xhat1 = np.zeros((M,1)) # signal estimates in Denoising step
        xhat2 = np.zeros((M,1)) # signal estimates in LMMSE step
        Sigma2_u = np.zeros((M,1))
        gam1 = self.gam1
        rho = self.rho # Damping factor
        gamw = self.gamw # Precision of noise
        xhat1s = []
        I = scipy.sparse.identity(M) # Identity matrix
        gamws = []
        alpha1 = 0
        alpha2 = 0

        for it in range(iterations):
            if rank==0:
                logging.info(f"-----ITERATION {it} -----")
            # Denoising
            # collecting r1s from all the processes
            r1s = np.zeros((K, M)) # every column represents one marker
            r1s[rank,:] = np.reshape(r1, (1, -1))
            gam1s =np.zeros((K,1))
            gam1s[rank] = gam1

            Comm.Barrier()

            if (size_MPI > 1):
                for dest in range(K):
                    if dest!=rank:
                        reqr1S = Comm.Isend(r1s[rank], tag=0, dest=dest)
                        reqgam1S = Comm.Isend(gam1s[rank], tag=1, dest=dest)

                for source in range(K):
                    if source!=rank:
                        reqr1R = Comm.Irecv(r1s[source], tag=0, source=source)
                        reqgam1R = Comm.Irecv(gam1s[source], tag=1, source=source)

            Comm.Barrier()

            if rank==0:
                logging.info("...Denoising")
                logging.debug(f"gam1 = {gam1:0.9f}")
            xhat1_prev = xhat1
            alpha1_prev = alpha1
            vect_den_beta = lambda x: self.denoiser(x, gam1)
            # vect_den_beta_meta = lambda x: self.denoiser_meta(x, gam1s)
            xhat1 = vect_den_beta(r1)
            # xhat1 = np.apply_along_axis(vect_den_beta, axis=1, arr=r1s)
            xhat1 = rho * xhat1 + (1 - rho) * xhat1_prev # apply damping on xhat1
            xhat1s.append(xhat1)
            alpha1 = np.mean( self.der_denoiser(r1, gam1) )
            # vect_der_den_beta_meta = lambda x: self.der_denoiser_meta(x, gam1s)
            # xhat1 = np.mean( np.apply_along_axis(vect_der_den_beta, axis=1, arr=r1s) )
            alpha1 = rho * alpha1 + (1 - rho) * alpha1_prev # apply damping on alpha1
            if rank==0:
                logging.debug(f"alpha1 = {alpha1:0.9f}")
            gam2 = gam1 * (1 - alpha1) / alpha1
            if rank==0:
                logging.debug(f"gam2 = {gam2:0.9f}")
            r2 = (xhat1 - alpha1 * r1) / (1 - alpha1)

            # LMMSE
            if rank==0: 
                logging.info("...LMMSE")
            xhat2_prev = xhat2
            alpha2_prev = alpha2
            A = (gamw * R + gam2 * I) # Sigma2 = A^(-1)
            mu2 = (gamw * r + gam2 * r2)

            # Conjugate gradient for solving linear system A^(-1) @ mu2 = Sigma2 @ mu2
            xhat2, ret = con_grad(A, mu2, maxiter=cg_maxit, x0=xhat2_prev)
            
            if ret > 0 and rank==0: 
                logging.info(f"WARNING: CG 1 convergence after {ret} iterations not achieved!")
            xhat2.resize((M,1))

            if lmmse_damp:
                xhat2 = rho * xhat2 + (1 - rho) * xhat2_prev # damping on xhat2

            # Generate iid random vector [-1,1] of size M
            u = binomial(p=1/2, n=1, size=M) * 2 - 1

            # Hutchinson trace estimator
            # Sigma2 = (gamw * R + gam2 * I)^(-1)
            # Conjugate gradient for solving linear system (gamw * R + gam2 * I)^(-1) @ u

            Sigma2_u_prev = Sigma2_u
            Sigma2_u, ret = con_grad(A,u, maxiter=cg_maxit, x0=Sigma2_u_prev)

            if ret > 0 and rank==0: 
                logging.info(f"WARNING: CG 2 convergence after {ret} iterations not achieved!")

            TrSigma2 = u.T @ Sigma2_u # Tr[Sigma2] = u^T @ Sigma2 @ u 

            alpha2 = gam2 * TrSigma2 / M
            if lmmse_damp:
                alpha2 = rho * alpha2 + (1 - rho) * alpha2_prev # damping on alpha2
            if rank==0:
                logging.debug(f"alpha2 = {alpha2:0.9f}")
            gam1 = gam2 * (1 - alpha2) / alpha2
            r1 = (xhat2 - alpha2 * r2) / (1-alpha2)

            if learn_gamw:

                z = N - (2 * xhat2.T @ r) + (xhat2.T @ R @ xhat2)

                # Hutchinson trace estimator
                TrRSigma2 = u.T @ R @ Sigma2_u # u^T @ R @ [(gamw * R + gam2 * I)^(-1) @ u]

                # Update noise precison
                gamw_prev = gamw
                gamw = 1 / (z / N + TrRSigma2 / N)
                gamw = float(gamw.squeeze())
                gamw = rho * gamw + (1 - rho) * gamw_prev # damping on gamw
            gamws.append(gamw)
            if rank==0:
                logging.debug(f"gamw = {gamw:0.9f} \n")

            # Write parameters from current iteration to output file
            self.write_params_to_file([it, gamw, gam1, gam2, alpha1, alpha2])
            self.write_xhat_to_file(it, xhat1)

        return xhat1s