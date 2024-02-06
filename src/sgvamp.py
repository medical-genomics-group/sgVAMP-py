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
    def __init__(self, N, M, K, rho, gamw, gam1, a, prior_vars, prior_probs, out_dir, out_name, comm):
        self.eps = 1e-32
        self.N = N
        self.M = M
        self.K = K
        self.L = len(prior_probs)
        self.rho = rho
        self.gamw = gamw
        self.gam1 = gam1 #np.full(K, gam1)
        self.a = a
        self.lam = 1 - prior_probs[0]
        self.sigmas = np.array(prior_vars[1:]) # a vector containing variances of different groups except the first one, length = L-1
        self.omegas = np.array([ p / sum(prior_probs[1:]) for p in prior_probs[1:]])
        self.setup_io(out_dir, out_name)
        self.comm = comm
        self.gam = None

    def setup_io(self, out_dir, out_name):
        self.out_dir = out_dir
        self.out_name = out_name

        # Setup output CSV file for hyperparameters
        for i in range(self.K):
            csv_file = open(os.path.join(self.out_dir, "%s_cohort_%d.csv" % (self.out_name, i+1)), 'w', newline="")
            csv_writer = csv.writer(csv_file, delimiter='\t')
            header = ["it", "gamw", "gam1", "gam2", "alpha1", "alpha2", "lam"]
            csv_writer.writerow(header)
            csv_file.close()

        # Setup file for metrics output
        csv_file_metrics = open(os.path.join(self.out_dir, "%s_metrics.csv" % (self.out_name)), 'w', newline="")
        csv_metrics_writer = csv.writer(csv_file_metrics, delimiter='\t')
        header = ["it", "alignment", "l2"]
        csv_metrics_writer.writerow(header)
        csv_file_metrics.close()

    def write_params_to_file(self, params, cohort_idx):
        csv_file = open(os.path.join(self.out_dir, "%s_cohort_%d.csv" % (self.out_name, cohort_idx+1)), 'a', newline="")
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(params)
        csv_file.close()

    def write_metrics_to_file(self, metrics):
        csv_file = open(os.path.join(self.out_dir, "%s_metrics.csv" % (self.out_name)), 'a', newline="")
        csv_writer = csv.writer(csv_file, delimiter='\t')
        csv_writer.writerow(metrics)
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
        sigma2_meta = 1.0 / (sum(self.a * gam1s) + 1.0/self.sigmas)  # a vector of dimension L - 1
        mu_meta = np.inner(rs, self.a * gam1s) * sigma2_meta
        max_ind = (np.array( mu_meta * mu_meta / sigma2_meta)).argmax()
        EXP = np.exp(0.5 * (mu_meta * mu_meta * sigma2_meta[max_ind] - mu_meta[max_ind] * mu_meta[max_ind] * sigma2_meta) / ( sigma2_meta * sigma2_meta[max_ind]) )
        Num = self.lam * sum(self.omegas * EXP * mu_meta * np.sqrt(sigma2_meta / self.sigmas))
        EXP2 = np.exp(- 0.5 * ((mu_meta[max_ind])**2 / sigma2_meta[max_ind]))
        Den = (1-self.lam) * EXP2 + self.lam * sum(self.omegas * EXP * np.sqrt(sigma2_meta / self.sigmas))
        return Num/Den

    def der_denoiser_meta(self, rs, gam1s):
        sigma2_meta = 1.0 / (sum(self.a * gam1s) + 1.0/ self.sigmas) # a numpy vector of dimension L - 1
        mu_meta = np.inner(rs, self.a * gam1s) * sigma2_meta # a numpy vector of dimension L - 1
        max_ind = (np.array( mu_meta * mu_meta / sigma2_meta)).argmax()
        EXP = np.exp(0.5 * (mu_meta * mu_meta * sigma2_meta[max_ind] - mu_meta[max_ind] * mu_meta[max_ind] * sigma2_meta) / ( sigma2_meta * sigma2_meta[max_ind]) )
        Num = self.lam * sum( self.omegas * EXP * mu_meta * np.sqrt(sigma2_meta / self.sigmas))
        EXP2 = np.exp(- 0.5 * ((mu_meta[max_ind])**2 / sigma2_meta[max_ind]))
        Den = (1- self.lam) * EXP2 + self.lam * sum( self.omegas * EXP * np.sqrt(sigma2_meta / self.sigmas))
        DerNum = self.lam * sum( self.omegas * EXP * (mu_meta * mu_meta + sigma2_meta) * self.a[self.comm.Get_rank()] * gam1s[self.comm.Get_rank()] * np.sqrt(sigma2_meta / self.sigmas) )
        DerDen = self.lam * sum( self.omegas * mu_meta * EXP * self.a[self.comm.Get_rank()] * gam1s[self.comm.Get_rank()] * np.sqrt(sigma2_meta / self.sigmas) )
        return (DerNum * Den - DerDen * Num) / (Den * Den)
    
    def prior_update_em(self, r1s, gam1s):
        # r1s is a (K,M) numpy matrix
        # gam1s is a (K,) numpy array

        # converting to the right dimension
        Lm1 = self.L -1 
        prior_vars0 = self.sigmas.reshape(1, 1, Lm1)
        gam1s_rs = gam1s.reshape(self.K, 1, 1)
        gam1invs = 1.0/gam1s_rs
        r1s_rs = r1s.reshape(self.K, self.M, 1)

        exp_max = ( -np.power(r1s_rs,2).reshape(self.K, self.M, 1) / 2 / (prior_vars0 + gam1invs) ).max()
        xi = self.lam * self.omegas.reshape(1, 1, Lm1) * np.exp(- np.power(r1s_rs,2).reshape(self.K, self.M, 1) / 2 / (prior_vars0 + gam1invs) - exp_max) / np.sqrt(gam1invs + prior_vars0)
        sum_xi = xi.sum(axis=2).reshape(self.K, self.M,1)
        xi_tilde = xi / sum_xi
        pi = 1.0 / ( 1.0 + (1-self.lam) * np.exp(-np.power(r1s_rs,2) / 2 * gam1s_rs - exp_max) / np.sqrt(gam1invs) / sum_xi )
        
        #updating sparsity level
        self.lam = np.mean( np.average(pi, axis=0, weights=self.a) )
        #updating prior probabilities in the mixture
        self.omegas = np.sum(pi * xi_tilde * self.a.reshape(self.K,1,1), axis = (0,1)) / np.sum(pi * self.a.reshape(self.K,1,1), axis = (0,1))


    def Lagrangian_der(self, x, omega0, sigma2, r1s, gam1s):
        # r1s is a (K,M) numpy matrix
        # gam1s is a (K,) numpy array
        # omega0 is a (L,) numpy array

        y = np.zeros(self.L+1)
        omega = x[:self.L]
        gam = x[self.L]
        prior_vars0 = sigma2.reshape(1, 1, self.L)
        gam1s_rs = gam1s.reshape(self.K, 1, 1)
        gam1invs = 1.0/gam1s_rs
        r1s_rs = r1s.reshape(self.K, self.M, 1)
        omega_rs = omega.reshape(1,1,self.L)

        exp_max = ( -np.power(r1s_rs,2).reshape(self.K, self.M, 1) / 2 / (prior_vars0 + gam1invs) ).max()
        probs = np.exp(-np.power(r1s_rs,2) / 2 / (prior_vars0 + gam1invs) - exp_max) / np.sqrt(prior_vars0 + gam1invs)
        Num = self.a.reshape(self.K,1,1) * probs
        Den = np.sum(probs * omega_rs, axis=2).reshape(self.K, self.M, 1)

        y[:self.L] = np.sum(Num/Den, axis=(0,1)) + (omega0-1) / omega + gam
        y[self.L] = sum(omega) - 1.0 
        return y

    def prior_update_mle(self, r1s, gam1s):
        rank = self.comm.Get_rank()

        omega0 = np.zeros(self.L)
        omega0[0] = 1 - self.lam
        omega0[1:] = self.lam * self.omegas

        sigma2 = np.zeros(self.L)
        sigma2[0] = 1e-16
        sigma2[1:] = self.sigmas

        x0 = np.zeros(self.L + 1)
        x0[:-1] = omega0
        if self.gam == None:
            x0[-1] = 1
        else:   
            x0[-1] = self.gam

        x, _, ier, _ = scipy.optimize.fsolve(func=self.Lagrangian_der, x0=x0, args=(omega0,sigma2,r1s,gam1s), full_output=True)
        #logging.debug(f"x={x}")
        if ier != 1:
            if rank == 0:
                logging.info(f"WARNING: fsolve not converged. No prior update!")
            return
        elif any(s <= 0 for s in x[:-1]):
            if rank == 0:
                logging.info(f"WARNING: Negative values in MLE. No prior update!")
            return
        else:
            x[:-1] /= sum(x[:-1])
            self.lam = 1 - x[0]
            self.omegas = np.array([ w / sum(x[1:-1]) for w in x[1:-1]])
            self.gam = x[self.L]

    def infer(self,R,r,iterations, x0, cg_maxit=500, em_prior_maxit=100, learn_gamw=True, lmmse_damp=True, prior_update=None):

        # Initialization
        M = self.M
        N = self.N
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

        if rank == 0:
            logging.debug(f"a = {self.a}")

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

            # Update prior
            if it > 0: 
                if prior_update == "mle":
                    if rank == 0:
                        logging.info("...Updating prior parameters using MLE")
                    self.prior_update_mle(r1s, gam1s)
                elif prior_update == "em":
                    if rank == 0:
                        logging.info("...Updating prior parameters using EM")
                    for em_it in range(em_prior_maxit):
                        old_omegas = self.omegas
                        old_lam = self.lam
                        self.prior_update_em(r1s, gam1s)
                        omegas_rel_err = np.linalg.norm(self.omegas - old_omegas) / np.linalg.norm(old_omegas)
                        lam_rel_err = np.abs(self.lam - old_lam) / self.lam
                        if  omegas_rel_err < 1e-6 and lam_rel_err < 1e-6:
                            break
                    if rank == 0:
                        logging.info(f"... prior-learning EM algorithm performed {em_it+1} steps and had final relative error = {max(omegas_rel_err,lam_rel_err):0.9f}")
            
            if rank == 0:
                logging.debug(f"lam={self.lam}")
                logging.debug(f"omegas={self.omegas}")
                logging.debug(f"sigmas={self.sigmas}")

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
            delta = 1 - np.log(2*alpha1)
            if alpha1<0.5:
                alpha1 *= delta

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
            self.write_params_to_file([it, gamw, gam1, gam2, alpha1, alpha2, self.lam], rank)

            # Calculate error metrics
            alignment = np.inner(xhat1.squeeze(), x0.squeeze()) / np.linalg.norm(xhat1.squeeze()) / np.linalg.norm(x0.squeeze()) # Alignment
            l2 = np.linalg.norm(xhat1.squeeze() - x0.squeeze()) / np.linalg.norm(x0.squeeze()) # L2 norm error
        
            if rank==0:
                logging.debug(f"Alignment(xhat1, x0) = {alignment:0.9f} \n")
                logging.debug(f"L2_error(xhat1, x0) = {l2:0.9f} \n")
                self.write_metrics_to_file([it, alignment, l2])

        return xhat1s