# gVAMP for summary statistics

from scipy.stats import norm
import numpy as np
from numpy.linalg import inv

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
        ratio = gam1 * r / (gam1 + 1.0) * B / (A + B)
        return ratio

    def der_denoiser(self, r, gam1):
        A = (1-self.lam) * norm.pdf(r, loc=0, scale=np.sqrt(1.0/gam1))
        B = self.lam * norm.pdf(r, loc=0, scale=np.sqrt(1.0 + 1.0/gam1))
        Ader = A * (-r*gam1)
        Bder = B * (-r) / (1.0 + 1.0/gam1)
        BoverAplusBder = ( Bder * A - Ader * B ) / (A+B) / (A+B)
        ratio = gam1 / (gam1 + 1.0) * B / (A + B) + BoverAplusBder * r * gam1 / (gam1 + 1.0)
        return ratio

    def infer(self,R,r,iterations):
        [M,M] = np.shape(R)
        [N,_] = np.shape(r)

        # initialization
        r1 = np.zeros((M,1))
        xhat1 = np.zeros((M,1))
        gam1 = self.gam1
        rho=self.rho
        gamw = self.gamw
        xhat1s = []

        for it in range(iterations):
            print("-----ITERATION %d -----"%(it))
            # Denoising
            print("...Denoising")
            xhat1_prev = xhat1
            vect_den_beta = lambda x: self.denoiser(x, gam1)
            xhat1 = vect_den_beta(r1)
            xhat1 = rho * xhat1 + (1 - rho) * xhat1_prev
            xhat1s.append(xhat1)
            alpha1 = np.mean( self.der_denoiser(r1, gam1) )
            gam2 = gam1 * (1 - alpha1) / alpha1
            r2 = (xhat1 - alpha1 * r1) / (1 - alpha1)

            # LMMSE
            print("...LMMSE")
            xhat2 = inv(gamw * R + gam2 * np.identity(M)) @ (gamw * r + gam2 * r2)
            alpha2 = gam2 * np.trace(inv(gamw * R + gam2 * np.identity(M))) / M
            gam1 = gam2 * (1 - alpha2) / alpha2
            r1 = (xhat2 - alpha2 * r2) / (1-alpha2)

        return xhat1s