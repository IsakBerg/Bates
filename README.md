# Bates
Bates model

from __future__ import division
import numpy as np
from scipy.stats import norm
from numpy import linalg

def bates(S0, r, d, T, Vinst, Vlong, kappa, epsilon, rho, lambdas, muj, sigj, NTime, NSim, NBatches):

    # Discretization for the Bates model
    # Using QE Scheme


    dT = T/NTime                                              # time step
    pathS = np.zeros(shape = (NSim, NTime+1, NBatches))       # output pathS
    pathV = np.zeros(shape = (NSim, NTime+1, NBatches))         # output pathV

    lnS1 = np.zeros(shape = (NSim, NTime+1))                   # logspot price path
    lnS1[:,0] = np.log(S0*np.exp(-d*T))             # set S(0) adjust with dividend

    V2 = np.zeros(shape = (NSim,NTime+1))                     # variance paths
    V2[:,0] = Vinst                                 # set V0

    k1 = np.exp(-kappa*dT)
    k2 = epsilon**2*k1*(1-k1)/kappa
    k3 = np.exp(kappa*dT)*0.5*k2*(1-k1)*Vlong

    psiC = 1.5                                      # psi in (1,2)
    gamma1 = 0.5                                    # For PredictorCorrector
    gamma2 = 0.5                                    # For PredictorCorrector

    c1 = (r-d)*dT                                  # drift adjustment
    c2 = -rho*kappa/epsilon*dT                     # used to determine K0

    K0 = c1 + c2*Vlong                                      # drift adjusted K0
    K1 = gamma1*dT*(kappa*rho/epsilon - 0.5)-rho/epsilon    # K1
    K2 = gamma2*dT*(kappa*rho/epsilon - 0.5)+rho/epsilon    # K2
    K3 = gamma1*dT*(1-rho**2)                                # K3
    K4 = gamma2*dT*(1-rho**2)                                # K4

    for l in range(NBatches):                       #batch loop
        UV1 = np.random.rand(NSim,NTime)            #uniforms
        UV2 = np.random.rand(NSim,NTime)            #uniforms
        dW2 = norm.ppf(UV2)                         #Gaussians

        for i in range(1,NTime+1):                  #timeloop
            m = Vlong + (V2[:,i-1]-Vlong)*k1        # mean (moment matching)
            s2 = V2[:,i-1]*k2 + k3                  # var (moment mathing)
            m2 = np.square(m)                              # psi compared to psiC
            psi = np.true_divide(s2,m2)

            psihat = np.true_divide(1,psi)
            b2 = 2*psihat-1+np.sqrt(np.multiply(2*psihat,2*psihat-1))
            a = np.true_divide(m,(1+b2))
            V2[psi<=psiC,i] = np.multiply(a[psi<=psiC],(np.square(np.sqrt(b2[psi<=psiC])+norm.ppf(UV1[psi<=psiC,i-1])))) # non-central chi squared approximation

            p = np.true_divide(psi-1,psi+1)         # for switching rule

            V2[(UV1[:,i-1]<=p) & (psi>psiC),i]=0   #case u<=p&psi>psiC
            
            I1b = np.where((UV1[:,i-1]>p) & (psi>psiC))


            beta = np.true_divide(1-p,m)         #for switching rule
            if len(I1b)==0:
                print "Hej"
            else:

                v2 = np.log(np.true_divide(1-p[I1b],1-UV1[I1b,i-1]))
                V2[I1b,i] = np.true_divide(v2,beta[I1b])

            #Jumps

            P = np.random.poisson(lambdas*dT,size=(NSim,1))       #randoms for jumps
            lnY = muj*P + np.multiply(sigj*np.sqrt(P),np.random.randn(NSim,1))   #log jumps

            #log Euler Predictor-Corrector step
            l3 = np.sqrt(K3*V2[:,i-1] + K4*V2[:,i-1])
            lnS1[:,i] = lnS1[:,i-1] + K0 + K1*V2[:,i-1] + K2*V2[:,i] + np.multiply(l3, dW2[:,i-1]) + lnY - dT*lambdas*(np.exp(np.log(1+muj)+0.5*sigj**2)-1)

        pathS[:,:,l] = np.exp(lnS1)
        pathV[:,:,l] = V2


    return pathS, pathV

