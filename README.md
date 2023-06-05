# Regime_Switching

General problem: PSD matrix is required for covariance matrix, but they become singular in the iterative process for some reason, hence halts the algorithm.

1. det(jP2) becomes negative -- why?
werid thing is: 

both jP and I-KGLmd are positive semi-definite (psd)
- det(jP) is positive
- det(I-KGLmd) is positive

Hence jP2 should also be psd by theory

Tried with data without missing values, still not working

**Joseph form of Kalman Filter solved the issue with jP2**

2. some of the det(jF) elements cannot be computed -- why?

Solved by changing the parameter initializations for **B** and **Lmd** as that keeps the numerical overflow of jF

3. to do

- how to aggreate the likelihood function where possibly FIML can help
- taking gradients
- adding Adam?
