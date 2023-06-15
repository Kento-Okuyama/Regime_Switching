# Regime_Switching

Current problems
- optimization is interrupted by errors (mostly p.d. violation of covariance matrix)
- calibration of stopping criterion

Remedies
- Diagonal purtubation: add a small number to diagonal elements whenever it is not p.d.
- Eigendecomposition: replace non-positive eigenvalues witha small number 

Future:
- check estimated S_{it} and parameters
- simulation studies - multiple experiments? (not for master thesis)

## important files

- data file (strictly confidential)
- sam_preprocess: loading data & getting factor scores (Bartlett scores) of the data
- adam: function that runs adam optimization
- algo1_sam7_try: Kim Filter and parameter optimization (Eigendecomposition + Diagonal purtubation)