# Regime_Switching

Current problems
- optimization is interrupted by errors (mostly p.d. violation of covariance matrix)
- calibration of stopping criterion
- adding interaction of intra- and inter-individual latent factors to Kim filter
- evaluation of the estimated models (latent variables, dynamic regime indicators, parameters)

Remedies for p.d. violation
- Diagonal purtubation: add a small number to diagonal elements whenever it is not p.d.
- Eigendecomposition: replace non-positive eigenvalues witha small number 

Adding interaction effects
- added interaction effects to the transition probability (Hamilton filter part)

Evaluation of the estimated models
- compared trajectories of estimated S_{it} and parameters against dropout occurences

Future:
- simulation studies - multiple experiments? (not for master thesis)

## important files
#### only main effects
- data file (strictly confidential)
- sam_preprocess: loading data & getting factor scores (Bartlett scores) of the data
- adam: function that runs adam optimization
- algo1_sam7_try: Kim filter and parameter optimization (Eigendecomposition + Diagonal purtubation)
- algo1_sam7_try2_after: Kim filter for the etstimated model, and some evaluation

#### partially interaction (Hamilton filter contains interaction)
- sam_preprocess_interact: loading data & getting factor scores (Bartlett scores) of the data
- adam2: function that runs adam optimization
- algo1_sam7_try: Kim Filter and parameter optimization (Eigendecomposition + Diagonal purtubation)
- algo1_sam7_try2_after_interact: Kim filter for the etstimated model, and some evaluation (not yet)
