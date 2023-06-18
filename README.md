# Regime_Switching

Current problems
- evaluation of the estimated models (latent variables, dynamic regime indicators, parameters)
- dropout indicator to be used as covariate
- adding interaction of intra- and inter-individual latent factors (in both state-space model and Hamilton filter)?
- optimization is interrupted by errors (errors happen at random)
(- calibration of stopping criterion)

Remedies for p.d. violation
- Diagonal purtubation: add a small number to diagonal elements whenever it is not p.d.
- Eigendecomposition: replace non-positive eigenvalues witha small number 

Evaluation of the estimated models
- compare trajectories of estimated S_{it} and parameters against dropout occurences
... no reliable estimate produced
Lmd not used


Adding interaction effects
- added interaction effects to the transition probability (Hamilton filter part)

Future:
- simulation studies - multiple experiments? (not for master thesis)

## latest important files
#### partial interaction (Hamilton filter contains interaction) -> dropout as covariate
- sam_preprocessing_interact: loading data & getting factor scores (Bartlett scores) of the data
- adam_interact_do: function that runs adam optimization
- algo1_sam7_try2_interact_do: Kim Filter and parameter optimization (Eigendecomposition + Diagonal purtubation)
- algo1_sam7_try2_after_interact_do: Kim filter for the etstimated model, and some evaluation (not yet)