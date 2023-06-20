# Regime_Switching

Current problems
- evaluation of the estimated models (latent variables, dynamic regime indicators, parameters)
- optimization is interrupted by errors which happen at random
(- calibration of stopping criterion)

Remedies for p.d. violation
- Diagonal purtubation: add a small number to diagonal elements whenever it is not p.d.
- Eigendecomposition: replace non-positive eigenvalues witha small number 

Evaluation of the estimated models
- compare trajectories of estimated S_{it} and parameters against dropout occurences
... no reliable estimate produced
Lmd not used

Adding dropout indicator
- AR(1) model (Kalman filter)

Adding interaction effects
- transition probability (Hamilton filter)
- AR(1) model (Kalman filter)

Future:
- simulation studies - multiple experiments? (not for master thesis)

## file description (interaction in both the kalman filter and hamilton filter + dropout as a covariate in the Kalman filter)
- sam_preprocessing: loading data & getting factor scores (Bartlett scores) of the data
- adam: function that runs adam optimization
- algo: Kim Filter and parameter optimization (Eigendecomposition + Diagonal purtubation)
- algo_after: Kim filter for the etstimated model, and some evaluation