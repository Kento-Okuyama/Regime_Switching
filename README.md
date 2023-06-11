# Regime_Switching

Current problem: backward path (to compute derivative) takes time to be computed within adam optimization

Remedies
- optimized code by 'switching off' backward path tracking whenever appropriate
- mini-batch gradient descent

Next problem: calibration of stopping criterion

- autoregressive parameters B and Lmd should be clipped (at line 72, 73, 76, 77)?
- Line 210-226 could be speeded up using vector-indexing notations?

## important files

- data file (strictly confidential)
- sam_preprocess4: loading data
- CFA: getting factor scores of the data
- adam: function that runs stochastic optimization
- algo1_sam7: Kim Filter and parameter optimization
- algo1_sam7_try: same as above (with error catch)
- algo1_sam7_batch: same as above (mini-batch version)
