# Regime_Switching

Current problems
- optimization is interrupted by errors (cause not found; possibly undesirable parameter initialization or numerical error in the backward pass calculations?)
- backward path (to compute derivative) takes time to be computed within adam optimization
- calibration of stopping criterion

Remedies
- optimized code by 'switching off' backward path tracking whenever appropriate
- mini-batch gradient descent

Small notes:
- autoregressive parameters B and Lmd should be clipped (at line 72, 73, 76, 77)? -> check stationarity propertires of VAR -> eigenvalues lie within unit circle? ...  the absolute value of each eigenvalue should be less than one 

Future:
- add parameters
- simulation studies - multiple experiments?



## important files

- data file (strictly confidential)
- sam_preprocess: loading data & getting factor scores (Bartlett scores) of the data
- adam: function that runs adam optimization
- algo1_sam7_try: Kim Filter and parameter optimization