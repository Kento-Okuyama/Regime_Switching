# Regime_Switching

Current problem: backward path (to compute derivative) takes forever to be computed within adam optimization

Possible remedy -> batch samples

## important files

- sam_preprocess4: loading data
- CFA: getting factor scores of the data
- adam: function that runs stochastic optimization
- algo1_sam7: Kim Filter and parameter optimization
