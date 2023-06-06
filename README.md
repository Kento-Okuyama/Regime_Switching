# Regime_Switching

Current problem: 
- error says that inplace modification prevents derivative from being calculated
- aggreated likelihood is computed, but the backward path (to compute derivative) takes forever to be computed at line 309 (see algo_sam7.R) + there's no counterpart to torch_detach() which might be helpful to reduce the computation costs 


For first problem:
Figure out where to add torch_clone() <- might have been solved

For second problem:

Possible remedies...
- reduce number of parameters?
- pararrel computing (high performance computing resource)?

Or different approach
- approximate Σexp(f(Theta)) by exp(Σf(Theta)) and get analytical form?