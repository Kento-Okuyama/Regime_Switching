# Regime_Switching

Current problem: aggreated likelihood is computed, but the backward path (to compute derivative) takes forever to be computed at line 309 (see algo_sam7.R).

Possible remedies...
- reduce number of parameters?
- pararrel computing (high performance computing resource)?

Or different approach
- approximate Σexp(f(Theta)) by exp(Σf(Theta)) and get analytical form?