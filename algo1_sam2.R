#############
# algo1_sam #
#############

# sam_preprocess3
# treat missing values as being filled using most recent available entry

# sam_preprocess4
# treat missing values as it is
# CFA cannot be applied?


# change from the last work
# 1. substituted y_{t} with eta_{t} in the Hamilton Filter
# 2. substituted univariate notation with multivariate (bold) notation
# 3. apply lavaan to obtain the "marginal" eta_{t} (for #1)

# 4. handle missing values without imputation in the Kim Filter 
# Requirement:
# eta_{t|t}^{s,s'} := eta_{t|t-1}^{s,s'}
# delta_{t}^{s,s'} -> 0, 
# P{t|t}^{s,s'} := P{t|t-1}^{s,s'} -> inf,
# v_{t}^{s,s'} -> 0,
# F_{t}^{s,s'} -> inf
# Result:
# K_{t}^{s} -> 0
# f(eta_{t}|eta{t-1}) := f(eta_{t}|s,s',eta{t-1}) -> 0
#

# FIML
