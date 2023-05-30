# Regime_Switching

PSD matrix is required for covariance matrix,
but they become singular in the iterative process

det(jP2) becomes negative -- why?
werid thing is: 

both jP and I-KGLmd are positive semi-definite (psd)
- det(jP) is positive
- det(I-KGLmd) is positive

Hence jP2 should also be psd by theory

Tried with data without missing values, still not working