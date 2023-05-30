# step 1: input {Y_{T}}
# step 2: compute {eta_{t}}_{t=1:T}

######################### 
##  
# y3D: N x Nt x nC3D
## intra- and inter-individual observed variables
# eta3D: N x Nt x Nf
## intra- and inter-individual latent factors
#########################

# install.packages("torch")
library(torch)
# install.packages("reticulate")
library(reticulate)

# for reproducibility 
set.seed(init)
# number of parameter initialization
nInit <- 1
# max number of optimization steps
nIter <- 1
# a very small number
epsilon <- 1e-3

###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

dropout <- y3D[,,dim(y3D)[3]]
y <- y3D[,,1:(dim(y3D)[3]-1)]
N <- dim(y)[1]
Nt <- dim(y)[2]
No <- dim(y)[3]
eta <- eta3D
for (t in 1:Nt) {
  for (i in 1:N) {eta[i,t,] <- eta[i,t,] - colMeans(eta[,1,])} }
Nf <- dim(eta)[3]

###################################
# Algorithm 1
###################################

for (init in 1:nInit) {
  print(paste0('Initialization step '), init)
  
  # optimization step count
  iter <- 1
  # stopping criterion count
  count <- 0 
  # moment estimates 
  m <- v <- NULL
  
  # step 3: initialize parameters
  a1 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE) 
  a2 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE) 
  a <- list(a1, a2)
  B1d <- torch_tensor(torch_randn(Nf), requires_grad=TRUE)
  B2d <- torch_tensor(torch_randn(Nf), requires_grad=TRUE)
  B1 <- torch_diag(B1d)
  B2 <- torch_diag(B2d)
  B <- list(B1, B2)
  k1 <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  k2 <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  k <- list(k1, k2)
  Lmd1v <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  Lmd2v <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  Lmd1 <- Lmd2 <- torch_full(c(Nf,No), 0)
  Lmd1[1,1:3] <- Lmd1v[1:3]; Lmd1[2,4:5] <- Lmd1v[4:5]
  Lmd1[3,6:7] <- Lmd1v[6:7]; Lmd1[4,8:9] <- Lmd1v[8:9]
  Lmd1[5,10:11] <- Lmd1v[10:11]; Lmd1[6,12:14] <- Lmd1v[12:14]
  Lmd1[7,15:17] <- Lmd1v[15:17]; Lmd1[8,18] <- Lmd2v[18]
  Lmd2[1,1:3] <- Lmd2v[1:3]; Lmd2[2,4:5] <- Lmd2v[4:5]
  Lmd2[3,6:7] <- Lmd2v[6:7]; Lmd2[4,8:9] <- Lmd2v[8:9]
  Lmd2[5,10:11] <- Lmd2v[10:11]; Lmd2[6,12:14] <- Lmd2v[12:14]
  Lmd2[7,15:17] <- Lmd2v[15:17]; Lmd2[8,18] <- Lmd2v[18]
  Lmd <- list(Lmd1, Lmd2)
  alpha1 <- torch_tensor(torch_randn(1), requires_grad=TRUE)
  alpha2 <- torch_tensor(torch_randn(1), requires_grad=TRUE)
  alpha <- list(alpha1, alpha2)
  beta1 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE)
  beta2 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE)
  beta <- list(beta1, beta2)
  Q1d <- torch_tensor(torch_randn(Nf)**2, requires_grad=TRUE)
  Q2d <- torch_tensor(torch_randn(Nf)**2, requires_grad=TRUE)
  Q1 <- torch_diag(Q1d)
  Q2 <- torch_diag(Q2d)
  Q <- list(Q1, Q2)
  R1d <- torch_tensor(torch_randn(No)**2, requires_grad=TRUE)
  R2d <- torch_tensor(torch_randn(No)**2, requires_grad=TRUE)
  R1 <- torch_diag(R1d)
  R2 <- torch_diag(R2d)
  R <- list(R1, R2)
  theta <- torch_cat(list(a1, a2, B1d, B2d, k1, k2, Lmd1v, Lmd2v, alpha1, alpha2, beta1, beta2, Q1d, Q2d, R1d, R2d))
  
  # define variables
  jEta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.2 (LHS)
  jDelta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.3 (LHS)
  jP <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Eq.4 (LHS)
  jPChol <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) 
  jV <- torch_full(c(N,Nt,2,2,No), NaN) # Eq.5 (LHS)
  jF <- torch_full(c(N,Nt,2,2,No,No), NaN) # Eq.6 (LHS)
  jFChol <- torch_full(c(N,Nt,2,2,No,No), NaN) 
  jEta2 <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.7 (LHS)
  jP2 <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Eq.8 (LHS)
  mEta <- torch_full(c(N,Nt+1,2,Nf), NaN) # Eq.9-1 (LHS)
  mP <- torch_full(c(N,Nt+1,2,Nf,Nf), NaN) # Eq.9-2 (LHS)
  W <- torch_full(c(N,Nt,2,2), NaN) # Eq.9-3 (LHS)
  jPr <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-1 (LHS)
  mLik <- torch_full(c(N,Nt), NaN) # Eq.10-2 (LHS)
  jPr2 <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-3 (LHS)
  mPr <- torch_full(c(N,Nt+1), NaN) # Eq.10-4 (LHS)
  jLik <- torch_full(c(N,Nt,2,2), NaN) # Eq.11 (LHS)
  tPr <- torch_full(c(N,Nt,2), NaN) # Eq.12 (LHS)
  
  # rows that have non-NA values 
  noNaRows <- list()
  # rows that have NA values
  naRows <- list()
  
  # step 4: initialize latent variables
  for (s in 1:2) {
    for (i in 1:N) {
      mEta[i,1,s,] <- rep(x=0, times=Nf)
      mP[i,1,s,,] <- diag(x=1e3, nrow=Nf, ncol=Nf) } }
  
  # while (count < 3) {
  for (iter in 1:nIter) {
    print(paste0('   optimization step: ', as.numeric(iter)))
    
    # step 5: initialize P(s'|eta_0)
    mPr[,1] <- epsilon 
    
    # store the pair (s,s') as data frame 
    jS <- expand.grid(s1=c(1,2), s2=c(1,2))
    # step 6:
    # for (t in 1:Nt) { 
    for (t in 1:Nt) {
      # step 7: Kalman Filter
      for (js in 1:nrow(jS)) {
        s1 <- jS$s1[js]
        s2 <- jS$s2[js]
        
        # rows that does not have NA values 
        noNaRows[[t]] <- which(rowSums(is.na(y[,t,])) == 0)
        # rows that have NA values
        naRows[[t]] <- which(rowSums(is.na(y[,t,])) > 0)
        
        jEta[,t,s1,s2,] <- a[[s1]] + torch_matmul(mEta[,t,s2,], B[[s1]]) # Eq.2
        for (noNaRow in noNaRows[[t]]) {jDelta[noNaRow,t,s1,s2,] <- eta[noNaRow,t,] - jEta[noNaRow,t,s1,s2,]} # Eq.3
        
        jP[,t,s1,s2,,] <- torch_matmul(torch_matmul(B[[s1]], mP[,t,s2,,]), B[[s1]]) + Q[[s1]] # Eq.4
        jP[,t,s1,s2,,] <- (jP[,t,s1,s2,,] + torch_transpose(jP[,t,s1,s2,,], 2, 3)) / 2 # ensure symmetry
        jP[,t,s1,s2,,] <- jP[,t,s1,s2,,] + epsilon * torch_eye(Nf) # add a small constant to ensure p.s.d.
        for (i in 1:N){if (as.numeric(torch_det(jP[i,t,s1,s2,,])) < 0) print(paste0('(i,t,s1,s2): (', i, ',', t, ',', s1, ',', s2, '): ', 'det(jP[i,t,s1,s2,,]) = ', as.numeric(torch_det(jP[i,t,s1,s2,,]))))} 
        jPChol[,t,s1,s2,,] <- torch_cholesky(jP[,t,s1,s2,,], upper=FALSE) # Cholesky decomposition
        
        for (noNaRow in noNaRows[[t]]) {
          jV[noNaRow,t,s1,s2,] <- y[noNaRow,t,] - (k[[s1]] + torch_matmul(jEta[noNaRow,t,s1,s2,], Lmd[[s1]]))} # Eq.5
        jF[,t,s1,s2,,] <- torch_matmul(torch_matmul(torch_transpose(Lmd[[s1]], 1, 2), jP[,t,s1,s2,,]), Lmd[[s1]]) + R[[s1]] # Eq.6
        jF[,t,s1,s2,,] <- (jF[,t,s1,s2,,] + torch_transpose(jF[,t,s1,s2,,], 2, 3)) / 2 # ensure symmetry
        jF[,t,s1,s2,,] <- jF[,t,s1,s2,,] + epsilon * torch_eye(No) # add a small constant to ensure p.s.d.
        for (i in 1:N){if (as.numeric(torch_det(jF[i,t,s1,s2,,])) < 0) print(paste0('(i,t,s1,s2): (', i, ',', t, ',', s1, ',', s2, '): ', 'det(jF[i,t,s1,s2,,]) = ', as.numeric(torch_det(jF[i,t,s1,s2,,]))))} 
        jFChol[,t,s1,s2,,] <- torch_cholesky(jF[,t,s1,s2,,], upper=FALSE)
        
        for (noNaRow in noNaRows[[t]]) {
          # kalman gain function
          KG <- torch_matmul(torch_matmul(jP[noNaRow,t,s1,s2,,], Lmd[[s1]]), torch_cholesky_inverse(jFChol[noNaRow,t,s1,s2,,], upper=FALSE))
          jEta2[noNaRow,t,s1,s2,] <- jEta[noNaRow,t,s1,s2,] + torch_matmul(KG, jV[noNaRow,t,s1,s2,]) # Eq.7
          KGLmd <- torch_matmul(KG, torch_transpose(Lmd[[s1]], 1, 2))
          # for numerical stability: ensure the diagonal element of KGLmd to be between -1 and 1
          for (f in 1:Nf) {KGLmd[f,f] <- KGLmd[f,f] - epsilon} # KGLmd needs to be adjusted to avoid numerical error
          KGLmdP <- torch_matmul(KGLmd, jP[noNaRow,t,s1,s2,,]) 
          KGLmdP <- (KGLmdP + torch_transpose(KGLmdP, 1, 2)) / 2 # ensure symmetry
          jP2[noNaRow,t,s1,s2,,] <- jP[noNaRow,t,s1,s2,,] - KGLmdP } # Eq.8 
          jP2[noNaRow,t,s1,s2,,] <- jP2[noNaRow,t,s1,s2,,] + epsilon * torch_eye(Nf) # add a small constant to ensure p.s.d.
        
        for (naRow in naRows[[t]]) {
          jEta2[naRow,t,s1,s2,] <- jEta[naRow,t,s1,s2,] # Eq.7 (for missing entries)
          jP2[naRow,t,s1,s2,,] <- jP[naRow,t,s1,s2,,] } # Eq.8 (for missing entries)
          for (i in 1:N){if (as.numeric(torch_det(jP2[i,t,s1,s2,,])) < 0) print(paste0('(i,t,s1,s2): (', i, ',', t, ',', s1, ',', s2, '): ', 'det(jP2[i,t,s1,s2,,]) = ', as.numeric(torch_det(KGLmd[i,t,s1,s2,,]))))}
          for (i in 1:N){if (as.numeric(torch_det(jP2[i,t,s1,s2,,])) < 0) print(paste0('(i,t,s1,s2): (', i, ',', t, ',', s1, ',', s2, '): ', 'det(jP2[i,t,s1,s2,,]) = ', as.numeric(torch_det(jP[i,t,s1,s2,,]))))}
          for (i in 1:N){if (as.numeric(torch_det(jP2[i,t,s1,s2,,])) < 0) print(paste0('(i,t,s1,s2): (', i, ',', t, ',', s1, ',', s2, '): ', 'det(jP2[i,t,s1,s2,,]) = ', as.numeric(torch_det(jP2[i,t,s1,s2,,]))))} 
        
        # step 8: joint likelihood function f(eta_{t}|s,s',eta_{t-1})
        # is likelihood function different because I am dealing with latent variables instead of observed variables?
        for (noNaRow in noNaRows[[t]]) {
          jLik[noNaRow,t,s1,s2] <- 
            torch_exp(-.5 * torch_matmul(torch_matmul(jDelta[noNaRow,t,s1,s2,], torch_cholesky_inverse(jPChol[noNaRow,t,s1,s2,,], upper=FALSE)), jDelta[noNaRow,t,s1,s2,])) 
          jLik[noNaRow,t,s1,s2] <- min(jLik[noNaRow,t,s1,s2], epsilon**(-1)) * (-.5*pi)**(-Nf/2) * torch_prod(torch_diag(jPChol[noNaRow,t,s1,s2,,]))**(-1)} } # Eq.11
      
      # step 9: transition probability P(s|s',eta_{t-1})  
      if (t == 1) {
        tPr[,t,1] <- torch_sigmoid(alpha[[1]])
        tPr[,t,2] <- torch_sigmoid(alpha[[2]]) 
        jPr[,t,2,2] <- tPr[,t,2] * mPr[,t]
        jPr[,t,2,1] <- tPr[,t,1] * (1-mPr[,t])
        jPr[,t,1,2] <- (1-tPr[,t,2]) * mPr[,t]
        jPr[,t,1,1] <- (1-tPr[,t,1]) * (1-mPr[,t]) }
      else {
        for (noNaRow in noNaRows[[t-1]]) {
          tPr[noNaRow,t,1] <- torch_sigmoid(alpha[[1]] + torch_matmul(eta[noNaRow,t-1,], beta[[1]]))
          tPr[noNaRow,t,2] <- torch_sigmoid(alpha[[2]] + torch_matmul(eta[noNaRow,t-1,], beta[[2]])) 
          
          # step 10: Hamilton Filter
          # joint probability P(s,s'|eta_{t-1})
          jPr[noNaRow,t,2,2] <- tPr[noNaRow,t,2] * mPr[noNaRow,t]
          jPr[noNaRow,t,2,1] <- tPr[noNaRow,t,1] * (1-mPr[noNaRow,t])
          jPr[noNaRow,t,1,2] <- (1-tPr[noNaRow,t,2]) * mPr[noNaRow,t]
          jPr[noNaRow,t,1,1] <- (1-tPr[noNaRow,t,1]) * (1-mPr[noNaRow,t]) }
        for (naRow in naRows[[t-1]]) {jPr[naRow,t,,] <- jPr[naRow,t-1,,]} }
      
      # marginal likelihood function f(eta_{t}|eta_{t-1})
      if (length(noNaRows[[t]]) > 0) {
        for (noNaRow in noNaRows[[t]]) {
          mLik[noNaRow,t] <- torch_sum(jLik[noNaRow,t,,] * jPr[noNaRow,t,,])
          # (updated) joint probability P(s,s'|eta_{t})
          jPr2[noNaRow,t,,] <- jLik[noNaRow,t,,] * jPr[noNaRow,t,,] / max(mLik[noNaRow,t], epsilon)
          if (as.numeric(torch_sum(jPr2[noNaRow,t,,])) == 0) {jPr2[noNaRow,t,,] <- jPr[noNaRow,t,,]} } }
      for (naRow in naRows[[t]]) {jPr2[naRow,t,,] <- jPr[naRow,t,,]}
      mPr[,t+1] <- torch_sum(jPr2[,t,2,], dim=2)
      
      # step 11: collapsing procedure
      for (s2 in 1:2) { 
        denom <- 1 - mPr[,t+1]; denom[denom==0] <- denom[denom==0] + epsilon
        W[,t,1,s2] <- jPr2[,t,1,s2] / denom
        denom <- mPr[,t+1]; denom[denom==0] <- denom[denom==0] + epsilon
        W[,t,2,s2] <- jPr2[,t,2,s2] / denom }
      
      for (f in 1:Nf) {mEta[,t+1,,f] <- torch_sum(W[,t,,] * jEta2[,t,,,f], dim=3)}
      
      mEta_jEta2 <- torch_full_like(jEta2[,t,,,], NaN)
      for (s2 in 1:2) {mEta_jEta2[,,s2,] <- mEta[,t+1,,] - jEta2[,t,,s2,]}
      mEta_jEta2_expanded1 <- torch_unsqueeze(mEta_jEta2, dim=-1)
      mEta_jEta2_expanded2 <- torch_unsqueeze(mEta_jEta2, dim=-2)
      mEta_jEta2_square <- torch_matmul(mEta_jEta2_expanded1, mEta_jEta2_expanded2)
      mEta_jEta2_square <- (mEta_jEta2_square + torch_transpose(mEta_jEta2_square, 4, 5)) / 2 # ensure symmetry
      mEta_jEta2_square <- mEta_jEta2_square + epsilon * torch_eye(Nf) # add a small constant to ensure p.s.d.
      for (i in 1:N) {if (as.numeric(torch_det(mEta_jEta2_square[i,s1,s2])) < 0) print(paste0('(i,t,s1,s2): (', i, ',', t, ',', s1, ',', s2, '): ', 'det(mEta_jEta2_square[i,t,s1,s2]): ', as.numeric(torch_det(mEta_jEta2_square[i,s1,s2]))))}
      
      jNf <- expand.grid(f1=1:Nf, f2=1:Nf)
      for (jnf in 1:nrow(jNf)) {
        f1 <- jNf$f1[jnf]
        f2 <- jNf$f2[jnf] 
        mP[,t+1,,f1,f2] <- torch_sum(W[,t,,] * (jP2[,t,,,,] + mEta_jEta2_square)[,,,f1,f2], dim=3) }
      mP[,t+1,,,] <- (mP[,t+1,,,] + torch_transpose(mP[,t+1,,,], 3, 4)) / 2 # ensure symmetry
      mP[,t+1,,,] <- mP[,t+1,,,] + epsilon * torch_eye(Nf) # add a small constant to ensure p.s.d.
      for (i in 1:N) {for (s in 1:2) {if (as.numeric(torch_det(mP[i,t,s1,,])) < 0) print(paste0('(i,t,s1): (', i, ',', t, ',', s1, '): ', 'det(mP[i,t,s1,,]) = ', as.numeric(torch_det(mP[i,t,s1,,]))))} } 
    } } # continue to numerical optimization
  }


