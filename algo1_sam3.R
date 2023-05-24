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
epsilon <- 1e-30

###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

dropout <- y3D[,,dim(y3D)[3]]
y <- y3D[,,1:(dim(y3D)[3]-1)]
No <- dim(y)[3] 
eta <- eta3D
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
  theta <- torch_cat(list(a1,a2,B1d,B2d,k1,k2,Lmd1v,Lmd2v,alpha1,alpha2,beta1,beta2,Q1d,Q2d,R1d,R2d))
  
  # define variables
  jEta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.2 (LHS)
  jDelta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.3 (LHS)
  jP <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Eq.4 (LHS)
  jV <- torch_full(c(N,Nt,2,2,No), NaN) # Eq.5 (LHS)
  jF <- torch_full(c(N,Nt,2,2,No,No), NaN) # Eq.6 (LHS)
  jEta2 <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.7 (LHS)
  jP2 <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Eq.8 (LHS)
  mEta <- torch_full(c(N,Nt+1,2,Nf), NaN) # Eq.9-1 (LHS)
  mP <- torch_full(c(N,Nt+1,2,Nf,Nf), NaN) # Eq.9-2 (LHS)
  W <- torch_full(c(N,Nt,2,2), NaN) # Eq.9-3 (LHS)
  jPr <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-1 (LHS)
  mLik <- torch_full(c(N,Nt,No), NaN) # Eq.10-2 (LHS)
  jPr2 <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-3 (LHS)
  mPr <- torch_full(c(N,Nt+1), NaN) # Eq.10-4 (LHS)
  jLik <- torch_full(c(N,Nt,2,2,No), NaN) # Eq.11 (LHS)
  tPr <- torch_full(c(N,Nt,2), NaN) # Eq.12 (LHS)
  
  # step 4: initialize latent variables
  for (s in 1:2) {
     for (i in 1:N) {
       mEta[i,1,s,] <- rep(x=0, times=Nf)
       mP[i,1,s,,] <- diag(x=1e30, nrow=Nf, ncol=Nf) } }
  
  # while (count < 3) {
  for (iter in 1:nIter) {
    print(paste0('   optimization step: ', as.numeric(iter)))
    
    # step 5: initialize P(s'|eta_0)
    mPr[,1] <- epsilon 
    
    # joint regimes
    jS <- expand.grid(s1=c(1,2), s2=c(1,2))
    # step 6:
    # for (t in 1:Nt) {
    for (t in 1:1) {
      # step 7: Kalman Filter
      for (js in 1:nrow(jS)) {
        s1 <- jS$s1[js]
        s2 <- jS$s2[js]
        
        # rows that have non-NA values 
        noNaRows <- which(rowSums(is.na(y[,t,])) == 0)
        # rows that have NA values
        naRows <- which(rowSums(is.na(y[,t,])) > 0)
        
        jEta[,t,s1,s2,] <- a[[s1]] + torch_matmul(torch_clone(mEta[,t,s2,]), B[[s1]]) # Eq.2
        for (noNaRow in noNaRows) {jDelta[noNaRow,t,s1,s2,] <- eta[noNaRow,t,] - torch_clone(jEta[noNaRow,t,s1,s2,])} # Eq.3
        jP[,t,s1,s2,,] <- torch_matmul(torch_matmul(B[[s1]], torch_clone(mP[,t,s2,,])), B[[s1]]) + Q[[s1]] # Eq.4
        for (noNaRow in noNaRows) {jV[noNaRow,t,s1,s2,] <- y[noNaRow,t,] - (k[[s1]] + torch_matmul(torch_clone(jEta[noNaRow,t,s1,s2,]), Lmd[[s1]]))} # Eq.5
        jF[,t,s1,s2,,] <- torch_matmul(torch_matmul(torch_transpose(Lmd[[s1]], dim0=2, dim1=1), torch_clone(jP[,t,s1,s2,,])), Lmd[[s1]]) + R[[s1]] # Eq.6
        for (noNaRow in noNaRows) {
          Ks <- torch_matmul(torch_matmul(torch_clone(jP[noNaRow,t,s1,s2,,]), Lmd[[s1]]), torch_pinverse(torch_clone(jF[noNaRow,t,s1,s2,,])))
          jEta2[noNaRow,t,s1,s2,] <- torch_clone(jEta[noNaRow,t,s1,s2,]) + torch_matmul(torch_clone(Ks), torch_clone(jV[noNaRow,t,s1,s2,])) # Eq.7
          jP2[noNaRow,t,s1,s2,,] <- 
            torch_clone(jP[noNaRow,t,s1,s2,,]) - torch_matmul(torch_matmul(torch_clone(Ks), torch_transpose(Lmd[[s1]], dim0=2, dim1=1)), torch_clone(jP[noNaRow,t,s1,s2,,])) } # Eq.8 
        for (naRow in naRows) {
          jEta2[naRow,t,s1,s2,] <- torch_clone(jEta[naRow,t,s1,s2,]) # Eq.7 (for missing entries)
          jP2[naRow,t,s1,s2,,] <- torch_clone(jP[naRow,t,s1,s2,,]) } # Eq.8 (for missing entries)

        # step 8: joint likelihood function f(eta_{t} | s,s', eta_{t-1})
        
        
        # Hamilton Filter: if likelihood ratio f(eta_{t} | s,s', eta_{t-1}) / f(eta_{t} | eta_{t-1}) = 0 / 0,
        # let P(s,s'|eta_t) = P(s,s'|eta_{t-1})
        
        # 1. easily impute data 
        # 2. find how to handle missing data 
      } } }

}


