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
  B2 <- torch_reshape(B2d)
  B <- list(B1, B2)
  k1 <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  k2 <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  k <- list(k1, k2)
  Lmd1v <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  Lmd2v <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  Lmd1 <- Lmd2 <- torch_full(c(No,Nf), 0)
  Lmd1[1:3,1] <- Lmd1v[1:3]; Lmd1[4:5,2] <- Lmd1v[4:5]
  Lmd1[6:7,3] <- Lmd1v[6:7]; Lmd1[8:9,4] <- Lmd1v[8:9]
  Lmd1[10:11,5] <- Lmd1v[10:11]; Lmd1[12:14,6] <- Lmd1v[12:14]
  Lmd1[15:17,7] <- Lmd1v[15:17]; Lmd1[18,8] <- Lmd2v[18]
  Lmd2[1:3,1] <- Lmd2v[1:3]; Lmd2[4:5,2] <- Lmd2v[4:5]
  Lmd2[6:7,3] <- Lmd2v[6:7]; Lmd2[8:9,4] <- Lmd2v[8:9]
  Lmd2[10:11,5] <- Lmd2v[10:11]; Lmd2[12:14,6] <- Lmd2v[12:14]
  Lmd2[15:17,7] <- Lmd2v[15:17]; Lmd2[18,8] <- Lmd2v[18]
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
  jEta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Equation 2 (LHS)
  jDelta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Equation 3 (LHS)
  jP <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Equation 4 (LHS)
  jV <- torch_full(c(N,Nt,2,2,No), NaN) # Equation 5 (LHS)
  jF <- torch_full(c(N,Nt,2,2,No,No), NaN) # Equation 6 (LHS)
  jEta2 <- torch_full(c(N,Nt,2,2,Nf), NaN) # Equation 7 (LHS)
  jP2 <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Equation 8 (LHS)
  mEta <- torch_full(c(N,Nt+1,2,Nf), NaN) # Equation 9-1 (LHS)
  mP <- torch_full(c(N,Nt+1,2,Nf,Nf), NaN) # Equation 9-2 (LHS)
  W <- torch_full(c(N,Nt,2,2), NaN) # Equation 9-3 (LHS)
  jPr <- torch_full(c(N,Nt,2,2), NaN) # Equation 10-1 (LHS)
  mLik <- torch_full(c(N,Nt,No), NaN) # Equation 10-2 (LHS)
  jPr2 <- torch_full(c(N,Nt,2,2), NaN) # Equation 10-3 (LHS)
  mPr <- torch_full(c(N,Nt+1), NaN) # Equation 10-4 (LHS)
  jLik <- torch_full(c(N,Nt,2,2,No), NaN) # Equation 11 (LHS)
  tPr <- torch_full(c(N,Nt,2), NaN) # Equation 12 (LHS)
  
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
        
        jEta[,t,s1,s2,] <- a[[s1]] + torch_matmul(torch_clone(mEta[,t,s2,]), B[[s1]])
        jDelta[,t,s1,s2,] <- eta
        jP[,t,s1,s2,,] <- torch_matmul(torch_matmul(B[[s1]], torch_clone(mP[,t,s2,,])), torch_transpose(B[[s1]], dim0=1, dim1=2)) + Q[[s1]] 
        print(jP[,t,s1,s2,,])
      } } }

}


