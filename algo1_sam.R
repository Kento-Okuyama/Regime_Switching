# install.packages("torch")
library(torch)
# install.packages("reticulate")
library(reticulate)

epsilon <- 1e-30
nparams <- 15

# number of parameter initialization
nInit <- 1
# max number of optimization steps
nIter <- 1

###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

Yt <- df$Yt

###################################
# define variables
###################################
# E[eta_{i,t|t-1}^{s,s'}]
jEta <- torch_full(c(N, Nt, 2, 2, Nf), NaN)
# Cov[eta_{i,t|t-1}^{s,s'}]
jP <- torch_full(c(N, Nt, 2, 2, Nf, Nf), NaN)
# E[eta_{i,t-1|t-1}^{s'}]
mEta <- torch_full(c(N, Nt+1, 2, Nf), NaN)
# Cov[eta_{i,t-1|t-1}^{s'}]
mP <- torch_full(c(N, Nt+1, 2, Nf, Nf), NaN)
# v_{i,t}^{s,s'} 
jV <- torch_full(c(N, Nt, 2, 2, No), NaN)
# F_{i,t}^{s,s'}
jF <- torch_full(c(N, Nt, 2, 2, No, No), NaN)
# E[eta_{i,t|t}^{s,s'}]
jEta2 <- torch_full(c(N, Nt, 2, 2, Nf), NaN)
# Cov[eta_{i,t|t}^{s,s'}]
jP2 <- torch_full(c(N, Nt, 2, 2, Nf, Nf), NaN)
# P(s=2 | s', y_{i,t-1})
tPr <- torch_full(c(N, Nt, 2), NaN)
# P(s, s' | y_{i,t-1})
jPr <- torch_full(c(N, Nt, 2, 2), NaN)
# f(y_{i,t} | s, s', y_{i,t-1})
jLik <- torch_full(c(N, Nt, 2, 2, No), NaN)
# f(y_{i,t} | y_{i,t-1})
mLik <- torch_full(c(N, Nt, No), NaN)
# P(s, s' | y_{i,t})
jPr2 <- torch_full(c(N, Nt, 2, 2), NaN)
# W_{i,t}^{s,s'}
W <- torch_full(c(N, Nt, 2, 2), NaN)
# f(Y | theta)
sumLik <- list()
sumLikBest <- epsilon

###################################
# Algorithm 1
###################################

for (init in 1:nInit) {
  
  # initialization of optimization step count
  iter <- 1
  # initialization of stopping criterion
  count <- 0
  
  # first and second moment estimates
  m <- NULL
  v <- NULL
  
  # for reproducibility
  set.seed(init)
  
  print(paste0('Initialization step: ', init))
  
  # step 1: input {y_{it}}
  # step 2: initialize set of parameters
  
  # define parameters
  a1 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE) 
  a2 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE)
  B1v <- torch_tensor(torch_randn(Nf*Nf), requires_grad=TRUE)
  B2v <- torch_tensor(torch_randn(Nf*Nf), requires_grad=TRUE)
  k1 <- torch_tensor(torch_randn(No), requires_grad=TRUE) 
  k2 <- torch_tensor(torch_randn(No), requires_grad=TRUE) 
  Lmd1v <- torch_tensor(torch_randn(No*Nf), requires_grad=TRUE) 
  Lmd2v <- torch_tensor(torch_randn(No*Nf), requires_grad=TRUE) 
  alpha1 <- torch_tensor(torch_randn(1), requires_grad=TRUE) 
  alpha2 <- torch_tensor(torch_randn(1), requires_grad=TRUE) 
  beta <- torch_tensor(torch_randn(No), requires_grad=TRUE) 
  Q1d <- torch_tensor(torch_normal(mean=0, std=1e-1, size=Nf)**2, requires_grad=TRUE)
  Q2d <- torch_tensor(torch_normal(mean=0, std=1e-1, size=Nf)**2, requires_grad=TRUE) 
  R1d <- torch_tensor(torch_normal(mean=0, std=1e-1, size=No)**2, requires_grad=TRUE)
  R2d <- torch_tensor(torch_normal(mean=0, std=1e-1, size=No)**2, requires_grad=TRUE) 
  
  # reshape some vectors into a matrix
  B1 <- torch_reshape(B1v, shape=c(Nf,Nf))
  B2 <- torch_reshape(B2v, shape=c(Nf,Nf))
  Lmd1 <- torch_reshape(Lmd1v, shape=c(No,Nf))
  Lmd2 <- torch_reshape(Lmd2v, shape=c(No,Nf))
  Q1 <- torch_diag(Q1d)
  Q2 <- torch_diag(Q2d)
  R1 <- torch_diag(R1d)
  R2 <- torch_diag(R2d)
  
  # collect some parameters in a list
  a <- list(a1, a2)
  B <- list(B1, B2)
  k <- list(k1, k2)
  Lmd <- list(Lmd1, Lmd2)
  alpha <- list(alpha1, alpha2)
  Q <- list(Q1, Q2)
  R <- list(R1, R2)
  
  # with gradient tracking
  a1 <- torch_tensor(a1, requires_grad=TRUE)
  a2 <- torch_tensor(a2, requires_grad=TRUE)
  b1 <- torch_tensor(b1, requires_grad=TRUE)
  b2 <- torch_tensor(b2, requires_grad=TRUE)
  k1 <- torch_tensor(k1, requires_grad=TRUE)
  k2 <- torch_tensor(k2, requires_grad=TRUE)
  Lmd1 <- torch_tensor(Lmd1, requires_grad=TRUE)
  Lmd2 <- torch_tensor(Lmd2, requires_grad=TRUE)
  alpha1 <- torch_tensor(alpha1, requires_grad=TRUE)
  alpha2 <- torch_tensor(alpha2, requires_grad=TRUE)
  beta <- torch_tensor(beta, requires_grad=TRUE)
  
  # step 3: initialize latent variables
  # latent variable score at initial time point is assumed to follow N(0, 1e3) 
  for (s in 1:2) {
    for (i in 1:N) {
      mEta[i,1,s,] <- rep(x=0, times=Nf)
      mP[i,1,s,,] <- diag(x=1e3, nrow=Nf, ncol=Nf) }
  }
  
  # vectorize parameters
  theta <- torch_cat(list(a1, a2, B1v, B2v, k1, k2, Lmd1v, Lmd2v, alpha1, alpha2, beta, Q1d, Q2d, R1d, R2d))
  
  # while (count < 3) {
  for (iter in 1:nIter) {
    
    # joint one-step ahead forecast
    jyth <- torch_full(c(N, Nt, 2, 2, No), NaN)
    # one-step ahead forecast
    yth <- torch_full(c(N, Nt, No), NaN)
    
    # P(s=2 | y_{i,t})
    mPr <- torch_zeros(N, Nt+1)
    
    # fixed prob of switching back
    tPr[,,2] <- torch_sigmoid(alpha2)
    
    # step 4: initialize marginal probability
    mPr[,1] <- epsilon # no drop out at t=0
    
    print(paste0('   optimization step: ', as.numeric(iter)))
    
    # step 5
    for (t in 1:Nt) {
      
      # step 6
      for (s1 in 1:2) {
        for (s2 in 1:2) {
          
          jEta[,t,s1,s2,] <- a[[s1]] + torch_matmul(torch_clone(mEta[,t,s2,]), B[[s1]])
          print('ok')
          jP[,t,s1,s2,,] <- torch_matmul(torch_matmul(B[[s1]], torch_clone(mP[,t,s2,,])), torch_transpose(B[[s1]])) + Q[[s1]]
          
          jyth[,t,s1,s2,] <- k[[s1]] + torch_matmul(Lmd[[s1]], torch_clone(jEta[,t,s1,s2,]))
          jV[,t,s1,s2,] <- Yt[,t,] - (k[[s1]] + torch_matmul(Lmd[[s1]], torch_clone(jEta[,t,s1,s2,])))
          
          jF[,t,s1,s2,,] <- torch_matmul(torch_matmul(Lmd[[s1]], torch_clone(jP[,t,s1,s2,,])), torch_transpose(Lmd[[s1]])) + R[[s1]]
          
          Ks <- torch_matmul(torch_matmul(torch_clone(jP[,t,s1,s2,,]), Lmd[[s1]]), torch_inverse(torch_clone(jF[,t,s1,s2,,])))
          
          jEta2[,t,s1,s2,] <- torch_clone(jEta[,t,s1,s2,]) + torch_matmul(torch_clone(Ks), torch_clone(jV[,t,s1,s2,])) 
          jP2[,t,s1,s2,,] <- torch_clone(jP[,t,s1,s2,,]) - torch_matmul(torch_matmul(torch_clone(Ks), Lmd[[s1]]), torch_clone(jP[,t,s1,s2,,]))
          

        }
      }
    }
  }
}
  