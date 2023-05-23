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

###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

y <- y3D
eta <- eta3D

###################################
# define variables
###################################

jEta <- torch_full(c(N,Nt,2,2,Nf), NaN)
jP <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN)
mEta <- torch_full(c(N,Nt+1,2,Nf), NaN)
mP <- torch_full(c(N,Nt+12,Nf,Nf), NaN)
jV <- torch_full(c(N,Nt,2,2,No), NaN)
jF <- torch_full(c(N,Nt,2,2,No,No), NaN)
jEta2 <- torch_full(c(N,Nt,2,2,Nf), NaN)
jP2 <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN)
tPr <- torch_full(c(N,Nt,2), NaN)
jPr <- torch_full(c(N,Nt,2,2), NaN)
jLik <- torch_full(c(N,Nt,2,2,No), NaN)
mLik <- torch_full(c(N,Nt,No), NaN)
jPr2 <- torch_full(c(N,Nt,2,2), NaN)
W <- torch_full(c(N,Nt,2,2), NaN)

###################################
# Algorithm 1
###################################

for (init in 1:nInit) {
  # optimization step count
  iter <- 1
  # stopping criterion count
  count <- 0 
  # moment estimates 
  m <- v <- NULL
  print(paste0('Initialization step '), init)
  # step 3: initialize parameters
  a1 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE) 
  a2 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE) 
  a <- list(a1, a2)
  B1v <- torch_tensor(torch_randn(Nf**2), requires_grad=TRUE)
  B2v <- torch_tensor(torch_randn(Nf**2), requires_grad=TRUE)
  B1 <- torch_reshape(B1v, shape=c(Nf,Nf))
  B2 <- torch_reshape(B2v, shape=c(Nf,Nf))
  B <- list(B1, B2)
  k1 <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  k2 <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  k <- list(k1, k2)
  Lmd1v <- torch_tensor(torch_randn(No*Nf), requires_grad=TRUE)
  Lmd2v <- torch_tensor(torch_randn(No*Nf), requires_grad=TRUE)
  Lmd1 <- torch_reshape(Lmd1v, shape=c(No,Nf))
  Lmd2 <- torch_reshape(Lmd2v, shape=c(No,Nf))
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
  R1d <- torch_tensor(torch_randn(No)**2, requires_grad=TRUE)
  R1 <- torch_diag(R1d)
  R2 <- torch_diag(R2d)
  R <- list(R1, R2)
  
  # step 4: initialize latent variables
  for (s in 1:2) {
     for (i in 1:N) {
       
     }
  }
}


