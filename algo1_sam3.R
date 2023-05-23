######################### 
## input 
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
  }


