# for reproducibility
set.seed(42)

# number of subjects
N <- 7
# number of time responses
Nt <- 5

state <- array(NA, c(N,Nt))

for (i in 1:N){
  # start from state = 1
  state[i,1] <- 1
  for (t in 2:Nt){
    # no switch back from state = 2 to state = 1
    if (state[i,t-1] == 2) state[i,t] <- state[i,t-1]
    # P(S_{i,t} = 2 | S_{i,t-1} = 1) = 0.01
    else state[i,t] <- rbinom(n=1, size=1, prob=0.01) + 1
  }
}

# install.packages("torch")
library(torch)

# for reproducibility
set.seed(42)

###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

###################################
# define variables
###################################
# E[eta_{i,t|t-1}^{s,s'}]
jEta <- array(NA, c(N, Nt, 2, 2))
# Cov[eta_{i,t|t-1}^{s,s'}]
jP <- array(NA, c(N, Nt, 2, 2))
# E[eta_{i,t-1|t-1}^{s'}]
mEta <- array(NA, c(N, Nt+1, 2))
# Cov[eta_{i,t-1|t-1}^{s'}]
mP <- array(NA, c(N, Nt+1, 2))
# v_{i,t}^{s,s'} 
jV <- array(NA, c(N, Nt, 2, 2))
# F_{i,t}^{s,s'}
jF <- array(NA, c(N, Nt, 2, 2))
# E[eta_{i,t|t}^{s,s'}]
jEta2 <- array(NA, c(N, Nt, 2, 2))
# Cov[eta_{i,t|t}^{s,s'}]
jP2 <- array(NA, c(N, Nt, 2, 2))
# P(s=2 | s', y_{i,t-1})
tPr <- array(NA, c(N, Nt, 2))
# P(s, s' | y_{i,t-1})
jPr <- array(NA, c(N, Nt, 2, 2))
# f(y_{i,t} | s, s', y_{i,t-1})
jLik <- array(NA, c(N, Nt, 2, 2))
# P(s, s' | y_{i,t})
jPr2 <- array(NA, c(N, Nt, 2, 2))
# W_{i,t}^{s,s'}
W <- array(NA, c(N, Nt, 2, 2))

###################################
# Algorithm 1
###################################

# step 1: input {y_{it}}
# step 2: initialize set of parameters

# define parameters
a <- c(-1,1)
b <- c(1,2)
k <- c(-1,1)
Lmd <- c(1,2)
alpha <- 1
beta <- 1

# step 3: initialize latent variables
# latent variable score at initial time point is assumed to follow N(0, 1e3) 
for (s in 1:2) {
  mEta[,1,s] <- rep(x=0, times=N)
  mP[,1,s] <- rep(x=1e3, times=N)
}

# step 4: initialize residual variances
Qs <- c(1,10)
Rs <- c(1,10)

# vectorize parameters
theta <- c(a, b, k, Lmd, alpha, beta, Qs, Rs)

# f(y_{i,t} | y_{i,t-1})
mLik <- array(0, c(N, Nt))

# P(s=2 | y_{i,t})
mPr <- array(0, c(N, Nt+1))

# step 5: initialize marginal probability
mPr[,1] <- 1e-8 # no drop out at t=0

# step 6
for (t in 1:Nt) {
  
  # step 7 
  tPr[,t,1] <- alpha + beta * yt[,t]
  tPr[,t,2] <- 1
  
  # step 8 
  jPr[,t,1,1] <- (1-tPr[,t,1]) * (1-mPr[,t])
  jPr[,t,2,1] <- tPr[,t,1] * (1-mPr[,t])
  jPr[,t,1,2] <- (1-tPr[,t,2]) * mPr[,t]
  jPr[,t,2,2] <- tPr[,t,2] * mPr[,t]
  
  # step 9 
  for (s1 in 1:2) {
    for (s2 in 1:2) {
      
      jEta[,t,s1,s2] <- a[s1] + b[s1] * mEta[,t,s2]
      jP[,t,s1,s2] <- b[s1]**2 * mP[,t,s2] + Qs[s1]
      
      jV[,t,s1,s2] <- yt[,t] - (k[s1] + Lmd[s1] * jEta[,t,s1,s2])
      jF[,t,s1,s2] <- Lmd[s1]**2 * jP[,t,s1,s2] + Rs[s1]
      
      Ks <- jP[,t,s1,s2] * Lmd[s1] / jF[,t,s1,s2]
      
      jEta2[,t,s1,s2] <- jEta[,t,s1,s2] + Ks * jV[,t,s1,s2] 
      jP2[,t,s1,s2] <- jP[,t,s1,s2] - Ks * Lmd[s1] * jP[,t,s1,s2]
      
      # step 10 
      jLik[,t,s1,s2] <- (2*pi)**(-1/2) * (jF[,t,s1,s2])**(-1/2) *
        exp(-1/2 * jV[,t,s1,s2]**2 / jF[,t,s1,s2])
      
      # step 11 
      mLik[,t] <- mLik[,t] + jLik[,t,s1,s2] * jPr[,t,s1,s2]
    }
  }
  
  for (s1 in 1:2) {
    for (s2 in 1:2) {
      # step 11 
      jPr2[,t,s1,s2] <- jLik[,t,s1,s2] * jPr[,t,s1,s2] / mLik[,t] 
      if (s1 == 2) {
        mPr[,t+1] <- mPr[,t+1] + jPr2[,t,s1,s2] }
    }
  }   
  
  for (s1 in 1:2) {
    for (s2 in 1:2) {
      # step 12
      if (s1 == 1) {
        if (mPr[,t+1] == 1) { W[,t,1,s2] <- (jPr2[,t,1,s2] + 1e-8) / 1e-8 }
        else { W[,t,1,s2] <- jPr2[,t,1,s2] / (1-mPr[,t+1]) }
      }
      else if (s1 == 2) {
        if (mPr[,t+1] == 0) { W[,t,2,s2] <- (jPr2[,t,2,s2] + 1e-8) / 1e-8 }
        else { W[,t,2,s2] <- jPr2[,t,2,s2] / mPr[,t+1] }
      }
    }
    
    # step 12 (continuation)
    mEta[,t+1,s1] <- sum( W[,t,s1,] * jEta2[,t,s1,], dim=2)
    mP[,t+1,s1] <- sum(W[,t,s1,] * (jP2[,t,s1,] + (cbind(mEta[,t+1,s1], mEta[,t+1,s1]) - jEta2[,t,s1,])**2), dim=2)
  }
  
  
} # this line relates to the beginnnig of step 6
(df <- list(state=state, yt=yt))