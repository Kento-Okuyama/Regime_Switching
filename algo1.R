str(df)

# for reproducibility
set.seed(42)

###################################
# s=1: non-drop out state
# s=2: drop out state
###################################

# logistic function 
# to avoid exp overflow 
# x >=0: Logistic(x) = 1 / (1 + exp(-x))
# x < 0: Logistic(x) = exp(x) / (1 + exp(x))
Logistic <- function(x) { exp(min(x,0)) / (1 + exp(-abs(x))) }

# E[eta_{i,t|t-1}^{s,s'}]
jEta <- array(NA, c(N,Nt,2,2))
# Cov[eta_{i,t|t-1}^{s,s'}]
jP <- array(NA, c(N,Nt,2,2))

# E[eta_{i,t-1|t-1}^{s'}]
mEta <- array(NA, c(N,Nt+1,2))
# Cov[eta_{i,t-1|t-1}^{s'}]
mP <- array(NA, c(N,Nt+1,2))


# v_{i,t}^{s,s'} 
jV <- array(NA, c(N,Nt,2,2))
# F_{i,t}^{s,s'}
jF <- array(NA, c(N,Nt,2,2))

# v_{i,t} 
mV <- array(0, c(N,Nt))
# F_{i,t}
mF <- array(0, c(N,Nt))

# E[eta_{i,t|t}^{s,s'}]
jEta2 <- array(NA, c(N,Nt,2,2))
# Cov[eta_{i,t|t}^{s,s'}]
jP2 <- array(NA, c(N,Nt,2,2))

# marginal probability
# P(s=2 | y_{i,t})
mPr <- array(0, c(N,Nt))

# transition probability
# P(s=2 | s', y_{i,t-1})
tPr <- array(NA, c(N,Nt,2))

# joint probability
# P(s, s' | y_{i,t-1})
jPr <- array(NA, c(N,Nt,2,2))

# f(y_{i,t} | s, s', y_{i,t-1})
jLik <- array(NA, c(N,Nt,2,2))
# f(y_{i,t} | y_{i,t-1})
mLik <- array(0, c(N,Nt))

# P(s, s' | y_{i,t})
jPr2 <- array(NA, c(N,Nt,2,2))

W <- array(NA, c(N,Nt,2,2))

###################################

# step 1: input {y_{it}}

# step 2: initialize set of parameters

a <- rnorm(2, mean=0, sd=1e2) 
b <- abs(rnorm(2, mean=0, sd=1e2))
k <- rnorm(2, mean=0, sd=1e2) 
Lmd <- abs(rnorm(2, mean=0, sd=1e2))
alpha <- abs(rnorm(2, mean=0, sd=1e2))
beta <- abs(rnorm(2, mean=0, sd=1e2))
gamma <- abs(rnorm(2, mean=0, sd=1e2))
delta <- abs(rnorm(2, mean=0, sd=1e2))

# step 3: initialize latent variables
# latent variable score at initial time point is assumed to follow N(0, 1e3) 
mEta[,1,1] <- rep(x=0, times=N)
mP[,1,1] <- rep(x=1e3, times=N)

# step 4: initialize residual variances
Qs <- abs(rnorm(2, mean=0, sd=1e2))
Rs <- abs(rnorm(2, mean=0, sd=1e2))

# step 5: initialize marginal probability
# mPr[, 0] <- 0 # no drop out at t=0

# step 6
for (i in 1:N){
  for (t in 1:Nt){
    
    # step 7 
    tPr[i,t,1] <- Logistic(alpha[1] + beta[1] * 0 + gamma[1] * yt[i,t] + delta[1] * 0 * yt[i,t])
    tPr[i,t,2] <- Logistic(alpha[2] + beta[2] * 1 + gamma[2] * yt[i,t] + delta[2] * 1 * yt[i,t])
    
    # step 8 
    jPr[i,t,1,1] <- (1-tPr[i,t,1]) * (1-mPr[i,t])
    jPr[i,t,2,1] <- tPr[i,t,1] * (1-mPr[i,t])
    jPr[i,t,1,2] <- (1-tPr[i,t,2]) * mPr[i,t]
    jPr[i,t,2,2] <- tPr[i,t,2] * mPr[i,t]
    
    # step 9 
    for (s1 in 1:2){
      for (s2 in 1:2){
        
        jEta[i,t,s1,s2] <- a[s1] + b[s1] * mEta[i,t,s2]
        jP[i,t,s1,s2] <- b[s1]**2 * mP[i,t,s2] + Qs[s1]
        
        jV[i,t,s1,s2] <- yt[i,t] - (k[s1] + Lmd[s1] * jEta[i,t,s1,s2])
        jF[i,t,s1,s2] <- Lmd[s1]**2 * jP[i,t,s1,s2] + Rs[s1]
        
        Ks <- jP[i,t,s1,s2] * Lmd[s1] / jF[i,t,s1,s2]
        
        jEta2[i,t,s1,s2] <- jEta[i,t,s1,s2] + Ks * jV[i,t,s1,s2] 
        jP2[i,t,s1,s2] <- jP[i,t,s1,s2] - Ks * Lmd[s1] * jP[i,t,s1,s2]
        
        # step 10 
        jLik[i,t,s1,s2] <- (2*pi)**(-1/2) * (jF[i,t,s1,s2])**(-1/2) * exp(-1/2 * jV[i,t,s1,s2]**2 / jF[i,t,s1,s2])
        
        # step 11 
        if (is.na(jLik[i,t,s1,s2]) == FALSE){
          mLik[i,t] <- mLik[i,t] + jLik[i,t,s1,s2] * jPr[i,t,s1,s2] }
      }
    }
       
    for (s1 in 1:2){
      for (s2 in 1:2){
        
        if (is.na(jLik[i,t,s1,s2]) == FALSE){
          jPr2[i,t,s1,s2] <- jLik[i,t,s1,s2] * jPr[i,t,s1,s2] / mLik[i,t] }
        
        if (s1 == 2 & is.na(jPr2[i,t,s1,s2]) == FALSE){
          mPr[i,t] <- mPr[i,t] + jPr2[i,t,s1,s2] }
      }
    }
    
    for (s1 in 1:2){
      for (s2 in 1:2){
        
        # step 12
        if (s1 == 1 & is.na(jPr2[i,t,1,s2]) == FALSE){
          if (mPr[i,t] == 1) { W[i,t,1,s2] <- jPr2[i,t,1,s2] / 1e-5 }
          else { W[i,t,1,s2] <- jPr2[i,t,1,s2] / (1-mPr[i,t]) }
        }
        
        else if (s1 == 2 & is.na(jPr2[i,t,2,s2]) == FALSE){
          if (mPr[i,t] == 0) { W[i,t,2,s2] <- jPr2[i,t,2,s2] / 1e-5 }
          else { W[i,t,2,s2] <- jPr2[i,t,2,s2] / mPr[i,t] }
        }
      }
      
      # step 12 (continuation)
      mEta[i,t+1,s1] <- sum( W[i,t,s1,] * jEta2[i,t,s1,], na.rm=TRUE )
      mP[i,t+1,s1] <- sum( W[i,t,s1,] * ( jP2[i,t,s1,] + (mEta[i,t+1,s1] - jEta2[i,t,s1,])**2 ), na.rm=TRUE )
    }
  }
}
print(W[1,1,1,])
print(jEta2[1,1,1,])
jPr2[1,1,1,1]
mPr[1,1]
