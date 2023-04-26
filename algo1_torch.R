library(torch)
str(df)

# for reproducibility
set.seed(42)

###################################
# s=1: non-drop out state
# s=2: drop out state
###################################

# E[eta_{i,t|t-1}^{s,s'}]
jEta = torch_full(c(N, Nt, 2, 2), NaN)
# Cov[eta_{i,t|t-1}^{s,s'}]
jP = torch_full(c(N, Nt, 2, 2), NaN)

# E[eta_{i,t-1|t-1}^{s'}]
mEta = torch_full(c(N, Nt+1, 2), NaN)
# Cov[eta_{i,t-1|t-1}^{s'}]
mP = torch_full(c(N, Nt+1, 2), NaN)

# v_{i,t}^{s,s'} 
jV = torch_full(c(N, Nt, 2, 2), NaN)
# F_{i,t}^{s,s'}
jF = torch_full(c(N, Nt, 2, 2), NaN)

# E[eta_{i,t|t}^{s,s'}]
jEta2 = torch_full(c(N, Nt, 2, 2), NaN)
# Cov[eta_{i,t|t}^{s,s'}]
jP2 = torch_full(c(N, Nt, 2, 2), NaN)

# marginal probability
# P(s=2 | y_{i,t})
mPr = torch_full(c(N, Nt), 1e-2)

# transition probability
# P(s=2 | s', y_{i,t-1})
tPr = torch_full(c(N, Nt, 2), NaN)

# joint probability
# P(s, s' | y_{i,t-1})
jPr = torch_full(c(N, Nt, 2, 2), NaN)

# f(y_{i,t} | s, s', y_{i,t-1})
jLik = torch_full(c(N, Nt, 2, 2), NaN)
# f(y_{i,t} | y_{i,t-1})
mLik = torch_zeros(N, Nt)

# P(s, s' | y_{i,t})
jPr2 = torch_full(c(N, Nt, 2, 2), NaN)

W = torch_full(c(N, Nt, 2, 2), NaN)

###################################

# step 1: input {y_{it}}
# step 2: initialize set of parameters
a <- torch_randn(2)
b <- torch_abs(torch_randn(2))
k <- torch_randn(2)
Lmd <- torch_randn(2)
alpha <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))
beta <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))
gamma <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))
delta <- torch_normal(mean=0, std=1e-1, size=2)

# step 3: initialize latent variables
# latent variable score at initial time point is assumed to follow N(0, 1e3) 

for (s in 1:2){
  mEta[,1,s] <- rep(x=0, times=N)
  mP[,1,s] <- rep(x=1e3, times=N)
}

# step 4: initialize residual variances
Qs <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))
Rs <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))

# step 5: initialize marginal probability
# mPr[, 0] <- 0 # no drop out at t=0

a <- torch_tensor(a, requires_grad=TRUE)
b <- torch_tensor(b, requires_grad=TRUE)
k <- torch_tensor(k, requires_grad=TRUE)
Lmd <- torch_tensor(Lmd, requires_grad=TRUE)
alpha <- torch_tensor(alpha, requires_grad=TRUE)
beta <- torch_tensor(beta, requires_grad=TRUE)
gamma <- torch_tensor(gamma, requires_grad=TRUE)
delta <- torch_tensor(delta, requires_grad=TRUE)

# step 6
for (i in 1:N){
  print(i)
  for (t in 1:Nt){
    
    # step 7 
    tPr[i,t,1] <- torch_sigmoid(alpha[1] + beta[1] * 0 + gamma[1] * yt[i,t] + delta[1] * 0 * yt[i,t])
    tPr[i,t,2] <- torch_sigmoid(alpha[2] + beta[2] * 1 + gamma[2] * yt[i,t] + delta[2] * 1 * yt[i,t])
    
    # step 8 
    jPr[i,t,1,1] <- (1-torch_clone(tPr[i,t,1])) * (1-torch_clone(mPr[i,t]))
    jPr[i,t,2,1] <- torch_clone(tPr[i,t,1]) * (1-torch_clone(mPr[i,t]))
    jPr[i,t,1,2] <- (1-torch_clone(tPr[i,t,2])) * torch_clone(mPr[i,t])
    jPr[i,t,2,2] <- torch_clone(tPr[i,t,2]) * torch_clone(mPr[i,t])
    
    # step 9 
    for (s1 in 1:2){
      for (s2 in 1:2){
        
        jEta[i,t,s1,s2] <- a[s1] + b[s1] * torch_clone(mEta[i,t,s2])
        jP[i,t,s1,s2] <- b[s1]**2 * torch_clone(mP[i,t,s2]) + Qs[s1]
        
        jV[i,t,s1,s2] <- yt[i,t] - (k[s1] + Lmd[s1] * torch_clone(jEta[i,t,s1,s2]))
        jF[i,t,s1,s2] <- Lmd[s1]**2 * torch_clone(jP[i,t,s1,s2]) + Rs[s1]
        
        Ks <- torch_clone(jP[i,t,s1,s2]) * Lmd[s1] / torch_clone(jF[i,t,s1,s2])
        
        jEta2[i,t,s1,s2] <- torch_clone(jEta[i,t,s1,s2]) + torch_clone(Ks) * torch_clone(jV[i,t,s1,s2]) 
        jP2[i,t,s1,s2] <- torch_clone(jP[i,t,s1,s2]) - torch_clone(Ks) * Lmd[s1] * torch_clone(jP[i,t,s1,s2])
        
        # step 10 
        jLik[i,t,s1,s2] <- (2*pi)**(-1/2) * (torch_clone(jF[i,t,s1,s2]))**(-1/2) *
          torch_exp(-1/2 * torch_clone(jV[i,t,s1,s2])**2 / torch_clone(jF[i,t,s1,s2]))
        
        # step 11 
        if (torch_allclose(torch_isnan(jLik[i,t,s1,s2]), FALSE)){
          mLik[i,t] <- torch_clone(mLik[i,t]) + torch_clone(jLik[i,t,s1,s2]) * torch_clone(jPr[i,t,s1,s2]) }
      }
    }
    
    for (s1 in 1:2){
      for (s2 in 1:2){
        
        if (torch_allclose(torch_isnan(jLik[i,t,s1,s2]), FALSE)){
          jPr2[i,t,s1,s2] <- torch_clone(jLik[i,t,s1,s2]) * torch_clone(jPr[i,t,s1,s2]) / torch_clone(mLik[i,t]) }
        
        if (s1 == 2 & torch_allclose(torch_isnan(jPr2[i,t,s1,s2]), FALSE)){
          mPr[i,t] <- torch_clone(mPr[i,t]) + torch_clone(jPr2[i,t,s1,s2]) }
      }
    }   
    
    for (s1 in 1:2){
      for (s2 in 1:2){
        
        # step 12
        if (s1 == 1 & torch_allclose(torch_isnan(jPr2[i,t,1,s2]), FALSE)){
          if (torch_allclose(mPr[i,t], 1)) { W[i,t,1,s2] <- max(torch_clone(Pr2[i,t,1,s2]), 1e-5) / 1e-5 }
          else { W[i,t,1,s2] <- torch_clone(jPr2[i,t,1,s2]) / (1-torch_clone(mPr[i,t])) }
        }
        
        else if (s1 == 2 & torch_allclose(torch_isnan(jPr2[i,t,2,s2]), FALSE)){
          if (torch_allclose(mPr[i,t], 0)) { W[i,t,2,s2] <- max(torch_clone(jPr2[i,t,2,s2]), 1e-5) / 1e-5 }
          else { W[i,t,2,s2] <- torch_clone(jPr2[i,t,2,s2]) / torch_clone(mPr[i,t]) }
        }
      }
      
      # step 12 (continuation)
      mEta[i,t+1,s1] <- torch_sum( torch_clone(W[i,t,s1,]) * torch_clone(jEta2[i,t,s1,]))
      mP[i,t+1,s1] <- torch_sum( torch_clone(W[i,t,s1,]) * ( torch_clone(jP2[i,t,s1,]) + (torch_clone(mEta[i,t+1,s1]) - torch_clone(jEta2[i,t,s1,]))**2 ))
    }
    
    
    
  }
}

sumLik <- sum(mLik)
sumLik$grad_fn
sumLik$backward()

a$grad
b$grad
k$grad
Lmd$grad
alpha$grad
beta$grad
gamma$grad
delta$grad
