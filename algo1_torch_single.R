# install.packages("torch")
library(torch)
# install.packages("reticulate")
library(reticulate)

###################################
# function for Adam optimization 
###################################
# input: initial parameter values and gradients 
# output: updated parameter values
adam <- function(theta, grad, iter, m, v, lr=1e-2, beta1=0.9, beta2=0.999, epsilon=1e-8) {
  # theta: parameters values
  # grad: gradient of the objective function with respect to the parameters at the current iteration
  # lr: learning rate
  # beta1, beta2: hyper-parameters controlling the exponential decay rates for the moment estimates
  # epsilon: small constant added to the denominator to avoid division by zero
  # iter: current iteration number
  # m, v: first and second moment estimates
  
  # initialize moment estimates
  if (is.null(m) || is.null(v)) {
    m <- rep(0, length(theta))
    v <- rep(0, length(theta))
  }
  
  # update moment estimates
  m <- beta1 * m + (1 - beta1) * grad
  v <- beta2 * v + (1 - beta2) * grad**2
  
  # update bias corrected moment estimates
  m_hat <- m / (1 - beta1**iter)
  v_hat <- v / (1 - beta2**iter)
  
  # Update parameters using Adam update rule
  theta <- theta + lr * m_hat / (sqrt(v_hat) + epsilon)
  
  return(list(theta=theta, m=m, v=v))
}

# number of parameter initialization
nInit <- 5
# max number of optimization steps
nIter <- 50
# initialization for Adam optimization
m <- NULL
v <- NULL

###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

###################################
# define variables
###################################
# E[eta_{i,t|t-1}^{s,s'}]
jEta <- torch_full(c(N, Nt, 2, 2), NaN)
# Cov[eta_{i,t|t-1}^{s,s'}]
jP <- torch_full(c(N, Nt, 2, 2), NaN)
# E[eta_{i,t-1|t-1}^{s'}]
mEta <- torch_full(c(N, Nt+1, 2), NaN)
# Cov[eta_{i,t-1|t-1}^{s'}]
mP <- torch_full(c(N, Nt+1, 2), NaN)
# v_{i,t}^{s,s'} 
jV <- torch_full(c(N, Nt, 2, 2), NaN)
# F_{i,t}^{s,s'}
jF <- torch_full(c(N, Nt, 2, 2), NaN)
# E[eta_{i,t|t}^{s,s'}]
jEta2 <- torch_full(c(N, Nt, 2, 2), NaN)
# Cov[eta_{i,t|t}^{s,s'}]
jP2 <- torch_full(c(N, Nt, 2, 2), NaN)
# P(s=2 | s', y_{i,t-1})
tPr <- torch_full(c(N, Nt, 2), NaN)
# P(s, s' | y_{i,t-1})
jPr <- torch_full(c(N, Nt, 2, 2), NaN)
# f(y_{i,t} | s, s', y_{i,t-1})
jLik <- torch_full(c(N, Nt, 2, 2), NaN)
# P(s, s' | y_{i,t})
jPr2 <- torch_full(c(N, Nt, 2, 2), NaN)
# W_{i,t}^{s,s'}
W <- torch_full(c(N, Nt, 2, 2), NaN)
# f(Y | theta)
sumLik <- list()

###################################
# Algorithm 1
###################################

# for reproducibility
set.seed(42)

# step 1: input {y_{it}}
# step 2: initialize set of parameters

# define parameters
# a <- torch_randn(2)
# b <- torch_randn(2)
# k <- torch_randn(2)
# Lmd <- torch_randn(2)
# alpha <- torch_normal(mean=0, std=1e-1, size=1)
# beta <- torch_normal(mean=0, std=1e-1, size=1)

# with gradient tracking
a <- torch_tensor(a, requires_grad=TRUE)
b <- torch_tensor(b, requires_grad=TRUE)
k <- torch_tensor(k, requires_grad=TRUE)
Lmd <- torch_tensor(Lmd, requires_grad=TRUE)
alpha <- torch_tensor(alpha, requires_grad=TRUE)
beta <- torch_tensor(beta, requires_grad=TRUE)

# step 3: initialize latent variables
# latent variable score at initial time point is assumed to follow N(0, 1e3) 
for (s in 1:2) {
  mEta[,1,s] <- rep(x=0, times=N)
  mP[,1,s] <- rep(x=1e3, times=N)
}

# step 4: initialize residual variances
# Qs <- torch_normal(mean=0, std=1e-1, size=2)**2
# Rs <- torch_normal(mean=0, std=1e-1, size=2)**2

Qs <- torch_tensor(Qs, requires_grad=TRUE)
Rs <- torch_tensor(Rs, requires_grad=TRUE)

# vectorize parameters
theta <- torch_cat(list(a, b, k, Lmd, alpha, beta, Qs, Rs))

# f(y_{i,t} | y_{i,t-1})
mLik <- torch_zeros(N, Nt)

# P(s=2 | y_{i,t})
mPr <- torch_zeros(N, Nt+1)

# step 5: initialize marginal probability
mPr[,1] <- 1e-8 # no drop out at t=0
print(paste0('   optimization step: ', as.numeric(iter)))

# step 6
for (t in 1:Nt) {
  
  # step 7 
  tPr[,t,1] <- torch_sigmoid(alpha + beta * yt[,t])
  tPr[,t,2] <- 1
  
  # step 8 
  jPr[,t,1,1] <- (1-torch_clone(tPr[,t,1])) * (1-torch_clone(mPr[,t]))
  jPr[,t,2,1] <- torch_clone(tPr[,t,1]) * (1-torch_clone(mPr[,t]))
  jPr[,t,1,2] <- (1-torch_clone(tPr[,t,2])) * torch_clone(mPr[,t])
  jPr[,t,2,2] <- torch_clone(tPr[,t,2]) * torch_clone(mPr[,t])
  
  # step 9 
  for (s1 in 1:2) {
    for (s2 in 1:2) {
      
      jEta[,t,s1,s2] <- a[s1] + b[s1] * torch_clone(mEta[,t,s2])
      jP[,t,s1,s2] <- b[s1]**2 * torch_clone(mP[,t,s2]) + Qs[s1]
      
      jV[,t,s1,s2] <- yt[,t] - (k[s1] + Lmd[s1] * torch_clone(jEta[,t,s1,s2]))
      jF[,t,s1,s2] <- Lmd[s1]**2 * torch_clone(jP[,t,s1,s2]) + Rs[s1]
      
      Ks <- torch_clone(jP[,t,s1,s2]) * Lmd[s1] / torch_clone(jF[,t,s1,s2])
      
      jEta2[,t,s1,s2] <- torch_clone(jEta[,t,s1,s2]) + torch_clone(Ks) * torch_clone(jV[,t,s1,s2]) 
      jP2[,t,s1,s2] <- torch_clone(jP[,t,s1,s2]) - torch_clone(Ks) * Lmd[s1] * torch_clone(jP[,t,s1,s2])
      
      # step 10 
      jLik[,t,s1,s2] <- (2*pi)**(-1/2) * (torch_clone(jF[,t,s1,s2]))**(-1/2) *
        torch_exp(-1/2 * torch_clone(jV[,t,s1,s2])**2 / torch_clone(jF[,t,s1,s2]))
      
      # step 11 
      mLik[,t] <- torch_clone(mLik[,t]) + torch_clone(jLik[,t,s1,s2]) * torch_clone(jPr[,t,s1,s2])
    }
  }
  
  for (s1 in 1:2) {
    for (s2 in 1:2) {
      # step 11 
      jPr2[,t,s1,s2] <- torch_clone(jLik[,t,s1,s2]) * torch_clone(jPr[,t,s1,s2]) / torch_clone(mLik[,t]) 
      if (s1 == 2) {
        mPr[,t+1] <- torch_clone(mPr[,t+1]) + torch_clone(jPr2[,t,s1,s2]) }
    }
  }   
  
  for (s1 in 1:2) {
    for (s2 in 1:2) {
      # step 12
      if (s1 == 1) {
        if (torch_allclose(mPr[,t+1], 1)) { W[,t,1,s2] <- (torch_clone(jPr2[,t,1,s2]) + 1e-8) / 1e-8 }
        else { W[,t,1,s2] <- torch_clone(jPr2[,t,1,s2]) / (1-torch_clone(mPr[,t+1])) }
      }
      else if (s1 == 2) {
        if (torch_allclose(mPr[,t+1], 0)) { W[,t,2,s2] <- (torch_clone(jPr2[,t,2,s2]) + 1e-8) / 1e-8 }
        else { W[,t,2,s2] <- torch_clone(jPr2[,t,2,s2]) / torch_clone(mPr[,t+1]) }
      }
    }
    
    # step 12 (continuation)
    mEta[,t+1,s1] <- torch_sum( torch_clone(W[,t,s1,]) * torch_clone(jEta2[,t,s1,]), dim=2)
    mP[,t+1,s1] <- 
      torch_sum(torch_clone(W[,t,s1,]) 
                * (torch_clone(jP2[,t,s1,]) 
                   + (torch_transpose(torch_vstack(list(torch_clone(mEta[,t+1,s1]), torch_clone(mEta[,t+1,s1]))), 1, 2) 
                      - torch_clone(jEta2[,t,s1,]))**2), dim=2)
  }
  
  
} # this line relates to the beginnig of step 6

if (count < 3) {
  if (as.numeric(torch_sum(torch_isnan(mLik))) > 0) { # is mLik has NaN values
    count <- 0
    print('   optimization terminated: mLik has null values')
    print(paste0(c('a1', 'a2', 'b1', 'b2', 'k1', 'k2', 'Lmd1', 'Lmd2', 'alpha', 'beta', 'Qs1', 'Qs2', 'Rs1', 'Rs2'), ': ', as.matrix(theta)))
  }
  else {
    # sum likelihood at each optimization step
    sumLik[iter] <- as.numeric(torch_sum(mLik))
    
    if (sumLik[iter][[1]] > sumLikBest) { # if sumLik beats the best score
      sumLikBest <- sumLik[iter]
      thetaBest <- theta
    }
    
    # stopping criterion
    if (iter > 1) {
      # add count if sumLik does not beat the best score 
      if (sumLik[iter][[1]] < sumLikBest) {
        count <- count + 1 }
      else {count <- 0} 
    }
  }
  
  if (count==3) {
    print('   optimizaiton terminated: stopping criterion is met')
    count <- 0
  }
  print(paste0('   sum of likelihood = ', sumLik[iter]))
  
  # backward propagation
  torch_sum(mLik)$backward(retain_graph=TRUE)
  # store gradients
  grad <- torch_cat(list(a$grad, b$grad, k$grad, Lmd$grad, alpha$grad, beta$grad, Qs$grad, Rs$grad))
  # run adam function definied above
  result <- adam(theta, grad, iter, m, v)
  # update parameters
  theta <- result$theta
  theta[11:12] <- torch_tensor(max(torch_tensor(c(0,0)), theta[11:12]))
  theta[13:14] <- torch_tensor(max(torch_tensor(c(0,0)), theta[13:14]))
  a <- torch_tensor(theta[1:2], requires_grad=TRUE)
  b <- torch_tensor(theta[3:4], requires_grad=TRUE)
  k <- torch_tensor(theta[5:6], requires_grad=TRUE)
  Lmd <- torch_tensor(theta[7:8], requires_grad=TRUE)
  alpha <- torch_tensor(theta[9:9], requires_grad=TRUE)
  beta <- torch_tensor(theta[10:10], requires_grad=TRUE)
  Qs <- torch_tensor(theta[11:12], requires_grad=TRUE)
  Rs <- torch_tensor(theta[13:14], requires_grad=TRUE)
  m <- result$m 
  v <- result$v
}