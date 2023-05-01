# install.packages("torch")
library(torch)

# for reproducibility
set.seed(42)

# Adam optimization 
# input: initial parameter values and gradients 
# output: updated parameter values
adam <- function(theta, grad, iter, m, v, lr = 1e-3, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8) {
  # theta: parameters values
  # grad: gradient of the objective function with respect to the parameters at the current iteration
  # lr: learning rate
  # beta1, beta2: hyperparameters controlling the exponential decay rates for the moment estimates
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
  
  return(list(theta = theta, m = m, v = v))
}

# max number of optimization steps
nIter <- 50
# initialization of stopping criterion
count <- 0
# initialization for Adam optimization
m <- NULL
v <- NULL

###################################
# s=1: non-drop out state
# s=2: drop out state
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
sumLik <- torch_full(nIter, NaN)
  
###################################

# step 1: input {y_{it}}
# step 2: initialize set of parameters
a <- torch_randn(2)
b <- torch_abs(torch_randn(2))
k <- torch_randn(2)
Lmd <- torch_randn(2)
alpha <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))
beta <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))

theta <- torch_cat(list(a, b, k, Lmd, alpha, beta))

# step 3: initialize latent variables
# latent variable score at initial time point is assumed to follow N(0, 1e3) 
for (s in 1:2) {
  mEta[,1,s] <- rep(x=0, times=N)
  mP[,1,s] <- rep(x=1e3, times=N)
}

# step 4: initialize residual variances
Qs <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))
Rs <- torch_abs(torch_normal(mean=0, std=1e-1, size=2))

# step 5: initialize marginal probability
# mPr[, 0] <- 0 # no drop out at t=0

# activate gradient tracking for each parameters
a <- torch_tensor(a, requires_grad=TRUE)
b <- torch_tensor(b, requires_grad=TRUE)
k <- torch_tensor(k, requires_grad=TRUE)
Lmd <- torch_tensor(Lmd, requires_grad=TRUE)
alpha <- torch_tensor(alpha, requires_grad=TRUE)
beta <- torch_tensor(beta, requires_grad=TRUE)
gamma <- torch_tensor(gamma, requires_grad=TRUE)
delta <- torch_tensor(delta, requires_grad=TRUE)

for (iter in 1:nIter) { 
  # marginal likelihood
  # f(y_{i,t} | y_{i,t-1})
  mLik <- torch_zeros(N, Nt)
  
  # marginal probability
  # P(s=2 | y_{i,t})
  mPr <- torch_zeros(N, Nt+1)
  
  # initial drop out probability
  mPr[,1] <- 1e-2
  
  print(paste0('optimization step: ', as.numeric(iter)))
  
  # step 6
  for (i in 1:N) {
    for (t in 1:Nt) {
      
      # step 7 
      tPr[i,t,1] <- torch_sigmoid(alpha[1] + beta[1] * yt[i,t])
      tPr[i,t,2] <- 1
      
      # step 8 
      jPr[i,t,1,1] <- (1-torch_clone(tPr[i,t,1])) * (1-torch_clone(mPr[i,t]))
      jPr[i,t,2,1] <- torch_clone(tPr[i,t,1]) * (1-torch_clone(mPr[i,t]))
      jPr[i,t,1,2] <- (1-torch_clone(tPr[i,t,2])) * torch_clone(mPr[i,t])
      jPr[i,t,2,2] <- torch_clone(tPr[i,t,2]) * torch_clone(mPr[i,t])
      
      # step 9 
      for (s1 in 1:2) {
        for (s2 in 1:2) {
          
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
          mLik[i,t] <- torch_clone(mLik[i,t]) + torch_clone(jLik[i,t,s1,s2]) * torch_clone(jPr[i,t,s1,s2])
        }
      }
      
      for (s1 in 1:2) {
        for (s2 in 1:2) {
          # step 11 
          jPr2[i,t,s1,s2] <- torch_clone(jLik[i,t,s1,s2]) * torch_clone(jPr[i,t,s1,s2]) / torch_clone(mLik[i,t]) 
          if (s1 == 2) {
            mPr[i,t+1] <- torch_clone(mPr[i,t+1]) + torch_clone(jPr2[i,t,s1,s2]) }
        }
      }   
      
      for (s1 in 1:2) {
        for (s2 in 1:2) {
          # step 12
          if (s1 == 1) {
            if (torch_allclose(mPr[i,t+1], 1)) { W[i,t,1,s2] <- max(torch_clone(jPr2[i,t,1,s2]), 1e-5) / 1e-5 }
            else { W[i,t,1,s2] <- torch_clone(jPr2[i,t,1,s2]) / (1-torch_clone(mPr[i,t+1])) }
          }
          else if (s1 == 2) {
            if (torch_allclose(mPr[i,t+1], 0)) { W[i,t,2,s2] <- max(torch_clone(jPr2[i,t,2,s2]), 1e-5) / 1e-5 }
            else { W[i,t,2,s2] <- torch_clone(jPr2[i,t,2,s2]) / torch_clone(mPr[i,t+1]) }
          }
        }
        # step 12 (continuation)
        mEta[i,t+1,s1] <- torch_sum( torch_clone(W[i,t,s1,]) * torch_clone(jEta2[i,t,s1,]))
        mP[i,t+1,s1] <- torch_sum( torch_clone(W[i,t,s1,]) * ( torch_clone(jP2[i,t,s1,]) + (torch_clone(mEta[i,t+1,s1]) - torch_clone(jEta2[i,t,s1,]))**2 ))
      }
    }
  }
  # store sum likelihood in each optimization step
  sumLik[iter] <- sum(mLik)
  
  # stopping criterion
  if (iter > 1) {
    if ( as.numeric(sumLik[iter]) <= as.numeric(sumLik[iter - (1+count)]) ) { 
      if (count > 1) {break} 
      else {count <- count + 1} }
    else {count <- 0}
  }
  
  # sumLik[iter]$grad_fn
  
  print(paste0('sum of likelihood = ', as.numeric(sumLik[iter])))
  
  # backward propagation
  sumLik[iter]$backward(retain_graph=TRUE)
  # store gradients
  grad <- torch_cat(list(a$grad, b$grad, k$grad, Lmd$grad, alpha$grad, beta$grad, gamma$grad, delta$grad))
  # run adam function definied above
  result <- adam(theta, grad, iter, m, v)
  # update parameters
  a <- torch_tensor(result$theta[1:2], requires_grad=TRUE)
  b <- torch_tensor(result$theta[3:4], requires_grad=TRUE)
  k <- torch_tensor(result$theta[5:6], requires_grad=TRUE)
  Lmd <- torch_tensor(result$theta[7:8], requires_grad=TRUE)
  alpha <- torch_tensor(result$theta[9:10], requires_grad=TRUE)
  beta <- torch_tensor(result$theta[11:12], requires_grad=TRUE)
  m <- result$m 
  v <- result$v
}

# plot optimization process w.r.t sum likelihood
plot(sumLik[1:iter], type='b', xlab='optimization step', ylab='sum of the likelihood', main='sum likelihood in each optimization step')
