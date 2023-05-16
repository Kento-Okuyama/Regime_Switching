# install.packages("torch")
library(torch)
# install.packages("reticulate")
library(reticulate)

epsilon <- 1e-30
nparams <- 15

###################################
# function for Adam optimization 
###################################
# input: initial parameter values and gradients 
# output: updated parameter values
adam <- function(theta, grad, iter, m, v, lr=3e-2, beta1=0.9, beta2=0.999, epsilon=1e-30) {
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
  if (as.numeric(torch_sum(torch_isnan(m))) > 0) {
    m[torch_isnan(m)] <- 0 }
  
  if (as.numeric(torch_sum(torch_isnan(v))) > 0) {
    v[torch_isnan(v)] <- 0 }
  
  # update moment estimates
  m <- beta1 * m + (1 - beta1) * grad
  v <- beta2 * v + (1 - beta2) * grad**2
  
  # update bias corrected moment estimates
  m_hat <- m / (1 - beta1**iter)
  v_hat <- v / (1 - beta2**iter)
  
  # Update parameters using Adam update rule
  
  compared <- (sqrt(v_hat) + epsilon) > epsilon
  denom <- torch_full(nparams, epsilon)
  if (as.numeric(torch_sum(compared)) > 0) {
    denom[compared] <- (sqrt(v_hat) + epsilon)[compared] }
  theta <- theta + lr * m_hat / denom
  
  return(list(theta=theta, m=m, v=v))
}

# number of parameter initialization
nInit <- 5
# max number of optimization steps
nIter <- 10
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
# f(y_{i,t} | y_{i,t-1})
mLik <- torch_full(c(N, Nt), NaN)
# P(s, s' | y_{i,t})
jPr2 <- torch_full(c(N, Nt, 2, 2), NaN)
# W_{i,t}^{s,s'}
W <- torch_full(c(N, Nt, 2, 2), NaN)
# f(Y | theta)
sumLik <- list()
sumLikBest <- epsilon

thetaBest <- torch_full(nparams, NaN)
sumLikBestNow <- epsilon
thetaBestNow <- torch_full(nparams, NaN)
ythBest <-  torch_full(c(N, Nt), NaN)
mPrBest <-  torch_full(c(N, Nt), NaN)


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
  a <- torch_randn(2)
  b <- torch_randn(2)
  k <- torch_randn(2)
  Lmd <- torch_randn(2)
  alpha <- torch_randn(2)
  beta <- torch_randn(1)
  
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
  Q <- torch_normal(mean=0, std=1e-1, size=2)**2
  R <- torch_normal(mean=0, std=1e-1, size=2)**2
  
  Q <- torch_tensor(Q, requires_grad=TRUE)
  R <- torch_tensor(R, requires_grad=TRUE)
  
  # vectorize parameters
  theta <- torch_cat(list(a, b, k, Lmd, alpha, beta, Q, R))
  
  while (count < 3) {
    # for (iter in 1:nIter) {
    
    # joint one-step ahead forecast
    jyth <- torch_full(c(N, Nt, 2, 2), NaN)
    # one-step ahead forecast
    yth <- torch_full(c(N, Nt), NaN)
    
    # P(s=2 | y_{i,t})
    mPr <- torch_zeros(N, Nt+1)
    
    # fixed prob of switching back
    tPr[,,2] <- torch_sigmoid(alpha[2])
    
    # step 5: initialize marginal probability
    mPr[,1] <- epsilon # no drop out at t=0
    
    print(paste0('   optimization step: ', as.numeric(iter)))
    # step 6
    for (t in 1:Nt) {
      
      # step 9 
      for (s1 in 1:2) {
        for (s2 in 1:2) {
          
          jEta[,t,s1,s2] <- a[s1] + b[s1] * torch_clone(mEta[,t,s2])
          jP[,t,s1,s2] <- b[s1]**2 * torch_clone(mP[,t,s2]) + Q[s1]
          
          jyth[,t,s1,s2] <- k[s1] + Lmd[s1] * torch_clone(jEta[,t,s1,s2])
          jV[,t,s1,s2] <- yt[,t] - (k[s1] + Lmd[s1] * torch_clone(jEta[,t,s1,s2]))
          
          jF[,t,s1,s2] <- Lmd[s1]**2 * torch_clone(jP[,t,s1,s2]) + R[s1]
          
          Ks <- torch_clone(jP[,t,s1,s2]) * Lmd[s1] / torch_clone(jF[,t,s1,s2])
          
          jEta2[,t,s1,s2] <- torch_clone(jEta[,t,s1,s2]) + torch_clone(Ks) * torch_clone(jV[,t,s1,s2]) 
          jP2[,t,s1,s2] <- torch_clone(jP[,t,s1,s2]) - torch_clone(Ks) * Lmd[s1] * torch_clone(jP[,t,s1,s2])
          
          # step 10 
          compared <- ((2*pi)**(-1/2) * (torch_clone(jF[,t,s1,s2]))**(-1/2) *
                         torch_exp(-1/2 * torch_clone(jV[,t,s1,s2])**2 / torch_clone(jF[,t,s1,s2]))) > epsilon
          jLik[,t,s1,s2] <- torch_full(N, epsilon)
          if (as.numeric(torch_sum(compared)) > 0) {
            jLik[,t,s1,s2][compared] <- ((2*pi)**(-1/2) * (torch_clone(jF[,t,s1,s2]))**(-1/2) *
                                           torch_exp(-1/2 * torch_clone(jV[,t,s1,s2])**2 / torch_clone(jF[,t,s1,s2])))[compared]  
          }
        }
      }
      
      # step 7 
      tPr[,t,1] <- torch_sigmoid(alpha[1] + beta * yt[,t])
      
      # step 8 
      jPr[,t,1,1] <- (1-torch_clone(tPr[,t,1])) * (1-torch_clone(mPr[,t]))
      jPr[,t,2,1] <- torch_clone(tPr[,t,1]) * (1-torch_clone(mPr[,t]))
      jPr[,t,1,2] <- (1-torch_clone(tPr[,t,2])) * torch_clone(mPr[,t])
      jPr[,t,2,2] <- torch_clone(tPr[,t,2]) * torch_clone(mPr[,t])
      
      yth[,t] <- torch_sum(jyth[,t,,] * jPr[,t,,])
      
      # step 11 
      mLik[,t] <- torch_sum(torch_clone(jLik[,t,,]) * torch_clone(jPr[,t,,]))
      
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
            compared <- 1 - torch_clone(mPr[,t+1]) > epsilon
            denom <- torch_full(N, epsilon)
            if (as.numeric(torch_sum(compared)) > 0) {
              denom[compared] <- (1 - torch_clone(mPr[,t+1]))[compared] }
            W[,t,1,s2] <- torch_clone(jPr2[,t,1,s2]) / denom }
          
          else if (s1 == 2) {
            compared <- torch_clone(mPr[,t+1]) > epsilon
            denom <- torch_full(N, epsilon)
            if (as.numeric(torch_sum(compared)) > 0) {
              denom[compared] <- torch_clone(mPr[,t+1])[compared] }
            W[,t,2,s2] <- torch_clone(jPr2[,t,2,s2]) / denom }
        }
      }
      
      # step 12 (continuation)
      mEta[,t+1,] <- torch_sum(torch_clone(W[,t,,]) * torch_clone(jEta2[,t,,]), dim=3)
      mEtaVec <- torch_cat(list(torch_unsqueeze(torch_clone(mEta[,t+1,]), dim=3), torch_unsqueeze(torch_clone(mEta[,t+1,]), dim=3)), dim=3)
      mP[,t+1,] <- torch_sum(torch_clone(W[,t,,]) * ( torch_clone(jP2[,t,,]) + (mEtaVec - torch_clone(jEta2[,t,,]))**2 ), dim=3)
      
    } # this line relates to the beginning of step 6
    
    if (count < 3) {
      if (as.numeric(torch_sum(torch_isnan(mLik))) > 0) { # is mLik has NaN values
        print('   optimization terminated: mLik has null values')
        break
      }
      else {
        # sum likelihood at each optimization step
        sumLik[iter] <- as.numeric(torch_sum(mLik))
        
        if (sumLik[iter][[1]] > sumLikBest) { # if sumLik beats the best score
          sumLikBest <- sumLik[iter]
          thetaBest <- theta
          ythBest <- yth
          mPrBest <- mPr[,2:Nt]
        }
        
        # stopping criterion
        if (iter > 1) {
          if (sumLik[iter][[1]] - sumLik[1][[1]] != 0) {
            crit <- (sumLik[iter][[1]] - sumLik[iter-1][[1]]) / (sumLik[iter][[1]] - sumLik[1][[1]]) }
          else {crit <- 0}
          
          # add count if sumLik does not beat the best score 
          if (crit < 1e-2) {
            count <- count + 1 }
          else {
            sumLikBestNow <- sumLik[iter]
            thetaBestNow <- theta
            count <- 0
          } 
        }
      }
      
      if (count==3) {
        print('   optimizaiton terminated: stopping criterion is met')
        break }
      print(paste0('   sum of likelihood = ', sumLik[iter]))
      
      # backward propagation
      torch_sum(mLik)$backward(retain_graph=TRUE)
      # store gradients
      grad <- torch_cat(list(a$grad, b$grad, k$grad, Lmd$grad, alpha$grad, beta$grad, Q$grad, R$grad))
      
      if (as.numeric(torch_sum(torch_isnan(grad))) > 0) {
        print(as.numeric(torch_sum(mLik)))
        print('   optimizaiton terminated: gradient has nan values')
        break }
      
      # run adam function detorch finied above
      result <- adam(theta, grad, iter, m, v)
      # update parameters
      theta <- result$theta
      theta[12:13] <- torch_tensor(max(torch_tensor(c(0,0)), theta[12:13]))
      theta[14:15] <- torch_tensor(max(torch_tensor(c(0,0)), theta[14:15]))
      a <- torch_tensor(theta[1:2], requires_grad=TRUE)
      b <- torch_tensor(theta[3:4], requires_grad=TRUE)
      k <- torch_tensor(theta[5:6], requires_grad=TRUE)
      Lmd <- torch_tensor(theta[7:8], requires_grad=TRUE)
      alpha <- torch_tensor(theta[9:10], requires_grad=TRUE)
      beta <- torch_tensor(theta[11:11], requires_grad=TRUE)
      Q <- torch_tensor(theta[12:13], requires_grad=TRUE)
      R <- torch_tensor(theta[14:15], requires_grad=TRUE)
      m <- result$m 
      v <- result$v
    }
    iter <- iter + 1
  } # nIter
  
  sumLikBestNow <- epsilon
  thetaBestNow <- torch_full(nparams, NaN)
  
} # nInit

# return the best result
print(paste0('Best sum likelihood = ', sumLikBest[[1]]))
thetaBest <- as.data.frame(t(as.matrix(thetaBest)))
colnames(thetaBest) <- c('a1', 'a2', 'b1', 'b2', 'k1', 'k2', 'Lmd1', 'Lmd2', 'alpha1', 'alpha2', 'beta', 'Q1', 'Q2', 'R1', 'R2')
print('Optimal parameters found:')
print(paste0(colnames(thetaBest), ": ", as.matrix(thetaBest)))

print('yt (true scores):')
print(yt)
print('yth (One-step-ahead predictions):')
print(yth)

print('St (true scores)')
print(state)
print('mPr')
print(mPrBest)

# PoC: if you want to test the feasibility of the model
# yt <- as.matrix(yth)
# state <- torch_bernoulli(mPrBest) + 1
