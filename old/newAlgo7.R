# install.packages('torch')
library(torch)
# install.packages('reticulate')
library(reticulate)
# install.packages('truncnorm')
library(truncnorm)

# number of parameter initialization
nInit <- 50
# max number of iterations
maxIter <- 250
# some small number
sEpsilon <- 1e-6
lEpsilon <- 1e-1
# some large number
ceil <- 1e6
# hyperparameters for adam optimization
lr <- 1e-3
betas <- c(.9, .999) 

###############
# import data #
###############
x <- df$x
y1 <- df$y3D1
y2 <- df$y3D2
eta1 <- df$eta3D1
eta2 <- df$eta3D2
N <- df$N
Nt <- df$Nt
No1 <- df$No1 
No2 <- df$No2 
Nf1 <- df$Nf1
Nf2 <- df$Nf2

y1 <- torch_tensor(y1)
x <- torch_tensor(x)
eta1 <- torch_tensor(eta1)
eta2 <- torch_tensor(eta2)

#############
# algorithm #
#############
# for reproducibility 
set.seed(42)

sumLikBest <- 0

for (init in 1:nInit) {
  cat('Initialization step ', init, '\n')
  
  # optimization step count
  iter <- 1
  # stopping criterion count
  count <- 0 
  # store sum-likelihood 
  sumLik <- list()
  # initialize moment estimates
  m <- v <- list()
  m_hat <- v_hat <- list()
  
  # initialize parameters
  B11 <- torch_tensor(rep(times=Nf1, x=0))
  B12 <- torch_tensor(rep(times=Nf1, x=0))
  B21d <- B22d <- torch_tensor(rep(times=Nf1, x=1))
  B31 <- torch_tensor(0)
  B32 <- torch_tensor(0)
  d <- torch_tensor(df$icept1)
  Lmdv <- torch_tensor(df$coef1)
  Lmd <- Lmd <- torch_full(c(No1,Nf1), 0)
  Lmd[1:3,1] <- Lmdv[1:3]; Lmd[4:5,2] <- Lmdv[4:5]
  Lmd[6:7,3] <- Lmdv[6:7]; Lmd[8:9,4] <- Lmdv[8:9]
  Lmd[10:11,5] <- Lmdv[10:11]; Lmd[12:14,6] <- Lmdv[12:14]
  Lmd[15:17,7] <- Lmdv[15:17]
  Q1d <- torch_tensor(abs(rnorm(Nf1, mean=0, sd=.1)))
  Q2d <- torch_tensor(abs(rnorm(Nf1, mean=0, sd=.1)))
  Q1 <- Q1d$diag()
  Q2 <- Q2d$diag()
  R1d <- torch_tensor(abs(rnorm(No1, mean=0, sd=.1)))
  R2d <- torch_tensor(abs(rnorm(No1, mean=0, sd=.1)))
  R1 <- R1d$diag()
  R2 <- R2d$diag()
  gamma11 <- torch_tensor(-2.95)
  gamma21 <- torch_tensor(rep(times=Nf1, x=0))

  # with_detect_anomaly ({
  try (silent = FALSE, {
    while (count <= 3 && iter <= maxIter) {
      cat('   optimization step: ', as.numeric(iter), '\n')
      
      B11$requires_grad_()
      B12$requires_grad_()
      B1 <- list(B11, B12)
      B21 <- B21d$diag()
      B22 <- B22d$diag()
      B21$requires_grad_()
      B22$requires_grad_()
      B2 <- list(B21, B22)
      B31$requires_grad_()
      B32$requires_grad_()
      B3 <- list(B31, B32)
      Q1$requires_grad_()
      Q2$requires_grad_()
      Q <- list(Q1, Q2)
      R1$requires_grad_()
      R2$requires_grad_()
      R <- list(R1, R2)
      gamma11$requires_grad_()
      gamma21$requires_grad_()

      theta <- list(B11=B11, B12=B12, B21=B21, B22=B22, B31=B31, B32=B32, Q1=Q1, Q2=Q2, R1=R1, R2=R2,
                    gamma11=gamma11, gamma21=gamma21)  
      
      # define variables
      jEta <- torch_full(c(N,Nt,2,2,Nf1), NaN) # Eq.2 (LHS)
      jDelta <- torch_full(c(N,Nt,2,2,Nf1), NaN) # Eq.3 (LHS)
      jP <- jP_ <- jPChol <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN) # Eq.4 (LHS)
      jV <- torch_full(c(N,Nt,2,2,No1), NaN) # Eq.5 (LHS)
      jF <- jF_ <- jFChol <- torch_full(c(N,Nt,2,2,No1,No1), NaN) # Eq.6 (LHS)
      jEta2 <- torch_full(c(N,Nt,2,2,Nf1), NaN) # Eq.7 (LHS)
      jP2 <- jP2_ <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN) # Eq.8 (LHS)
      mEta <- torch_full(c(N,Nt+1,2,Nf1), NaN) # Eq.9-1 (LHS)
      mP <- mP_ <- torch_full(c(N,Nt+1,2,Nf1,Nf1), NaN) # Eq.9-2 (LHS)
      W <- torch_full(c(N,Nt,2,2), NaN) # Eq.9-3 (LHS)
      jPr <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-1 (LHS)
      mLik <- torch_full(c(N,Nt), NaN) # Eq.10-2 (LHS)
      jPr2 <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-3 (LHS)
      mPr <- torch_full(c(N,Nt+1), NaN) # Eq.10-4 (LHS)
      jLik <- torch_full(c(N,Nt,2,2), NaN) # Eq.11 (LHS)
      tPr <- torch_full(c(N,Nt,2), NaN) # Eq.12 (LHS)
      KG <- torch_full(c(N,Nt,2,2,Nf1,No1), NaN) # Kalman gain function
      I_KGLmd <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN) 
      denom1 <- torch_full(c(N,Nt), NaN)
      denom2 <- torch_full(c(N,Nt), NaN)
      subEta <- torch_full(c(N,Nt,2,2,Nf1), NaN) 
      subEtaSq <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN)
      
      # initialize latent variables
      mEta[,1,,] <- 0
      mP[,1,,,] <- 0; mP[,1,,,]$add_(torch_eye(Nf1)) 
      
      # initialize P(s'|eta_0)
      mPr[,1] <- sEpsilon 
      
      #######################
      # extended Kim filter #
      #######################
      for (t in 1:Nt) { 
        if (t%%10==0) {cat('   t=', t, '\n')}
        
        # Kalman Filter
        for (s1 in 1:2) {
          # Eq.2
          jEta[,t,s1,,] <- B1[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) + 
            mEta[,t,,]$clone()$matmul(B2[[s1]]) + 
            B3[[s1]] * eta2$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) 
          
          # Eq.3
          jDelta[,t,s1,,] <- eta1[,t,]$clone()$unsqueeze(dim=2) - jEta[,t,s1,,]$clone() 
          
          # Eq.4
          jP[,t,s1,,,] <- B2[[s1]]$matmul(mP[,t,,,]$clone())$matmul(B2[[s1]]$transpose(1, 2)) + 
            Q[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1)
          with_no_grad ({ 
            jPEig <- linalg_eigh(jP[,t,s1,,,])
            jPEig[[1]]$real$clip_(lEpsilon, ceil)
            for (s2 in 1:2) {
              for (row in 1:N) {
                jP[row,t,s1,s2,,] <- jPEig[[2]]$real[row,s2,,]$matmul(jPEig[[1]]$real[row,s2,]$diag())$matmul(jPEig[[2]]$real[row,s2,,]$transpose(1, 2)) } } })
          jPChol[,t,s1,,,] <- linalg_cholesky_ex(jP[,t,s1,,,]$clone())$L
          
          # Eq.5
          jV[,t,s1,,] <- y1[,t,]$clone()$unsqueeze(dim=2) -
            (d$unsqueeze(dim=1)$unsqueeze(dim=1) + 
               jEta[,t,s1,,]$clone()$matmul(Lmd$transpose(1, 2))) 
          
          # Eq.6
          jF[,t,s1,,,] <- Lmd$matmul(jP[,t,s1,,,]$clone())$matmul(Lmd$transpose(1, 2)) + 
            R[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1)
          with_no_grad ({
            jFEig <- linalg_eigh(jF[,t,s1,,,])
            jFEig[[1]]$real$clip_(lEpsilon, ceil)
            for (s2 in 1:2) {
              for (row in 1:N) {
                jF[row,t,s1,s2,,] <- jFEig[[2]]$real[row,s2,,]$matmul(jFEig[[1]]$real[row,s2,]$diag())$matmul(jFEig[[2]]$real[row,s2,,]$transpose(1, 2)) } } }) 
          jFChol[,t,s1,,,] <- linalg_cholesky_ex(jF[,t,s1,,,]$clone())$L
          
          # kalman gain function
          KG[,t,s1,,,] <- jP[,t,s1,,,]$clone()$matmul(Lmd$transpose(1, 2))$matmul(jFChol[,t,s1,,,]$clone()$cholesky_inverse())
          
          for (s2 in 1:2) {
            # Eq.7
            jEta2[,t,s1,s2,] <- jEta[,t,s1,s2,]$clone() + KG[,t,s1,s2,,]$clone()$matmul(jV[,t,s1,s2,]$clone()$unsqueeze(dim=-1))$squeeze()
            I_KGLmd[,t,s1,s2,,] <- torch_eye(Nf1)$unsqueeze(dim=1) - KG[,t,s1,s2,,]$clone()$matmul(Lmd)
            
            # Eq.9
            jP2[,t,s1,s2,,] <- I_KGLmd[,t,s1,s2,,]$clone()$matmul(jP[,t,s1,s2,,]$clone())$matmul(I_KGLmd[,t,s1,s2,,]$clone()$transpose(2, 3)) + 
              KG[,t,s1,s2,,]$clone()$matmul(R[[s1]])$matmul(KG[,t,s1,s2,,]$clone()$transpose(2, 3))
            with_no_grad ({
              jP2Eig <- linalg_eigh(jP2[,t,s1,s2,,]) 
              jP2Eig[[1]]$real$clip_(lEpsilon, ceil)
              for (row in 1:N) {
                jP2[row,t,s1,s2,,] <- jP2Eig[[2]]$real[s2,,]$matmul(jP2Eig[[1]]$real[s2,]$diag())$matmul(jP2Eig[[2]]$real[s2,,]$transpose(1, 2)) } }) 
            
            # joint likelihood f(eta_{t}|s,s',eta_{t-1})
            # Eq.12
            jLik[,t,s1,s2] <- (2*pi)**(-Nf1/2) * jP[,t,s1,s2,,]$clone()$det()**(-1) * 
              (-.5 * jDelta[,t,s1,s2,]$clone()$unsqueeze(dim=2)$matmul(jPChol[,t,s1,s2,,]$clone()$cholesky_inverse())$matmul(jDelta[,t,s1,s2,]$clone()$unsqueeze(dim=-1))$squeeze()$squeeze())$exp()
            jLik[,t,s1,s2]$clip_(min=sEpsilon) } }
        
        # transition probability P(s|s',eta_{t-1})  
        if (t == 1) {
          tPr[,t,1] <- (gamma11)$sigmoid() 
          tPr[,t,2] <- 1 
          
        } else {
          tPr[,t,1] <- (1 - x[,t-1]) * (gamma11 + eta1[,t-1,]$clone()$matmul(gamma21))$sigmoid() + x[,t-1]
          tPr[,t,2] <- 1 }
        
        jPr[,t,2,2] <- tPr[,t,2]$clone() * mPr[,t]$clone()
        jPr[,t,2,1] <- tPr[,t,1]$clone() * (1-mPr[,t]$clone())
        jPr[,t,1,2] <- (1-tPr[,t,2]$clone()) * mPr[,t]$clone()
        jPr[,t,1,1] <- (1-tPr[,t,1]$clone()) * (1-mPr[,t]$clone()) 
        div <- jPr[,t,,]$sum(dim=c(2,3))
        with_no_grad({div$clip_(sEpsilon, ceil)})
        jPr[,t,,]$div_(div$unsqueeze(dim=-1)$unsqueeze(dim=-1))
        
        # marginal likelihood function f(eta_{t}|eta_{t-1})
        mLik[,t] <- (jLik[,t,,]$clone() * jPr[,t,,]$clone())$sum(dim=c(2,3))
        with_no_grad(mLik[,t]$clip_(min=sEpsilon))
        
        # (updated) joint probability P(s,s'|eta_{t})
        jPr2[,t,,] <- jLik[,t,,]$clone() * jPr[,t,,]$clone() / mLik[,t]$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) 
        div <- jPr2[,t,,]$sum(dim=c(2,3))
        with_no_grad({div$clip_(sEpsilon, ceil)})
        jPr2[,t,,]$div_(div$unsqueeze(dim=-1)$unsqueeze(dim=-1))  
        
        # marginal probability P(s|eta_{t})
        mPr[,t+1] <- jPr2[,t,2,]$clone()$sum(dim=2)
        
        # step 11: collapsing procedure
        for (s2 in 1:2) { 
          denom1[,t] <- 1 - mPr[,t+1]$clone()
          with_no_grad({denom1[,t]$clip_(sEpsilon, ceil)})
          W[,t,1,s2] <- jPr2[,t,1,s2]$clone() / denom1[,t]$clone()
          
          denom2[,t] <- mPr[,t+1]$clone()
          with_no_grad({denom2[,t]$clip_(sEpsilon, ceil)})
          W[,t,2,s2] <- jPr2[,t,2,s2]$clone() / denom2[,t]$clone()
          
          with_no_grad({W[,t,,s2]$clip_(sEpsilon, 1-sEpsilon)}) }
        
        mEta[,t+1,,] <- (W[,t,,]$clone()$unsqueeze(dim=-1) * jEta2[,t,,,]$clone())$sum(dim=3)
        subEta[,t,,,] <- mEta[,t+1,,]$clone()$unsqueeze(dim=-2) - jEta2[,t,,,]$clone()
        subEtaSq[,t,,,,] <- subEta[,t,,,]$clone()$unsqueeze(dim=-1)$matmul(subEta[,t,,,]$clone()$unsqueeze(dim=-2))
        
        mP[,t+1,,,] <- (W[,t,,]$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) * (jP2[,t,,,,]$clone() + subEtaSq[,t,,,,]$clone()))$sum(dim=3) 
        with_no_grad ({
          for (s1 in 1:2) {
            mPEig <- linalg_eigh(mP[,t+1,s1,,]) 
            mPEig[[1]]$real$clip_(lEpsilon, ceil)
            for (row in 1:N) {
              mP[row,t+1,s1,,] <- mPEig[[2]]$real[row,,]$matmul(mPEig[[1]]$real[row,]$diag())$matmul(mPEig[[2]]$real[row,,]$transpose(1, 2)) } } }) }
      
      # aggregated (summed) likelihood at each optimization step
      loss <- -mLik[,]$sum()
      sumLik[iter] <- as.numeric(-loss)
      
      # stopping criterion
      crit <- ifelse(abs(sumLik[iter][[1]] - sumLik[1][[1]]) > sEpsilon, (sumLik[iter][[1]] - sumLik[iter-1][[1]]) / abs(sumLik[iter][[1]] - sumLik[1][[1]]), 0)
      # add count if the new sumLik does not beat the best score
      count <- ifelse(crit < .01, count + 1, 0)
      
      cat('   sum likelihood = ', sumLik[iter][[1]], '\n')
      plot(unlist(sumLik), xlab='optimization step', ylab='sum likelihood', type='b')
      
      if (count == 3 || iter == maxIter) {
        print('   stopping criterion is met')
        
        # switch off the gradient tracking
        with_no_grad ({
          for (var in 1:length(theta)) {theta[[var]]$requires_grad_(requires_grad=FALSE)} })
        break
        
      } else if (sumLikBest < sumLik[iter][[1]]) {
        with_no_grad (thetaBest <- as.list(theta))
        sumLikBest <- sumLik[iter][[1]] } 
      
      # backward propagation
      loss$backward() 
      
      # store gradients
      grad <- list()
      
      with_no_grad ({
        for (var in 1:length(theta)) {
          
          grad[[var]] <- theta[[var]]$grad
          
          # run adam function defined above
          # initialize moment estimates
          if (iter == 1) {m[[var]] <- v[[var]] <- torch_zeros_like(grad[[var]])} 
          
          # update moment estimates
          m[[var]] <- betas[1] * m[[var]] + (1 - betas[1]) * grad[[var]]
          v[[var]] <- betas[2] * v[[var]] + (1 - betas[2]) * grad[[var]]**2 
          
          # update bias corrected moment estimates
          m_hat[[var]] <- m[[var]] / (1 - betas[1]**iter)
          v_hat[[var]] <- v[[var]] / (1 - betas[2]**iter) 
          
          denom <- sqrt(v_hat[[var]]) + sEpsilon
          denom[denom < sEpsilon] <- sEpsilon 
          
          # switch off the gradient tracking
          theta[[var]]$requires_grad_(requires_grad=FALSE)
          # Update parameters using Adam update rule
          theta[[var]]$sub_(lr * m_hat[[var]] / denom) } })
      
      # switch off the gradient tracking
      B11 <- torch_tensor(theta$B11)
      B12 <- torch_tensor(theta$B12)
      B21 <- torch_tensor(theta$B21)
      B22 <- torch_tensor(theta$B22)
      B31 <- torch_tensor(theta$B31)
      B32 <- torch_tensor(theta$B32)
      gamma11 <- torch_tensor(theta$gamma11)
      gamma21 <- torch_tensor(theta$gamma21)
      Q1 <- torch_tensor(theta$Q1)
      Q2 <- torch_tensor(theta$Q2)
      R1 <- torch_tensor(theta$R1)
      R2 <- torch_tensor(theta$R2)
      
      iter <- iter + 1 } }) }