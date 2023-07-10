# install.packages('torch')
library(torch)
# install.packages('reticulate')
library(reticulate)

# number of parameter initialization
nInit <- 3
# max number of iterations
maxIter <- 300
# a very small number
epsilon <- 1e-6
# a very large number
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

####################
# standardize data #
####################
y1Mean <- colMeans(y1[,1,], na.rm=TRUE)
y1Sd <- sqrt(diag(var(y1[,1,], na.rm=TRUE)))
for (o in 1:No1) {
  y1[,,o] <- ifelse(y1Sd[o]==0, y1[,,o] - y1Mean[o], (y1[,,o] - y1Mean[o]) / y1Sd[o]) }

# y2Mean <- colMeans(y2[,1,], na.rm=TRUE)
# y2Sd <- sqrt(diag(var(y2[,1,], na.rm=TRUE)))
# for (o in 1:No2) {
#   y2[,,o] <- ifelse(y2Sd[o]==0, y2[,,o] - y2Mean[o], (y2[,,o] - y2Mean[o]) / y2Sd[o]) }

eta1Mean <- colMeans(eta1[,1,], na.rm=TRUE)
eta1Sd <- sqrt(diag(var(eta1[,1,], na.rm=TRUE)))
for (f in 1:Nf1) {
  eta1[,,f] <- ifelse(eta1Sd[f]==0, eta1[,,f] - eta1Mean[f], (eta1[,,f] - eta1Mean[f]) / eta1Sd[f]) }

eta2Mean <- mean(eta2, na.rm=TRUE)
eta2Sd <- sd(eta2, na.rm=TRUE)
eta2 <- ifelse(eta2Sd==0, eta2 - eta2Mean, (eta2 - eta2Mean) / eta2Sd)

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
  m <- v <- NULL
  
  # initialize parameters
  a1 <- torch_tensor(rnorm(Nf1))
  a2 <- torch_tensor(rnorm(Nf1))
  B1d <- torch_tensor(runif(Nf1, min=0, max=1))
  B2d <- torch_tensor(runif(Nf1, min=0, max=1))
  C1d <- torch_tensor(runif(Nf1, min=0, max=1))
  C2d <- torch_tensor(runif(Nf1, min=0, max=1))
  D1 <- torch_tensor(rnorm(Nf1))
  D2 <- torch_tensor(rnorm(Nf1))
  k1 <- torch_tensor(rnorm(No1))
  k2 <- torch_tensor(rnorm(No1))
  Lmd1v <- torch_tensor(runif(No1, min=0, max=1))
  Lmd2v <- torch_tensor(runif(No1, min=0, max=1))
  Omega1v <- torch_tensor(runif(No1, min=0, max=1))
  Omega2v <- torch_tensor(runif(No1, min=0, max=1))
  M1 <- torch_tensor(rnorm(No1))
  M2 <- torch_tensor(rnorm(No1))
  alpha_1 <- torch_tensor(rnorm(1))
  alpha_2 <- torch_tensor(rnorm(1))
  alpha1 <- min(alpha_1, alpha_2)$unsqueeze(dim=1)
  alpha2 <- max(alpha_1, alpha_2)$unsqueeze(dim=1)
  beta1 <- torch_tensor(rnorm(Nf1))
  beta2 <- torch_tensor(rnorm(Nf1))
  gamma1 <- torch_tensor(rnorm(1))
  gamma2 <- torch_tensor(rnorm(1))
  rho1 <- torch_tensor(rnorm(Nf1))
  rho2 <- torch_tensor(rnorm(Nf1))
  tau1 <- torch_tensor(rnorm(1))
  tau2 <- torch_tensor(rnorm(1))
  Q1d <- torch_tensor(rchisq(Nf1, df=1))
  Q2d <- torch_tensor(rchisq(Nf1, df=1))
  R1d <- torch_tensor(rchisq(No1, df=1))
  R2d <- torch_tensor(rchisq(No1, df=1))
  
  # try (silent = FALSE, {
  with_detect_anomaly ({
    while (count <= 3 && iter <= maxIter) {
      cat('   optimization step: ', as.numeric(iter), '\n')
      
      a1$requires_grad_()
      a2$requires_grad_()
      a <- list(a1, a2)
      B1d$requires_grad_()
      B2d$requires_grad_()
      B1 <- torch_diag(B1d)
      B2 <- torch_diag(B2d)
      B <- list(B1, B2)
      C1d$requires_grad_()
      C2d$requires_grad_()
      C1 <- torch_diag(C1d)
      C2 <- torch_diag(C2d)
      C <- list(C1, C2)
      D1$requires_grad_()
      D2$requires_grad_()
      D <- list(D1, D2)
      k1$requires_grad_()
      k2$requires_grad_()
      k <- list(k1, k2)
      Lmd1v$requires_grad_()
      Lmd2v$requires_grad_()
      Lmd1 <- Lmd2 <- torch_full(c(No1,Nf1), 0)
      Lmd1[1:3,1] <- Lmd1v[1:3]; Lmd1[4:5,2] <- Lmd1v[4:5]
      Lmd1[6:7,3] <- Lmd1v[6:7]; Lmd1[8:9,4] <- Lmd1v[8:9]
      Lmd1[10:11,5] <- Lmd1v[10:11]; Lmd1[12:14,6] <- Lmd1v[12:14]
      Lmd1[15:17,7] <- Lmd1v[15:17]
      Lmd2[1:3,1] <- Lmd2v[1:3]; Lmd2[4:5,2] <- Lmd2v[4:5]
      Lmd2[6:7,3] <- Lmd2v[6:7]; Lmd2[8:9,4] <- Lmd2v[8:9]
      Lmd2[10:11,5] <- Lmd2v[10:11]; Lmd2[12:14,6] <- Lmd2v[12:14]
      Lmd2[15:17,7] <- Lmd2v[15:17]
      Lmd <- list(Lmd1, Lmd2)
      Omega1v$requires_grad_()
      Omega2v$requires_grad_()
      Omega1 <- Omega2 <- torch_full(c(No1,Nf1), 0)
      Omega1[1:3,1] <- Omega1v[1:3]; Omega1[4:5,2] <- Omega1v[4:5]
      Omega1[6:7,3] <- Omega1v[6:7]; Omega1[8:9,4] <- Omega1v[8:9]
      Omega1[10:11,5] <- Omega1v[10:11]; Omega1[12:14,6] <- Omega1v[12:14]
      Omega1[15:17,7] <- Omega1v[15:17]
      Omega2[1:3,1] <- Omega2v[1:3]; Omega2[4:5,2] <- Omega2v[4:5]
      Omega2[6:7,3] <- Omega2v[6:7]; Omega2[8:9,4] <- Omega2v[8:9]
      Omega2[10:11,5] <- Omega2v[10:11]; Omega2[12:14,6] <- Omega2v[12:14]
      Omega2[15:17,7] <- Omega2v[15:17]
      Omega <- list(Omega1, Omega2)
      M1$requires_grad_()
      M2$requires_grad_()
      M <- list(M1, M2)
      alpha1$requires_grad_()
      alpha2$requires_grad_()
      alpha <- list(alpha1, alpha2)
      beta1$requires_grad_()
      beta2$requires_grad_()
      beta <- list(beta1, beta2)
      gamma1$requires_grad_()
      gamma2$requires_grad_()
      gamma <- list(gamma1, gamma2)
      rho1$requires_grad_()
      rho2$requires_grad_()
      rho <- list(rho1, rho2)
      tau1$requires_grad_()
      tau2$requires_grad_()
      tau <- list(tau1, tau2)
      Q1d$requires_grad_()
      Q2d$requires_grad_()
      Q1 <- Q1d$diag()
      Q2 <- Q2d$diag()
      Q <- list(Q1, Q2)
      R1d$requires_grad_()
      R2d$requires_grad_()
      R1 <- R1d$diag()
      R2 <- R2d$diag()
      R <- list(R1, R2)
      theta <- list(a1=a1, a2=a2, B1d=B1d, B2d=B2d, C1d=C1d, C2d=C2d, D1=D1, D2=D2, 
                    k1=k1, k2=k2, Lmd1v=Lmd1v, Lmd2v=Lmd2v, Omega1v=Omega1v, Omega2v=Omega2v, M1=M1, M2=M2, 
                    alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, gamma1=gamma1, gamma2=gamma2, rho1=rho1, rho2=rho2, tau1=tau1, tau2=tau2, 
                    Q1d=Q1d, Q2d=Q2d, R1d=R1d, R2d=R2d)
      
      # define variables
      jEta <- torch_full(c(N,Nt,2,2,Nf1), NaN) # Eq.2 (LHS)
      jDelta <- torch_full(c(N,Nt,2,2,Nf1), NaN) # Eq.3 (LHS)
      jP <- jPChol <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN) # Eq.4 (LHS)
      jV <- torch_full(c(N,Nt,2,2,No1), NaN) # Eq.5 (LHS)
      jF <- jFChol <- torch_full(c(N,Nt,2,2,No1,No1), NaN) # Eq.6 (LHS)
      jEta2 <- torch_full(c(N,Nt,2,2,Nf1), NaN) # Eq.7 (LHS)
      jP2 <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN) # Eq.8 (LHS)
      mEta <- torch_full(c(N,Nt+1,2,Nf1), NaN) # Eq.9-1 (LHS)
      mP <- torch_full(c(N,Nt+1,2,Nf1,Nf1), NaN) # Eq.9-2 (LHS)
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
      mP[,1,,,] <- 0; mP[,1,,,]$add_(1e2 * torch_eye(Nf1)) 
      
      # initialize P(s'|eta_0)
      mPr[,1] <- epsilon 
      
      #######################
      # extended Kim filter #
      #######################
      for (t in 1:Nt) { 
        if (t%%10==0) {cat('   t=', t, '\n')}
        # cat('      t=', t, '\n') 
        
        # Kalman Filter
        for (s1 in 1:2) {
          # Eq.2
          jEta[,t,s1,,] <- a[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) + 
            mEta[,t,,]$clone()$matmul(B[[s1]]) + 
            mEta[,t,,]$clone()$matmul(C[[s1]]) * 
            eta2$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) + 
            x[,t]$clone()$outer(D[[s1]])$unsqueeze(dim=2) 
          jEta[,t,s1,,]$clip_(-ceil, ceil)

          # Eq.3
          jDelta[,t,s1,,] <- eta1[,t,]$clone()$unsqueeze(dim=2) - jEta[,t,s1,,]$clone() 
          jDelta[,t,s1,,]$clip_(-ceil, ceil)
          
          # Eq.4
          jP[,t,s1,,,] <- mP[,t,,,]$clone()$matmul(B[[s1]])$matmul(B[[s1]]$transpose(1, 2)) + 
            Q[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) 
          with_no_grad ({ 
            jP[,t,s1,,,] <- (jP[,t,s1,,,] + jP[,t,s1,,,]$transpose(3, 4)) / 2
            jP[,t,s1,,,]$clip_(-ceil, ceil)
            jPEig <- linalg_eigh(jP[,t,s1,,,])
            jPEig[[1]]$real$clip_(epsilon, ceil)
            for (row in 1:N) {
              for (s2 in 1:2) {
                jP[row,t,s1,s2,,] <- jPEig[[2]]$real[row,s2,,]$matmul(jPEig[[1]]$real[row,s2,]$diag())$matmul(jPEig[[2]]$real[row,s2,,]$transpose(1, 2))
                while (as.numeric(jP[row,t,s1,s2,,]$det()) < epsilon) {
                  jP[row,t,s1,s2,,]$add_(2e-1 * torch_eye(Nf1)) } } } }) 
        
          # Eq.5
          jV[,t,s1,,] <- y1[,t,]$clone()$unsqueeze(dim=2) -
            (k[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) + 
               jEta[,t,s1,,]$clone()$matmul(Lmd[[s1]]$transpose(1, 2)) + 
               jEta[,t,s1,,]$clone()$matmul(Omega[[s1]]$transpose(1, 2)) * 
               eta2$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) + 
               x[,t]$clone()$outer(M[[s1]])$unsqueeze(dim=2))        
          jV[,t,s1,,]$clip_(-ceil, ceil)
          
          # Eq.6
          jF[,t,s1,,,] <- Lmd[[s1]]$matmul(jP[,t,s1,,,]$clone())$matmul(Lmd[[s1]]$transpose(1, 2)) + 
            R[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) 
          jF[,t,s1,,,]$clip_(-ceil, ceil)
          with_no_grad ({
            jF[,t,s1,,,] <- (jF[,t,s1,,,] + jF[,t,s1,,,]$transpose(3, 4)) / 2
            jFEig <- linalg_eigh(jF[,t,s1,,,])
            jFEig[[1]]$real$clip_(epsilon, ceil)
            for (row in 1:N) {
              for (s2 in 1:2) {
                jF[row,t,s1,s2,,] <- jFEig[[2]]$real[row,s2,,]$matmul(jFEig[[1]]$real[row,s2,]$diag())$matmul(jFEig[[2]]$real[row,s2,,]$transpose(1, 2))
                while (as.numeric(jF[row,t,s1,s2,,]$det()) < epsilon) {
                  jF[row,t,s1,s2,,]$add_(5e-1 * torch_eye(No1)) } } } }) 
        
          # kalman gain function
          KG[,t,s1,,,] <- jP[,t,s1,,,]$clone()$matmul(Lmd[[s1]]$transpose(1, 2))$matmul(linalg_inv_ex(jF[,t,s1,,,]$clone())$inverse)
          KG[,t,s1,,,]$clip_(-ceil, ceil)
          
          for (s2 in 1:2) {
            # Eq.7
            jEta2[,t,s1,s2,] <- jEta[,t,s1,s2,]$clone() + KG[,t,s1,s2,,]$clone()$matmul(jV[,t,s1,s2,]$clone()$unsqueeze(dim=-1))$squeeze()
            jEta2[,t,s1,s2,]$clip_(-ceil, ceil)
            I_KGLmd[,t,s1,s2,,] <- torch_eye(Nf1)$unsqueeze(dim=1) - KG[,t,s1,s2,,]$clone()$matmul(Lmd[[s1]])
            I_KGLmd[,t,s1,s2,,]$clip_(-ceil, ceil)
            
            # Eq.9
            jP2[,t,s1,s2,,] <- I_KGLmd[,t,s1,s2,,]$clone()$matmul(jP[,t,s1,s2,,]$clone())$matmul(I_KGLmd[,t,s1,s2,,]$clone()$transpose(2, 3)) + 
              KG[,t,s1,s2,,]$clone()$matmul(R[[s1]])$matmul(KG[,t,s1,s2,,]$clone()$transpose(2, 3))
            jP2[,t,s1,s2,,]$clip_(-ceil, ceil)
            
            with_no_grad ({
              jP2Eig <- linalg_eigh(jP2[,t,s1,s2,,]) 
              jP2Eig[[1]]$real$clip_(epsilon, ceil)
              for (row in 1:N) {
                jP2[row,t,s1,s2,,] <- jP2Eig[[2]]$real[s2,,]$matmul(jP2Eig[[1]]$real[s2,]$diag())$matmul(jP2Eig[[2]]$real[s2,,]$transpose(1, 2)) 
                while (as.numeric(jP2[row,t,s1,s2,,]$det()) < epsilon) {jP2[row,t,s1,s2,,]$add_(2e-1 * torch_eye(Nf1))} } }) 
            
            # joint likelihood f(eta_{t}|s,s',eta_{t-1})
            # Eq.12
            jLik[,t,s1,s2] <- (2*pi)**(-Nf1/2) * jP[,t,s1,s2,,]$clone()$det()**(-1) * 
              (-.5 * jDelta[,t,s1,s2,]$clone()$unsqueeze(dim=2)$matmul(linalg_inv_ex(jP[,t,s1,s2,,]$clone())$inverse)$matmul(jDelta[,t,s1,s2,]$clone()$unsqueeze(dim=-1))$squeeze()$squeeze())$exp()
            jLik[,t,s1,s2]$clip_(0, ceil)
            jLik[,t,s1,s2]$retain_grad() } }
        
        # transition probability P(s|s',eta_{t-1})  
        if (t == 1) {
          tPr[,t,1] <- (alpha[[1]] + gamma[[1]] * eta2$clone())$sigmoid()
          tPr[,t,2] <- (alpha[[2]] + gamma[[2]] * eta2$clone())$sigmoid()
          
        } else {
          tPr[,t,1] <- (alpha[[1]] + eta1[,t-1,]$clone()$matmul(beta[[1]]) + gamma[[1]] * eta2$clone() + eta1[,t-1,]$clone()$matmul(rho[[1]]) * eta2$clone() + tau[[1]] * x[,t-1]$clone())$sigmoid() 
          tPr[,t,2] <- (alpha[[2]] + eta1[,t-1,]$clone()$matmul(beta[[2]]) + gamma[[2]] * eta2$clone() + eta1[,t-1,]$clone()$matmul(rho[[2]]) * eta2$clone() + tau[[2]] * x[,t-1]$clone())$sigmoid() }
        
        jPr[,t,2,2] <- tPr[,t,2]$clone() * mPr[,t]$clone()
        jPr[,t,2,1] <- tPr[,t,1]$clone() * (1-mPr[,t]$clone())
        jPr[,t,1,2] <- (1-tPr[,t,2]$clone()) * mPr[,t]$clone()
        jPr[,t,1,1] <- (1-tPr[,t,1]$clone()) * (1-mPr[,t]$clone()) 
        div <- jPr[,t,,]$sum(dim=c(2,3))
        div$clip_(epsilon, ceil)
        jPr[,t,,]$div_(div$unsqueeze(dim=-1)$unsqueeze(dim=-1))

        # marginal likelihood function f(eta_{t}|eta_{t-1})
        mLik[,t] <- (jLik[,t,,]$clone() * jPr[,t,,]$clone())$sum(dim=c(2,3))
        mLik[,t]$retain_grad()
        
        # (updated) joint probability P(s,s'|eta_{t})
        jPr2[,t,,] <- jLik[,t,,]$clone() * jPr[,t,,]$clone() / mLik[,t]$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) 
        div <- jPr2[,t,,]$sum(dim=c(2,3))
        div$clip_(epsilon, ceil)
        jPr2[,t,,]$div_(div$unsqueeze(dim=-1)$unsqueeze(dim=-1))  
      
        # marginal probability P(s|eta_{t})
        mPr[,t+1] <- jPr2[,t,2,]$clone()$sum(dim=2)
        
        # step 11: collapsing procedure
        for (s2 in 1:2) { 
          denom1[,t] <- 1 - mPr[,t+1]$clone()
          denom1[,t]$clip_(epsilon, ceil)
          W[,t,1,s2] <- jPr2[,t,1,s2]$clone() / denom1[,t]$clone()
          
          denom2[,t] <- mPr[,t+1]$clone()
          denom2[,t]$clip_(epsilon, ceil)
          W[,t,2,s2] <- jPr2[,t,2,s2]$clone() / denom2[,t]$clone()
          
          W[,t,,s2]$clip_(epsilon, 1-epsilon) }
        
        mEta[,t+1,,] <- (W[,t,,]$clone()$unsqueeze(dim=-1) * jEta2[,t,,,]$clone())$sum(dim=3)
        mEta[,t+1,,]$clip_(-ceil, ceil)
        
        subEta[,t,,,] <- mEta[,t+1,,]$clone()$unsqueeze(dim=-2) - jEta2[,t,,,]$clone()
        subEta[,t,,,]$clip_(-ceil, ceil)
        
        subEtaSq[,t,,,,] <- subEta[,t,,,]$clone()$unsqueeze(dim=-1)$matmul(subEta[,t,,,]$clone()$unsqueeze(dim=-2))
        subEtaSq[,t,,,,]$clip_(-ceil, ceil)
        with_no_grad(subEtaSq[,t,,,,] <- (subEtaSq[,t,,,,] + subEtaSq[,t,,,,]$transpose(4, 5)) / 2)
          
        # store the pair (s,s') as data frame 
        jS <- expand.grid(s1=c(1,2), s2=c(1,2))
          
        with_no_grad ({
          for (js in 1:nrow(jS)) {
            s1 <- jS$s1[js]; s2 <- jS$s2[js]
            subEtaSqEig <- linalg_eigh(subEtaSq[,t,s1,s2,,]) 
            subEtaSqEig[[1]]$real$clip_(epsilon, ceil)
            for (row in 1:N) {
              subEtaSq[row,t,s1,s2,,] <- subEtaSqEig[[2]]$real[row,,]$matmul(subEtaSqEig[[1]]$real[row,]$diag())$matmul(subEtaSqEig[[2]]$real[row,,]$transpose(1, 2)) 
              while (as.numeric(subEtaSq[row,t,s1,s2,,]$det()) < epsilon) {
                subEtaSq[row,t,s1,s2,,]$add_(2e-1 * torch_eye(Nf1)) } } } })
        
        mP[,t+1,,,] <- (W[,t,,]$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) * (jP2[,t,,,,]$clone() + subEtaSq[,t,,,,]$clone()))$sum(dim=3) 
        mP[,t+1,,,]$clip_(-ceil, ceil)
        with_no_grad ({
          mP[,t+1,,,] <- (mP[,t+1,,,] + mP[,t+1,,,]$transpose(3, 4)) / 2
          for (s1 in 1:2) {
            mPEig <- linalg_eigh(mP[,t+1,s1,,]) 
            mPEig[[1]]$real$clip_(epsilon, ceil)
            for (row in 1:N) {
              mP[row,t+1,s1,,] <- mPEig[[2]]$real[row,,]$matmul(mPEig[[1]]$real[row,]$diag())$matmul(mPEig[[2]]$real[row,,]$transpose(1, 2))
              while (as.numeric(mP[row,t+1,s1,,]$det()) < epsilon) {
                mP[row,t+1,s1,,]$add_(2e-1 * torch_eye(Nf1)) } } } }) } 
      
      # aggregated (summed) likelihood at each optimization step
      loss <- -mLik[,]$sum()
      sumLik[iter] <- as.numeric(-loss$clone())
      
      # stopping criterion
      crit <- ifelse(abs(sumLik[iter][[1]] - sumLik[1][[1]]) > epsilon, (sumLik[iter][[1]] - sumLik[iter-1][[1]]) / abs(sumLik[iter][[1]] - sumLik[1][[1]]), 0)
      # add count if the new sumLik does not beat the best score
      count <- ifelse(crit < 5e-2, count + 1, 0)
      
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
      
      # run adam function defined above
      # initialize moment estimates
      with_no_grad ({
        if (is.null(m) || is.null(v)) {m <- v <- rep(0, length(torch_cat(theta)))} })
      
      # backward propagation
      loss$backward() 
      
      # store gradients
      grad <- list()
      with_no_grad ({
        for (var in 1:length(theta)) {grad <- append(grad, theta[[var]]$grad)} })
      grad <- torch_cat(grad)
      
      # update moment estimates
      m <- betas[1] * m + (1 - betas[1]) * grad
      v <- betas[2] * v + (1 - betas[2]) * grad**2
      
      # update bias corrected moment estimates
      m_hat <- m / (1 - betas[1]**iter)
      v_hat <- v / (1 - betas[2]**iter)
      
      denom <- sqrt(v_hat) + epsilon
      denom[denom < epsilon] <- epsilon
      
      index <- 0
      for (var in 1:length(theta)) {
        begin <- index + 1
        end <- index + length(theta[[var]])
        # switch off the gradient tracking
        theta[[var]]$requires_grad_(requires_grad=FALSE)
        # Update parameters using Adam update rule
        theta[[var]]$sub_(lr * m_hat[begin:end] / denom[begin:end])
        index <- end }
      
      # switch off the gradient tracking
      a1 <- torch_tensor(theta$a1)
      a2 <- torch_tensor(theta$a2)
      B1d <- torch_tensor(theta$B1d)
      B2d <- torch_tensor(theta$B2d)
      C1d <- torch_tensor(theta$C1d)
      C2d <- torch_tensor(theta$C2d)
      D1 <- torch_tensor(theta$D1)
      D2 <- torch_tensor(theta$D2)
      k1 <- torch_tensor(theta$k1)
      k2 <- torch_tensor(theta$k2)
      Lmd1v <- torch_tensor(theta$Lmd1v)
      Lmd2v <- torch_tensor(theta$Lmd2v)
      Omega1v <- torch_tensor(theta$Omega1v)
      Omega2v <- torch_tensor(theta$Omega2v)
      M1 <- torch_tensor(theta$M1)
      M2 <- torch_tensor(theta$M2)
      alpha1 <- torch_tensor(theta$alpha1)
      alpha2 <- torch_tensor(theta$alpha2)
      beta1 <- torch_tensor(theta$beta1)
      beta2 <- torch_tensor(theta$beta2)
      gamma1 <- torch_tensor(theta$gamma1)
      gamma2 <- torch_tensor(theta$gamma2)
      rho1 <- torch_tensor(theta$rho1)
      rho2 <- torch_tensor(theta$rho2)
      tau1 <- torch_tensor(theta$tau1)
      tau2 <- torch_tensor(theta$tau2)
      Q1d <- torch_tensor(theta$Q1d)
      Q2d <- torch_tensor(theta$Q2d)
      R1d <- torch_tensor(theta$R1d)
      R2d <- torch_tensor(theta$R2d) 
      
      iter <- iter + 1 } }) }