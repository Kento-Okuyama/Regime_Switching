set.seed(42)

nInit <- 30
maxIter <- 500
lEpsilon <- 1e-3
sEpsilon <- 1e-8
stopCrit <- 1e-4
lr <- 1e-3
epsilon <- 1e-8
betas <- c(.9, .999)

y1 <- df$y1
y2 <- df$y2
N <- df$N
Nt <- df$Nt
O1 <- df$O1
O2 <- df$O2
L1 <- df$L1

#####################
# Measurement model #
#####################
model_cfa <- '
# latent variables
lv =~ ov1 + ov2 + ov3 '

y2_df <- as.data.frame(y2)
colnames(y2_df) <- c('ov1', 'ov2', 'ov3')
fit_cfa <- cfa(model_cfa, data=y2_df)
eta2_score <- lavPredict(fit_cfa, method='Bartlett')
eta2 <- as.array(eta2_score[,1])

y1 <- torch_tensor(y1)
eta2 <- torch_tensor(eta2)

sumLikBest <- 0

init <- 0
for (init in 1:nInit) {
  cat('Init step ', init, '\n')
  iter <- 1
  count <- 0
  m <- v <- m_hat <- v_hat <- list()
  
  # initialize parameters
  B11 <- torch_tensor(abs(rnorm(L1, 0, .3)))
  B12 <- torch_tensor(-abs(rnorm(L1, 0, .2))) # B11 - constant
  B21d <- torch_tensor(runif(L1, .6, 1)) 
  B22d <- torch_tensor(runif(L1, .2, .6)) # B21d - constant
  B31 <- torch_tensor(abs(rnorm(L1, 0, .15)))
  B32 <- torch_tensor(-abs(rnorm(L1, 0, .1))) # B31 - constant
  Lmdd <- torch_tensor(runif(O1*L1, .5, 1.5)) # fixed all to 1
  Lmdd[c(1,8)] <- 1; Lmdd[c(2,4,6,7,9,11)] <- 0 
  Qd <- torch_tensor(abs(rnorm(L1, 0, .3))) # fixed
  Rd <- torch_tensor(abs(rnorm(O1, 0, .8))) # fixed
  gamma1 <- torch_tensor(runif(1, 2, 5)) # fixed to 3
  gamma2 <- torch_tensor(abs(rnorm(L1, 0, 1)))
  gamma3 <- torch_tensor(0) # not estimated
  gamma4 <- torch_tensor(rep(0, L1)) # not estimated
  
  # with_detect_anomaly ({
  try (silent=FALSE, {
    while (count <=3 && iter <= maxIter) {
      cat('   optim step ', iter, '\n')
      
      B11$requires_grad_()
      B12$requires_grad_()
      B21d$requires_grad_()
      B22d$requires_grad_()
      B31$requires_grad_()
      B32$requires_grad_()
      Lmdd$requires_grad_()
      Qd$requires_grad_()
      Rd$requires_grad_()
      gamma1$requires_grad_()
      gamma2$requires_grad_()
      
      theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                    Lmdd=Lmdd, Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2)
      
      jEta <- torch_full(c(N,Nt+1,2,2,L1), NaN)
      jP <- torch_full(c(N,Nt+1,2,2,L1,L1), NaN)
      jV <- torch_full(c(N,Nt,2,2,O1), NaN)
      jF <- torch_full(c(N,Nt,2,2,O1,O1), NaN)
      jEta2 <- torch_full(c(N,Nt,2,2,L1), NaN)
      jP2 <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
      mEta <- torch_full(c(N,Nt+1,2,L1), NaN)
      mP <- torch_full(c(N,Nt+1,2,L1,L1), NaN)
      W <- torch_full(c(N,Nt,2,2), NaN)
      jPr <- torch_full(c(N,Nt+1,2,2), NaN)
      mLik <- torch_full(c(N,Nt), NaN)
      jPr2 <- torch_full(c(N,Nt,2,2), NaN)
      mPr <- torch_full(c(N,Nt+1), NaN)
      jLik <- torch_full(c(N,Nt,2,2), NaN)
      tPr <- torch_full(c(N,Nt+1,2), NaN)
      KG <- torch_full(c(N,Nt,2,2,L1,O1), NaN)
      I_KGLmd <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
      denom1 <- torch_full(c(N,Nt), NaN)
      denom2 <- torch_full(c(N,Nt), NaN)
      subEta <- torch_full(c(N,Nt,2,2,L1), NaN)
      subEtaSq <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
      eta1_pred <- torch_full(c(N,Nt+1,L1), NaN)
      P_pred <- torch_full(c(N,Nt+1,L1,L1), NaN)
      
      mEta[,1,,] <- 0
      mP[,1,,,] <- torch_eye(L1)
      mPr[,1] <- 1 - lEpsilon
      W[,,1,1] <- 1
      
      B21 <- B21d$clone()$diag()
      B22 <- B22d$clone()$diag()
      Lmd <- Lmdd$clone()$reshape(c(O1, L1))
      Lmd1 <- Lmd$clone()
      Lmd2 <- Lmd$clone()
      Q1 <- Qd$clone()$diag()
      Q2 <- Qd$clone()$diag()
      R1 <- Rd$clone()$diag()
      R2 <- Rd$clone()$diag()
      
      for (t in 1:Nt) {
        
        #################
        # Kalman filter #
        #################

        jEta[,t,1,1,] <- B11$clone() + mEta[,t,1,]$clone()$matmul(B21$clone()) + eta2$clone()$outer(B31$clone())
        jEta[,t,2,1,] <- B12$clone() + mEta[,t,1,]$clone()$matmul(B22$clone()) + eta2$clone()$outer(B32$clone())
        jEta[,t,2,2,] <- B12$clone() + mEta[,t,2,]$clone()$matmul(B22$clone()) + eta2$clone()$outer(B32$clone())
        
        jP[,t,1,1,,] <- B21$clone()$matmul(mP[,t,1,,]$clone())$matmul(B21$clone()) + Q1$clone()
        jP[,t,2,1,,] <- B22$clone()$matmul(mP[,t,1,,]$clone())$matmul(B22$clone()) + Q2$clone()
        jP[,t,2,2,,] <- B22$clone()$matmul(mP[,t,2,,]$clone())$matmul(B22$clone()) + Q2$clone()

        jV[,t,1,1,] <- y1[,t,]$clone() - jEta[,t,1,1,]$clone()$matmul(Lmd1$clone()$transpose(1, 2))
        jV[,t,2,1,] <- y1[,t,]$clone() - jEta[,t,2,1,]$clone()$matmul(Lmd2$clone()$transpose(1, 2))
        jV[,t,2,2,] <- y1[,t,]$clone() - jEta[,t,2,2,]$clone()$matmul(Lmd2$clone()$transpose(1, 2))
        
        jF[,t,1,1,,] <- Lmd1$clone()$matmul(jP[,t,1,1,,]$clone())$matmul(Lmd1$clone()$transpose(1, 2)) + R1$clone()
        jF[,t,2,1,,] <- Lmd2$clone()$matmul(jP[,t,2,1,,]$clone())$matmul(Lmd1$clone()$transpose(1, 2)) + R2$clone()
        jF[,t,2,2,,] <- Lmd2$clone()$matmul(jP[,t,2,2,,]$clone())$matmul(Lmd2$clone()$transpose(1, 2)) + R2$clone()
        
        KG[,t,1,1,,] <- jP[,t,1,1,,]$clone()$matmul(Lmd1$clone()$transpose(1, 2))$matmul(jF[,t,1,1,,]$clone()$cholesky_inverse())
        KG[,t,2,1,,] <- jP[,t,2,1,,]$clone()$matmul(Lmd2$clone()$transpose(1, 2))$matmul(jF[,t,2,1,,]$clone()$cholesky_inverse())
        KG[,t,2,2,,] <- jP[,t,2,2,,]$clone()$matmul(Lmd2$clone()$transpose(1, 2))$matmul(jF[,t,2,2,,]$clone()$cholesky_inverse())
        
        jEta2[,t,1,1,] <- jEta[,t,1,1,]$clone() + KG[,t,1,1,,]$clone()$matmul(jV[,t,1,1,]$clone()$unsqueeze(-1))$squeeze()
        jEta2[,t,2,1,] <- jEta[,t,2,1,]$clone() + KG[,t,2,1,,]$clone()$matmul(jV[,t,2,1,]$clone()$unsqueeze(-1))$squeeze()
        jEta2[,t,2,2,] <- jEta[,t,2,2,]$clone() + KG[,t,2,2,,]$clone()$matmul(jV[,t,2,2,]$clone()$unsqueeze(-1))$squeeze()
        
        I_KGLmd[,t,1,1,,] <- torch_eye(L1) - KG[,t,1,1,,]$clone()$matmul(Lmd1$clone())
        I_KGLmd[,t,2,1,,] <- torch_eye(L1) - KG[,t,2,1,,]$clone()$matmul(Lmd2$clone())
        I_KGLmd[,t,2,2,,] <- torch_eye(L1) - KG[,t,2,2,,]$clone()$matmul(Lmd2$clone())
        
        jP2[,t,1,1,,] <- I_KGLmd[,t,1,1,,]$clone()$matmul(jP[,t,1,1,,]$clone())$matmul(I_KGLmd[,t,1,1,,]$clone()$transpose(2, 3)) +
          KG[,t,1,1,,]$clone()$matmul(R1$clone())$matmul(KG[,t,1,1,,]$clone()$transpose(2, 3))
        jP2[,t,2,1,,] <- I_KGLmd[,t,2,1,,]$clone()$matmul(jP[,t,2,1,,]$clone())$matmul(I_KGLmd[,t,2,1,,]$clone()$transpose(2, 3)) +
          KG[,t,2,1,,]$clone()$matmul(R2$clone())$matmul(KG[,t,2,1,,]$clone()$transpose(2, 3))
        jP2[,t,2,2,,] <- I_KGLmd[,t,2,2,,]$clone()$matmul(jP[,t,2,2,,]$clone())$matmul(I_KGLmd[,t,2,2,,]$clone()$transpose(2, 3)) +
          KG[,t,2,2,,]$clone()$matmul(R2$clone())$matmul(KG[,t,2,2,,]$clone()$transpose(2, 3))
        
        jLik[,t,1,1] <- sEpsilon + (2*pi)**(-O1/2) * jF[,t,1,1,,]$clone()$det()**(-1) *
          (-.5 * jV[,t,1,1,]$clone()$unsqueeze(2)$matmul(jF[,t,1,1,,]$clone()$cholesky_inverse())$matmul(jV[,t,1,1,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
        jLik[,t,2,1] <- sEpsilon + (2*pi)**(-O1/2) * jF[,t,2,1,,]$clone()$det()**(-1) *
          (-.5 * jV[,t,2,1,]$clone()$unsqueeze(2)$matmul(jF[,t,2,1,,]$clone()$cholesky_inverse())$matmul(jV[,t,2,1,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
        jLik[,t,2,2] <- sEpsilon + (2*pi)**(-O1/2) * jF[,t,2,2,,]$clone()$det()**(-1) *
          (-.5 * jV[,t,2,2,]$clone()$unsqueeze(2)$matmul(jF[,t,2,2,,]$clone()$cholesky_inverse())$matmul(jV[,t,2,2,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
        
        ###################
        # Hamilton filter #
        ###################
        
        if (t == 1) {tPr[,t,1] <- (gamma1$clone() + gamma3$clone() * eta2$clone())$sigmoid()$clip(min=lEpsilon, max=1-lEpsilon)
        } else {
          eta1_pred[,t-1,] <- mPr[,t]$clone()$unsqueeze(-1) * mEta[,t,1,]$clone() + (1 - mPr[,t]$clone())$unsqueeze(-1) * mEta[,t,2,]$clone()
          tPr[,t,1] <- (gamma1$clone() + eta1_pred[,t-1,]$clone()$matmul(gamma2$clone()) + gamma3$clone() * eta2$clone() + eta1_pred[,t-1,]$clone()$matmul(gamma4$clone()) * eta2$clone())$sigmoid()$clip(min=lEpsilon, max=1-lEpsilon) }
        
        jPr[,t,1,1] <- tPr[,t,1]$clone() * mPr[,t]$clone()
        jPr[,t,2,1] <- (1 - tPr[,t,1]$clone()) * mPr[,t]$clone()
        jPr[,t,2,2] <- (1 - mPr[,t]$clone())
        jPr[,t,,]$clip_(min=lEpsilon, max=1-lEpsilon)
        
        mLik[,t] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() +
          jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() +
          jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone()
        
        jPr2[,t,1,1] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() / mLik[,t]$clone()
        jPr2[,t,2,1] <- jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() / mLik[,t]$clone()
        jPr2[,t,2,2] <- jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone() / mLik[,t]$clone()
        jPr2[,t,,]$clip_(min=lEpsilon, max=1-lEpsilon)
        
        mPr[,t+1] <- jPr2[,t,1,1]$clone()
        
        W[,t,2,1] <- jPr2[,t,2,1]$clone() / (1 - mPr[,t+1]$clone())
        W[,t,2,2] <- jPr2[,t,2,2]$clone() / (1 - mPr[,t+1]$clone())
        
        mEta[,t+1,1,] <- jEta2[,t,1,1,]$clone()
        mEta[,t+1,2,] <- (W[,t,2,]$clone()$unsqueeze(-1) * jEta2[,t,2,,]$clone())$sum(2)
        
        subEta[,t,1,1,] <- mEta[,t+1,1,]$clone() - jEta2[,t,1,1,]$clone()
        subEta[,t,2,1,] <- mEta[,t+1,2,]$clone() - jEta2[,t,2,1,]$clone()
        subEta[,t,2,2,] <- mEta[,t+1,2,]$clone() - jEta2[,t,2,2,]$clone()
        
        subEtaSq[,t,1,1,,] <- subEta[,t,1,1,]$clone()$unsqueeze(-1)$matmul(subEta[,t,1,1,]$clone()$unsqueeze(-2))
        subEtaSq[,t,2,1,,] <- subEta[,t,2,1,]$clone()$unsqueeze(-1)$matmul(subEta[,t,2,1,]$clone()$unsqueeze(-2))
        subEtaSq[,t,2,2,,] <- subEta[,t,2,2,]$clone()$unsqueeze(-1)$matmul(subEta[,t,2,2,]$clone()$unsqueeze(-2))
        
        mP[,t+1,1,,] <- jP2[,t,1,1,,]$clone() + subEtaSq[,t,1,1,,]$clone()
        mP[,t+1,2,,] <- (W[,t,2,]$clone()$unsqueeze(-1)$unsqueeze(-1) * (jP2[,t,2,,,]$clone() + subEtaSq[,t,2,,,]$clone()))$sum(2) }
      
      eta1_pred[,Nt,] <- mPr[,Nt+1]$clone()$unsqueeze(-1) * mEta[,Nt+1,1,]$clone() + (1 - mPr[,Nt+1]$clone())$unsqueeze(-1) * mEta[,Nt+1,2,]$clone()
      P_pred[,1:Nt,,] <- mPr[,2:(Nt+1)]$clone()$unsqueeze(-1)$unsqueeze(-1) * mP[,2:(Nt+1),1,,]$clone() + (1 - mPr[,2:(Nt+1)]$clone())$unsqueeze(-1)$unsqueeze(-1) * mP[,2:(Nt+1),2,,]$clone()
      
      jEta[,Nt+1,1,1,] <- B11$clone() + mEta[,Nt+1,1,]$clone()$matmul(B21$clone()) + eta2$clone()$outer(B31$clone())
      jEta[,Nt+1,2,1,] <- B12$clone() + mEta[,Nt+1,1,]$clone()$matmul(B22$clone()) + eta2$clone()$outer(B32$clone())
      jEta[,Nt+1,2,2,] <- B12$clone() + mEta[,Nt+1,2,]$clone()$matmul(B22$clone()) + eta2$clone()$outer(B32$clone())
      
      jP[,Nt+1,1,1,,] <- B21$clone()$matmul(mP[,Nt+1,1,,]$clone())$matmul(B21$clone()) + Q1$clone()
      jP[,Nt+1,2,1,,] <- B22$clone()$matmul(mP[,Nt+1,1,,]$clone())$matmul(B22$clone()) + Q2$clone()
      jP[,Nt+1,2,2,,] <- B22$clone()$matmul(mP[,Nt+1,2,,]$clone())$matmul(B22$clone()) + Q2$clone()
      
      tPr[,Nt+1,1] <- (gamma1$clone() + eta1_pred[,Nt,]$clone()$matmul(gamma2$clone()) + gamma3$clone() * eta2$clone() + eta1_pred[,Nt,]$clone()$matmul(gamma4$clone()) * eta2$clone())$sigmoid()$clip(min=lEpsilon, max=1-lEpsilon)
      
      jPr[,Nt+1,1,1] <- tPr[,Nt+1,1]$clone() * mPr[,Nt+1]$clone()
      jPr[,Nt+1,2,1] <- (1 - tPr[,Nt+1,1]$clone()) * mPr[,Nt+1]$clone()
      jPr[,Nt+1,2,2] <- (1 - mPr[,Nt+1]$clone())
      jPr[,Nt+1,,]$clip_(min=lEpsilon, max=1-lEpsilon)
      
      eta1_pred[,Nt+1,] <- jEta[,Nt+1,1,1,]$clone() * jPr[,Nt+1,1,1]$clone()$unsqueeze(-1) + jEta[,Nt+1,2,1,]$clone() * jPr[,Nt+1,2,1]$clone()$unsqueeze(-1) + jEta[,Nt+1,2,2,]$clone() * jPr[,Nt+1,2,2]$clone()$unsqueeze(-1)   
      P_pred[,Nt+1,,] <- jP[,Nt+1,1,1,,]$clone() * jPr[,Nt+1,1,1]$clone()$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,1,,]$clone() * jPr[,Nt+1,2,1]$clone()$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,2,,]$clone() * jPr[,Nt+1,2,2]$clone()$unsqueeze(-1)$unsqueeze(-1)
      
      loss <- -mLik[,1:Nt]$clone()$sum()
      
      if (is.nan(-as.numeric(loss))) {
        print('   error in calculating the sum likelihood')
        with_no_grad ({
          for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
        break }
      
      if (init == 1 && iter == 1) {
        
        theta_list <- list(theta)
        B11_1_list  <- melt(as.matrix(B11$clone()), nrow=1)[melt(as.matrix(B11$clone()), nrow=1)$X1==1,]; B11_1_list$X1 <- as.factor(init+1); B11_1_list$X2 <- iter
        B11_2_list  <- melt(as.matrix(B11$clone()), nrow=1)[melt(as.matrix(B11$clone()), nrow=1)$X1==2,]; B11_2_list$X1 <- as.factor(init+1); B11_2_list$X2 <- iter
        B12_1_list  <- melt(as.matrix(B12$clone()), nrow=1)[melt(as.matrix(B12$clone()), nrow=1)$X1==1,]; B12_1_list$X1 <- as.factor(init+1); B12_1_list$X2 <- iter
        B12_2_list  <- melt(as.matrix(B12$clone()), nrow=1)[melt(as.matrix(B12$clone()), nrow=1)$X1==2,]; B12_2_list$X1 <- as.factor(init+1); B12_2_list$X2 <- iter
        B21d_1_list  <- melt(as.matrix(B21d$clone()), nrow=1)[melt(as.matrix(B21d$clone()), nrow=1)$X1==1,]; B21d_1_list$X1 <- as.factor(init+1); B21d_1_list$X2 <- iter
        B21d_2_list  <- melt(as.matrix(B21d$clone()), nrow=1)[melt(as.matrix(B21d$clone()), nrow=1)$X1==2,]; B21d_2_list$X1 <- as.factor(init+1); B21d_2_list$X2 <- iter
        B22d_1_list  <- melt(as.matrix(B22d$clone()), nrow=1)[melt(as.matrix(B22d$clone()), nrow=1)$X1==1,]; B22d_1_list$X1 <- as.factor(init+1); B22d_1_list$X2 <- iter
        B22d_2_list  <- melt(as.matrix(B22d$clone()), nrow=1)[melt(as.matrix(B22d$clone()), nrow=1)$X1==2,]; B22d_2_list$X1 <- as.factor(init+1); B22d_2_list$X2 <- iter
        B31_1_list  <- melt(as.matrix(B31$clone()), nrow=1)[melt(as.matrix(B31$clone()), nrow=1)$X1==1,]; B31_1_list$X1 <- as.factor(init+1); B31_1_list$X2 <- iter
        B31_2_list  <- melt(as.matrix(B31$clone()), nrow=1)[melt(as.matrix(B31$clone()), nrow=1)$X1==2,]; B31_2_list$X1 <- as.factor(init+1); B31_2_list$X2 <- iter
        B32_1_list  <- melt(as.matrix(B32$clone()), nrow=1)[melt(as.matrix(B32$clone()), nrow=1)$X1==1,]; B32_1_list$X1 <- as.factor(init+1); B32_1_list$X2 <- iter
        B32_2_list  <- melt(as.matrix(B32$clone()), nrow=1)[melt(as.matrix(B32$clone()), nrow=1)$X1==2,]; B32_2_list$X1 <- as.factor(init+1); B32_2_list$X2 <- iter
        Lmdd_1_list  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==1,]; Lmdd_1_list$X1 <- as.factor(init+1); Lmdd_1_list$X2 <- iter
        Lmdd_2_list  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==3,]; Lmdd_2_list$X1 <- as.factor(init+1); Lmdd_2_list$X2 <- iter
        Lmdd_3_list  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==5,]; Lmdd_3_list$X1 <- as.factor(init+1); Lmdd_3_list$X2 <- iter
        Lmdd_4_list  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==8,]; Lmdd_4_list$X1 <- as.factor(init+1); Lmdd_4_list$X2 <- iter
        Lmdd_5_list  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==10,]; Lmdd_5_list$X1 <- as.factor(init+1); Lmdd_5_list$X2 <- iter
        Lmdd_6_list  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==12,]; Lmdd_6_list$X1 <- as.factor(init+1); Lmdd_6_list$X2 <- iter
        Qd_1_list  <- melt(as.matrix(Qd$clone()), nrow=1)[melt(as.matrix(Qd$clone()), nrow=1)$X1==1,]; Qd_1_list$X1 <- as.factor(init+1); Qd_1_list$X2 <- iter
        Qd_2_list  <- melt(as.matrix(Qd$clone()), nrow=1)[melt(as.matrix(Qd$clone()), nrow=1)$X1==2,]; Qd_2_list$X1 <- as.factor(init+1); Qd_2_list$X2 <- iter
        Rd_1_list  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==1,]; Rd_1_list$X1 <- as.factor(init+1); Rd_1_list$X2 <- iter
        Rd_2_list  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==2,]; Rd_2_list$X1 <- as.factor(init+1); Rd_2_list$X2 <- iter
        Rd_3_list  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==3,]; Rd_3_list$X1 <- as.factor(init+1); Rd_3_list$X2 <- iter
        Rd_4_list  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==4,]; Rd_4_list$X1 <- as.factor(init+1); Rd_4_list$X2 <- iter
        Rd_5_list  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==5,]; Rd_5_list$X1 <- as.factor(init+1); Rd_5_list$X2 <- iter
        Rd_6_list  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==6,]; Rd_6_list$X1 <- as.factor(init+1); Rd_6_list$X2 <- iter
        gamma1_list  <- melt(as.matrix(gamma1$clone()), nrow=1); gamma1_list$X1 <- as.factor(init+1); gamma1_list$X2 <- iter
        gamma21_list  <- melt(as.matrix(gamma2$clone()), nrow=1)[melt(as.matrix(gamma2$clone()), nrow=1)$X1==1,]; gamma21_list$X1 <- as.factor(init+1); gamma21_list$X2 <- iter
        gamma22_list  <- melt(as.matrix(gamma2$clone()), nrow=1)[melt(as.matrix(gamma2$clone()), nrow=1)$X1==2,]; gamma22_list$X1 <- as.factor(init+1); gamma22_list$X2 <- iter
        sumLik_list <- sumLik_init <- sumLik_new <- melt(as.matrix(-loss$clone()), nrow=1); sumLik_list$X1 <- as.factor(init+1); sumLik_list$X2 <- iter
        
      } else {
        
        theta_list <- append(theta_list, list(theta))
        B11_1_new  <- melt(as.matrix(B11$clone()), nrow=1)[melt(as.matrix(B11$clone()), nrow=1)$X1==1,]; B11_1_new$X1 <- as.factor(init+1); B11_1_new$X2 <- iter
        B11_1_list  <- rbind(B11_1_list, B11_1_new)
        B11_2_new  <- melt(as.matrix(B11$clone()), nrow=1)[melt(as.matrix(B11$clone()), nrow=1)$X1==2,]; B11_2_new$X1 <- as.factor(init+1); B11_2_new$X2 <- iter
        B11_2_list  <- rbind(B11_2_list, B11_2_new)
        B12_1_new  <- melt(as.matrix(B12$clone()), nrow=1)[melt(as.matrix(B12$clone()), nrow=1)$X1==1,]; B12_1_new$X1 <- as.factor(init+1); B12_1_new$X2 <- iter
        B12_1_list  <- rbind(B12_1_list, B12_1_new)
        B12_2_new  <- melt(as.matrix(B12$clone()), nrow=1)[melt(as.matrix(B12$clone()), nrow=1)$X1==2,]; B12_2_new$X1 <- as.factor(init+1); B12_2_new$X2 <- iter
        B12_2_list  <- rbind(B12_2_list, B12_2_new)
        B21d_1_new  <- melt(as.matrix(B21d$clone()), nrow=1)[melt(as.matrix(B21d$clone()), nrow=1)$X1==1,]; B21d_1_new$X1 <- as.factor(init+1); B21d_1_new$X2 <- iter
        B21d_1_list  <- rbind(B21d_1_list, B21d_1_new)
        B21d_2_new  <- melt(as.matrix(B21d$clone()), nrow=1)[melt(as.matrix(B21d$clone()), nrow=1)$X1==2,]; B21d_2_new$X1 <- as.factor(init+1); B21d_2_new$X2 <- iter
        B21d_2_list  <- rbind(B21d_2_list, B21d_2_new)
        B22d_1_new  <- melt(as.matrix(B22d$clone()), nrow=1)[melt(as.matrix(B22d$clone()), nrow=1)$X1==1,]; B22d_1_new$X1 <- as.factor(init+1); B22d_1_new$X2 <- iter
        B22d_1_list  <- rbind(B22d_1_list, B22d_1_new)
        B22d_2_new  <- melt(as.matrix(B22d$clone()), nrow=1)[melt(as.matrix(B22d$clone()), nrow=1)$X1==2,]; B22d_2_new$X1 <- as.factor(init+1); B22d_2_new$X2 <- iter
        B22d_2_list  <- rbind(B22d_2_list, B22d_2_new)
        B31_1_new  <- melt(as.matrix(B31$clone()), nrow=1)[melt(as.matrix(B31$clone()), nrow=1)$X1==1,]; B31_1_new$X1 <- as.factor(init+1); B31_1_new$X2 <- iter
        B31_1_list  <- rbind(B31_1_list, B31_1_new)
        B31_2_new  <- melt(as.matrix(B31$clone()), nrow=1)[melt(as.matrix(B31$clone()), nrow=1)$X1==2,]; B31_2_new$X1 <- as.factor(init+1); B31_2_new$X2 <- iter
        B31_2_list  <- rbind(B31_2_list, B31_2_new)
        B32_1_new  <- melt(as.matrix(B32$clone()), nrow=1)[melt(as.matrix(B32$clone()), nrow=1)$X1==1,]; B32_1_new$X1 <- as.factor(init+1); B32_1_new$X2 <- iter
        B32_1_list  <- rbind(B32_1_list, B32_1_new)
        B32_2_new  <- melt(as.matrix(B32$clone()), nrow=1)[melt(as.matrix(B32$clone()), nrow=1)$X1==2,]; B32_2_new$X1 <- as.factor(init+1); B32_2_new$X2 <- iter
        B32_2_list  <- rbind(B32_2_list, B32_2_new)
        Lmdd_1_new  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==1,]; Lmdd_1_new$X1 <- as.factor(init+1); Lmdd_1_new$X2 <- iter
        Lmdd_1_list  <- rbind(Lmdd_1_list, Lmdd_1_new)
        Lmdd_2_new  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==3,]; Lmdd_2_new$X1 <- as.factor(init+1); Lmdd_2_new$X2 <- iter
        Lmdd_2_list  <- rbind(Lmdd_2_list, Lmdd_2_new)
        Lmdd_3_new  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==5,]; Lmdd_3_new$X1 <- as.factor(init+1); Lmdd_3_new$X2 <- iter
        Lmdd_3_list  <- rbind(Lmdd_3_list, Lmdd_3_new)
        Lmdd_4_new  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==8,]; Lmdd_4_new$X1 <- as.factor(init+1); Lmdd_4_new$X2 <- iter
        Lmdd_4_list  <- rbind(Lmdd_4_list, Lmdd_4_new)
        Lmdd_5_new  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==10,]; Lmdd_5_new$X1 <- as.factor(init+1); Lmdd_5_new$X2 <- iter
        Lmdd_5_list  <- rbind(Lmdd_5_list, Lmdd_5_new)
        Lmdd_6_new  <- melt(as.matrix(Lmdd$clone()), nrow=1)[melt(as.matrix(Lmdd$clone()), nrow=1)$X1==12,]; Lmdd_6_new$X1 <- as.factor(init+1); Lmdd_6_new$X2 <- iter
        Lmdd_6_list  <- rbind(Lmdd_6_list, Lmdd_6_new)
        Qd_1_new  <- melt(as.matrix(Qd$clone()), nrow=1)[melt(as.matrix(Qd$clone()), nrow=1)$X1==1,]; Qd_1_new$X1 <- as.factor(init+1); Qd_1_new$X2 <- iter
        Qd_1_list  <- rbind(Qd_1_list, Qd_1_new)
        Qd_2_new  <- melt(as.matrix(Qd$clone()), nrow=1)[melt(as.matrix(Qd$clone()), nrow=1)$X1==2,]; Qd_2_new$X1 <- as.factor(init+1); Qd_2_new$X2 <- iter
        Qd_2_list  <- rbind(Qd_2_list, Qd_2_new)
        Rd_1_new  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==1,]; Rd_1_new$X1 <- as.factor(init+1); Rd_1_new$X2 <- iter
        Rd_1_list  <- rbind(Rd_1_list, Rd_1_new)
        Rd_2_new  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==2,]; Rd_2_new$X1 <- as.factor(init+1); Rd_2_new$X2 <- iter
        Rd_2_list  <- rbind(Rd_2_list, Rd_2_new)
        Rd_3_new  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==3,]; Rd_3_new$X1 <- as.factor(init+1); Rd_3_new$X2 <- iter
        Rd_3_list  <- rbind(Rd_3_list, Rd_3_new)
        Rd_4_new  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==4,]; Rd_4_new$X1 <- as.factor(init+1); Rd_4_new$X2 <- iter
        Rd_4_list  <- rbind(Rd_4_list, Rd_4_new)
        Rd_5_new  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==5,]; Rd_5_new$X1 <- as.factor(init+1); Rd_5_new$X2 <- iter
        Rd_5_list  <- rbind(Rd_5_list, Rd_5_new)
        Rd_6_new  <- melt(as.matrix(Rd$clone()), nrow=1)[melt(as.matrix(Rd$clone()), nrow=1)$X1==6,]; Rd_6_new$X1 <- as.factor(init+1); Rd_6_new$X2 <- iter
        Rd_6_list  <- rbind(Rd_6_list, Rd_6_new)
        gamma1_new <- melt(as.matrix(gamma1$clone())); gamma1_new$X1 <- as.factor(init+1); gamma1_new$X2 <- iter
        gamma1_list  <- rbind(gamma1_list, gamma1_new)
        gamma21_new <- melt(as.matrix(gamma2$clone()))[melt(as.matrix(gamma2$clone()))$X1==1,]; gamma21_new$X1 <- as.factor(init+1); gamma21_new$X2 <- iter
        gamma21_list  <- rbind(gamma21_list, gamma21_new)
        gamma22_new <- melt(as.matrix(gamma2$clone()))[melt(as.matrix(gamma2$clone()))$X1==2,]; gamma22_new$X1 <- as.factor(init+1); gamma22_new$X2 <- iter
        gamma22_list  <- rbind(gamma22_list, gamma22_new)
        sumLik_prev <- sumLik_new
        sumLik_new <- melt(as.matrix(-loss$clone())); sumLik_new$X1 <- as.factor(init+1); sumLik_new$X2 <- iter
        
        if (iter == 1) {sumLik_init <- sumLik_new}
        sumLik_list <- rbind(sumLik_list, sumLik_new)
        
        crit <- ifelse(abs(sumLik_new$value - sumLik_init$value) > sEpsilon, (sumLik_new$value - sumLik_prev$value) / abs(sumLik_new$value - sumLik_init$value), 0)
        count <- ifelse(crit < stopCrit, count + 1, 0)
        
        if (count == 3 || iter == maxIter) {
          print('   stopping criterion is met')
          with_no_grad ({
            for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
          break
          
        } else if (sumLikBest < sumLik_new$value) {
          initBest <- init
          iterBest <- iter
          thetaBest <- theta
          sumLikBest <- sumLik_new$value }
        
        df_sumLik <- sumLik_list; colnames(df_sumLik) <- c('initialization', 'iteration', 'sumLik')
        df_B11_1 <- B11_1_list; colnames(df_B11_1) <- c('initialization', 'iteration', 'B11_1')
        df_B11_2 <- B11_2_list; colnames(df_B11_2) <- c('initialization', 'iteration', 'B11_2')
        df_B12_1 <- B12_1_list; colnames(df_B12_1) <- c('initialization', 'iteration', 'B12_1')
        df_B12_2 <- B12_2_list; colnames(df_B12_2) <- c('initialization', 'iteration', 'B12_2')
        df_B21d_1 <- B21d_1_list; colnames(df_B21d_1) <- c('initialization', 'iteration', 'B21d_1')
        df_B21d_2 <- B21d_2_list; colnames(df_B21d_2) <- c('initialization', 'iteration', 'B21d_2')
        df_B22d_1 <- B22d_1_list; colnames(df_B22d_1) <- c('initialization', 'iteration', 'B22d_1')
        df_B22d_2 <- B22d_2_list; colnames(df_B22d_2) <- c('initialization', 'iteration', 'B22d_2')
        df_B31_1 <- B31_1_list; colnames(df_B31_1) <- c('initialization', 'iteration', 'B31_1')
        df_B31_2 <- B31_2_list; colnames(df_B31_2) <- c('initialization', 'iteration', 'B31_2')
        df_B32_1 <- B32_1_list; colnames(df_B32_1) <- c('initialization', 'iteration', 'B32_1')
        df_B32_2 <- B32_2_list; colnames(df_B32_2) <- c('initialization', 'iteration', 'B32_2')
        df_Lmdd_1 <- Lmdd_1_list; colnames(df_Lmdd_1) <- c('initialization', 'iteration', 'Lmdd_1')
        df_Lmdd_2 <- Lmdd_2_list; colnames(df_Lmdd_2) <- c('initialization', 'iteration', 'Lmdd_2')
        df_Lmdd_3 <- Lmdd_3_list; colnames(df_Lmdd_3) <- c('initialization', 'iteration', 'Lmdd_3')
        df_Lmdd_4 <- Lmdd_4_list; colnames(df_Lmdd_4) <- c('initialization', 'iteration', 'Lmdd_4')
        df_Lmdd_5 <- Lmdd_5_list; colnames(df_Lmdd_5) <- c('initialization', 'iteration', 'Lmdd_5')
        df_Lmdd_6 <- Lmdd_6_list; colnames(df_Lmdd_6) <- c('initialization', 'iteration', 'Lmdd_6')
        df_Qd_1 <- Qd_1_list; colnames(df_Qd_1) <- c('initialization', 'iteration', 'Qd_1')
        df_Qd_2 <- Qd_2_list; colnames(df_Qd_2) <- c('initialization', 'iteration', 'Qd_2')
        df_Rd_1 <- Rd_1_list; colnames(df_Rd_1) <- c('initialization', 'iteration', 'Rd_1')
        df_Rd_2 <- Rd_2_list; colnames(df_Rd_2) <- c('initialization', 'iteration', 'Rd_2')
        df_Rd_3 <- Rd_3_list; colnames(df_Rd_3) <- c('initialization', 'iteration', 'Rd_3')
        df_Rd_4 <- Rd_4_list; colnames(df_Rd_4) <- c('initialization', 'iteration', 'Rd_4')
        df_Rd_5 <- Rd_5_list; colnames(df_Rd_5) <- c('initialization', 'iteration', 'Rd_5')
        df_Rd_6 <- Rd_6_list; colnames(df_Rd_6) <- c('initialization', 'iteration', 'Rd_6')
        df_gamma1 <- gamma1_list; colnames(df_gamma1) <- c('initialization', 'iteration', 'gamma1')
        df_gamma21 <- gamma21_list; colnames(df_gamma21) <- c('initialization', 'iteration', 'gamma21')
        df_gamma22 <- gamma22_list; colnames(df_gamma22) <- c('initialization', 'iteration', 'gamma22')
        
        if (iter %% 5 == 0) {
          plot_sumLik <- ggplot(data=df_sumLik, aes(iteration, sumLik, group=initialization, color=initialization)) + geom_line() + theme(legend.position='none') + scale_y_continuous(trans = "log10")
          plot_B11_1 <- ggplot(data=df_B11_1, aes(iteration, B11_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B11_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B11_2 <- ggplot(data=df_B11_2, aes(iteration, B11_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B11_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B12_1 <- ggplot(data=df_B12_1, aes(iteration, B12_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B12_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B12_2 <- ggplot(data=df_B12_2, aes(iteration, B12_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B12_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B21d_1 <- ggplot(data=df_B21d_1, aes(iteration, B21d_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B21d_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B21d_2 <- ggplot(data=df_B21d_2, aes(iteration, B21d_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B21d_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B22d_1 <- ggplot(data=df_B22d_1, aes(iteration, B22d_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B22d_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B22d_2 <- ggplot(data=df_B22d_2, aes(iteration, B22d_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B22d_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B31_1 <- ggplot(data=df_B31_1, aes(iteration, B31_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B31_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B31_2 <- ggplot(data=df_B31_2, aes(iteration, B31_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B31_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B32_1 <- ggplot(data=df_B32_1, aes(iteration, B32_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B32_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_B32_2 <- ggplot(data=df_B32_2, aes(iteration, B32_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$B32_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Lmdd_1 <- ggplot(data=df_Lmdd_1, aes(iteration, Lmdd_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Lmdd_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Lmdd_2 <- ggplot(data=df_Lmdd_2, aes(iteration, Lmdd_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Lmdd_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Lmdd_3 <- ggplot(data=df_Lmdd_3, aes(iteration, Lmdd_3, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Lmdd_true[3]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Lmdd_4 <- ggplot(data=df_Lmdd_4, aes(iteration, Lmdd_4, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Lmdd_true[4]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Lmdd_5 <- ggplot(data=df_Lmdd_5, aes(iteration, Lmdd_5, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Lmdd_true[5]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Lmdd_6 <- ggplot(data=df_Lmdd_6, aes(iteration, Lmdd_6, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Lmdd_true[6]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Qd_1 <- ggplot(data=df_Qd_1, aes(iteration, Qd_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Qd_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Qd_2 <- ggplot(data=df_Qd_2, aes(iteration, Qd_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Qd_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Rd_1 <- ggplot(data=df_Rd_1, aes(iteration, Rd_1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Rd_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Rd_2 <- ggplot(data=df_Rd_2, aes(iteration, Rd_2, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Rd_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Rd_3 <- ggplot(data=df_Rd_3, aes(iteration, Rd_3, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Rd_true[3]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Rd_4 <- ggplot(data=df_Rd_4, aes(iteration, Rd_4, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Rd_true[4]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Rd_5 <- ggplot(data=df_Rd_5, aes(iteration, Rd_5, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Rd_true[5]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_Rd_6 <- ggplot(data=df_Rd_6, aes(iteration, Rd_6, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$Rd_true[6]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_gamma1 <- ggplot(data=df_gamma1, aes(iteration, gamma1, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$gamma1_true), color='black', linetype='dashed') + theme(legend.position='none')
          plot_gamma21 <- ggplot(data=df_gamma21, aes(iteration, gamma21, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$gamma2_true[1]), color='black', linetype='dashed') + theme(legend.position='none')
          plot_gamma22 <- ggplot(data=df_gamma22, aes(iteration, gamma22, group=initialization, color=initialization)) + geom_line() + geom_line(aes(y=df$gamma2_true[2]), color='black', linetype='dashed') + theme(legend.position='none')
          print(plot_grid(plot_sumLik, plot_B11_1, plot_B11_2, plot_B12_1, plot_B12_2,
                          plot_B21d_1, plot_B21d_2, plot_B22d_1, plot_B22d_2,
                          plot_B31_1, plot_B31_2, plot_B32_1, plot_B32_2,
                          # plot_Lmdd_1, plot_Lmdd_2, plot_Lmdd_3, plot_Lmdd_4, plot_Lmdd_5, plot_Lmdd_6, 
                          # plot_Qd_1, plot_Qd_2,
                          # plot_Rd_1, plot_Rd_2, plot_Rd_3, plot_Rd_4, plot_Rd_5, plot_Rd_6, 
                          plot_gamma1, plot_gamma21, plot_gamma22), labels='AUTO') } }
      
      cat('   sumLik = ', sumLik_new$value, '\n')
      loss$backward()
      
      grad <- list()
      
      with_no_grad ({
        for (var in 1:length(theta)) {
          grad[[var]] <- theta[[var]]$grad
          
          if (iter == 1) {m[[var]] <- v[[var]] <- torch_zeros_like(grad[[var]])}
          
          # update moment estimates
          m[[var]] <- betas[1] * m[[var]] + (1 - betas[1]) * grad[[var]]
          v[[var]] <- betas[2] * v[[var]] + (1 - betas[2]) * grad[[var]]**2
          
          # update bias corrected moment estimates
          m_hat[[var]] <- m[[var]] / (1 - betas[1]**iter)
          v_hat[[var]] <- v[[var]] / (1 - betas[2]**iter)
          
          theta[[var]]$requires_grad_(FALSE)
          theta[[var]]$sub_(lr * m_hat[[var]] / (sqrt(v_hat[[var]]) + epsilon)) } })
      
      B11 <- torch_tensor(theta$B11)
      B12 <- torch_tensor(theta$B12)
      B21d <- torch_tensor(theta$B21d)
      B22d <- torch_tensor(theta$B22d)
      B31 <- torch_tensor(theta$B31)
      B32 <- torch_tensor(theta$B32)
      Lmdd <- torch_tensor(theta$Lmdd); Lmdd[c(1,8)] <- 1; Lmdd[c(2,4,6,7,9,11)] <- 0
      Qd <- torch_tensor(theta$Qd); Qd$clip_(min=lEpsilon)
      Rd <- torch_tensor(theta$Rd); Rd$clip_(min=lEpsilon)
      gamma1 <- torch_tensor(theta$gamma1)
      gamma2 <- torch_tensor(theta$gamma2)
      
      theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                    Lmdd=Lmdd, Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2)
      
      iter <- iter + 1 }
    }) }
