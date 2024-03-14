filtering <- function(seed, N, Nt, O1, O2, L1, y1, y2, nInit, maxIter) {
  
  set.seed(101*seed)
  
  lEpsilon <- 1e-3
  sEpsilon <- 1e-8
  stopCrit <- 1e-4
  lr <- 1e-3
  epsilon <- 1e-8
  betas <- c(.9, .999)
  const <- (2*pi)**(-O1/2)
  
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
  
  Qd <- torch_tensor(rep(.3, L1)) # fixed
  Rd <- torch_tensor(rep(.5, O1)) # fixed
  gamma3 <- torch_tensor(0) # not estimated
  gamma4 <- torch_tensor(rep(0, L1)) # not estimated
  
  for (init in 1:nInit) {
    cat('Init step ', init, '\n')
    iter <- 1
    count <- 0
    m <- v <- m_hat <- v_hat <- list()
    
    # initialize parameters
    B11 <- torch_tensor(abs(rnorm(L1, 0, .3)))
    B12 <- torch_tensor(-abs(rnorm(L1, 0, .2)))
    B21d <- torch_tensor(runif(L1, .6, 1))
    B22d <- torch_tensor(runif(L1, .2, .6))
    B31 <- torch_tensor(abs(rnorm(L1, 0, .15)))
    B32 <- torch_tensor(-abs(rnorm(L1, 0, .1)))
    Lmdd <- torch_tensor(runif(O1*L1, .5, 1.5))
    Lmdd[c(1,8)] <- 1; Lmdd[c(2,4,6,7,9,11)] <- 0
    gamma1 <- torch_tensor(runif(1, 3, 5))
    gamma2 <- torch_tensor(abs(rnorm(L1, 0, 1)))
    
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
        mEta <- torch_full(c(N,Nt+1,2,L1), 0)
        mP <- torch_full(c(N,Nt+1,2,L1,L1), NaN)
        W <- torch_full(c(N,Nt,2,2), NaN)
        jPr <- torch_full(c(N,Nt+1,2,2), 0)
        mLik <- torch_full(c(N,Nt), NaN)
        jPr2 <- torch_full(c(N,Nt,2,2), 0)
        mPr <- torch_full(c(N,Nt+1), NaN)
        jLik <- torch_full(c(N,Nt,2,2), 0)
        tPr <- torch_full(c(N,Nt+1,2), NaN)
        KG <- torch_full(c(N,Nt,2,2,L1,O1), NaN)
        I_KGLmd <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
        subEta <- torch_full(c(N,Nt,2,2,L1), NaN)
        eta1_pred <- torch_full(c(N,Nt,L1), NaN)
        
        mP[,1,,,] <- torch_eye(L1)
        mPr[,1] <- 1 - lEpsilon
        
        B21 <- B21d$diag()
        B22 <- B22d$diag()
        Lmd <- Lmdd$reshape(c(O1, L1))
        LmdT <- Lmd$transpose(1, 2)
        Q <- Qd$diag()
        R <- Rd$diag()
        
        B1 <- c(B11, B12)
        B2 <- c(B21, B22)
        B3 <- c(B31, B32)
        
        time_1 <- Sys.time()
        
        for (t in 1:Nt) {
          
          #################
          # Kalman filter #
          #################
          
          jEta[,t,1,1,] <- B1[[1]] + mEta[,t,1,]$clone()$matmul(B2[[1]]) + eta2$outer(B3[[1]])
          jEta[,t,2,,] <- B1[[2]]$unsqueeze(1)$unsqueeze(1) + mEta[,t,,]$clone()$matmul(B2[[2]]) + eta2$outer(B3[[2]])$unsqueeze(-2)
          
          jP[,t,1,1,,] <- mP[,t,1,,]$matmul(B2[[1]]**2) + Q
          jP[,t,2,,,] <- mP[,t,,,]$matmul(B2[[2]]**2) + Q
          
          jV[,t,1,1,] <- y1[,t,] - jEta[,t,1,1,]$clone()$matmul(LmdT)
          jV[,t,2,,] <- y1[,t,]$unsqueeze(-2) - jEta[,t,2,,]$clone()$matmul(LmdT)
          
          jF[,t,1,1,,] <- Lmd$matmul(jP[,t,1,1,,])$matmul(LmdT) + R
          jF[,t,2,,,] <- Lmd$matmul(jP[,t,2,,,])$matmul(LmdT) + R
          
          KG[,t,1,1,,] <- jP[,t,1,1,,]$matmul(LmdT)$matmul(jF[,t,1,1,,]$clone()$cholesky_inverse())
          KG[,t,2,,,] <- jP[,t,2,,,]$matmul(LmdT)$matmul(jF[,t,2,,,]$clone()$cholesky_inverse())
          
          jEta2[,t,1,1,] <- jEta[,t,1,1,] + KG[,t,1,1,,]$clone()$matmul(jV[,t,1,1,]$clone()$unsqueeze(-1))$squeeze()
          jEta2[,t,2,,] <- jEta[,t,2,,] + KG[,t,2,,,]$clone()$matmul(jV[,t,2,,]$clone()$unsqueeze(-1))$squeeze()
          
          I_KGLmd[,t,1,1,,] <- torch_eye(L1) - KG[,t,1,1,,]$clone()$matmul(Lmd)
          I_KGLmd[,t,2,,,] <- torch_eye(L1) - KG[,t,2,,,]$clone()$matmul(Lmd)
          
          jP2[,t,1,1,,] <- I_KGLmd[,t,1,1,,]$clone()$matmul(jP[,t,1,1,,]$clone())$matmul(I_KGLmd[,t,1,1,,]$clone()$transpose(2, 3)) +
            KG[,t,1,1,,]$clone()$matmul(R)$matmul(KG[,t,1,1,,]$clone()$transpose(2, 3))
          jP2[,t,2,,,] <- I_KGLmd[,t,2,,,]$clone()$matmul(jP[,t,2,,,]$clone())$matmul(I_KGLmd[,t,2,,,]$clone()$transpose(3, 4)) +
            KG[,t,2,,,]$clone()$matmul(R)$matmul(KG[,t,2,,,]$clone()$transpose(3, 4))
          
          jLik[,t,1,1] <- sEpsilon + const * jF[,t,1,1,,]$clone()$det()**(-1) *
            (-.5 * jV[,t,1,1,]$clone()$unsqueeze(2)$matmul(jF[,t,1,1,,]$clone()$cholesky_inverse())$matmul(jV[,t,1,1,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
          jLik[,t,2,] <- sEpsilon + const * jF[,t,2,,,]$clone()$det()**(-1) *
            (-.5 * jF[,t,2,,,]$clone()$cholesky_inverse()$matmul(jV[,t,2,,]$clone()$unsqueeze(-1))$squeeze()$unsqueeze(-2)$matmul(jV[,t,2,,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
          
          ###################
          # Hamilton filter #
          ###################
          
          eta1_pred[,t,] <- mPr[,t]$clone()$unsqueeze(-1) * mEta[,t,1,]$clone() + (1 - mPr[,t])$unsqueeze(-1) * mEta[,t,2,]$clone()
          tPr[,t,1] <- (gamma1 + eta1_pred[,t,]$clone()$matmul(gamma2))$sigmoid()
          
          jPr[,t,1,1] <- tPr[,t,1]$clone() * mPr[,t]$clone()
          jPr[,t,2,1] <- (1 - tPr[,t,1]) * mPr[,t]$clone()
          jPr[,t,2,2] <- (1 - mPr[,t])
          
          mLik[,t] <- (jLik[,t,,]$clone() * jPr[,t,,]$clone())$sum(c(2,3)) 
          jPr2[,t,1,1] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() / mLik[,t]$clone()
          jPr2[,t,2,] <- jLik[,t,2,]$clone() * jPr[,t,2,]$clone() / mLik[,t]$clone()$unsqueeze(-1)
          
          mPr[,t+1] <- jPr2[,t,1,1]
          W[,t,2,] <- jPr2[,t,2,]$clone() / (1 - mPr[,t+1])$unsqueeze(-1)
          
          mEta[,t+1,1,] <- jEta2[,t,1,1,]
          mEta[,t+1,2,] <- (W[,t,2,]$clone()$unsqueeze(-1) * jEta2[,t,2,,]$clone())$sum(2)
          
          subEta[,t,1,1,] <- mEta[,t+1,1,] - jEta2[,t,1,1,]
          subEta[,t,2,,] <- mEta[,t+1,2,]$unsqueeze(-2) - jEta2[,t,2,,]
          
          mP[,t+1,1,,] <- jP2[,t,1,1,,] + subEta[,t,1,1,]$clone()$unsqueeze(-1)$matmul(subEta[,t,1,1,]$clone()$unsqueeze(-2))
          mP[,t+1,2,,] <- (W[,t,2,]$clone()$unsqueeze(-1)$unsqueeze(-1) * (jP2[,t,2,,,] + subEta[,t,2,,]$clone()$unsqueeze(-1)$matmul(subEta[,t,2,,]$clone()$unsqueeze(-2))))$sum(2) }
        
        time_2 <- Sys.time()
        
        loss <- -mLik$sum()
        
        if (is.nan(-as.numeric(loss))) {
          print('   error in calculating the sum likelihood')
          with_no_grad ({
            for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
          break }
        
        if (init == 1 && iter == 1) {
          sumLik_list <- sumLik_init <- sumLik_new <- melt(as.matrix(-loss), nrow=1); sumLik_list$X1 <- as.factor(init+1); sumLik_list$X2 <- iter
          
        } else {
          
          sumLik_prev <- sumLik_new
          sumLik_new <- melt(as.matrix(-loss)); sumLik_new$X1 <- as.factor(init+1); sumLik_new$X2 <- iter
          
          if (iter == 1) {sumLik_init <- sumLik_new}
          sumLik_list <- rbind(sumLik_list, sumLik_new)
          
          crit <- ifelse(abs(sumLik_new$value - sumLik_init$value) > sEpsilon, (sumLik_new$value - sumLik_prev$value) / abs(sumLik_new$value - sumLik_init$value), 0)
          count <- ifelse(crit < stopCrit, count + 1, 0)
          
          if (iter == 100) {if (sumLik_new$value < max(sumLik_list[sumLik_list$X1==init-1,]$value) * .8) {count <- 3} }
          
          if (count == 3 || iter == maxIter) {
            print('   stopping criterion is met')
            with_no_grad ({
              for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
            break
            
          } else if (sumLikBest < sumLik_new$value) {
            initBest <- init
            iterBest <- iter
            thetaBest <- theta
            sumLikBest <- sumLik_new$value } }
        
        cat('   sumLik = ', sumLik_new$value, '\n')
        
        time_3 <- Sys.time()
        
        loss$backward()
        
        time_4 <- Sys.time()
        
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
        
        print(paste0('time_2 - time_1 = ', time_2 - time_1))
        print(paste0('time_3 - time_2 = ', time_3 - time_2))
        print(paste0('time_4 - time_3 = ', time_4 - time_3))
        
        iter <- iter + 1 }
    }) } 
  
  params <- list()
  for (l in 1:length(thetaBest)) params[[l]] <- as.numeric(thetaBest[[l]])
  names(params) <- names(thetaBest)
  
  return(params)
}
