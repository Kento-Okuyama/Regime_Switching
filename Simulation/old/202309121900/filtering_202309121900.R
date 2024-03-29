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
        
        jEta <- torch_full(c(N,Nt+1,2,2,L1), 0)
        jP <- torch_full(c(N,Nt+1,2,2,L1,L1), 0)
        jV <- torch_full(c(N,Nt,2,2,O1), NaN)
        jF <- torch_full(c(N,Nt,2,2,O1,O1), NaN)
        jEta2 <- torch_full(c(N,Nt,2,2,L1), 0)
        jP2 <- torch_full(c(N,Nt,2,2,L1,L1), 0)
        mEta <- torch_full(c(N,Nt+1,2,L1), 0)
        mP <- torch_full(c(N,Nt+1,2,L1,L1), NaN)
        W <- torch_full(c(N,Nt,2,2), NaN)
        jPr <- torch_full(c(N,Nt+1,2,2), 0)
        mLik <- torch_full(c(N,Nt), NaN)
        jPr2 <- torch_full(c(N,Nt,2,2), 0)
        mPr <- torch_full(c(N,Nt+1,2), NaN)
        jLik <- torch_full(c(N,Nt,2,2), 0)
        tPr <- torch_full(c(N,Nt+1,2,2), NaN)
        KG <- torch_full(c(N,Nt,2,2,L1,O1), 0)
        I_KGLmd <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
        subEta <- torch_full(c(N,Nt,2,2,L1), NaN)
        eta1_pred <- torch_full(c(N,Nt,L1), NaN)
        
        mP[,1,,,] <- torch_eye(L1)
        mPr[,1,1] <- 1
        mPr[,1,2] <- 0
        tPr[,,1,2] <- 0
        tPr[,,2,2] <- 1
        
        B21 <- B21d$diag()
        B22 <- B22d$diag()
        Lmd <- Lmdd$reshape(c(O1, L1))
        LmdT <- Lmd$transpose(1, 2)
        Q <- Qd$diag()
        R <- Rd$diag()
        
        B1 <- torch_cat(c(B11,B12))$reshape(c(2,L1))
        B2 <- torch_cat(c(B21,B22))$reshape(c(2,L1,L1))
        B3 <- torch_cat(c(B31,B32))$reshape(c(2,L1))
        
        time_1 <- Sys.time()
        
        for (t in 1:Nt) {
          
          #################
          # Kalman filter #
          #################
          
          jEta[,t,,,] <- B1$expand(c(N, -1, -1))$unsqueeze(-2) + mEta[,t,,]$clone()$unsqueeze(2)$matmul(B2) + eta2$unsqueeze(-1)$unsqueeze(-1)$unsqueeze(-1) * B3$expand(c(N, -1, -1))$unsqueeze(-2)
          jP[,t,,,,] <- mP[,t,,,]$unsqueeze(2)$matmul(B2[2,,]**2) + Q$expand(c(N, 2, 2, -1, -1))
          jV[,t,,,] <- y1[,t,]$unsqueeze(-2)$unsqueeze(-2) - jEta[,t,,,]$clone()$matmul(LmdT)
          jF[,t,,,,] <- Lmd$matmul(jP[,t,,,,]$matmul(LmdT)) + R
          KG[,t,,,,] <- jP[,t,,,,]$matmul(LmdT)$matmul(jF[,t,,,,]$clone()$cholesky_inverse())
          jEta2[,t,,,] <- jEta[,t,,,] + KG[,t,,,,]$clone()$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()
          I_KGLmd[,t,,,,] <- torch_eye(L1)$expand(c(N,2,2,-1,-1)) - KG[,t,,,,]$clone()$matmul(Lmd)
          jP2[,t,,,,] <- I_KGLmd[,t,,,,]$clone()$matmul(jP[,t,,,,]$clone())$matmul(I_KGLmd[,t,,,,]$clone()$transpose(4, 5)) +
            KG[,t,,,,]$clone()$matmul(R)$matmul(KG[,t,,,,]$clone()$transpose(4, 5))
          jLik[,t,,] <- const * jF[,t,,,,]$clone()$det()**(-1) *
            (-.5 * jF[,t,,,,]$clone()$cholesky_inverse()$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()$unsqueeze(-2)$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
          
          ###################
          # Hamilton filter #
          ###################
          
          eta1_pred[,t,] <- mPr[,t,1]$clone()$unsqueeze(-1) * mEta[,t,1,]$clone() + mPr[,t,2]$clone()$unsqueeze(-1) * mEta[,t,2,]$clone()
          tPr[,t,1,1] <- (gamma1 + eta1_pred[,t,]$clone()$matmul(gamma2))$sigmoid()
          tPr[,t,2,1] <- 1 - tPr[,t,1,1]
          jPr[,t,,] <- tPr[,t,,]$clone() * mPr[,t,]$clone()$unsqueeze(-1)
          mLik[,t] <- (jLik[,t,,]$clone() * jPr[,t,,]$clone())$sum(c(2,3)) 
          jPr2[,t,,] <- jLik[,t,,]$clone() * jPr[,t,,]$clone() / mLik[,t]$clone()$unsqueeze(-1)$unsqueeze(-1)
          mPr[,t+1,] <- jPr2[,t,,]$sum(3)
          W[,t,,] <- jPr2[,t,,]$clone() / mPr[,t+1,]$clone()$unsqueeze(-1)$clip(min=lEpsilon, max=1-lEpsilon)
          mEta[,t+1,,] <- (W[,t,,]$clone()$unsqueeze(-1) * jEta2[,t,,,]$clone())$sum(3)
          subEta[,t,,,] <- mEta[,t+1,,]$unsqueeze(2) - jEta2[,t,,,]
          mP[,t+1,,,] <- (W[,t,,]$clone()$unsqueeze(-1)$unsqueeze(-1) * (jP2[,t,,,,] + subEta[,t,,,]$clone()$unsqueeze(-1)$matmul(subEta[,t,,,]$clone()$unsqueeze(-2))))$sum(3) }
        
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
