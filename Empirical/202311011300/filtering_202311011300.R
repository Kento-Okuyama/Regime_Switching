filtering <- function(seed, N, Nt, O1, O2, L1, y1, y2, nInit, maxIter) {
  set.seed(seed)
  
  lEpsilon <- 1e-3
  ceil <- 1e15
  sEpsilon <- 1e-15
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
  IQ =~ abiMath + TIMMS + totIQ'

  y2_df <- as.data.frame(y2[,2:4])
  colnames(y2_df) <- c('abiMath', 'TIMMS', 'totIQ')
  fit_cfa <- cfa(model_cfa, data=y2_df)
  eta2_score <- lavPredict(fit_cfa, method='Bartlett')
  eta2 <- as.array(eta2_score[,1])

  y1 <- torch_tensor(y1[,,1:O1])
  eta2 <- torch_tensor(eta2)
  
  sumLik_best <- 0
  
  for (init in 1:nInit) {
    # cat('Init step ', init, '\n')
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
    Lmdd <- torch_tensor(runif(O1-L1, .5, 1.5))
    Lmd <- torch_full(c(O1,L1), 0)
    Lmd[1,1] <- Lmd[4,2] <- Lmd[6,3] <- Lmd[8,4] <- Lmd[10,5] <- Lmd[12,6] <- Lmd[15,7] <- 1 
    Lmd[2:3,1] <- Lmdd[1:2]
    Lmd[5,2] <- Lmdd[3]
    Lmd[7,3] <- Lmdd[4]
    Lmd[9,4] <- Lmdd[5]
    Lmd[11,5] <- Lmdd[6]
    Lmd[13:14,6] <- Lmdd[7:8]
    Lmd[16:17,7] <- Lmdd[9:10]
    gamma1 <- torch_tensor(4) # fixed
    gamma2 <- torch_tensor(abs(rnorm(L1, 0, 1)))
    Qd <- torch_tensor(rep(.3, L1)) # fixed
    Rd <- torch_tensor(rep(.5, O1)) # fixed
    
    with_detect_anomaly ({
    # try (silent=FALSE, {
      while (count <=3 && iter <= maxIter) {
        # cat('   optim step ', iter, '\n')
        
        B11$requires_grad_()
        B12$requires_grad_()
        B21d$requires_grad_()
        B22d$requires_grad_()
        B31$requires_grad_()
        B32$requires_grad_()
        #Lmdd$requires_grad_()
        Qd$requires_grad_()
        Rd$requires_grad_()
        gamma1$requires_grad_()
        gamma2$requires_grad_()
        
        theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                      #Lmdd=Lmdd, 
                      Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2)
        q <- length(torch_cat(theta))
        
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
        mPr <- torch_full(c(N,Nt+2,2), NaN)
        jLik <- torch_full(c(N,Nt,2,2), 0)
        tPr <- torch_full(c(N,Nt+1,2,2), NaN)
        KG <- torch_full(c(N,Nt,2,2,L1,O1), 0)
        I_KGLmd <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
        subEta <- torch_full(c(N,Nt,2,2,L1), NaN)
        eta1_pred <- torch_full(c(N,Nt+2,L1), NaN)
        P_pred <- torch_full(c(N,Nt+2,L1,L1), NaN)
        
        mP[,1,,,] <- torch_eye(L1)
        mPr[,1,1] <- 1
        mPr[,1,2] <- 0
        tPr[,,1,2] <- 0
        tPr[,,2,2] <- 1
        
        B21 <- B21d$diag()
        B22 <- B22d$diag()
        LmdT <- Lmd$transpose(1, 2)
        Q <- Qd$diag()
        R <- Rd$diag() 
        B1 <- torch_cat(c(B11, B12))$reshape(c(2, L1))
        B2 <- torch_cat(c(B21, B22))$reshape(c(2, L1, L1))
        B3 <- torch_cat(c(B31, B32))$reshape(c(2, L1))
        
        for (t in 1:Nt) {
          
          #################
          # Kalman filter #
          #################
          
          jEta[,t,,,] <- B1$expand(c(N, -1, -1))$unsqueeze(-2) + mEta[,t,,]$clone()$unsqueeze(2)$matmul(B2) + eta2$unsqueeze(-1)$unsqueeze(-1)$unsqueeze(-1) * B3$expand(c(N, -1, -1))$unsqueeze(-2)
          jP[,t,,,,] <- mP[,t,,,]$unsqueeze(2)$matmul(B2[2,,]**2) + Q$expand(c(N, 2, 2, -1, -1))
          jV[,t,,,] <- y1[,t,]$unsqueeze(-2)$unsqueeze(-2) - jEta[,t,,,]$clone()$matmul(LmdT) # possible missingness
          jF[,t,,,,] <- Lmd$matmul(jP[,t,,,,]$clone()$matmul(LmdT)) + R
          KG[,t,,,,] <- jP[,t,,,,]$clone()$matmul(LmdT)$matmul(jF[,t,,,,]$clone()$cholesky_inverse())
          jEta2[,t,,,] <- jEta[,t,,,] + KG[,t,,,,]$clone()$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()
          I_KGLmd[,t,,,,] <- torch_eye(L1)$expand(c(N,2,2,-1,-1)) - KG[,t,,,,]$clone()$matmul(Lmd)
          jP2[,t,,,,] <- I_KGLmd[,t,,,,]$clone()$matmul(jP[,t,,,,]$clone())$matmul(I_KGLmd[,t,,,,]$clone()$transpose(4, 5)) +
            KG[,t,,,,]$clone()$matmul(R)$matmul(KG[,t,,,,]$clone()$transpose(4, 5))
          jLik[,t,,] <- sEpsilon + const * jF[,t,,,,]$clone()$det()$clip(min=sEpsilon, max=ceil)**(-1) *
            (-.5 * jF[,t,,,,]$clone()$cholesky_inverse()$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()$unsqueeze(-2)$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp() # possible missingness
          
          ###################
          # Hamilton filter #
          ###################
          
          eta1_pred[,t,] <- mPr[,t,1]$clone()$unsqueeze(-1) * mEta[,t,1,]$clone() + mPr[,t,2]$clone()$unsqueeze(-1) * mEta[,t,2,]$clone()
          P_pred[,t,,] <- mPr[,t,1]$unsqueeze(-1)$unsqueeze(-1) * mP[,t,1,,] + mPr[,t,2]$unsqueeze(-1)$unsqueeze(-1) * mP[,t,2,,]
          tPr[,t,1,1] <- (gamma1 + eta1_pred[,t,]$clone()$matmul(gamma2))$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
          tPr[,t,2,1] <- 1 - tPr[,t,1,1]
          jPr[,t,,] <- tPr[,t,,]$clone() * mPr[,t,]$clone()$unsqueeze(-1)
          mLik[,t] <- (jLik[,t,,]$clone() * jPr[,t,,]$clone())$sum(c(2,3)) 
          jPr2[,t,,] <- jLik[,t,,]$clone() * jPr[,t,,]$clone() / mLik[,t]$clone()$unsqueeze(-1)$unsqueeze(-1) # possible missingness
          mPr[,t+1,] <- jPr2[,t,,]$sum(3)$clip(min=sEpsilon, max=1-sEpsilon)
          W[,t,,] <- jPr2[,t,,]$clone() / mPr[,t+1,]$clone()$unsqueeze(-1)
          mEta[,t+1,,] <- (W[,t,,]$clone()$unsqueeze(-1) * jEta2[,t,,,]$clone())$sum(3) 
          subEta[,t,,,] <- mEta[,t+1,,]$unsqueeze(2) - jEta2[,t,,,]
          mP[,t+1,,,] <- (W[,t,,]$clone()$unsqueeze(-1)$unsqueeze(-1) * (jP2[,t,,,,] + subEta[,t,,,]$clone()$unsqueeze(-1)$matmul(subEta[,t,,,]$clone()$unsqueeze(-2))))$sum(3) 
          
          for (i in 1:N) {if (as.logical(sum(torch_isnan(y1[i,t,]))) > 0) {
            jEta2[i,t,,,] <- jEta2[i,t-1,,,]
            jP2[i,t,,,,] <- jP2[i,t-1,,,,]
            mEta[i,t+1,,] <- mEta[i,t,,]
            mP[i,t+1,,,] <- mP[i,t,,,] 
            jPr2[i,t,,] <- jPr2[i,t-1,,]  
            mPr[i,t+1,] <- mPr[i,t,]
          } } 
        }

        eta1_pred[,Nt+1,] <- mPr[,Nt+1,1]$unsqueeze(-1) * mEta[,Nt+1,1,] + mPr[,Nt+1,2]$unsqueeze(-1) * mEta[,Nt+1,2,]
        P_pred[,Nt+1,,] <- mPr[,Nt+1,1]$unsqueeze(-1)$unsqueeze(-1) * mP[,Nt+1,1,,] + mPr[,Nt+1,2]$unsqueeze(-1)$unsqueeze(-1) * mP[,Nt+1,2,,]
        
        jEta[,Nt+1,1,1,] <- B11 + mEta[,Nt+1,1,]$matmul(B21) + eta2$outer(B31)
        jEta[,Nt+1,2,1,] <- B12 + mEta[,Nt+1,1,]$matmul(B22) + eta2$outer(B32)
        jEta[,Nt+1,2,2,] <- B12 + mEta[,Nt+1,2,]$matmul(B22) + eta2$outer(B32)
        
        jP[,Nt+1,1,1,,] <- B21$matmul(mP[,Nt+1,1,,])$matmul(B21) + Q
        jP[,Nt+1,2,1,,] <- B22$matmul(mP[,Nt+1,1,,])$matmul(B22) + Q
        jP[,Nt+1,2,2,,] <- B22$matmul(mP[,Nt+1,2,,])$matmul(B22) + Q
        
        tPr[,Nt+1,1,1] <- (gamma1 + eta1_pred[,Nt+1,]$matmul(gamma2))$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
        tPr[,Nt+1,2,1] <- 1 - tPr[,Nt+1,1,1]
        
        jPr[,Nt+1,1,1] <- tPr[,Nt+1,1,1] * mPr[,Nt+1,1]
        jPr[,Nt+1,2,1] <- tPr[,Nt+1,2,1] * mPr[,Nt+1,1]
        jPr[,Nt+1,2,2] <- mPr[,Nt+1,2]
        
        eta1_pred[,Nt+2,] <- jEta[,Nt+1,1,1,] * jPr[,Nt+1,1,1]$unsqueeze(-1) + jEta[,Nt+1,2,1,] * jPr[,Nt+1,2,1]$unsqueeze(-1) + jEta[,Nt+1,2,2,] * jPr[,Nt+1,2,2]$unsqueeze(-1)
        P_pred[,Nt+2,,] <- jP[,Nt+1,1,1,,] * jPr[,Nt+1,1,1]$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,1,,] * jPr[,Nt+1,2,1]$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,2,,] * jPr[,Nt+1,2,2]$unsqueeze(-1)$unsqueeze(-1)
        
        mPr[,Nt+2,1] <- jPr[,Nt+1,1,1]
        mPr[,Nt+2,2] <- jPr[,Nt+1,2,]$sum(2)
        
        loss <- -mLik$nansum()
        print('loss')
        print(loss)

        if (!is.finite(as.numeric(loss))) {
          # print('   error in calculating the sum likelihood')
          with_no_grad ({
            for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
          break }
        
        if (init == 1 && iter == 1) {
          theta_list <- data.frame(init=init, iter=iter, param=1:length(torch_cat(theta)), value=as.numeric(torch_cat(theta)))
          theta_list_arranged <- theta_list %>%
            group_by(init, iter) %>%
            mutate(param = param) %>%
            pivot_wider(id_cols = c(init, iter), names_from = param, values_from = value)
          
          sumLik <- sumLik_init <- -as.numeric(loss)
          output_list <- data.table(init=init, iter=iter, sumLik=sumLik)
          
        } else {
          
          theta_new <- data.frame(init=init, iter=iter, param=1:length(torch_cat(theta)), value=as.numeric(torch_cat(theta)))
          theta_new_arranged <- theta_new %>%
            group_by(init, iter) %>%
            mutate(param = param) %>%
            pivot_wider(id_cols = c(init, iter), names_from = param, values_from = value)
          
          sumLik_prev <- sumLik
          sumLik <- -as.numeric(loss)
          output_new <- data.frame(init=init, iter=iter, sumLik=sumLik)
          
          if (iter == 1) {sumLik_init <- sumLik}
          theta_list_arranged <- rbindlist(list(theta_list_arranged, theta_new_arranged))
          output_list <- rbindlist(list(output_list, output_new))
          
          crit <- ifelse(abs(sumLik - sumLik_init) > sEpsilon, (sumLik - sumLik_prev) / (sumLik - sumLik_init), 0)
          count <- ifelse(crit < stopCrit, count + 1, 0)
          
          if (init > 1 && iter == 100) {if (sumLik < .8 * max(output_list$sumLik[output_list$init==init-1], na.rm=TRUE)) {break} }
          
          if (count >= 3 || iter == maxIter) {
            # print('   stopping criterion is met')
            with_no_grad ({
              for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
            break
            
          } else if (sumLik_best < sumLik) {
            init_best <- init
            iter_best <- iter
            theta_best <- as.numeric(torch_cat(theta))
            sumLik_best_NT <- sumLik / (N * Nt)
            AIC_best <- -2 * log(sumLik) + 2 * q
            BIC_best <- -2 * log(sumLik) + q * log(N * Nt)
            
            output_best <- output_new } }

        print('sumLik')
        print(sumLik)
        
        # cat('   sumLik = ', sumLik, '\n')
        loss$backward()
        
        grad <- list()

        with_no_grad ({
          for (var in 1:length(theta)) {
            grad[[var]] <- theta[[var]]$grad
            
            if (max(!is.finite(as.numeric(torch_cat(grad[[var]]))))) {
              count <- 3; theta[[var]]$requires_grad_(FALSE); break}
            
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
        # Lmdd <- torch_tensor(theta$Lmdd)
        Lmd[2:3,1] <- Lmdd[1:2]
        Lmd[5,2] <- Lmdd[3]
        Lmd[7,3] <- Lmdd[4]
        Lmd[9,4] <- Lmdd[5]
        Lmd[11,5] <- Lmdd[6]
        Lmd[13:14,6] <- Lmdd[7:8]
        Lmd[16:17,7] <- Lmdd[9:10]
        Qd <- torch_tensor(theta$Qd); Qd$clip_(min=lEpsilon)
        Rd <- torch_tensor(theta$Rd); Rd$clip_(min=lEpsilon)
        gamma1 <- torch_tensor(theta$gamma1)
        gamma2 <- torch_tensor(theta$gamma2)
        
        theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                      #Lmdd=Lmdd, 
                      Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2)
        
        iter <- iter + 1 }
    }) } 
  
  filter <- list(init_best=init_best, iter_best=iter_best, theta_best=theta_best, sumLik_best_NT=sumLik_best_NT, AIC_best=AIC_best, BIC_best=BIC_best, output_best=output_best, theta_list=data.table(theta_list_arranged), output_list=output_list)
  gc()
  return(filter)
}

