filtering <- function(seed, N, Nt, O1, O2, L1, y1, y2, init, maxIter) {
  set.seed(seed + init)
  # list2env(as.list(df), envir=.GlobalEnv)
  lEpsilon <- 1e-3
  ceil <- 1e8
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
  
  y2_df <- as.data.frame(y2)
  colnames(y2_df) <- c('abiMath', 'TIMMS', 'totIQ')
  fit_cfa <- cfa(model_cfa, data=y2_df)
  eta2_score <- lavPredict(fit_cfa, method='Bartlett')
  eta2 <- as.array(eta2_score[,1])
  
  y1 <- torch_tensor(y1[,,1:O1])
  eta2 <- torch_tensor(eta2)
  
  sumLik_best <- -ceil
  output_best <- NULL
  
  iter <- 1
  count <- 0
  m <- v <- m_hat <- v_hat <- list()
  
  # Initialize parameters
  B11_0 <- rnorm(L1, 0, 1)
  B11 <- torch_tensor(B11_0, requires_grad=FALSE)
  B12 <- torch_tensor(B11_0 + abs(rnorm(L1, 0, 1)), requires_grad=FALSE)
  B21d <- torch_tensor(runif(L1, 0, 1), requires_grad=FALSE)
  B22d <- torch_tensor(runif(L1, 0, 1), requires_grad=FALSE)
  B31_0 <- rnorm(L1, 0, 1)
  B31 <- torch_tensor(B31_0, requires_grad=FALSE)
  B32 <- torch_tensor(B31_0 + rnorm(L1, 0, 1), requires_grad=FALSE)
  Lmdd1 <- torch_tensor(1, requires_grad=FALSE)
  Lmdd2 <- torch_tensor(1, requires_grad=FALSE)
  Lmdd3 <- torch_tensor(1, requires_grad=FALSE)
  Lmdd4 <- torch_tensor(1, requires_grad=FALSE)
  Lmdd5 <- torch_tensor(1, requires_grad=FALSE)
  gamma1 <- torch_tensor(runif(1, 0, 3), requires_grad=FALSE)
  gamma2 <- torch_tensor(abs(rnorm(L1, 0, 1)), requires_grad=FALSE)
  Qd <- torch_tensor(abs(runif(L1, 0, 1)), requires_grad=FALSE)
  Rd <- torch_tensor(abs(runif(O1, 0, 1)), requires_grad=FALSE)
  mP_DO <- torch_tensor(min(abs(rnorm(1, 0, 0.5)), 1), requires_grad=FALSE)
  
  while (count <= 3 && iter <= maxIter) {
    cat('  optim step ', iter, '\n')
    
    theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                  Lmdd1=Lmdd1, Lmdd2=Lmdd2, Lmdd3=Lmdd3, Lmdd4=Lmdd4, Lmdd5=Lmdd5, 
                  Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2, mP_DO=mP_DO)
    
    B11$requires_grad_(TRUE)
    B12$requires_grad_(TRUE)
    B21d$requires_grad_(TRUE)
    B22d$requires_grad_(TRUE)
    B31$requires_grad_(TRUE)
    B32$requires_grad_(TRUE)
    Lmdd1$requires_grad_(TRUE)
    Lmdd2$requires_grad_(TRUE)
    Lmdd3$requires_grad_(TRUE)
    Lmdd4$requires_grad_(TRUE)
    Lmdd5$requires_grad_(TRUE)
    Qd$requires_grad_(TRUE)
    Rd$requires_grad_(TRUE)
    gamma1$requires_grad_(TRUE)
    gamma2$requires_grad_(TRUE)
    mP_DO$requires_grad_(TRUE)
    
    q <- length(torch_cat(theta))
    
    jEta <- torch_full(c(N, Nt+1, 2, 2, L1), 0)
    jP <- torch_full(c(N, Nt+1, 2, 2, L1, L1), 0)
    jV <- torch_full(c(N, Nt, 2, 2, O1), NaN)
    jF <- torch_full(c(N, Nt, 2, 2, O1, O1), NaN)
    jEta2 <- torch_full(c(N, Nt, 2, 2, L1), 0)
    jP2 <- torch_full(c(N, Nt, 2, 2, L1, L1), 0)
    mEta <- torch_full(c(N, Nt+1, 2, L1), 0)
    mP <- torch_full(c(N, Nt+1, 2, L1, L1), NaN)
    W <- torch_full(c(N, Nt, 2, 2), NaN)
    jPr <- torch_full(c(N, Nt+1, 2, 2), 0)
    mLik <- torch_full(c(N, Nt), NaN)
    jPr2 <- torch_full(c(N, Nt, 2, 2), 0)
    mPr <- torch_full(c(N, Nt+2, 2), NaN)
    jLik <- torch_full(c(N, Nt, 2, 2), 0)
    tPr <- torch_full(c(N, Nt+1, 2, 2), NaN)
    KG <- torch_full(c(N, Nt, 2, 2, L1, O1), 0)
    I_KGLmd <- torch_full(c(N, Nt, 2, 2, L1, L1), NaN)
    subEta <- torch_full(c(N, Nt, 2, 2, L1), NaN)
    eta1_pred <- torch_full(c(N, Nt+2, L1), NaN)
    P_pred <- torch_full(c(N, Nt+2, L1, L1), NaN)
    
    mP[,1,,,] <- torch_eye(L1)
    mPr[,1,1] <- 1 - mP_DO
    mPr[,1,2] <- mP_DO
    tPr[,,1,2] <- 0.05
    tPr[,,2,2] <- 1 - tPr[,,1,2]       
    
    Lmd <- torch_full(c(O1, L1), 0)
    Lmd[1,1] <- Lmd[3,2] <- Lmd[5,3] <- Lmd[7,4] <- 1 
    Lmd[2,1] <- Lmdd1
    Lmd[4,2] <- Lmdd2
    Lmd[6,3] <- Lmdd3
    Lmd[8,4] <- Lmdd4
    Lmd[9,4] <- Lmdd5
    B21 <- B21d$diag()
    B22 <- B22d$diag()
    LmdT <- Lmd$transpose(1, 2)
    Q <- Qd$diag()
    R <- Rd$diag() 
    B1 <- torch_cat(c(B11, B12))$reshape(c(2, L1))
    B2 <- torch_cat(c(B21, B22))$reshape(c(2, L1, L1))
    B3 <- torch_cat(c(B31, B32))$reshape(c(2, L1))
    
    for (t in 1:Nt) {
      for (i in 1:N) {
        
        #################
        # Kalman filter #
        #################
        
        if (as.logical(sum(torch_isnan(y1[i,t,]))) <= 0) {
          
          jEta[i,t,,,] <- B1$unsqueeze(-2) + mEta[i,t,,]$clone()$unsqueeze(2)$matmul(B2) + eta2[i]$clone()$unsqueeze(-1)$unsqueeze(-1)$unsqueeze(-1) * B3$unsqueeze(-2)
          jP[i,t,,,,] <- mP[i,t,,,]$clone()$unsqueeze(2)$matmul(B2[2,,]**2) + Q$expand(c(2, 2, -1, -1))
          jV[i,t,,,] <- y1[i,t,]$clone()$unsqueeze(-2)$unsqueeze(-2) - jEta[i,t,,,]$clone()$matmul(LmdT) # possible missingness
          jF[i,t,,,,] <- Lmd$matmul(jP[i,t,,,,]$clone()$matmul(LmdT)) + R$expand(c(2, 2, -1, -1))
          KG[i,t,,,,] <- jP[i,t,,,,]$clone()$matmul(LmdT)$matmul(jF[i,t,,,,]$clone()$cholesky_inverse())
          jEta2[i,t,,,] <- jEta[i,t,,,] + KG[i,t,,,,]$clone()$matmul(jV[i,t,,,]$clone()$unsqueeze(-1))$squeeze()
          I_KGLmd[i,t,,,,] <- torch_eye(L1)$expand(c(2,2,-1,-1)) - KG[i,t,,,,]$clone()$matmul(Lmd)
          jP2[i,t,,,,] <- I_KGLmd[i,t,,,,]$clone()$matmul(jP[i,t,,,,]$clone())$matmul(I_KGLmd[i,t,,,,]$clone()$transpose(3, 4)) +
            KG[i,t,,,,]$clone()$matmul(R)$matmul(KG[i,t,,,,]$clone()$transpose(3, 4))
          
          log_det_jF <- jF[i,t,,,,]$clone()$det()$clip(min=-ceil, max=ceil)$log()
          quadratic_term <- -.5 * jF[i,t,,,,]$clone()$cholesky_inverse()$matmul(jV[i,t,,,]$clone()$unsqueeze(-1))$squeeze()$unsqueeze(-2)$matmul(jV[i,t,,,]$clone()$unsqueeze(-1))$squeeze()$squeeze()
          jLik[i,t,,] <- log(sEpsilon + const) - log_det_jF + quadratic_term
          
          ###################
          # Hamilton filter #
          ###################
          
          eta1_pred[i,t,] <- mPr[i,t,1]$clone()$unsqueeze(-1) * mEta[i,t,1,]$clone() + mPr[i,t,2]$clone()$unsqueeze(-1) * mEta[i,t,2,]$clone()
          P_pred[i,t,,] <- mPr[i,t,1]$clone()$unsqueeze(-1)$unsqueeze(-1) * mP[i,t,1,,]$clone() + mPr[i,t,2]$clone()$unsqueeze(-1)$unsqueeze(-1) * mP[i,t,2,,]$clone()
          tPr[i,t,1,1] <- (gamma1 + eta1_pred[i,t,]$clone()$matmul(gamma2))$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
          tPr[i,t,2,1] <- 1 - tPr[i,t,1,1]
          jPr[i,t,,] <- tPr[i,t,,]$clone() * mPr[i,t,]$clone()$unsqueeze(-1)
          mLik[i,t] <- (jLik[i,t,,]$clone() * jPr[i,t,,]$clone())$sum() 
          jPr2[i,t,,] <- jLik[i,t,,]$clone() * jPr[i,t,,]$clone() / mLik[i,t]$clone()$unsqueeze(-1)$unsqueeze(-1) # possible missingness
          mPr[i,t+1,] <- jPr2[i,t,,]$sum(2)$clip(min=sEpsilon, max=1-sEpsilon)
          W[i,t,,] <- jPr2[i,t,,]$clone() / mPr[i,t+1,]$clone()$unsqueeze(-1)
          mEta[i,t+1,,] <- (W[i,t,,]$clone()$unsqueeze(-1) * jEta2[i,t,,,]$clone())$sum(2) 
          subEta[i,t,,,] <- mEta[i,t+1,,]$unsqueeze(2) - jEta2[i,t,,,]
          mP[i,t+1,,,] <- (W[i,t,,]$clone()$unsqueeze(-1)$unsqueeze(-1) * (jP2[i,t,,,,] + subEta[i,t,,,]$clone()$unsqueeze(-1)$matmul(subEta[i,t,,,]$clone()$unsqueeze(-2))))$sum(2) 
        }
        
        if (as.logical(sum(torch_isnan(y1[i,t,]))) > 0) {
          jEta2[i,t,,,] <- jEta2[i,t-1,,,]$clone()
          jP2[i,t,,,,] <- jP2[i,t-1,,,,]$clone()
          mEta[i,t+1,,] <- mEta[i,t,,]$clone()
          mP[i,t+1,,,] <- mP[i,t,,,]$clone()
          jPr2[i,t,,] <- jPr2[i,t-1,,]$clone()  
          mPr[i,t+1,] <- mPr[i,t,]$clone()
        }
      }
    }
    
    eta1_pred[,Nt+1,] <- mPr[,Nt+1,1]$unsqueeze(-1) * mEta[,Nt+1,1,] + mPr[,Nt+1,2]$unsqueeze(-1) * mEta[,Nt+1,2,]
    P_pred[,Nt+1,,] <- mPr[,Nt+1,1]$unsqueeze(-1)$unsqueeze(-1) * mP[,Nt+1,1,,] + mPr[,Nt+1,2]$unsqueeze(-1)$unsqueeze(-1) * mP[,Nt+1,2,,]
    
    jEta[,Nt+1,1,1,] <- B11 + mEta[,Nt+1,1,]$matmul(B21) + eta2$outer(B31)
    jEta[,Nt+1,1,2,] <- B11 + mEta[,Nt+1,2,]$matmul(B21) + eta2$outer(B31)
    jEta[,Nt+1,2,1,] <- B12 + mEta[,Nt+1,1,]$matmul(B22) + eta2$outer(B32)
    jEta[,Nt+1,2,2,] <- B12 + mEta[,Nt+1,2,]$matmul(B22) + eta2$outer(B32)
    
    jP[,Nt+1,1,1,,] <- B21$matmul(mP[,Nt+1,1,,])$matmul(B21) + Q
    jP[,Nt+1,1,2,,] <- B21$matmul(mP[,Nt+1,2,,])$matmul(B21) + Q
    jP[,Nt+1,2,1,,] <- B22$matmul(mP[,Nt+1,1,,])$matmul(B22) + Q
    jP[,Nt+1,2,2,,] <- B22$matmul(mP[,Nt+1,2,,])$matmul(B22) + Q
    
    tPr[,Nt+1,1,1] <- (gamma1 + eta1_pred[,Nt+1,]$matmul(gamma2))$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
    tPr[,Nt+1,2,1] <- 1 - tPr[,Nt+1,1,1]
    
    jPr[,Nt+1,1,1] <- tPr[,Nt+1,1,1] * mPr[,Nt+1,1]
    jPr[,Nt+1,2,1] <- tPr[,Nt+1,2,1] * mPr[,Nt+1,1]
    jPr[,Nt+1,1,2] <- tPr[,Nt+1,1,2] * mPr[,Nt+1,2]
    jPr[,Nt+1,2,2] <- tPr[,Nt+1,2,2] * mPr[,Nt+1,2]
    
    eta1_pred[,Nt+2,] <- jEta[,Nt+1,1,1,] * jPr[,Nt+1,1,1]$unsqueeze(-1) + jEta[,Nt+1,2,1,] * jPr[,Nt+1,2,1]$unsqueeze(-1) + jEta[,Nt+1,2,2,] * jPr[,Nt+1,2,2]$unsqueeze(-1)
    P_pred[,Nt+2,,] <- jP[,Nt+1,1,1,,] * jPr[,Nt+1,1,1]$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,1,,] * jPr[,Nt+1,2,1]$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,2,,] * jPr[,Nt+1,2,2]$unsqueeze(-1)$unsqueeze(-1)
    
    mPr[,Nt+2,1] <- jPr[,Nt+1,1,]$sum(2)
    mPr[,Nt+2,2] <- jPr[,Nt+1,2,]$sum(2)
    
    loss <- -mLik$nansum()
    
    if (!is.finite(as.numeric(loss))) {
      cat('   error in calculating the sum likelihood', '\n')
      with_no_grad ({
        for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
      break }
    
    if (iter == 1) {
      theta_list <- data.frame(init=init, iter=iter, param=1:length(torch_cat(theta)), value=as.numeric(torch_cat(theta)))
      theta_list_arranged <- theta_list %>%
      group_by(init, iter) %>%
      mutate(param = param) %>%
      pivot_wider(id_cols = c(init, iter), names_from = param, values_from = value)
      
      sumLik <- sumLik_init <- -as.numeric(loss)
      AIC <- -2 * sumLik + 2 * q
      BIC <- -2 * sumLik + q * log(N * Nt)
      output_list <-
      output_new <- data.table(init=init, iter=iter, sumLik=sumLik, AIC=AIC, BIC=BIC)
      
    } else {
      
      theta_new <- data.frame(init=init, iter=iter, param=1:length(torch_cat(theta)), value=as.numeric(torch_cat(theta)))
      theta_new_arranged <- theta_new %>%
        group_by(init, iter) %>%
        mutate(param = param) %>%
        pivot_wider(id_cols = c(init, iter), names_from = param, values_from = value)
      
      sumLik_prev <- sumLik
      sumLik <- -as.numeric(loss)
      output_new <- data.frame(init=init, iter=iter, sumLik=sumLik, AIC=AIC, BIC=BIC)
      
      if (iter == 1) {sumLik_init <- sumLik}
      theta_list_arranged <- rbindlist(list(theta_list_arranged, theta_new_arranged))
      output_list <- rbindlist(list(output_list, output_new))
      
      crit <- ifelse(abs(sumLik - sumLik_init) > sEpsilon, (sumLik - sumLik_prev) / (sumLik - sumLik_init), 0)
      count <- ifelse(crit < stopCrit, count + 1, 0) }
    
    cat('   count =', count, '\n')
    cat('   iter =', iter, '\n')
    if (count >= 3 || iter == maxIter) {
      cat('   stopping criterion is met', '\n')
      with_no_grad ({
        for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
      break
      
    } else if (sumLik_best < sumLik) {
      init_best <- init
      iter_best <- iter
      theta_best <- as.numeric(torch_cat(theta))
      sumLik_best_NT <- sumLik / (N * Nt)
      AIC_best <- AIC
      BIC_best <- BIC
      
      output_best <- output_new
    } 
    
    cat('   sumLik = ', sumLik, '\n')
    loss$backward()
    
    grad <- list()
    
    with_no_grad ({
      for (var in 1:length(theta)) {
        grad[[var]] <- theta[[var]]$grad
        
        if (max(!is.finite(as.numeric(torch_cat(grad[[var]]))))) {
          cat('numerical blowup', '\n')
          count <- 3; theta[[var]]$requires_grad_(FALSE); break}
        
        if (iter == 1) {m[[var]] <- v[[var]] <- torch_zeros_like(grad[[var]])}
        
        # Update moment estimates
        m[[var]] <- betas[1] * m[[var]] + (1 - betas[1]) * grad[[var]]
        v[[var]] <- betas[2] * v[[var]] + (1 - betas[2]) * grad[[var]]**2
        
        # Update bias corrected moment estimates
        m_hat[[var]] <- m[[var]] / (1 - betas[1]**iter)
        v_hat[[var]] <- v[[var]] / (1 - betas[2]**iter)
        
        theta[[var]]$requires_grad_(FALSE)
        theta[[var]]$sub_(lr * m_hat[[var]] / (sqrt(v_hat[[var]]) + epsilon))$detach_() } })
    
    B11 <- torch_tensor(theta$B11, requires_grad=FALSE)
    B12 <- torch_tensor(theta$B12, requires_grad=FALSE)
    B21d <- torch_tensor(theta$B21d, requires_grad=FALSE)
    B22d <- torch_tensor(theta$B22d, requires_grad=FALSE)
    B31 <- torch_tensor(theta$B31, requires_grad=FALSE)
    B32 <- torch_tensor(theta$B32, requires_grad=FALSE)
    Lmdd1 <- torch_tensor(theta$Lmdd1, requires_grad=FALSE)
    Lmdd2 <- torch_tensor(theta$Lmdd2, requires_grad=FALSE)
    Lmdd3 <- torch_tensor(theta$Lmdd3, requires_grad=FALSE)
    Lmdd4 <- torch_tensor(theta$Lmdd4, requires_grad=FALSE)
    Lmdd5 <- torch_tensor(theta$Lmdd5, requires_grad=FALSE)
    Qd <- torch_tensor(theta$Qd, requires_grad=FALSE); Qd$clip_(min=lEpsilon)
    Rd <- torch_tensor(theta$Rd, requires_grad=FALSE); Rd$clip_(min=lEpsilon)
    gamma1 <- torch_tensor(theta$gamma1, requires_grad=FALSE)
    gamma2 <- torch_tensor(theta$gamma2, requires_grad=FALSE)
    mP_DO <- torch_tensor(theta$mP_DO, requires_grad=FALSE); mP_DO$clip_(min=sEpsilon, max=1-sEpsilon)
    
    theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                  Lmdd1=Lmdd1, Lmdd2=Lmdd2, Lmdd3=Lmdd3, Lmdd4=Lmdd4, Lmdd5=Lmdd5, 
                  Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2, mP_DO=mP_DO)
    
    B11$requires_grad_(TRUE)
    B12$requires_grad_(TRUE)
    B21d$requires_grad_(TRUE)
    B22d$requires_grad_(TRUE)
    B31$requires_grad_(TRUE)
    B32$requires_grad_(TRUE)
    Lmdd1$requires_grad_(TRUE)
    Lmdd2$requires_grad_(TRUE)
    Lmdd3$requires_grad_(TRUE)
    Lmdd4$requires_grad_(TRUE)
    Lmdd5$requires_grad_(TRUE)
    Qd$requires_grad_(TRUE)
    Rd$requires_grad_(TRUE)
    gamma1$requires_grad_(TRUE)
    gamma2$requires_grad_(TRUE)
    mP_DO$requires_grad_(TRUE)
    
    iter <- iter + 1 
    rm(grad)
    gc() }
  
  filter <- list(init_best=init_best, iter_best=iter_best, theta_best=theta_best, sumLik_best_NT=sumLik_best_NT, AIC_best=AIC_best, BIC_best=BIC_best, output_best=output_best, theta_list=data.table(theta_list_arranged), output_list=output_list)
  gc()
  return(filter)
}
