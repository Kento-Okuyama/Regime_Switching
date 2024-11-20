filtering <- function(seed, N, Nt, O1, O2, L1, y1, y2, DO, init, maxIter) {
  set.seed(seed + init)
  # list2env(as.list(df), envir=.GlobalEnv)
  lEpsilon <- 1e-3
  ceil <- 1e8
  sEpsilon <- 1e-8
  stopCrit <- 1e-4
  lr <- 1e-2
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
  
  sumLik_best <- -Inf
  output_best <- NULL
  
  iter <- 1
  count <- 0
  m <- v <- m_hat <- v_hat <- list()
  
  # Initialize parameters
  B11_0 <- rnorm(L1, 0, 1)
  B11 <- torch_tensor(B11_0, requires_grad=FALSE)
  B12 <- torch_tensor(B11_0 + rnorm(L1, 0, 1), requires_grad=FALSE)
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
  gamma1 <- torch_tensor(rnorm(1, 0, 1), requires_grad=FALSE)
  gamma2 <- torch_tensor(rnorm(L1, 0, 1), requires_grad=FALSE)
  gamma3 <- torch_tensor(rnorm(1, 0, 1), requires_grad=FALSE)
  # gamma4 <- torch_tensor(rnorm(L1, 0, 1), requires_grad=FALSE)
  Qd <- torch_tensor(abs(runif(L1, 0, 1)), requires_grad=FALSE)
  Rd <- torch_tensor(abs(runif(O1, 0, 1)), requires_grad=FALSE)
  mP_DO <- torch_tensor(0, requires_grad=FALSE)
  tP_SB <- torch_tensor(0.05, requires_grad=FALSE)
  
  while (count <= 3 && iter <= maxIter) {
    cat('  optim step ', iter, '\n')
    
    theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                  Lmdd1=Lmdd1, Lmdd2=Lmdd2, Lmdd3=Lmdd3, Lmdd4=Lmdd4, Lmdd5=Lmdd5, 
                  Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, tP_SB=tP_SB)
    
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
    gamma3$requires_grad_(TRUE)
    # gamma4$requires_grad_(TRUE)
    tP_SB$requires_grad_(TRUE)
    
    jEta <- torch_full(c(N, Nt, 2, 2, L1), 0)
    jP <- torch_full(c(N, Nt, 2, 2, L1, L1), 0)
    jV <- torch_full(c(N, Nt, 2, 2, O1), NaN)
    jF <- torch_full(c(N, Nt, 2, 2, O1, O1), NaN)
    jEta2 <- torch_full(c(N, Nt, 2, 2, L1), 0)
    jP2 <- torch_full(c(N, Nt, 2, 2, L1, L1), 0)
    mEta <- torch_full(c(N, Nt+1, 2, L1), 0)
    mP <- torch_full(c(N, Nt+1, 2, L1, L1), NaN)
    W <- torch_full(c(N, Nt, 2, 2), NaN)
    jPr <- torch_full(c(N, Nt, 2, 2), 0)
    mLik <- torch_full(c(N, Nt), NaN)
    jPr2 <- torch_full(c(N, Nt, 2, 2), 0)
    mPr_filtered <- torch_full(c(N, Nt+1, 2), NaN)
    mPr_pred <- torch_full(c(N, Nt+1, 2), NaN)
    jLik <- torch_full(c(N, Nt, 2, 2), 0)
    tPr <- torch_full(c(N, Nt, 2, 2), NaN)
    KG <- torch_full(c(N, Nt, 2, 2, L1, O1), 0)
    I_KGLmd <- torch_full(c(N, Nt, 2, 2, L1, L1), NaN)
    subEta <- torch_full(c(N, Nt, 2, 2, L1), NaN)
    eta1_pred <- torch_full(c(N, Nt, L1), NaN)
    P_pred <- torch_full(c(N, Nt, L1, L1), NaN)
    
    mP[,1,,,] <- torch_eye(L1)
    mPr_filtered[,1,1] <- 1 - mP_DO
    mPr_filtered[,1,2] <- mP_DO
    tPr[,,1,2] <- tP_SB
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
        
        if (as.logical(sum(torch_isnan(y1[i,t,]))) <= 0 && DO[i,t] == 0) {
          
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
          
          eta1_pred[i,t,] <- mPr_filtered[i,t,1]$clone()$unsqueeze(-1) * mEta[i,t,1,]$clone() + mPr_filtered[i,t,2]$clone()$unsqueeze(-1) * mEta[i,t,2,]$clone()
          P_pred[i,t,,] <- mPr_filtered[i,t,1]$clone()$unsqueeze(-1)$unsqueeze(-1) * mP[i,t,1,,]$clone() + mPr_filtered[i,t,2]$clone()$unsqueeze(-1)$unsqueeze(-1) * mP[i,t,2,,]$clone()
          tPr[i,t,1,1] <- (gamma1 + eta1_pred[i,t,]$clone()$matmul(gamma2) + eta2[i] * gamma3)$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
          # tPr[i,t,1,1] <- (gamma1 + eta1_pred[i,t,]$clone()$matmul(gamma2) + eta2[i] * gamma3 + eta1_pred[i,t,]$clone()$matmul(gamma4) * eta2[i])$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
          tPr[i,t,2,1] <- 1 - tPr[i,t,1,1]
          jPr[i,t,,] <- tPr[i,t,,]$clone() * mPr_filtered[i,t,]$clone()$unsqueeze(-1)
          mLik[i,t] <- (jLik[i,t,,]$clone() * jPr[i,t,,]$clone())$sum() 
          jPr2[i,t,,] <- jLik[i,t,,]$clone() * jPr[i,t,,]$clone() / mLik[i,t]$clone()$unsqueeze(-1)$unsqueeze(-1) # possible missingness
          mPr_pred[i,t+1,] <- jPr[i,t,,]$sum(2)$clip(min=sEpsilon, max=1-sEpsilon)
          mPr_filtered[i,t+1,] <- jPr2[i,t,,]$sum(2)$clip(min=sEpsilon, max=1-sEpsilon)
          W[i,t,,] <- jPr2[i,t,,]$clone() / mPr_filtered[i,t+1,]$clone()$unsqueeze(-1)
          mEta[i,t+1,,] <- (W[i,t,,]$clone()$unsqueeze(-1) * jEta2[i,t,,,]$clone())$sum(2) 
          subEta[i,t,,,] <- mEta[i,t+1,,]$unsqueeze(2) - jEta2[i,t,,,]
          mP[i,t+1,,,] <- (W[i,t,,]$clone()$unsqueeze(-1)$unsqueeze(-1) * (jP2[i,t,,,,] + subEta[i,t,,,]$clone()$unsqueeze(-1)$matmul(subEta[i,t,,,]$clone()$unsqueeze(-2))))$sum(2) 
        }
        
        if (as.logical(sum(torch_isnan(y1[i,t,]))) > 0 || DO[i,t] == 1) {
          jEta2[i,t,,,] <- jEta2[i,t-1,,,]$clone()
          jP2[i,t,,,,] <- jP2[i,t-1,,,,]$clone()
          mEta[i,t+1,,] <- mEta[i,t,,]$clone()
          mP[i,t+1,,,] <- mP[i,t,,,]$clone()
          jPr2[i,t,,] <- jPr2[i,t-1,,]$clone()  
          if (DO[i,t]==1) {mPr_filtered[i,t+1,1] <- 0; mPr_filtered[i,t+1,2] <- 1} else {mPr_filtered[i,t+1,] <- mPr_filtered[i,t,]$clone()}
          if (DO[i,t]==1) {mPr_pred[i,t+1,1] <- 0; mPr_pred[i,t+1,2] <- 1} else {mPr_pred[i,t+1,] <- mPr_pred[i,t,]$clone()}
          
        }
      }
    }
    
    ######################################################## 
    # Calculate sensitivity and specificity-based loss
    ########################################################
    
    # Convert predicted probabilities to binary predictions based on the threshold
    predicted_dropout <- as.array(mPr_pred[,2:(Nt+1),2]$clone() > 0.5)  # Binary predictions
    
    DO_success <- DO_total <- 0
    NDO_success <- NDO_total <- 0
    
    for (i in 1:N) {
      if (max(predicted_dropout[i,] - DO[i,]) == 1) {
        DO_success <- DO_success + 1
      }
      DO_total <- DO_total + 1
    }
    
    for (i in 1:N) {
      if (max(DO[i,])==0 && max(predicted_dropout[i,])==0) {
        NDO_success <- NDO_success + 1
      }
      NDO_total <- NDO_total + 1
    }
    
    # Calculate sensitivity and specificity
    sensitivity <- DO_success / (DO_total + sEpsilon)  # Adding sEpsilon to avoid division by zero
    specificity <- NDO_success / (NDO_total + sEpsilon)
    
    # Define the loss 
    loss <- -mLik$nanmean()
    
    if (!is.finite(as.numeric(loss))) {
      cat('   error in calculating the sum likelihood', '\n')
      if (iter == 1) {iter_best <- theta_best <- sumLik_best <- output_best <- mPr_pred_best <- mPr_filtered_best <- eta1_best <- P_best <- theta_list_arranged <- output_list <- NULL}
      with_no_grad ({
        for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
      break }
    
    if (iter == 1) {
      theta_list <- data.frame(iter=iter, param=1:length(torch_cat(theta)), value=as.numeric(torch_cat(theta)))
      theta_list_arranged <- theta_list %>%
        group_by(iter) %>%
        mutate(param = param) %>%
        pivot_wider(id_cols = c(iter), names_from = param, values_from = value)
      
      sumLik <- sumLik_init <- -as.numeric(loss)
      output_list <-
        output_new <- data.table(iter=iter, sumLik=sumLik)
      
    } else {
      
      theta_new <- data.frame(iter=iter, param=1:length(torch_cat(theta)), value=as.numeric(torch_cat(theta)))
      theta_new_arranged <- theta_new %>%
        group_by(iter) %>%
        mutate(param = param) %>%
        pivot_wider(id_cols = c(iter), names_from = param, values_from = value)
      
      sumLik_prev <- sumLik
      sumLik <- -as.numeric(loss)
      output_new <- data.frame(iter=iter, sumLik=sumLik)
      
      if (iter == 1) {sumLik_init <- sumLik}
      theta_list_arranged <- rbindlist(list(theta_list_arranged, theta_new_arranged))
      output_list <- rbindlist(list(output_list, output_new))
      
      crit <- ifelse(abs(sumLik - sumLik_init) > sEpsilon, (sumLik - sumLik_prev) / (sumLik - sumLik_init), 0)
      count <- ifelse(crit < stopCrit, count + 1, 0) }
    
    cat('   count =', count, '\n')
    if (count >= 3 || iter == maxIter) {
      cat('   stopping criterion is met', '\n')
      with_no_grad ({
        for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
      break
      
    } else if (sumLik_best < sumLik) {
      iter_best <- iter
      theta_best <- as.numeric(torch_cat(theta))
      sumLik_best <- sumLik
      eta1_best <- as.array(eta1_pred)
      P_best <- as.array(P_pred)
      mPr_pred_best <- as.array(mPr_pred[,2:(Nt+1),2])
      mPr_filtered_best <- as.array(mPr_filtered[,2:(Nt+1),2])
      
      output_best <- output_new
    } 
    
    cat('     sumLik = ', sumLik, '\n')
    cat('     sensitivity = ', sensitivity, '\n')
    cat('     specificity = ', specificity, '\n')
    loss$backward()
    grad <- list()
    
    with_no_grad ({
      for (var in 1:length(theta)) {
        grad[[var]] <- theta[[var]]$grad
        
        if (max(!is.finite(as.numeric(torch_cat(grad[[var]]))))) {
          cat('     numerical blowup', '\n')
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
    
    # for (l1 in 1:L1) {if (as.numeric(Qd[l1]) < lEpsilon) {Qd[l1] <- lEpsilon}}
    # for (o1 in 1:O1) {if (as.numeric(Rd[o1]) < lEpsilon) {Rd[o1] <- lEpsilon}}
    
    gamma1 <- torch_tensor(theta$gamma1, requires_grad=FALSE)
    gamma2 <- torch_tensor(theta$gamma2, requires_grad=FALSE)
    gamma3 <- torch_tensor(theta$gamma3, requires_grad=FALSE)
    # gamma4 <- torch_tensor(theta$gamma4, requires_grad=FALSE)
    tP_SB <- torch_tensor(theta$tP_SB, requires_grad=FALSE); tP_SB$clip_(min=sEpsilon, max=1-sEpsilon)
    theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                  Lmdd1=Lmdd1, Lmdd2=Lmdd2, Lmdd3=Lmdd3, Lmdd4=Lmdd4, Lmdd5=Lmdd5, 
                  Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2, gamma3=gamma3, tP_SB=tP_SB)
    
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
    gamma3$requires_grad_(TRUE)
    # gamma4$requires_grad_(TRUE)
    tP_SB$requires_grad_(TRUE)
    iter <- iter + 1 
    rm(grad)
    gc() }
  
  filter <- list(iter_best=iter_best, theta_best=theta_best, sumLik_best=sumLik_best, output_best=output_best, mPr_pred_best=mPr_pred_best, mPr_filtered_best=mPr_filtered_best, eta1_best=eta1_best, P_best=P_best, theta_list=data.table(theta_list_arranged), output_list=output_list)
  gc()
  return(filter)
}
