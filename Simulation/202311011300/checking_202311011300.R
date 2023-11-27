checking <- function(seed, N, Nt, O1, O2, L1, y1, y2, S, eta1_true, theta) {
  set.seed(101*seed)
  
  lEpsilon <- 1e-3
  ceil <- 1e15
  sEpsilon <- 1e-15
  epsilon <- 1e-8
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
  
  sumLik_best <- 0
  
  # initialize parameters
  B11 <- torch_tensor(theta[1:2])
  B12 <- torch_tensor(theta[3:4])
  B21d <- torch_tensor(theta[5:6])
  B22d <- torch_tensor(theta[7:8])
  B31 <- torch_tensor(theta[9:10])
  B32 <- torch_tensor(theta[11:12])
  Lmdd <- torch_tensor(theta[13:24])
  gamma1 <- torch_tensor(theta[25]) # fixed
  gamma2 <- torch_tensor(theta[26:27])
  Qd <- torch_tensor(theta[28:29]) # fixed
  Rd <- torch_tensor(theta[30:35]) # fixed
  
  q <- length(torch_cat(theta)) - 8
  
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
  Lmd <- Lmdd$reshape(c(O1, L1))
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
    jV[,t,,,] <- y1[,t,]$unsqueeze(-2)$unsqueeze(-2) - jEta[,t,,,]$clone()$matmul(LmdT)
    jF[,t,,,,] <- Lmd$matmul(jP[,t,,,,]$matmul(LmdT)) + R
    KG[,t,,,,] <- jP[,t,,,,]$matmul(LmdT)$matmul(jF[,t,,,,]$clone()$cholesky_inverse())
    jEta2[,t,,,] <- jEta[,t,,,] + KG[,t,,,,]$clone()$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()
    I_KGLmd[,t,,,,] <- torch_eye(L1)$expand(c(N,2,2,-1,-1)) - KG[,t,,,,]$clone()$matmul(Lmd)
    jP2[,t,,,,] <- I_KGLmd[,t,,,,]$clone()$matmul(jP[,t,,,,]$clone())$matmul(I_KGLmd[,t,,,,]$clone()$transpose(4, 5)) +
      KG[,t,,,,]$clone()$matmul(R)$matmul(KG[,t,,,,]$clone()$transpose(4, 5))
    jLik[,t,,] <- sEpsilon + const * jF[,t,,,,]$clone()$det()$clip(min=sEpsilon, max=ceil)**(-1) *
      (-.5 * jF[,t,,,,]$clone()$cholesky_inverse()$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()$unsqueeze(-2)$matmul(jV[,t,,,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
    
    ###################
    # Hamilton filter #
    ###################
    
    eta1_pred[,t,] <- mPr[,t,1]$clone()$unsqueeze(-1) * mEta[,t,1,]$clone() + mPr[,t,2]$clone()$unsqueeze(-1) * mEta[,t,2,]$clone()
    P_pred[,t,,] <- mPr[,t,1]$unsqueeze(-1)$unsqueeze(-1) * mP[,t,1,,] + mPr[,t,2]$unsqueeze(-1)$unsqueeze(-1) * mP[,t,2,,]
    tPr[,t,1,1] <- (gamma1 + eta1_pred[,t,]$clone()$matmul(gamma2))$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
    tPr[,t,2,1] <- 1 - tPr[,t,1,1]
    jPr[,t,,] <- tPr[,t,,]$clone() * mPr[,t,]$clone()$unsqueeze(-1)
    mLik[,t] <- (jLik[,t,,]$clone() * jPr[,t,,]$clone())$sum(c(2,3)) 
    jPr2[,t,,] <- jLik[,t,,]$clone() * jPr[,t,,]$clone() / mLik[,t]$clone()$unsqueeze(-1)$unsqueeze(-1)
    mPr[,t+1,] <- jPr2[,t,,]$sum(3)$clip(min=sEpsilon, max=1-sEpsilon)
    W[,t,,] <- jPr2[,t,,]$clone() / mPr[,t+1,]$clone()$unsqueeze(-1)
    mEta[,t+1,,] <- (W[,t,,]$clone()$unsqueeze(-1) * jEta2[,t,,,]$clone())$sum(3)
    subEta[,t,,,] <- mEta[,t+1,,]$unsqueeze(2) - jEta2[,t,,,]
    mP[,t+1,,,] <- (W[,t,,]$clone()$unsqueeze(-1)$unsqueeze(-1) * (jP2[,t,,,,] + subEta[,t,,,]$clone()$unsqueeze(-1)$matmul(subEta[,t,,,]$clone()$unsqueeze(-2))))$sum(3) }
  
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
  
  loss <- -mLik$sum()
  
  if (!is.finite(as.numeric(loss))) {
    # print('   error in calculating the sum likelihood')
    with_no_grad ({
      for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
    break }
  
  # contingency table
  # cTable <- table(factor(S[,Nt+1], levels=c(1,2)), factor(1 + round(as.numeric(mPr[,Nt+2,2])), levels=c(1,2)))
  cTable <- table(factor(S[,Nt+1], levels=c(1,2)), factor(1 + as.numeric(as.numeric(mPr[,Nt+2,2]) > quantile(as.numeric(mPr[,Nt+2,2]), 0.15)), levels=c(1,2)))
  TP <- cTable[2,2]
  TN <- cTable[1,1]
  FP <- cTable[1,2]
  FN <- cTable[2,1]
  sensitivity <- TP / (TP + FN)
  specificity <- TN / (TN + FP)
  
  # mean score function
  delta <- as.numeric(sum((eta1_pred[,Nt+2,] - eta1_true[,Nt+1,])**2))
  
  check <- list(TP=TP, TN=TN, FP=FP, FN=FN, sensitivity=sensitivity, specificity=specificity, cTable=cTable, S=S, mPr=as.numeric(mPr[,Nt+2,2]))
  gc()
  return(check)
}


setwd('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311011300')

source('library_202311011300.R')
source('DGP_202311011300.R')
source('filtering_202311011300.R')

library_load()

seeds <- seeds <- c(1:3, 5:69)
N <- 75
Nt <- 25
O1 <- 6
O2 <- 3
L1 <- 2

for (seed in seeds) {
  df <- readRDS(paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311011300/output/df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
  filter <- readRDS(paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311011300/output/filter__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
  check <- checking(seed, N, Nt, O1, O2, L1, df$y1, df$y2, df$S, df$eta1_true, filter$theta_best)
  saveRDS(check, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311011300/output/check__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))
} 