result <- function(N, Nt, O1, O2, L1, y1, y2, params, S, eta1_true) {

  lEpsilon <- 1e-3
  ceil <- 1e10
  sEpsilon <- 1e-10
  stopCrit <- 1e-4
  thetaBest <- params
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
  
  B11 <- torch_tensor(thetaBest$B11)
  B12 <- torch_tensor(thetaBest$B12)
  B21d <- torch_tensor(thetaBest$B21d)
  B22d <- torch_tensor(thetaBest$B22d)
  B31 <- torch_tensor(thetaBest$B31)
  B32 <- torch_tensor(thetaBest$B32)
  Lmdd <- torch_tensor(thetaBest$Lmdd); Lmdd[c(1,8)] <- 1; Lmdd[c(2,4,6,7,9,11)] <- 0
  Qd <- torch_tensor(thetaBest$Qd); Qd$clip_(min=lEpsilon)
  Rd <- torch_tensor(thetaBest$Rd); Rd$clip_(min=lEpsilon)
  gamma1 <- torch_tensor(thetaBest$gamma1)
  gamma2 <- torch_tensor(thetaBest$gamma2)
  gamma3 <- torch_tensor(0)
  gamma4 <- torch_tensor(rep(0, L1))
  thetaBest <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
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
  eta1_pred <- torch_full(c(N,Nt+1,L1), NaN)
  P_pred <- torch_full(c(N,Nt+1,L1,L1), NaN)
  
  mP[,1,,,] <- torch_eye(L1)
  mPr[,1,1] <- 1 - sEpsilon
  mPr[,1,2] <- sEpsilon
  tPr[,,1,2] <- sEpsilon
  tPr[,,2,2] <- 1 - sEpsilon
  
  B21 <- B21d$diag()
  B22 <- B22d$diag()
  Lmd <- Lmdd$reshape(c(O1, L1))
  LmdT <- Lmd$transpose(1, 2)
  Q <- Qd$diag()
  R <- Rd$diag()
  
  B1 <- torch_cat(c(B11,B12))$reshape(c(2,L1))
  B2 <- torch_cat(c(B21,B22))$reshape(c(2,L1,L1))
  B3 <- torch_cat(c(B31,B32))$reshape(c(2,L1))
  
  for (t in 1:Nt) {
    
    #################
    # Kalman filter #
    #################
    
    jEta[,t,,,] <- B1$expand(c(N, -1, -1))$unsqueeze(-2) + mEta[,t,,]$unsqueeze(2)$matmul(B2) + eta2$unsqueeze(-1)$unsqueeze(-1)$unsqueeze(-1) * B3$expand(c(N, -1, -1))$unsqueeze(-2)
    jP[,t,,,,] <- mP[,t,,,]$unsqueeze(2)$matmul(B2[2,,]**2) + Q$expand(c(N, 2, 2, -1, -1))
    jV[,t,,,] <- y1[,t,]$unsqueeze(-2)$unsqueeze(-2) - jEta[,t,,,]$matmul(LmdT)
    jF[,t,,,,] <- Lmd$matmul(jP[,t,,,,]$matmul(LmdT)) + R
    KG[,t,,,,] <- jP[,t,,,,]$matmul(LmdT)$matmul(jF[,t,,,,]$cholesky_inverse())
    jEta2[,t,,,] <- jEta[,t,,,] + KG[,t,,,,]$matmul(jV[,t,,,]$unsqueeze(-1))$squeeze()
    I_KGLmd[,t,,,,] <- torch_eye(L1)$expand(c(N,2,2,-1,-1)) - KG[,t,,,,]$matmul(Lmd)
    jP2[,t,,,,] <- I_KGLmd[,t,,,,]$matmul(jP[,t,,,,])$matmul(I_KGLmd[,t,,,,]$transpose(4, 5)) +
      KG[,t,,,,]$matmul(R)$matmul(KG[,t,,,,]$transpose(4, 5))
    jLik[,t,,] <- sEpsilon + const * jF[,t,,,,]$det()**(-1) *
      (-.5 * jF[,t,,,,]$cholesky_inverse()$matmul(jV[,t,,,]$unsqueeze(-1))$squeeze()$unsqueeze(-2)$matmul(jV[,t,,,]$unsqueeze(-1))$squeeze()$squeeze())$exp()
    
    ###################
    # Hamilton filter #
    ###################
    
    eta1_pred[,t,] <- mPr[,t,1]$unsqueeze(-1) * mEta[,t,1,] + mPr[,t,2]$unsqueeze(-1) * mEta[,t,2,]
    if (t > 1) {
      for (i in 1:N) {
        for (j in 1:L1) {
          if (!is.finite(as.numeric(eta1_pred[i,t,j]))) {eta1_pred[i,t,j] <- eta1_pred[i,t-1,j]} } } }
    
    P_pred[,t,,] <- mPr[,t,1]$unsqueeze(-1)$unsqueeze(-1) * mP[,t,1,,] + mPr[,t,2]$unsqueeze(-1)$unsqueeze(-1) * mP[,t,2,,]
    if (t > 1) {
      for (i in 1:N) {
        for (j in 1:L1) {
          if (!is.finite(as.numeric(P_pred[i,t,j,j]))) {P_pred[i,t,j,j] <- P_pred[i,t-1,j,j]} } } }

    tPr[,t,1,1] <- (gamma1 + eta1_pred[,t,]$matmul(gamma2))$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
    tPr[,t,2,1] <- 1 - tPr[,t,1,1]
    jPr[,t,,] <- tPr[,t,,] * mPr[,t,]$unsqueeze(-1)
    mLik[,t] <- (jLik[,t,,] * jPr[,t,,])$sum(c(2,3)) 
    jPr2[,t,,] <- jLik[,t,,] * jPr[,t,,] / mLik[,t]$unsqueeze(-1)$unsqueeze(-1)
    mPr[,t+1,] <- jPr2[,t,,]$sum(3)$clip(min=sEpsilon, max=1-sEpsilon)
    for (i in 1:N) {
      for (j in 1:2) {
        if (!is.finite(as.numeric(mPr[i,t+1,j]))) {mPr[i,t+1,j] <- mPr[i,t,j]} } }

    W[,t,,] <- jPr2[,t,,] / mPr[,t+1,]$unsqueeze(-1)
    mEta[,t+1,,] <- (W[,t,,]$unsqueeze(-1) * jEta2[,t,,,])$sum(3)
    subEta[,t,,,] <- mEta[,t+1,,]$unsqueeze(2) - jEta2[,t,,,]
    mP[,t+1,,,] <- (W[,t,,]$unsqueeze(-1)$unsqueeze(-1) * (jP2[,t,,,,] + subEta[,t,,,]$unsqueeze(-1)$matmul(subEta[,t,,,]$unsqueeze(-2))))$sum(3) }
  
  eta1_pred[,1:(Nt-1),] <- eta1_pred[,2:Nt,]  
  eta1_pred[,Nt,] <- mPr[,Nt+1,1]$unsqueeze(-1) * mEta[,Nt+1,1,] + mPr[,Nt+1,2]$unsqueeze(-1) * mEta[,Nt+1,2,]
  for (i in 1:N) {
    for (j in 1:L1) {
      if (!is.finite(as.numeric(eta1_pred[i,Nt,j]))) {eta1_pred[i,Nt,j] <- eta1_pred[i,Nt-1,j]} } }

  P_pred[,1:(Nt-1),,] <- P_pred[,2:Nt,,]  
  P_pred[,Nt,,] <- mPr[,Nt+1,1]$unsqueeze(-1)$unsqueeze(-1) * mP[,Nt+1,1,,] + mPr[,Nt+1,2]$unsqueeze(-1)$unsqueeze(-1) * mP[,Nt+1,2,,]
  for (i in 1:N) {
    for (j in 1:L1) {
      if (!is.finite(as.numeric(P_pred[i,Nt,j,j]))) {P_pred[i,Nt,j,j] <- P_pred[i,Nt-1,j,j]} } }
  
  jEta[,Nt+1,1,1,] <- B11 + mEta[,Nt+1,1,]$matmul(B21) + eta2$outer(B31)
  jEta[,Nt+1,2,1,] <- B12 + mEta[,Nt+1,1,]$matmul(B22) + eta2$outer(B32)
  jEta[,Nt+1,2,2,] <- B12 + mEta[,Nt+1,2,]$matmul(B22) + eta2$outer(B32)
  
  jP[,Nt+1,1,1,,] <- B21$matmul(mP[,Nt+1,1,,])$matmul(B21) + Q
  jP[,Nt+1,2,1,,] <- B22$matmul(mP[,Nt+1,1,,])$matmul(B22) + Q
  jP[,Nt+1,2,2,,] <- B22$matmul(mP[,Nt+1,2,,])$matmul(B22) + Q
  
  tPr[,Nt+1,1,1] <- (gamma1 + eta1_pred[,Nt,]$matmul(gamma2) + gamma3 * eta2 + eta1_pred[,Nt,]$matmul(gamma4) * eta2)$sigmoid()$clip(min=lEpsilon, max=1-lEpsilon)
  tPr[,Nt+1,2,1] <- 1 - tPr[,Nt+1,1,1]
  
  jPr[,Nt+1,1,1] <- tPr[,Nt+1,1,1] * mPr[,Nt+1,1]
  jPr[,Nt+1,2,1] <- tPr[,Nt+1,2,1] * mPr[,Nt+1,1]
  jPr[,Nt+1,2,2] <- mPr[,Nt+1,2]

  eta1_pred[,Nt+1,] <- jEta[,Nt+1,1,1,] * jPr[,Nt+1,1,1]$unsqueeze(-1) + jEta[,Nt+1,2,1,] * jPr[,Nt+1,2,1]$unsqueeze(-1) + jEta[,Nt+1,2,2,] * jPr[,Nt+1,2,2]$unsqueeze(-1)
  for (i in 1:N) {
    for (j in 1:L1) {
      if (!is.finite(as.numeric(eta1_pred[i,Nt+1,j]))) {eta1_pred[i,Nt+1,j] <- eta1_pred[i,Nt,j]} } }

  P_pred[,Nt+1,,] <- jP[,Nt+1,1,1,,] * jPr[,Nt+1,1,1]$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,1,,] * jPr[,Nt+1,2,1]$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,2,,] * jPr[,Nt+1,2,2]$unsqueeze(-1)$unsqueeze(-1)
  for (i in 1:N) {
    for (j in 1:L1) {
      if (!is.finite(as.numeric(P_pred[i,Nt+1,j,j]))) {P_pred[i,Nt+1,j,j] <- P_pred[i,Nt,j,j]} } }

  ################
  # Kim smoother #
  ################
  
  jPr3 <- torch_full(c(N,Nt,2,2), 0)
  mPr2 <- torch_full(c(N,Nt+1,2), NaN)
  mPr2[,Nt+1,] <- mPr[,Nt+1,]

  for (t in (Nt-1):1) {
    jPr3[,t+1,,] <- mPr2[,t+2,]$unsqueeze(-1) * mPr[,t+1,]$unsqueeze(-2) * tPr[,t+1,,] / jPr[,t+1,,]
    jPr3[,t+1,1,2] <- 0; jPr3[,t+1,,]$clip_(min=lEpsilon, max=1-lEpsilon)
    mPr2[,t+1,] <- jPr3[,t+1,,]$sum(2)$clip(min=lEpsilon, max=1-lEpsilon)
  }
  
  # information criterion
  q <- length(torch_cat(thetaBest)) - 8
  sumLikBest <- as.numeric(torch_sum(mLik[,t]))
  AIC <- -2 * log(sumLikBest) + 2 * q
  BIC <- -2 * log(sumLikBest) + q * log(N * Nt)
  sumLikBest_NxT <- sumLikBest / (N*Nt) 
  
  # contingency table
  cTable1 <- table(factor(S[,Nt+1], levels=c(1,2)), factor(round(as.array(2 - jPr[,Nt+1,1,1])), levels=c(1,2)))
  sensitivity1 <- cTable1[2,2] / sum(cTable1[2,])
  specificity1 <- cTable1[1,1] / sum(cTable1[1,])
  
  # mean score function
  delta1 <- as.numeric(torch_sum((eta1_pred[,Nt+1,] - eta1_true[,Nt+1,])**2))
  delta1_N <- delta1 / N
  
  P_pred_vec <- array(NA, dim=dim(as.array(eta1_pred)))
  for (j in 1:L1) {P_pred_vec[,,j] <- as.array(P_pred[,,j,j])}
  
  output <- list(eta1_pred=as.array(eta1_pred), P_pred=P_pred_vec, mPr=as.array(mPr[,2:(Nt+1),2]), mPr_sm=as.array(mPr2[,2:(Nt+1),2]), sumLik=sumLikBest, sumLik_NxT=sumLikBest_NxT, cTable=cTable1, sensitivity=sensitivity1, specificity=specificity1, delta1=delta1, delta1_N=delta1_N)
  return(output)
}