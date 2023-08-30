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
thetaBest <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
              Lmdd=Lmdd, Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2)

jEta <- torch_full(c(N,Nt+H,2,2,L1), NaN)
jP <- torch_full(c(N,Nt+H,2,2,L1,L1), NaN)
jV <- torch_full(c(N,Nt,2,2,O1), NaN)
jF <- torch_full(c(N,Nt,2,2,O1,O1), NaN)
jEta2 <- torch_full(c(N,Nt,2,2,L1), NaN)
jP2 <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
mEta <- torch_full(c(N,Nt+1+H,2,L1), NaN)
mP <- torch_full(c(N,Nt+1+H,2,L1,L1), NaN)
W <- torch_full(c(N,Nt,2,2), NaN)
jPr <- torch_full(c(N,Nt,2,2), NaN)
mLik <- torch_full(c(N,Nt), NaN)
jPr2 <- torch_full(c(N,Nt,2,2), NaN)
mPr <- torch_full(c(N,Nt+1), NaN)
jLik <- torch_full(c(N,Nt,2,2), NaN)
tPr <- torch_full(c(N,Nt,2), NaN)
KG <- torch_full(c(N,Nt,2,2,L1,O1), NaN)
I_KGLmd <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
denom1 <- torch_full(c(N,Nt), NaN)
denom2 <- torch_full(c(N,Nt), NaN)
subEta <- torch_full(c(N,Nt,2,2,L1), NaN)
subEtaSq <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
eta1_pred <- torch_full(c(N,Nt,L1), NaN)

mEta[,1,,] <- 0
mP[,1,,,] <- torch_eye(L1)
mPr[,1] <- 1
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
  
  if (t == 1) {tPr[,t,1] <- (gamma1$clone() + gamma3$clone() * eta2$clone())$sigmoid()
  } else {
    eta1_pred[,t-1,] <- mPr[,t]$clone()$unsqueeze(-1) * mEta[,t,1,]$clone() + (1 - mPr[,t]$clone())$unsqueeze(-1) * mEta[,t,2,]$clone()
    tPr[,t,1] <- (gamma1$clone() + eta1_pred[,t-1,]$clone()$matmul(gamma2$clone()) + gamma3$clone() * eta2$clone() + eta1_pred[,t-1,]$clone()$matmul(gamma4$clone()) * eta2$clone())$sigmoid() }
  
  jPr[,t,1,1] <- tPr[,t,1]$clone()$clip(min=lEpsilon, max=1-lEpsilon) * mPr[,t]$clone()$clip(min=lEpsilon, max=1-lEpsilon)
  jPr[,t,2,1] <- (1 - tPr[,t,1]$clone()$clip(min=lEpsilon, max=1-sEpsilon)) * mPr[,t]$clone()$clip(min=lEpsilon, max=1-lEpsilon)
  jPr[,t,2,2] <- (1 - mPr[,t]$clone()$clip(min=lEpsilon, max=1-lEpsilon))
  
  mLik[,t] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() +
    jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() +
    jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone()
  
  jPr2[,t,1,1] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() / mLik[,t]$clone()
  jPr2[,t,2,1] <- jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() / mLik[,t]$clone()
  jPr2[,t,2,2] <- jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone() / mLik[,t]$clone()
  
  mPr[,t+1] <- jPr2[,t,1,1]$clone()$clip(min=lEpsilon, max=1-lEpsilon)
  
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

jPr3 <- torch_full(c(N,Nt,2,2), NaN)
mPr2 <- torch_full(c(N,Nt+1), NaN)
jEta3 <- torch_full(c(N,Nt,2,2,L1), NaN)
jP3 <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
jPtilde <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
mEta2 <- torch_full(c(N,Nt+1+H,2,L1), NaN)
mP2 <- torch_full(c(N,Nt+1+H,2,L1,L1), NaN)
subEta2 <- torch_full(c(N,Nt,2,2,L1), NaN)
subEtaSq2 <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
eta1_sm <- torch_full(c(N,Nt,L1), NaN)
P_sm <- torch_full(c(N,Nt,L1,L1), NaN)
subEta3 <- torch_full(c(N,Nt,2,L1), NaN)
subEtaSq3 <- torch_full(c(N,Nt,2,L1,L1), NaN)

mPr2[,Nt+1] <- mPr[,Nt+1]$clone()
mEta2[,Nt+1,,] <- mEta[,Nt+1,,]$clone()
mP2[,Nt+1,,] <- mP[,Nt+1,,]$clone()

for (t in (Nt-1):1) {
  
  jPr3[,t+1,1,1] <- mPr2[,t+2]$clone() * mPr[,t+1]$clone() * tPr[,t+1,1]$clone() / jPr[,t+1,1,1]$clone()
  jPr3[,t+1,2,1] <- (1 - mPr2[,t+2]$clone()) * mPr[,t+1]$clone() * (1 - tPr[,t+1,1]$clone()) / jPr[,t+1,2,]$clone()$sum(2)
  jPr3[,t+1,2,2] <- (1 - mPr2[,t+2]$clone()) * (1 - mPr[,t+1]$clone()) / jPr[,t+1,2,]$clone()$sum(2)

  mPr2[,t+1] <- jPr3[,t+1,,1]$clone()$sum(2)$clip(min=lEpsilon, max=1-lEpsilon)
  
  jPtilde[,t,1,1,,] <- mP[,t+1,1,,]$clone()$matmul(B21$clone())$matmul(jP[,t+1,1,1,,]$clone()$cholesky_inverse())
  jPtilde[,t,2,1,,] <- mP[,t+1,1,,]$clone()$matmul(B22$clone())$matmul(jP[,t+1,2,1,,]$clone()$cholesky_inverse())
  jPtilde[,t,2,2,,] <- mP[,t+1,2,,]$clone()$matmul(B22$clone())$matmul(jP[,t+1,2,2,,]$clone()$cholesky_inverse())
  
  jEta3[,t,1,1,] <- mEta[,t+1,1,]$clone() + (mEta2[,t+2,1,]$clone() - jEta[,t+1,1,1,]$clone())$unsqueeze(-2)$matmul(jPtilde[,t,1,1,,]$clone())$squeeze()
  jEta3[,t,2,1,] <- mEta[,t+1,2,]$clone() + (mEta2[,t+2,2,]$clone() - jEta[,t+1,2,1,]$clone())$unsqueeze(-2)$matmul(jPtilde[,t,2,1,,]$clone())$squeeze()
  jEta3[,t,2,2,] <- mEta[,t+1,2,]$clone() + (mEta2[,t+2,2,]$clone() - jEta[,t+1,2,2,]$clone())$unsqueeze(-2)$matmul(jPtilde[,t,2,2,,]$clone())$squeeze()

  jP3[,t,1,1,,] <- mP[,t+1,1,,]$clone() + jPtilde[,t,1,1,,]$clone()$matmul(mP2[,t+2,1,,]$clone() - jP[,t+1,1,1,,])$matmul(jPtilde[,t,1,1,,]$clone())
  jP3[,t,2,1,,] <- mP[,t+1,2,,]$clone() + jPtilde[,t,2,1,,]$clone()$matmul(mP2[,t+2,2,,]$clone() - jP[,t+1,2,1,,])$matmul(jPtilde[,t,2,1,,]$clone())
  jP3[,t,2,2,,] <- mP[,t+1,2,,]$clone() + jPtilde[,t,2,2,,]$clone()$matmul(mP2[,t+2,2,,]$clone() - jP[,t+1,2,2,,])$matmul(jPtilde[,t,2,2,,]$clone())
  
  mEta2[,t+1,1,] <- (jPr3[,t+1,1,1]$clone() / mPr2[,t+1]$clone())$unsqueeze(-1) * jEta3[,t,1,1,]$clone() + (jPr3[,t+1,2,1]$clone() / mPr2[,t+1]$clone())$unsqueeze(-1) * jEta3[,t,2,1,]$clone()
  mEta2[,t+1,2,] <- (jPr3[,t+1,2,2]$clone() / (1 - mPr2[,t+1]$clone()))$unsqueeze(-1) * jEta3[,t,2,2,]$clone()
    
  subEta2[,t,1,1,] <- mEta2[,t+1,1,]$clone() - jEta3[,t,1,1,]$clone() 
  subEta2[,t,2,1,] <- mEta2[,t+1,1,]$clone() - jEta3[,t,2,1,]$clone()
  subEta2[,t,2,2,] <- mEta2[,t+1,2,]$clone() - jEta3[,t,2,2,]$clone()
  
  subEtaSq2[,t,1,1,,] <- subEta2[,t,1,1,]$clone()$unsqueeze(-1)$matmul(subEta2[,t,1,1,]$clone()$unsqueeze(-2))
  subEtaSq2[,t,2,1,,] <- subEta2[,t,2,1,]$clone()$unsqueeze(-1)$matmul(subEta2[,t,2,1,]$clone()$unsqueeze(-2))
  subEtaSq2[,t,2,2,,] <- subEta2[,t,2,2,]$clone()$unsqueeze(-1)$matmul(subEta2[,t,2,2,]$clone()$unsqueeze(-2))
  
  mP2[,t+1,1,,] <- (jPr3[,t+1,1,1]$clone() / mPr2[,t+1]$clone())$unsqueeze(-1)$unsqueeze(-1) * (jP3[,t,1,1,,]$clone() + subEtaSq2[,t,1,1,,]$clone()) + (jPr3[,t+1,2,1]$clone() / mPr2[,t+1]$clone())$unsqueeze(-1)$unsqueeze(-1) * (jP3[,t,2,1,,]$clone() + subEtaSq2[,t,2,1,,]$clone()) 
  mP2[,t+1,2,,] <- (jPr3[,t+1,2,2]$clone() / (1 - mPr2[,t+1]$clone()))$unsqueeze(-1)$unsqueeze(-1) * (jP3[,t,2,2,,]$clone() + subEtaSq2[,t,2,2,,]$clone()) 
  
  eta1_sm[,t,] <- mPr2[,t+1]$clone()$unsqueeze(-1) * mEta2[,t+1,1,]$clone() + (1 - mPr2[,t+1]$clone())$unsqueeze(-1) * mEta2[,t+1,2,]$clone() 
  
  subEta3[,t,1,] <- eta1_sm[,t,]$clone() - mEta2[,t+1,1,]$clone() 
  subEta3[,t,2,] <- eta1_sm[,t,]$clone() - mEta2[,t+1,2,]$clone() 

  subEtaSq3[,t,1,,] <- subEta3[,t,1,]$clone()$unsqueeze(-1)$matmul(subEta3[,t,1,]$clone()$unsqueeze(-2))
  subEtaSq3[,t,2,,] <- subEta3[,t,2,]$clone()$unsqueeze(-1)$matmul(subEta3[,t,2,]$clone()$unsqueeze(-2))

  P_sm[,t,,] <- mPr2[,t+1]$clone()$unsqueeze(-1)$unsqueeze(-1) * (mP2[,t+1,1,,]$clone() + subEtaSq3[,t,1,,]$clone()) + (1 - mPr2[,t+1]$clone())$unsqueeze(-1)$unsqueeze(-1) * (mP2[,t+1,2,,]$clone() + subEtaSq3[,t,2,,]$clone()) }

DO <- 2 - mPr[,2:(Nt+1)]
DO2 <- 2 - mPr2[,2:(Nt+1)]

df_S <- melt(S); colnames(df_S) <- c('ID', 'time', 'S')
plot_S <- ggplot(data=df_S, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')

df_Pr <- melt(as.array(DO)); colnames(df_Pr) <- c('ID', 'time', 'S')
plot_Pr <- ggplot(data=df_Pr, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')

df_Pr2 <- melt(as.array(DO2)); colnames(df_Pr2) <- c('ID', 'time', 'S')
plot_Pr2 <- ggplot(data=df_Pr2, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')

df_Pr_diff <- melt(as.array(DO2)); colnames(df_Pr_diff) <- c('ID', 'time', 'S')
df_Pr_diff$S <- abs(df_Pr_diff$S - df_S$S)
plot_Pr_diff <- ggplot(data=df_Pr_diff, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')

plot_grid(plot_S, plot_Pr, plot_Pr2, plot_Pr_diff, labels = "AUTO")

df_eta1 <- melt(as.array(eta1_sm[3:4,(Nt-20):(Nt-1),1])); colnames(df_eta1) <- c('ID', 'time', 'eta1')
plot_eta1 <- ggplot(data=df_eta1, aes(time, eta1, group=ID, color=as.factor(ID))) + geom_line() + ylim(-3, 3) + theme(legend.position='none')
plot_grid(plot_eta1, labels = "AUTO")

