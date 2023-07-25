theta <- thetaBest

# switch off the gradient tracking
B11 <- torch_tensor(theta$B11)
B12 <- torch_tensor(theta$B12)
B21 <- torch_tensor(theta$B21)
B22 <- torch_tensor(theta$B22)
B31 <- torch_tensor(theta$B31)
B32 <- torch_tensor(theta$B32)
Q1 <- torch_tensor(theta$Q1)
Q2 <- torch_tensor(theta$Q2)
R1 <- torch_tensor(theta$R1)
R2 <- torch_tensor(theta$R2)
gamma11 <- torch_tensor(theta$gamma11)
gamma21 <- torch_tensor(theta$gamma21)

d <- torch_tensor(df$icept1)
Lmdv <- torch_tensor(df$coef1)
Lmd <- torch_full(c(No1,Nf1), 0)
Lmd[1:3,1] <- Lmdv[1:3]
Lmd[4:5,2] <- Lmdv[4:5]
Lmd[6:7,3] <- Lmdv[6:7]
Lmd[8:9,4] <- Lmdv[8:9]
Lmd[10:11,5] <- Lmdv[10:11]
Lmd[12:14,6] <- Lmdv[12:14]
Lmd[15:17,7] <- Lmdv[15:17]

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

mEta[,1,,] <- 0
mP[,1,,,] <- torch_eye(Nf1)
mPr[,1] <- sEpsilon 

#######################
# extended Kim filter #
#######################
for (t in 1:Nt) {
  if (t%%10==0) {cat('   t=', t, '\n')}
  
  jEta[,t,1,1,] <- B11 + mEta[,t,1,]$clone()$matmul(B21) + (eta2$clone() * B31)$unsqueeze(-1)
  jEta[,t,2,1,] <- B12 + mEta[,t,1,]$clone()$matmul(B22) + (eta2$clone() * B32)$unsqueeze(-1)
  jEta[,t,2,2,] <- B12 + mEta[,t,2,]$clone()$matmul(B22) + (eta2$clone() * B32)$unsqueeze(-1)
  
  jDelta[,t,1,1,] <- eta1[,t,]$clone() - jEta[,t,1,1,]$clone() 
  jDelta[,t,2,1,] <- eta1[,t,]$clone() - jEta[,t,2,1,]$clone() 
  jDelta[,t,2,2,] <- eta1[,t,]$clone() - jEta[,t,2,2,]$clone() 
  
  jP[,t,1,1,,] <- B21$matmul(mP[,t,1,,]$clone())$matmul(B21$transpose(1, 2)) + Q1
  jP[,t,2,1,,] <- B22$matmul(mP[,t,1,,]$clone())$matmul(B22$transpose(1, 2)) + Q2
  jP[,t,2,2,,] <- B22$matmul(mP[,t,2,,]$clone())$matmul(B22$transpose(1, 2)) + Q2
  
  jV[,t,1,1,] <- y1[,t,]$clone() - (d + jEta[,t,1,1,]$clone()$matmul(Lmd$transpose(1, 2))) 
  jV[,t,2,1,] <- y1[,t,]$clone() - (d + jEta[,t,2,1,]$clone()$matmul(Lmd$transpose(1, 2))) 
  jV[,t,2,2,] <- y1[,t,]$clone() - (d + jEta[,t,2,2,]$clone()$matmul(Lmd$transpose(1, 2))) 
  
  jF[,t,1,1,,] <- Lmd$matmul(jP[,t,1,1,,]$clone())$matmul(Lmd$transpose(1, 2)) + R1
  jF[,t,2,1,,] <- Lmd$matmul(jP[,t,2,1,,]$clone())$matmul(Lmd$transpose(1, 2)) + R2
  jF[,t,2,2,,] <- Lmd$matmul(jP[,t,2,2,,]$clone())$matmul(Lmd$transpose(1, 2)) + R2
  
  KG[,t,1,1,,] <- jP[,t,1,1,,]$clone()$matmul(Lmd$transpose(1, 2))$matmul(jF[,t,1,1,,]$clone()$cholesky_inverse())
  KG[,t,2,1,,] <- jP[,t,2,1,,]$clone()$matmul(Lmd$transpose(1, 2))$matmul(jF[,t,2,1,,]$clone()$cholesky_inverse())
  KG[,t,2,2,,] <- jP[,t,2,2,,]$clone()$matmul(Lmd$transpose(1, 2))$matmul(jF[,t,2,2,,]$clone()$cholesky_inverse())
  
  jEta2[,t,1,1,] <- jEta[,t,1,1,]$clone() + KG[,t,1,1,,]$clone()$matmul(jV[,t,1,1,]$clone()$unsqueeze(-1))$squeeze()
  jEta2[,t,2,1,] <- jEta[,t,2,1,]$clone() + KG[,t,2,1,,]$clone()$matmul(jV[,t,2,1,]$clone()$unsqueeze(-1))$squeeze()
  jEta2[,t,2,2,] <- jEta[,t,2,2,]$clone() + KG[,t,2,2,,]$clone()$matmul(jV[,t,2,2,]$clone()$unsqueeze(-1))$squeeze()
  
  I_KGLmd[,t,1,1,,] <- torch_eye(Nf1) - KG[,t,1,1,,]$clone()$matmul(Lmd)
  I_KGLmd[,t,2,1,,] <- torch_eye(Nf1) - KG[,t,2,1,,]$clone()$matmul(Lmd)
  I_KGLmd[,t,2,2,,] <- torch_eye(Nf1) - KG[,t,2,2,,]$clone()$matmul(Lmd)
  
  jP2[,t,1,1,,] <- I_KGLmd[,t,1,1,,]$clone()$matmul(jP[,t,1,1,,]$clone())$matmul(I_KGLmd[,t,1,1,,]$clone()$transpose(2, 3)) + 
    KG[,t,1,1,,]$clone()$matmul(R1)$matmul(KG[,t,1,1,,]$clone()$transpose(2, 3))
  jP2[,t,2,1,,] <- I_KGLmd[,t,2,1,,]$clone()$matmul(jP[,t,2,1,,]$clone())$matmul(I_KGLmd[,t,2,1,,]$clone()$transpose(2, 3)) + 
    KG[,t,2,1,,]$clone()$matmul(R2)$matmul(KG[,t,2,1,,]$clone()$transpose(2, 3))
  jP2[,t,2,2,,] <- I_KGLmd[,t,2,2,,]$clone()$matmul(jP[,t,2,2,,]$clone())$matmul(I_KGLmd[,t,2,2,,]$clone()$transpose(2, 3)) + 
    KG[,t,2,2,,]$clone()$matmul(R2)$matmul(KG[,t,2,2,,]$clone()$transpose(2, 3))
  
  jLik[,t,1,1] <- sEpsilon + jP[,t,1,1,,]$clone()$det()**(-1) * 
    (-.5 * jDelta[,t,1,1,]$clone()$unsqueeze(2)$matmul(jP[,t,1,1,,]$clone()$cholesky_inverse())$matmul(jDelta[,t,1,1,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
  jLik[,t,2,1] <- sEpsilon + jP[,t,2,1,,]$clone()$det()**(-1) * 
    (-.5 * jDelta[,t,2,1,]$clone()$unsqueeze(2)$matmul(jP[,t,2,1,,]$clone()$cholesky_inverse())$matmul(jDelta[,t,2,1,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
  jLik[,t,2,2] <- sEpsilon + jP[,t,2,2,,]$clone()$det()**(-1) * 
    (-.5 * jDelta[,t,2,2,]$clone()$unsqueeze(2)$matmul(jP[,t,2,2,,]$clone()$cholesky_inverse())$matmul(jDelta[,t,2,2,]$clone()$unsqueeze(-1))$squeeze()$squeeze())$exp()
  
  if (t == 1) {tPr[,t,1] <- gamma11$sigmoid() 
  } else {tPr[,t,1] <- (gamma11 + eta1[,t-1,]$clone()$matmul(gamma21) + tau11 * x[,t-1])$sigmoid() }
  
  jPr[,t,1,1] <- (1-tPr[,t,1]$clone()) * (1-mPr[,t]$clone())
  jPr[,t,2,1] <- tPr[,t,1]$clone() * (1-mPr[,t]$clone()) 
  jPr[,t,2,2] <- mPr[,t]$clone() 
  
  mLik[,t] <- (jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone()) +
    (jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone()) +
    (jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone())
  
  jPr2[,t,1,1] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() / mLik[,t]$clone() + sEpsilon
  jPr2[,t,2,1] <- jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() / mLik[,t]$clone() + sEpsilon
  jPr2[,t,2,2] <- jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone() / mLik[,t]$clone() + sEpsilon
  
  mPr[,t+1] <- jPr2[,t,2,]$clone()$sum(dim=2)
  
  W[,t,1,1] <- jPr2[,t,1,1]$clone() / (1 - mPr[,t+1]$clone())
  W[,t,2,1] <- jPr2[,t,2,1]$clone() / mPr[,t+1]$clone()
  W[,t,2,2] <- jPr2[,t,2,2]$clone() / mPr[,t+1]$clone()
  
  mEta[,t+1,1,] <- W[,t,1,1]$clone()$unsqueeze(-1) * jEta2[,t,1,1,]$clone()
  mEta[,t+1,2,] <- (W[,t,2,]$clone()$unsqueeze(-1) * jEta2[,t,2,,]$clone())$sum(2)
  
  subEta[,t,1,1,] <- mEta[,t+1,1,]$clone() - jEta2[,t,1,1,]$clone()
  subEta[,t,2,1,] <- mEta[,t+1,2,]$clone() - jEta2[,t,2,1,]$clone()
  subEta[,t,2,2,] <- mEta[,t+1,2,]$clone() - jEta2[,t,2,2,]$clone()
  
  subEtaSq[,t,1,1,,] <- subEta[,t,1,1,]$clone()$unsqueeze(dim=-1)$matmul(subEta[,t,1,1,]$clone()$unsqueeze(dim=-2))
  subEtaSq[,t,2,1,,] <- subEta[,t,2,1,]$clone()$unsqueeze(dim=-1)$matmul(subEta[,t,2,1,]$clone()$unsqueeze(dim=-2))
  subEtaSq[,t,2,2,,] <- subEta[,t,2,2,]$clone()$unsqueeze(dim=-1)$matmul(subEta[,t,2,2,]$clone()$unsqueeze(dim=-2))
  
  mP[,t+1,1,,] <- W[,t,1,1]$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) * (jP2[,t,1,1,,]$clone() + subEtaSq[,t,1,1,,]$clone()) 
  mP[,t+1,2,,] <- (W[,t,2,]$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) * (jP2[,t,2,,,]$clone() + subEtaSq[,t,2,,,]$clone()))$sum(2) }

colors <- rainbow(N)
c <- brewer.pal(8, "Dark2")

i <- 30 # person E {1, ... , N}
plot(x[i,], lwd=1.5, ylim=c(0,1), type="l")
lines(mPr[i,2:(Nt+1)], lwd=1.5, col=c[i%%8+1]) 

for (t in 1:Nt) {
  cat('\n', 't=', t, '\n')
  print(table(as.numeric(x[,t] - as.numeric(mPr[,t+1] > .5)) == 0)) }


mean(abs(mPr[,2:(Nt+1)] - torch_tensor(x)))
table(as.numeric(mPr[,2:(Nt+1)] > .5 - torch_tensor(x)))

as.numeric(x[1,]) # drop out at t=21
as.numeric(mPr[1,2:(Nt+1)] > .5) # drop out at t=19

as.numeric(x[2,]) # drop out at t=51
as.numeric(mPr[2,2:(Nt+1)] > .5) # drop out at t=28

as.numeric(x[3,]) # no drop out
as.numeric(mPr[3,2:(Nt+1)] > .5) # no drop out
