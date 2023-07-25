theta <- thetaBest

# switch off the gradient tracking
a1 <- torch_tensor(theta$a1)
a2 <- torch_tensor(theta$a2)
B1d <- torch_tensor(theta$B1d)
B2d <- torch_tensor(theta$B2d)
C1d <- torch_tensor(theta$C1d)
C2d <- torch_tensor(theta$C2d)
D1 <- torch_tensor(theta$D1)
D2 <- torch_tensor(theta$D2)
k1 <- torch_tensor(theta$k1)
k2 <- torch_tensor(theta$k2)
Lmd1v <- torch_tensor(theta$Lmd1v)
Lmd2v <- torch_tensor(theta$Lmd2v)
Omega1v <- torch_tensor(theta$Omega1v)
Omega2v <- torch_tensor(theta$Omega2v)
M1 <- torch_tensor(theta$M1)
M2 <- torch_tensor(theta$M2)
alpha1 <- torch_tensor(theta$alpha1)
alpha2 <- torch_tensor(theta$alpha2)
beta1 <- torch_tensor(theta$beta1)
beta2 <- torch_tensor(theta$beta2)
gamma1 <- torch_tensor(theta$gamma1)
gamma2 <- torch_tensor(theta$gamma2)
rho1 <- torch_tensor(theta$rho1)
rho2 <- torch_tensor(theta$rho2)
tau1 <- torch_tensor(theta$tau1)
tau2 <- torch_tensor(theta$tau2)
Q1d <- torch_tensor(theta$Q1d)
Q2d <- torch_tensor(theta$Q2d)
R1d <- torch_tensor(theta$R1d)
R2d <- torch_tensor(theta$R2d) 

a <- list(a1, a2)
B1 <- torch_diag(B1d)
B2 <- torch_diag(B2d)
B <- list(B1, B2)
C1 <- torch_diag(C1d)
C2 <- torch_diag(C2d)
C <- list(C1, C2)
D <- list(D1, D2)
k <- list(k1, k2)
Lmd1 <- Lmd2 <- torch_full(c(No1,Nf1), 0)
Lmd1[1:3,1] <- Lmd1v[1:3]; Lmd1[4:5,2] <- Lmd1v[4:5]
Lmd1[6:7,3] <- Lmd1v[6:7]; Lmd1[8:9,4] <- Lmd1v[8:9]
Lmd1[10:11,5] <- Lmd1v[10:11]; Lmd1[12:14,6] <- Lmd1v[12:14]
Lmd1[15:17,7] <- Lmd1v[15:17]
Lmd2[1:3,1] <- Lmd2v[1:3]; Lmd2[4:5,2] <- Lmd2v[4:5]
Lmd2[6:7,3] <- Lmd2v[6:7]; Lmd2[8:9,4] <- Lmd2v[8:9]
Lmd2[10:11,5] <- Lmd2v[10:11]; Lmd2[12:14,6] <- Lmd2v[12:14]
Lmd2[15:17,7] <- Lmd2v[15:17]
Lmd <- list(Lmd1, Lmd2)
Omega1 <- Omega2 <- torch_full(c(No1,Nf1), 0)
Omega1[1:3,1] <- Omega1v[1:3]; Omega1[4:5,2] <- Omega1v[4:5]
Omega1[6:7,3] <- Omega1v[6:7]; Omega1[8:9,4] <- Omega1v[8:9]
Omega1[10:11,5] <- Omega1v[10:11]; Omega1[12:14,6] <- Omega1v[12:14]
Omega1[15:17,7] <- Omega1v[15:17]
Omega2[1:3,1] <- Omega2v[1:3]; Omega2[4:5,2] <- Omega2v[4:5]
Omega2[6:7,3] <- Omega2v[6:7]; Omega2[8:9,4] <- Omega2v[8:9]
Omega2[10:11,5] <- Omega2v[10:11]; Omega2[12:14,6] <- Omega2v[12:14]
Omega2[15:17,7] <- Omega2v[15:17]
Omega <- list(Omega1, Omega2)
M <- list(M1, M2)
alpha <- list(alpha1, alpha2)
beta <- list(beta1, beta2)
gamma <- list(gamma1, gamma2)
rho <- list(rho1, rho2)
tau <- list(tau1, tau2)
Q1 <- Q1d$diag()
Q2 <- Q2d$diag()
Q <- list(Q1, Q2)
R1 <- R1d$diag()
R2 <- R2d$diag()
R <- list(R1, R2)

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

# initialize latent variables
mEta[,1,,] <- 0
mP[,1,,,] <- 0; mP[,1,,,]$add_(torch_eye(Nf1)) 

# initialize P(s'|eta_0)
mPr[,1] <- sEpsilon 

#######################
# extended Kim filter #
#######################
for (t in 1:Nt) { 
  if (t%%10==0) {cat('   t=', t, '\n')}
  
  # Kalman Filter
  for (s1 in 1:2) {
    # Eq.2
    jEta[,t,s1,,] <- a[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) + 
      mEta[,t,,]$clone()$matmul(B[[s1]]) + 
      mEta[,t,,]$clone()$matmul(C[[s1]]) * 
      eta2$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) + 
      x[,t]$clone()$outer(D[[s1]])$unsqueeze(dim=2) 
    with_no_grad(jEta[,t,s1,,]$clip_(-ceil, ceil))
    
    # Eq.3
    jDelta[,t,s1,,] <- eta1[,t,]$clone()$unsqueeze(dim=2) - jEta[,t,s1,,]$clone() 
    with_no_grad(jDelta[,t,s1,,]$clip_(-ceil, ceil))
    
    # Eq.4
    jP[,t,s1,,,] <- B[[s1]]$matmul(mP[,t,,,]$clone())$matmul(B[[s1]]$transpose(1, 2)) + 
      C[[s1]]$matmul(mP[,t,,,]$clone())$matmul(C[[s1]]$transpose(1, 2)) * eta2$clone()$square()$unsqueeze(dim=-1)$unsqueeze(dim=-1)$unsqueeze(dim=-1) + 
      Q[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) 
    with_no_grad ({ 
      jP[,t,s1,,,] <- (jP[,t,s1,,,] + jP[,t,s1,,,]$transpose(3, 4)) / 2
      jP[,t,s1,,,]$clip_(-ceil, ceil)
      jPEig <- linalg_eigh(jP[,t,s1,,,])
      jPEig[[1]]$real$clip_(sEpsilon, ceil)
      for (row in 1:N) {
        for (s2 in 1:2) {
          jP[row,t,s1,s2,,] <- jPEig[[2]]$real[row,s2,,]$matmul(jPEig[[1]]$real[row,s2,]$diag())$matmul(jPEig[[2]]$real[row,s2,,]$transpose(1, 2))
          while (as.numeric(jP[row,t,s1,s2,,]$det()) < 0) {
            jP[row,t,s1,s2,,]$add_(lEpsilon * torch_eye(Nf1)) } } } }) 
    
    # Eq.5
    jV[,t,s1,,] <- y1[,t,]$clone()$unsqueeze(dim=2) -
      (k[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) + 
         jEta[,t,s1,,]$clone()$matmul(Lmd[[s1]]$transpose(1, 2)) + 
         jEta[,t,s1,,]$clone()$matmul(Omega[[s1]]$transpose(1, 2)) * 
         eta2$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) + 
         x[,t]$clone()$outer(M[[s1]])$unsqueeze(dim=2))        
    with_no_grad(jV[,t,s1,,]$clip_(-ceil, ceil))
    
    # Eq.6
    jF[,t,s1,,,] <- Lmd[[s1]]$matmul(jP[,t,s1,,,]$clone())$matmul(Lmd[[s1]]$transpose(1, 2)) + 
      Omega[[s1]]$matmul(jP[,t,s1,,,]$clone())$matmul(Omega[[s1]]$transpose(1, 2)) * eta2$clone()$square()$unsqueeze(dim=-1)$unsqueeze(dim=-1)$unsqueeze(dim=-1) +
      R[[s1]]$unsqueeze(dim=1)$unsqueeze(dim=1) 
    with_no_grad ({
      jF[,t,s1,,,]$clip_(-ceil, ceil)
      jF[,t,s1,,,] <- (jF[,t,s1,,,] + jF[,t,s1,,,]$transpose(3, 4)) / 2
      jFEig <- linalg_eigh(jF[,t,s1,,,])
      jFEig[[1]]$real$clip_(sEpsilon, ceil)
      for (row in 1:N) {
        for (s2 in 1:2) {
          jF[row,t,s1,s2,,] <- jFEig[[2]]$real[row,s2,,]$matmul(jFEig[[1]]$real[row,s2,]$diag())$matmul(jFEig[[2]]$real[row,s2,,]$transpose(1, 2))
          while (as.numeric(jF[row,t,s1,s2,,]$det()) < 0) {
            jF[row,t,s1,s2,,]$add_(lEpsilon * torch_eye(No1)) } } } }) 
    
    # kalman gain function
    KG[,t,s1,,,] <- jP[,t,s1,,,]$clone()$matmul(Lmd[[s1]]$transpose(1, 2))$matmul(linalg_inv_ex(jF[,t,s1,,,]$clone())$inverse)
    with_no_grad(KG[,t,s1,,,]$clip_(-ceil, ceil))
    
    for (s2 in 1:2) {
      # Eq.7
      jEta2[,t,s1,s2,] <- jEta[,t,s1,s2,]$clone() + KG[,t,s1,s2,,]$clone()$matmul(jV[,t,s1,s2,]$clone()$unsqueeze(dim=-1))$squeeze()
      with_no_grad(jEta2[,t,s1,s2,]$clip_(-ceil, ceil))
      I_KGLmd[,t,s1,s2,,] <- torch_eye(Nf1)$unsqueeze(dim=1) - KG[,t,s1,s2,,]$clone()$matmul(Lmd[[s1]])
      with_no_grad(I_KGLmd[,t,s1,s2,,]$clip_(-ceil, ceil))
      
      # Eq.9
      jP2[,t,s1,s2,,] <- I_KGLmd[,t,s1,s2,,]$clone()$matmul(jP[,t,s1,s2,,]$clone())$matmul(I_KGLmd[,t,s1,s2,,]$clone()$transpose(2, 3)) + 
        KG[,t,s1,s2,,]$clone()$matmul(R[[s1]])$matmul(KG[,t,s1,s2,,]$clone()$transpose(2, 3))
      with_no_grad(jP2[,t,s1,s2,,]$clip_(-ceil, ceil))
      
      with_no_grad ({
        jP2Eig <- linalg_eigh(jP2[,t,s1,s2,,]) 
        jP2Eig[[1]]$real$clip_(sEpsilon, ceil)
        for (row in 1:N) {
          jP2[row,t,s1,s2,,] <- jP2Eig[[2]]$real[s2,,]$matmul(jP2Eig[[1]]$real[s2,]$diag())$matmul(jP2Eig[[2]]$real[s2,,]$transpose(1, 2)) 
          while (as.numeric(jP2[row,t,s1,s2,,]$det()) < 0) {jP2[row,t,s1,s2,,]$add_(sEpsilon * torch_eye(Nf1))} } }) 
      
      # joint likelihood f(eta_{t}|s,s',eta_{t-1})
      # Eq.12
      jLik[,t,s1,s2] <- (2*pi)**(-Nf1/2) * jP[,t,s1,s2,,]$clone()$det()**(-1) * 
        (-.5 * jDelta[,t,s1,s2,]$clone()$unsqueeze(dim=2)$matmul(linalg_inv_ex(jP[,t,s1,s2,,]$clone())$inverse)$matmul(jDelta[,t,s1,s2,]$clone()$unsqueeze(dim=-1))$squeeze()$squeeze())$exp()
      with_no_grad(jLik[,t,s1,s2]$clip_(min=sEpsilon)) } }
  
  # transition probability P(s|s',eta_{t-1})  
  if (t == 1) {
    tPr[,t,1] <- (alpha[[1]] + gamma[[1]] * eta2$clone())$sigmoid()
    tPr[,t,2] <- (alpha[[2]] + gamma[[2]] * eta2$clone())$sigmoid()
    
  } else {
    tPr[,t,1] <- (alpha[[1]] + eta1[,t-1,]$clone()$matmul(beta[[1]]) + gamma[[1]] * eta2$clone() + eta1[,t-1,]$clone()$matmul(rho[[1]]) * eta2$clone() + tau[[1]] * x[,t-1]$clone())$sigmoid() 
    tPr[,t,2] <- (alpha[[2]] + eta1[,t-1,]$clone()$matmul(beta[[2]]) + gamma[[2]] * eta2$clone() + eta1[,t-1,]$clone()$matmul(rho[[2]]) * eta2$clone() + tau[[2]] * x[,t-1]$clone())$sigmoid() 
  }
  
  jPr[,t,2,2] <- tPr[,t,2]$clone() * mPr[,t]$clone()
  jPr[,t,2,1] <- tPr[,t,1]$clone() * (1-mPr[,t]$clone())
  jPr[,t,1,2] <- (1-tPr[,t,2]$clone()) * mPr[,t]$clone()
  jPr[,t,1,1] <- (1-tPr[,t,1]$clone()) * (1-mPr[,t]$clone()) 
  div <- jPr[,t,,]$sum(dim=c(2,3))
  with_no_grad(div$clip_(sEpsilon, ceil))
  jPr[,t,,]$div_(div$unsqueeze(dim=-1)$unsqueeze(dim=-1))
  
  # marginal likelihood function f(eta_{t}|eta_{t-1})
  mLik[,t] <- (jLik[,t,,]$clone() * jPr[,t,,]$clone())$sum(dim=c(2,3))
  with_no_grad(mLik[,t]$clip_(min=sEpsilon))
  
  # (updated) joint probability P(s,s'|eta_{t})
  jPr2[,t,,] <- jLik[,t,,]$clone() * jPr[,t,,]$clone() / mLik[,t]$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) 
  div <- jPr2[,t,,]$sum(dim=c(2,3))
  with_no_grad(div$clip_(sEpsilon, ceil))
  jPr2[,t,,]$div_(div$unsqueeze(dim=-1)$unsqueeze(dim=-1))  
  
  # marginal probability P(s|eta_{t})
  mPr[,t+1] <- jPr2[,t,2,]$clone()$sum(dim=2)
  
  # step 11: collapsing procedure
  for (s2 in 1:2) { 
    denom1[,t] <- 1 - mPr[,t+1]$clone()
    with_no_grad(denom1[,t]$clip_(sEpsilon, ceil))
    W[,t,1,s2] <- jPr2[,t,1,s2]$clone() / denom1[,t]$clone()
    
    denom2[,t] <- mPr[,t+1]$clone()
    with_no_grad(denom2[,t]$clip_(sEpsilon, ceil))
    W[,t,2,s2] <- jPr2[,t,2,s2]$clone() / denom2[,t]$clone()
    
    with_no_grad(W[,t,,s2]$clip_(sEpsilon, 1-sEpsilon)) }
  
  mEta[,t+1,,] <- (W[,t,,]$clone()$unsqueeze(dim=-1) * jEta2[,t,,,]$clone())$sum(dim=3)
  with_no_grad(mEta[,t+1,,]$clip_(-ceil, ceil))
  
  subEta[,t,,,] <- mEta[,t+1,,]$clone()$unsqueeze(dim=-2) - jEta2[,t,,,]$clone()
  with_no_grad(subEta[,t,,,]$clip_(-ceil, ceil))
  
  subEtaSq[,t,,,,] <- subEta[,t,,,]$clone()$unsqueeze(dim=-1)$matmul(subEta[,t,,,]$clone()$unsqueeze(dim=-2))
  with_no_grad ({
    subEtaSq[,t,,,,]$clip_(-ceil, ceil)
    subEtaSq[,t,,,,] <- (subEtaSq[,t,,,,] + subEtaSq[,t,,,,]$transpose(4, 5)) / 2 })
  
  mP[,t+1,,,] <- (W[,t,,]$clone()$unsqueeze(dim=-1)$unsqueeze(dim=-1) * (jP2[,t,,,,]$clone() + subEtaSq[,t,,,,]$clone()))$sum(dim=3) 
  with_no_grad ({
    mP[,t+1,,,]$clip_(-ceil, ceil)
    mP[,t+1,,,] <- (mP[,t+1,,,] + mP[,t+1,,,]$transpose(3, 4)) / 2
    for (s1 in 1:2) {
      mPEig <- linalg_eigh(mP[,t+1,s1,,]) 
      mPEig[[1]]$real$clip_(sEpsilon, ceil)
      for (row in 1:N) {
        mP[row,t+1,s1,,] <- mPEig[[2]]$real[row,,]$matmul(mPEig[[1]]$real[row,]$diag())$matmul(mPEig[[2]]$real[row,,]$transpose(1, 2))
        while (as.numeric(mP[row,t+1,s1,,]$det()) < 0) {
          mP[row,t+1,s1,,]$add_(lEpsilon * torch_eye(Nf1))} } } }) } 

colors <- rainbow(N)
c <- brewer.pal(8, "Dark2")

i <- 100 # person E {1, ... , N}
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
