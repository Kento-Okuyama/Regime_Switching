# step 1: input {Y_{T}}
# step 2: compute {eta_{t}}_{t=1:T}

######################### 
##  
# y3D: N x Nt x nC3D
## intra- and inter-individual observed variables
# eta3D: N x Nt x Nf
## intra- and inter-individual latent factors
#########################

# install.packages('torch')
library(torch)
# install.packages('reticulate')
library(reticulate)

# for reproducibility 
set.seed(42)
# number of parameter initialization
nInit <- 3
# a very small number
epsilon <- 1e-6
# a very large number
ceil <- 1e6
###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

x <- y3D[,,dim(y3D)[3]] # dropout indicator
y <- y3D[,,1:(dim(y3D)[3]-1)]
N <- dim(y)[1] 
Nt <- dim(y)[2]
No <- dim(y)[3]
yMean <- colMeans(y[,1,], na.rm=TRUE)
ySd <- sqrt(diag(var(y[,1,], na.rm=TRUE)))
for (o in 1:No) {
  if (ySd[o] == 0) {y[,,o] <- y[,,o] - yMean[o]}
  else {y[,,o] <- (y[,,o] - yMean[o]) / ySd[o]} } 

y1 <- y3D[,,1:(dim(y3D1)[3])]
No1 <- dim(y1)[3]
y1Mean <- colMeans(y1[,1,], na.rm=TRUE)
y1Sd <- sqrt(diag(var(y1[,1,], na.rm=TRUE)))
for (o in 1:No1) {
  if (y1Sd[o] == 0) {y1[,,o] <- y1[,,o] - y1Mean[o]}
  else {y1[,,o] <- (y1[,,o] - y1Mean[o]) / y1Sd[o]} } 

y2 <- y3D[,,1:(dim(y3D2)[3])]
No2 <- dim(y2)[3]
y2Mean <- colMeans(y2[,1,], na.rm=TRUE)
y2Sd <- sqrt(diag(var(y2[,1,], na.rm=TRUE)))
for (o in 1:No2) {
  if (y2Sd[o] == 0) {y2[,,o] <- y2[,,o] - y2Mean[o]}
  else {y2[,,o] <- (y2[,,o] - y2Mean[o]) / y2Sd[o]} } 

eta <- eta3D
Nf <- dim(eta)[3]
etaMean <- colMeans(eta[,1,], na.rm=TRUE)
etaSd <- sqrt(diag(var(eta[,1,], na.rm=TRUE)))
for (f in 1:Nf) {
  if (etaSd[f] == 0) {eta[,,f] <- eta[,,f] - etaMean[f]}
  else {eta[,,f] <- (eta[,,f] - etaMean[f]) / etaSd[f]} } 

eta12 <- eta3D12
Nf12 <- dim(eta12)[3]
eta12Mean <- colMeans(eta12[,1,], na.rm=TRUE)
eta12Sd <- sqrt(diag(var(eta12[,1,], na.rm=TRUE)))
for (f in 1:Nf12) {
  if (eta12Sd[f] == 0) {eta12[,,f] <- eta12[,,f] - eta12Mean[f]}
  else {eta12[,,f] <- (eta12[,,f] - eta12Mean[f]) / eta12Sd[f]} } 

eta1 <- eta3D1
Nf1 <- dim(eta1)[3]
eta1Mean <- colMeans(eta1[,1,], na.rm=TRUE)
eta1Sd <- sqrt(diag(var(eta1[,1,], na.rm=TRUE)))
for (f in 1:Nf1) {
  if (etaSd[f] == 0) {eta1[,,f] <- eta1[,,f] - eta1Mean[f]}
  else {eta[,,f] <- (eta1[,,f] - eta1Mean[f]) / eta1Sd[f]} } 

eta2 <- eta3D2
Nf2 <- 1
eta2Mean <- mean(eta2, na.rm=TRUE)
eta2Sd <- sd(eta2)
if (eta2Sd == 0) {eta2 <- eta2 - eta2Mean
} else {eta2 <- (eta2 - eta2Mean) / eta2Sd} 

sumLikBest <- 0

###################################
# Algorithm 1
###################################

for (init in 1:nInit) {
  cat('Initialization step ', init, '\n')
    
  # store sum-likelihood 
  sumLik <- list()
  # optimization step count
  iter <- 1
  # stopping criterion count
  count <- 0 
  
  # step 3: initialize parameters
  a1 <- rnorm(Nf1)
  a2 <- rnorm(Nf1)
  B1d <- runif(Nf1, min=0, max=1)
  B2d <- runif(Nf1, min=0, max=1)
  C1d <- runif(Nf1, min=0, max=1)
  C2d <- runif(Nf1, min=0, max=1)
  D1 <- rnorm(Nf1)
  D2 <- rnorm(Nf1)
  k1 <- rnorm(No1)
  k2 <- rnorm(No1)
  Lmd1v <- runif(No1, min=0, max=1)
  Lmd2v <- runif(No1, min=0, max=1)
  Omega1v <- runif(No1, min=0, max=1)
  Omega2v <- runif(No1, min=0, max=1)
  A1 <- rnorm(No1)
  A2 <- rnorm(No1)
  alpha1 <- rnorm(1)
  alpha2 <- rnorm(1)
  beta1 <- rnorm(Nf12)
  beta2 <- rnorm(Nf12)
  Q1d <- rchisq(Nf1, df=1)
  Q2d <- rchisq(Nf1, df=1)
  R1d <- rchisq(No1, df=1)
  R2d <- rchisq(No1, df=1)
  
  # rows that have non-NA values 
  noNaRows <- list()
  # rows that have NA values
  naRows <- list()
  # initialize moment estimates
  m <- v <- NULL
  
  try ({
    while (count < 3) {
      cat('   optimization step: ', as.numeric(iter), '\n')
      a1 <- torch_tensor(a1, requires_grad=TRUE)
      a2 <- torch_tensor(a2, requires_grad=TRUE)
      a <- list(a1, a2)
      B1d <- torch_tensor(B1d, requires_grad=TRUE)
      B2d <- torch_tensor(B2d, requires_grad=TRUE)
      B1 <- torch_diag(B1d)
      B2 <- torch_diag(B2d)
      B <- list(B1, B2)
      C1d <- torch_tensor(C1d, requires_grad=TRUE)
      C2d <- torch_tensor(C2d, requires_grad=TRUE)
      C1 <- torch_diag(C1d)
      C2 <- torch_diag(C2d)
      C <- list(C1, C2)
      D1 <- torch_tensor(D1, requires_grad=TRUE)
      D2 <- torch_tensor(D2, requires_grad=TRUE)
      D <- list(D1, D2)
      k1 <- torch_tensor(k1, requires_grad=TRUE)
      k2 <- torch_tensor(k2, requires_grad=TRUE)
      k <- list(k1, k2)
      Lmd1v <- torch_tensor(Lmd1v, requires_grad=TRUE)
      Lmd2v <- torch_tensor(Lmd2v, requires_grad=TRUE)
      Lmd1 <- Lmd2 <- torch_full(c(Nf1,No1), 0)
      Lmd1[1,1:3] <- Lmd1v[1:3]; Lmd1[2,4:5] <- Lmd1v[4:5]
      Lmd1[3,6:7] <- Lmd1v[6:7]; Lmd1[4,8:9] <- Lmd1v[8:9]
      Lmd1[5,10:11] <- Lmd1v[10:11]; Lmd1[6,12:14] <- Lmd1v[12:14]
      Lmd1[7,15:17] <- Lmd1v[15:17]
      Lmd2[1,1:3] <- Lmd2v[1:3]; Lmd2[2,4:5] <- Lmd2v[4:5]
      Lmd2[3,6:7] <- Lmd2v[6:7]; Lmd2[4,8:9] <- Lmd2v[8:9]
      Lmd2[5,10:11] <- Lmd2v[10:11]; Lmd2[6,12:14] <- Lmd2v[12:14]
      Lmd2[7,15:17] <- Lmd2v[15:17]
      Lmd <- list(Lmd1, Lmd2)
      Omega1v <- torch_tensor(Omega1v, requires_grad=TRUE)
      Omega2v <- torch_tensor(Omega2v, requires_grad=TRUE)
      Omega1 <- Omega2 <- torch_full(c(Nf1,No1), 0)
      Omega1[1,1:3] <- Omega1v[1:3]; Omega1[2,4:5] <- Omega1v[4:5]
      Omega1[3,6:7] <- Omega1v[6:7]; Omega1[4,8:9] <- Omega1v[8:9]
      Omega1[5,10:11] <- Omega1v[10:11]; Omega1[6,12:14] <- Omega1v[12:14]
      Omega1[7,15:17] <- Omega1v[15:17]
      Omega2[1,1:3] <- Omega2v[1:3]; Omega2[2,4:5] <- Omega2v[4:5]
      Omega2[3,6:7] <- Omega2v[6:7]; Omega2[4,8:9] <- Omega2v[8:9]
      Omega2[5,10:11] <- Omega2v[10:11]; Omega2[6,12:14] <- Omega2v[12:14]
      Omega2[7,15:17] <- Omega2v[15:17]
      Omega <- list(Omega1, Omega2)
      A1 <- torch_tensor(A1, requires_grad=TRUE)
      A2 <- torch_tensor(A2, requires_grad=TRUE)
      A <- list(A1, A2)
      alpha1 <- torch_tensor(alpha1, requires_grad=TRUE)
      alpha2 <- torch_tensor(alpha2, requires_grad=TRUE)
      # alpha <- list(alpha1) 
      alpha <- list(alpha1, alpha2)
      beta1 <- torch_tensor(beta1, requires_grad=TRUE)
      beta2 <- torch_tensor(beta2, requires_grad=TRUE)
      # beta <- list(beta1) 
      beta <- list(beta1, beta2)
      Q1d <- torch_tensor(Q1d, requires_grad=TRUE)
      Q2d <- torch_tensor(Q2d, requires_grad=TRUE)
      Q1 <- torch_diag(Q1d)
      Q2 <- torch_diag(Q2d)
      Q <- list(Q1, Q2)
      R1d <- torch_tensor(R1d, requires_grad=TRUE)
      R2d <- torch_tensor(R2d, requires_grad=TRUE)
      R1 <- torch_diag(R1d)
      R2 <- torch_diag(R2d)
      R <- list(R1, R2)
      theta <- list(a1=a1, a2=a2, B1d=B1d, B2d=B2d, C1d=C1d, C2d=C2d, D1=D1, D2=D2, k1=k1, k2=k2, Lmd1v=Lmd1v, Lmd2v=Lmd2v, Omega1v=Omega1v, Omega2v=Omega2v, A1=A1, A2=A2, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, Q1d=Q1d, Q2d=Q2d, R1d=R1d, R2d=R2d)
      
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
      subEta <- torch_full(c(N,2,2,Nf1), NaN)
      
      # step 4: initialize latent variables
      mEta[,1,,] <- 0
      mP[,1,,,] <- 0; mP[,1,,,]$add_(1e2 * torch_eye(Nf1)) 
      
      # step 5: initialize P(s'|eta_0)
      mPr[,1] <- epsilon 
      
      # store the pair (s,s') as data frame 
      jS <- expand.grid(s1=c(1,2), s2=c(1,2))
      
      # step 6
      for (t in 1:Nt) { 
        cat('   t=', t, '\n')
        # rows that does not have NA values 
        noNaRows[[t]] <- which(rowSums(is.na(y1[,t,])) == 0)
        # rows that have NA values
        naRows[[t]] <- which(rowSums(is.na(y1[,t,])) > 0)
        
        # step 7: Kalman Filter
        for (js in 1:nrow(jS)) {
          s1 <- jS$s1[js]; s2 <- jS$s2[js]
          # Eq.2
          jEta[,t,s1,s2,] <- torch_unsqueeze(a[[s1]], dim=1) + torch_matmul(torch_clone(mEta[,t,s2,]), B[[s1]]) + torch_matmul(torch_clone(mEta[,t,s2,]), C[[s1]]) * torch_unsqueeze(torch_tensor(eta2), dim=-1) + torch_outer(torch_tensor(x[,t]), D[[s1]]) 
          with_no_grad({ 
            jEta[,t,s1,s2,][jEta[,t,s1,s2,] > ceil] <- ceil
            jEta[,t,s1,s2,][jEta[,t,s1,s2,] < -ceil] <- -ceil })
          
          # Eq.3
          jDelta[,t,s1,s2,] <- torch_tensor(eta1[,t,]) - torch_clone(jEta[,t,s1,s2,]) 
          with_no_grad({ 
            jDelta[,t,s1,s2,][jDelta[,t,s1,s2,] > ceil] <- ceil
            jDelta[,t,s1,s2,][jDelta[,t,s1,s2,] < -ceil] <- -ceil })
          
          # Eq.4
          jP[,t,s1,s2,,] <- torch_matmul(torch_matmul(B[[s1]], torch_clone(mP[,t,s2,,])), B[[s1]]) + Q[[s1]] 
          with_no_grad ({
            jP[,t,s1,s2,,] <- (jP[,t,s1,s2,,] + torch_transpose(jP[,t,s1,s2,,], 2, 3)) / 2
            jP[,t,s1,s2,,][jP[,t,s1,s2,,] > ceil] <- ceil
            jP[,t,s1,s2,,][jP[,t,s1,s2,,] < -ceil] <- -ceil
            jPEig <- linalg_eigh(jP[,t,s1,s2,,])
            jPEig[[1]]$real[jPEig[[1]]$real < epsilon] <- epsilon
            jPEig[[1]]$real[jPEig[[1]]$real > ceil] <- ceil
            for (row in 1:N) {jP[row,t,s1,s2,,] <- torch_matmul(torch_matmul(jPEig[[2]]$real[row,,], torch_diag(jPEig[[1]]$real[row,])), torch_transpose(jPEig[[2]]$real[row,,], 1, 2))} 
            while (sum(as.numeric(torch_det(jP[,t,s1,s2,,])) < epsilon) > 0) {
              jPInd <- which(as.numeric(torch_det(jP[,t,s1,s2,,])) < epsilon)
              for (ind in jPInd) {jP[ind,t,s1,s2,,]$add_(2e-1 * torch_eye(Nf1))} } }) 
          
          # Eq.5
          jV[,t,s1,s2,] <- torch_tensor(y1[,t,]) - (torch_unsqueeze(k[[s1]], dim=1) + torch_matmul(torch_clone(jEta[,t,s1,s2,]), Lmd[[s1]]) + torch_matmul(torch_clone(jEta[,t,s1,s2,]), Omega[[s1]]) * torch_unsqueeze(torch_tensor(eta2), dim=-1) + torch_outer(torch_tensor(x[,t]), A[[s1]]))        
          with_no_grad({ 
            jV[,t,s1,s2,][jV[,t,s1,s2,] > ceil] <- ceil
            jV[,t,s1,s2,][jV[,t,s1,s2,] < -ceil] <- -ceil })
          
          # Eq.6
          jF[,t,s1,s2,,] <- torch_matmul(torch_matmul(torch_transpose(Lmd[[s1]], 1, 2), torch_clone(jP[,t,s1,s2,,])), Lmd[[s1]]) + R[[s1]] 
          with_no_grad ({
            jF[,t,s1,s2,,][jF[,t,s1,s2,,] > ceil] <- ceil
            jF[,t,s1,s2,,][jF[,t,s1,s2,,] < -ceil] <- -ceil
            jF[,t,s1,s2,,] <- (jF[,t,s1,s2,,] + torch_transpose(jF[,t,s1,s2,,], 2, 3)) / 2 
            jFEig <- linalg_eigh(jF[,t,s1,s2,,])
            jFEig[[1]]$real[jFEig[[1]]$real > ceil] <- ceil 
            jFEig[[1]]$real[jFEig[[1]]$real < epsilon] <- epsilon
            for (row in 1:N) {jF[row,t,s1,s2,,] <- torch_matmul(torch_matmul(jFEig[[2]]$real[row,,], torch_diag(jFEig[[1]]$real[row,])), torch_transpose(jFEig[[2]]$real[row,,], 1, 2))} 
            while (sum(as.numeric(torch_det(jF[,t,s1,s2,,])) < epsilon) > 0) {
              jFInd <- which(as.numeric(torch_det(jF[,t,s1,s2,,])) < epsilon)
              for (ind in jFInd) {jF[ind,t,s1,s2,,]$add_(5e-1 * torch_eye(No1))} } })
          
          if (length(naRows[[t]]) == N) {
            # Eq.7 (for missing entries)
            jEta2[,t,s1,s2,] <- torch_clone(jEta[,t,s1,s2,]) 
            # Eq.8 (for missing entries)
            jP2[,t,s1,s2,,] <- torch_clone(jP[,t,s1,s2,,]) 
          } else {
            if (length(naRows[[t]]) > 0) {
              for (naRow in naRows[[t]]) {
                # Eq.7 (for missing entries)
                jEta2[naRow,t,s1,s2,] <- torch_clone(jEta[naRow,t,s1,s2,]) 
                # Eq.8 (for missing entries)
                jP2[naRow,t,s1,s2,,] <- torch_clone(jP[naRow,t,s1,s2,,]) } } 
            
            # kalman gain function
            KG <- torch_matmul(torch_matmul(torch_clone(jP[,t,s1,s2,,]), Lmd[[s1]]), linalg_inv_ex(torch_clone(jF[,t,s1,s2,,]))$inverse)
            with_no_grad ({KG[KG > ceil] <- ceil; KG[KG < -ceil] <- -ceil})
            
            for (noNaRow in noNaRows[[t]]) {
              # Eq.7
              jEta2[noNaRow,t,s1,s2,] <- torch_clone(jEta[noNaRow,t,s1,s2,]) + torch_matmul(torch_clone(KG[noNaRow,,]), torch_clone(jV[noNaRow,t,s1,s2,])) 
              with_no_grad ({
                jEta2[noNaRow,t,s1,s2,][jEta2[noNaRow,t,s1,s2,] > ceil] <- ceil
                jEta2[noNaRow,t,s1,s2,][jEta2[noNaRow,t,s1,s2,] < -ceil] <- -ceil })
              
              I_KGLmd <- torch_eye(Nf1) - torch_matmul(torch_clone(KG[noNaRow,,]), torch_transpose(Lmd[[s1]], 1, 2))
              with_no_grad ({I_KGLmd[I_KGLmd > ceil] <- ceil; I_KGLmd[I_KGLmd < -ceil] <- -ceil})
              
              # Eq.9
              jP2[noNaRow,t,s1,s2,,] <- torch_matmul(torch_matmul(torch_clone(I_KGLmd), torch_clone(jP[noNaRow,t,s1,s2,,])), torch_transpose(torch_clone(I_KGLmd), 1, 2)) + torch_matmul(torch_matmul(torch_clone(KG[noNaRow,,]), R[[s1]]), torch_transpose(torch_clone(KG[noNaRow,,]), 1, 2))
              with_no_grad ({
                jP2[noNaRow,t,s1,s2,,][jP2[noNaRow,t,s1,s2,,] > ceil] <- ceil
                jP2[noNaRow,t,s1,s2,,][jP2[noNaRow,t,s1,s2,,] < -ceil] <- -ceil
                jP2Eig <- linalg_eigh(jP2[noNaRow,t,s1,s2,,]) 
                jP2Eig[[1]]$real[jP2Eig[[1]]$real > ceil] <- ceil 
                jP2Eig[[1]]$real[jP2Eig[[1]]$real < epsilon] <- epsilon
                jP2[noNaRow,t,s1,s2,,] <- torch_matmul(torch_matmul(jP2Eig[[2]]$real, torch_diag(jP2Eig[[1]]$real)), torch_transpose(jP2Eig[[2]]$real, 1, 2)) 
                while (as.numeric(torch_det(jP2[noNaRow,t,s1,s2,,])) < epsilon) {jP2[noNaRow,t,s1,s2,,]$add_(2e-1 * torch_eye(Nf1)) } }) } }
          
          # step 8: joint likelihood function f(eta_{t}|s,s',eta_{t-1})
          # Eq.12
          for (noNaRow in noNaRows[[t]]) {
            jLik[noNaRow,t,s1,s2] <- torch_squeeze((-.5*pi)**(-Nf/2) * torch_det(torch_clone(jP[noNaRow,t,s1,s2,,]))**(-1) * torch_exp(-.5*torch_matmul(torch_matmul(torch_clone(jDelta[noNaRow,t,s1,s2,]), linalg_inv_ex(torch_clone(jP[noNaRow,t,s1,s2,,]))$inverse), torch_clone(jDelta[noNaRow,t,s1,s2,])))) 
            with_no_grad (jLik[noNaRow,t,s1,s2] <- min(jLik[noNaRow,t,s1,s2], ceil)) } } 
        
        # step 9: transition probability P(s|s',eta_{t-1})  
        if (t == 1) {
          tPr[,t,1] <- torch_sigmoid(torch_squeeze(alpha[[1]] + torch_tensor(eta2) * beta[[1]][(Nf1+1):Nf]))
          tPr[,t,2] <- torch_sigmoid(torch_squeeze(alpha[[2]] + torch_tensor(eta2) * beta[[2]][(Nf1+1):Nf]))
          
          jPr[,t,2,2] <- torch_clone(tPr[,t,2]) * torch_clone(mPr[,t])
          jPr[,t,2,1] <- torch_clone(tPr[,t,1]) * (1-torch_clone(mPr[,t]))
          jPr[,t,1,2] <- (1-torch_clone(tPr[,t,2])) * torch_clone(mPr[,t])
          jPr[,t,1,1] <- (1-torch_clone(tPr[,t,1])) * (1-torch_clone(mPr[,t])) 
          with_no_grad ({
            div <- torch_sum(jPr[,t,,], dim=c(2,3))
            div[div < epsilon] <- epsilon
            jPr[,t,,]$div_(torch_unsqueeze(torch_unsqueeze(div, dim=-1), dim=-1)) })
          
        } else {
          if (length(noNaRows[[t-1]]) == N) {
            tPr[,t,1] <- torch_sigmoid(torch_squeeze(alpha[[1]]) + torch_matmul(torch_tensor(eta12[,t-1,]), beta[[1]]))
            tPr[,t,2] <- torch_sigmoid(torch_squeeze(alpha[[2]]) + torch_matmul(torch_tensor(eta12[,t-1,]), beta[[2]])) 
            
            # step 10: Hamilton Filter
            # joint probability P(s,s'|eta_{t-1})
            jPr[,t,2,2] <- torch_clone(tPr[,t,2]) * torch_clone(mPr[,t])
            jPr[,t,2,1] <- torch_clone(tPr[,t,1]) * (1-torch_clone(mPr[,t]))
            jPr[,t,1,2] <- (1-torch_clone(tPr[,t,2])) * torch_clone(mPr[,t])
            jPr[,t,1,1] <- (1-torch_clone(tPr[,t,1])) * (1-torch_clone(mPr[,t])) 
            with_no_grad ({
              div <- torch_sum(jPr[,t,,], dim=c(2,3))
              div[div < epsilon] <- epsilon
              jPr[,t,,]$div_(torch_unsqueeze(torch_unsqueeze(div, dim=-1), dim=-1)) })
            
          } else if (length(naRows[[t-1]]) == N) {jPr[,t,,] <- torch_clone(jPr2[,t-1,,])
          
          } else { 
            for (noNaRow in noNaRows[[t-1]]) {
              tPr[noNaRow,t,1] <- torch_sigmoid(torch_squeeze(alpha[[1]]) + torch_matmul(torch_tensor(eta12[noNaRow,t-1,]), beta[[1]]))
              tPr[noNaRow,t,2] <- torch_sigmoid(torch_squeeze(alpha[[2]]) + torch_matmul(torch_tensor(eta12[noNaRow,t-1,]), beta[[2]])) 
             
              # step 10: Hamilton Filter
              # joint probability P(s,s'|eta_{t-1})
              jPr[noNaRow,t,2,2] <- torch_clone(tPr[noNaRow,t,2]) * torch_clone(mPr[noNaRow,t])
              jPr[noNaRow,t,2,1] <- torch_clone(tPr[noNaRow,t,1]) * (1-torch_clone(mPr[noNaRow,t]))
              jPr[noNaRow,t,1,2] <- (1-torch_clone(tPr[noNaRow,t,2])) * torch_clone(mPr[noNaRow,t])
              jPr[noNaRow,t,1,1] <- (1-torch_clone(tPr[noNaRow,t,1])) * (1-torch_clone(mPr[noNaRow,t])) 
              with_no_grad ({
                div <- max(torch_sum(jPr[noNaRow,t,,]), epsilon)
                jPr[noNaRow,t,,]$div_(torch_unsqueeze(torch_unsqueeze(div, dim=-1), dim=-1)) }) } 
            
            for (naRow in naRows[[t-1]]) {jPr[naRow,t,,] <- torch_clone(jPr2[naRow,t-1,,])} } }
          with_no_grad ({
            for (row in 1:N) {
              if (as.numeric(torch_sum(jPr[row,t,,])) < epsilon) {jPr[row,t,,] <- jPr2[row,t-1,,]} } })
          
        if (length(naRows[[t]]) == N) {jPr2[,t,,] <- torch_clone(jPr[,t,,])
        } else if (length(noNaRows[[t]]) == N) {
          # marginal likelihood function f(eta_{t}|eta_{t-1})
          mLik[,t] <- torch_sum(torch_clone(jLik[,t,,]) * torch_clone(jPr[,t,,]), dim=c(2,3))
          with_no_grad(mLik[,t][mLik[,t] < epsilon] <- epsilon)
          
          # (updated) joint probability P(s,s'|eta_{t})
          jPr2[,t,,] <- torch_clone(jLik[,t,,]) * torch_clone(jPr[,t,,]) / torch_unsqueeze(torch_unsqueeze(torch_clone(mLik[,t]), dim=-1), dim=-1)
          with_no_grad({
            div <- torch_sum(jPr2[,t,,], dim=c(2,3))
            div[div < epsilon] <- epsilon
            jPr2[,t,,]$div_(torch_unsqueeze(torch_unsqueeze(div, dim=-1), dim=-1)) 
            for (row in 1:N) {
              if (as.numeric(torch_sum(jPr2[row,t,,])) < epsilon) {jPr2[row,t,,] <- jPr[row,t,,]} } }) 
          
          } else {
            for (naRow in naRows[[t]]) {jPr2[naRow,t,,] <- torch_clone(jPr[naRow,t,,])} 
            for (noNaRow in noNaRows[[t]]) {
              mLik[noNaRow,t] <- torch_sum(torch_clone(jLik[noNaRow,t,,]) * torch_clone(jPr[noNaRow,t,,]))
              with_no_grad(mLik[noNaRow,t] <- max(mLik[noNaRow,t], epsilon))
            
              # (updated) joint probability P(s,s'|eta_{t})
              jPr2[noNaRow,t,,] <- torch_clone(jLik[noNaRow,t,,]) * torch_clone(jPr[noNaRow,t,,]) / torch_clone(mLik[noNaRow,t]) 
              with_no_grad ({
                if (as.numeric(torch_sum(jPr2[noNaRow,t,,])) < epsilon) {jPr2[noNaRow,t,,] <- jPr[noNaRow,t,,]} }) } }
        
        mPr[,t+1] <- torch_sum(torch_clone(jPr2[,t,2,]), dim=2)
        
        # step 11: collapsing procedure
        for (s2 in 1:2) { 
          denom1 <- 1 - torch_clone(mPr[,t+1])
          with_no_grad({
            dInd <- which(as.numeric(denom1) < epsilon) 
            for (ind in dInd) {denom1[ind] <- epsilon} })
          W[,t,1,s2] <- torch_clone(jPr2[,t,1,s2]) / torch_clone(denom1)
          
          denom2 <- torch_clone(mPr[,t+1])
          with_no_grad({
            dInd <- which(as.numeric(denom2) < epsilon) 
            for (ind in dInd) {denom2[ind] <- epsilon} })
          W[,t,2,s2] <- torch_clone(jPr2[,t,2,s2]) / torch_clone(denom2) 
          
          with_no_grad({
            for (s1 in 1:2) {
              while (sum(as.numeric(W[,t,s1,s2]) < 0) > 0) {
                WInd <- which(as.numeric(W[,t,s1,s2]) < 0) 
                for (ind in WInd) {W[ind,t,s1,s2] <- epsilon} }
              
              while (sum(as.numeric(W[,t,s1,s2]) > 1) > 0) {
                WInd <- which(as.numeric(W[,t,s1,s2]) >= 1)
                for (ind in WInd) {W[ind,t,s1,s2] <- 1 - epsilon} } } }) }
        
        mEta[,t+1,,] <- torch_sum(torch_unsqueeze(torch_clone(W[,t,,]), dim=-1) * torch_clone(jEta2[,t,,,]), dim=3)
        with_no_grad({
          mEta[,t+1,,][mEta[,t+1,,] > ceil] <- ceil
          mEta[,t+1,,][mEta[,t+1,,] < -ceil] <- -ceil })
        
        subEta <- torch_unsqueeze(torch_clone(mEta[,t+1,,]), dim=-2) - torch_clone(jEta2[,t,,,])
        with_no_grad({ 
          subEta[subEta > ceil] <- ceil
          subEta[subEta < -ceil] <- -ceil })
        
        subEtaSq <- torch_matmul(torch_unsqueeze(torch_clone(subEta), dim=-1), torch_unsqueeze(torch_clone(subEta), dim=-2))
        with_no_grad({ 
          subEtaSq[subEtaSq > ceil] <- ceil
          subEtaSq[subEtaSq < -ceil] <- -ceil 
          subEtaSq <- (subEtaSq + torch_transpose(subEtaSq, 4, 5)) / 2
          subEtaSqEig <- linalg_eigh(subEtaSq) 
          subEtaSqEig[[1]]$real[subEtaSqEig[[1]]$real > ceil] <- ceil
          subEtaSqEig[[1]]$real[subEtaSqEig[[1]]$real < epsilon] <- epsilon
          
          for (js in 1:nrow(jS)) {
            s1 <- jS$s1[js]; s2 <- jS$s2[js]
            for (row in 1:N) {
              subEtaSq[row,s1,s2,,] <- torch_matmul(torch_matmul(subEtaSqEig[[2]]$real[row,s1,s2,,], torch_diag(subEtaSqEig[[1]]$real[row,s1,s2,])), torch_transpose(subEtaSqEig[[2]]$real[row,s1,s2,,], 1, 2)) }
            while (sum(as.numeric(torch_det(subEtaSq[,s1,s2,,])) < epsilon) > 0) {
              subEtaSqInd <- which(as.numeric(torch_det(subEtaSq[,s1,s2,,])) < epsilon)
              for (ind in subEtaSqInd) {subEtaSq[ind,s1,s2,,]$add_(2e-1 * torch_eye(Nf1))} } } })  
        
        mP[,t+1,,,] <- torch_sum(torch_unsqueeze(torch_unsqueeze(torch_clone(W[,t,,]), dim=-1), dim=-1) * (torch_clone(jP2[,t,,,,]) + torch_clone(subEtaSq)), dim=3) 
        with_no_grad({
          mP[,t+1,,,][mP[,t+1,,,] > ceil] <- ceil
          mP[,t+1,,,][mP[,t+1,,,] < -ceil] <- -ceil
          mP[,t+1,,,] <- (mP[,t+1,,,] + torch_transpose(mP[,t+1,,,], 3, 4)) / 2
          for (s1 in 1:2) {
            mPEig <- linalg_eigh(mP[,t+1,s1,,]) 
            mPEig[[1]]$real[mPEig[[1]]$real > ceil] <- ceil
            mPEig[[1]]$real[mPEig[[1]]$real < epsilon] <- epsilon
            for (row in 1:N) {mP[row,t+1,s1,,] <- torch_matmul(torch_matmul(mPEig[[2]]$real[row,,], torch_diag(mPEig[[1]]$real[row,])), torch_transpose(mPEig[[2]]$real[row,,], 1, 2))}
            while (sum(as.numeric(torch_det(mP[,t+1,s1,,])) < epsilon) > 0) {
              mPInd <- which(as.numeric(torch_det(mP[,t+1,s1,,])) < epsilon)
              for (ind in mPInd) {mP[ind,t+1,s1,,]$add_(2e-1 * torch_eye(Nf1))} } } }) }
      
      # aggregated (summed) likelihood at each optimization step
      loss <- torch_nansum(-torch_clone(mLik))
      with_no_grad(sumLik[iter] <- as.numeric(-loss))
      
      # stopping criterion
      ifelse(abs(sumLik[iter][[1]] - sumLik[1][[1]]) > epsilon, crit <- (sumLik[iter][[1]] - sumLik[iter-1][[1]]) / abs(sumLik[iter][[1]] - sumLik[1][[1]]), crit <- 0)
      
      # add count if the new sumLik does not beat the best score
      ifelse(crit < 5e-2, count <- count + 1, count <- 0)
      
      cat('   sum likelihood = ', sumLik[iter][[1]], '\n')
      plot(unlist(sumLik), xlab='optimization step', ylab='sum likelihood', type='b')
      
      if (sumLikBest < sumLik[iter][[1]]) {with_no_grad({thetaBest <- as.list(theta)})}
      sumLikBest <- max(sumLikBest, sumLik[iter][[1]])
      
      # run adam function defined above
      with_no_grad({
        result <- adam4(loss=loss, theta=theta, m=m, v=v)
        theta <- result$theta
        m <- result$m 
        v <- result$v 
        
        # switch off the gradient tracking
        a1 <- torch_tensor(theta$a1, requires_grad=FALSE)
        a2 <- torch_tensor(theta$a2, requires_grad=FALSE)
        B1d <- torch_tensor(theta$B1d, requires_grad=FALSE)
        B2d <- torch_tensor(theta$B2d, requires_grad=FALSE)
        C1d <- torch_tensor(theta$C1d, requires_grad=FALSE)
        C2d <- torch_tensor(theta$C2d, requires_grad=FALSE)
        D1 <- torch_tensor(theta$D1, requires_grad=FALSE)
        D2 <- torch_tensor(theta$D2, requires_grad=FALSE)
        k1 <- torch_tensor(theta$k1, requires_grad=FALSE)
        k2 <- torch_tensor(theta$k2, requires_grad=FALSE)
        Lmd1v <- torch_tensor(theta$Lmd1v, requires_grad=FALSE)
        Lmd2v <- torch_tensor(theta$Lmd2v, requires_grad=FALSE)
        Omega1v <- torch_tensor(theta$Omega1v, requires_grad=FALSE)
        Omega2v <- torch_tensor(theta$Omega2v, requires_grad=FALSE)
        A1 <- torch_tensor(theta$A1, requires_grad=FALSE)
        A2 <- torch_tensor(theta$A2, requires_grad=FALSE)
        alpha1 <- torch_tensor(theta$alpha1, requires_grad=FALSE)
        alpha2 <- torch_tensor(theta$alpha2, requires_grad=FALSE)
        beta1 <- torch_tensor(theta$beta1, requires_grad=FALSE)
        beta2 <- torch_tensor(theta$beta2, requires_grad=FALSE)
        Q1d <- torch_tensor(theta$Q1d, requires_grad=FALSE)
        Q2d <- torch_tensor(theta$Q2d, requires_grad=FALSE)
        R1d <- torch_tensor(theta$R1d, requires_grad=FALSE)
        R2d <- torch_tensor(theta$R2d, requires_grad=FALSE) })
      
      if (count==3 || iter > 100) {print('   stopping criterion is met'); break}
      iter <- iter + 1 } }) # continue to numerical re-optimization 
} # continue to re-initialization of parameters