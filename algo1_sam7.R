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
set.seed(init)
# number of parameter initialization
nInit <- 1
# max number of optimization steps
nIter <- 1
# a very small number
epsilon <- 1e-6
# a small number
epsD <- 1 - epsilon 
# a very large number
ceil <- 1e6

###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

dropout <- y3D[,,dim(y3D)[3]]
y <- y3D[,,1:(dim(y3D)[3]-1)]
yMean <- mean(y[!is.na(y)])
ySd <- sd(y[!is.na(y)])
y <- (y - yMean) / ySd 
N <- dim(y)[1] 
Nt <- dim(y)[2]
No <- dim(y)[3]
eta <- eta3D
etaMean <- mean(eta[!is.na(eta)])
etaSd <-sd(eta[!is.na(eta)])
eta <- (eta - etaMean) / etaSd
for (t in 1:Nt) {for (i in 1:N) {eta[i,t,] <- eta[i,t,] - colMeans(eta[,1,])} }
Nf <- dim(eta)[3]

# f(Y|theta)
sumLik <- list()
sumLikBest <- 0

###################################
# Algorithm 1
###################################

N <- 5
y <- y[1:N,,]
eta <- eta[1:N,,]

for (init in 1:nInit) {
  print(paste0('Initialization step '), init)
  
  # optimization step count
  iter <- 1
  # stopping criterion count
  count <- 0 
  # moment estimates 
  m <- v <- NULL
  
  # step 3: initialize parameters
  a1 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE) 
  a2 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE) 
  a <- list(a1, a2)
  B1d <- torch_tensor(torch_rand(Nf), requires_grad=TRUE)
  B2d <- torch_tensor(torch_rand(Nf), requires_grad=TRUE)
  B1 <- torch_diag(B1d)
  B2 <- torch_diag(B2d)
  B <- list(B1, B2)
  k1 <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  k2 <- torch_tensor(torch_randn(No), requires_grad=TRUE)
  k <- list(k1, k2)
  Lmd1v <- torch_tensor(torch_rand(No), requires_grad=TRUE)
  Lmd2v <- torch_tensor(torch_rand(No), requires_grad=TRUE)
  Lmd1 <- Lmd2 <- torch_full(c(Nf,No), 0)
  Lmd1[1,1:3] <- Lmd1v[1:3]; Lmd1[2,4:5] <- Lmd1v[4:5]
  Lmd1[3,6:7] <- Lmd1v[6:7]; Lmd1[4,8:9] <- Lmd1v[8:9]
  Lmd1[5,10:11] <- Lmd1v[10:11]; Lmd1[6,12:14] <- Lmd1v[12:14]
  Lmd1[7,15:17] <- Lmd1v[15:17]; Lmd1[8,18] <- Lmd2v[18]
  Lmd2[1,1:3] <- Lmd2v[1:3]; Lmd2[2,4:5] <- Lmd2v[4:5]
  Lmd2[3,6:7] <- Lmd2v[6:7]; Lmd2[4,8:9] <- Lmd2v[8:9]
  Lmd2[5,10:11] <- Lmd2v[10:11]; Lmd2[6,12:14] <- Lmd2v[12:14]
  Lmd2[7,15:17] <- Lmd2v[15:17]; Lmd2[8,18] <- Lmd2v[18]
  Lmd <- list(Lmd1, Lmd2)
  alpha1 <- torch_tensor(torch_randn(1), requires_grad=TRUE)
  alpha2 <- torch_tensor(torch_randn(1), requires_grad=TRUE)
  alpha <- list(alpha1, alpha2)
  beta1 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE)
  beta2 <- torch_tensor(torch_randn(Nf), requires_grad=TRUE)
  beta <- list(beta1, beta2)
  Q1d <- torch_tensor(torch_rand(Nf)**2, requires_grad=TRUE)
  Q2d <- torch_tensor(torch_rand(Nf)**2, requires_grad=TRUE)
  Q1 <- torch_diag(Q1d)
  Q2 <- torch_diag(Q2d)
  Q <- list(Q1, Q2)
  R1d <- torch_tensor(torch_rand(No)**2, requires_grad=TRUE)
  R2d <- torch_tensor(torch_rand(No)**2, requires_grad=TRUE)
  R1 <- torch_diag(R1d)
  R2 <- torch_diag(R2d)
  R <- list(R1, R2)
  theta <- torch_cat(list(a1, a2, B1d, B2d, k1, k2, Lmd1v, Lmd2v, alpha1, alpha2, beta1, beta2, Q1d, Q2d, R1d, R2d))
  
  # define variables
  jEta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.2 (LHS)
  jDelta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.3 (LHS)
  jP <- jPsym <- jPE <- jPChol <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Eq.4 (LHS)
  jV <- torch_full(c(N,Nt,2,2,No), NaN) # Eq.5 (LHS)
  jF <- jFsym <- jFE <- jFChol <- torch_full(c(N,Nt,2,2,No,No), NaN) # Eq.6 (LHS)
  jEta2 <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.7 (LHS)
  jP2 <- jP2sym <- jP2E <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Eq.8 (LHS)
  mEta <- torch_full(c(N,Nt+1,2,Nf), NaN) # Eq.9-1 (LHS)
  mP <- mPsym <- mPE <- torch_full(c(N,Nt+1,2,Nf,Nf), NaN) # Eq.9-2 (LHS)
  W <- torch_full(c(N,Nt,2,2), NaN) # Eq.9-3 (LHS)
  jPr <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-1 (LHS)
  mLik <- torch_full(c(N,Nt), NaN) # Eq.10-2 (LHS)
  jPr2 <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-3 (LHS)
  mPr <- torch_full(c(N,Nt+1), NaN) # Eq.10-4 (LHS)
  jLik <- torch_full(c(N,Nt,2,2), NaN) # Eq.11 (LHS)
  tPr <- torch_full(c(N,Nt,2), NaN) # Eq.12 (LHS)
  
  # rows that have non-NA values 
  noNaRows <- list()
  # rows that have NA values
  naRows <- list()
  
  # step 4: initialize latent variables
  for (s in 1:2) {
    for (i in 1:N) {
      mEta[i,1,s,] <- rep(x=0, times=Nf)
      mP[i,1,s,,] <- diag(x=1e1, nrow=Nf, ncol=Nf)} }
  
  # while (count < 3) {
  for (iter in 1:nIter) {
    print(paste0('   optimization step: ', as.numeric(iter)))
    
    # step 5: initialize P(s'|eta_0)
    mPr[,1] <- epsilon 
    
    # store the pair (s,s') as data frame 
    jS <- expand.grid(s1=c(1,2), s2=c(1,2))
    # step 6:
    for (t in 1:Nt) { 
      print(paste0('   t=',t))
      # step 7: Kalman Filter
      print('      Kim Filter')
      for (js in 1:nrow(jS)) {
        s1 <- jS$s1[js]
        s2 <- jS$s2[js]
        
        # rows that does not have NA values 
        noNaRows[[t]] <- which(rowSums(is.na(y[,t,])) == 0)
        # rows that have NA values
        naRows[[t]] <- which(rowSums(is.na(y[,t,])) > 0)
        
        jEta[,t,s1,s2,] <- a[[s1]] + torch_matmul(torch_clone(mEta[,t,s2,]), B[[s1]]) # Eq.2
        for (noNaRow in noNaRows[[t]]) {jDelta[noNaRow,t,s1,s2,] <- eta[noNaRow,t,] - torch_clone(jEta[noNaRow,t,s1,s2,])} # Eq.3
        jP[,t,s1,s2,,] <- torch_matmul(torch_matmul(B[[s1]], torch_clone(mPE[,t,s2,,])), B[[s1]]) + Q[[s1]] # Eq.4
        jPsym[,t,s1,s2,,] <- (torch_clone(jP[,t,s1,s2,,]) + torch_transpose(torch_clone(jP[,t,s1,s2,,]), 2, 3)) / 2 # ensure symmetry
        if (sum(as.numeric(torch_det(torch_clone(jPsym[,t,s1,s2,,]))) < epsilon) > 0) {
          jPInd <- which(as.numeric(torch_det(torch_clone(jPsym[,t,s1,s2,,]))) < epsilon)
          for (ind in jPInd) {jPE[ind,t,s1,s2,,] <- torch_clone(jPsym[ind,t,s1,s2,,]) + epsD * torch_eye(Nf)} } # add a small constant to ensure p.s.d.
        if (sum(as.numeric(torch_det(torch_clone(jPsym[,t,s1,s2,,]))) >= epsilon) > 0) {
          jPInd <- which(as.numeric(torch_det(torch_clone(jPsym[,t,s1,s2,,]))) >= epsilon)
          for (ind in jPInd) {jPE[ind,t,s1,s2,,] <- torch_clone(jPsym[ind,t,s1,s2,,])} } 
        jPChol[,t,s1,s2,,] <- torch_cholesky(torch_clone(jPE[,t,s1,s2,,]), upper=FALSE) # Cholesky decomposition
        
        for (noNaRow in noNaRows[[t]]) {
          jV[noNaRow,t,s1,s2,] <- y[noNaRow,t,] - (k[[s1]] + torch_matmul(torch_clone(jEta[noNaRow,t,s1,s2,]), Lmd[[s1]])) } # Eq.5
        jF[,t,s1,s2,,] <- torch_matmul(torch_matmul(torch_transpose(Lmd[[s1]], 1, 2), torch_clone(jPE[,t,s1,s2,,])), Lmd[[s1]]) + R[[s1]] # Eq.6
        jFsym[,t,s1,s2,,] <- (torch_clone(jF[,t,s1,s2,,]) + torch_transpose(torch_clone(jF[,t,s1,s2,,]), 2, 3)) / 2 # ensure symmetry
        if (sum(as.numeric(torch_det(torch_clone(jFsym[,t,s1,s2,,]))) < epsilon) > 0) {
          jFInd <- which(as.numeric(torch_det(torch_clone(jFsym[,t,s1,s2,,]))) < epsilon)
          for (ind in jFInd) {jFE[ind,t,s1,s2,,] <- torch_clone(jFsym[ind,t,s1,s2,,]) + epsD * torch_eye(No)} } # add a small constant to ensure p.s.d.
        if (sum(as.numeric(torch_det(torch_clone(jFsym[,t,s1,s2,,]))) >= epsilon) > 0) {
          jFInd <- which(as.numeric(torch_det(torch_clone(jFsym[,t,s1,s2,,]))) >= epsilon)
          for (ind in jFInd) {jFE[ind,t,s1,s2,,] <- torch_clone(jFsym[ind,t,s1,s2,,])} } 
        jFChol[,t,s1,s2,,] <- torch_cholesky(torch_clone(jFE[,t,s1,s2,,]), upper=FALSE) # Cholesky decomposition
        
        for (noNaRow in noNaRows[[t]]) {
          # kalman gain function
          KG <- torch_matmul(torch_matmul(torch_clone(jPE[noNaRow,t,s1,s2,,]), Lmd[[s1]]), torch_cholesky_inverse(torch_clone(jFChol[noNaRow,t,s1,s2,,]), upper=FALSE))
          jEta2[noNaRow,t,s1,s2,] <- torch_clone(jEta[noNaRow,t,s1,s2,]) + torch_matmul(torch_clone(KG), torch_clone(jV[noNaRow,t,s1,s2,])) # Eq.7
          KGLmd <- torch_matmul(torch_clone(KG), torch_transpose(Lmd[[s1]], 1, 2))
          I_KGLmd <- torch_eye(Nf) - torch_clone(KGLmd)
          
          # jP2[noNaRow,t,s1,s2,,] <- torch_matmul(I_KGLmd, jP[noNaRow,t,s1,s2,,])} # Eq.8 
          jP2[noNaRow,t,s1,s2,,] <- torch_matmul(torch_matmul(torch_clone(I_KGLmd), torch_clone(jPE[noNaRow,t,s1,s2,,])), torch_transpose(torch_clone(I_KGLmd), 1, 2)) + torch_matmul(torch_matmul(torch_clone(KG), R[[s1]]), torch_transpose(torch_clone(KG), 1, 2)) # Eq.9
          if (as.numeric(torch_det(torch_clone(jP2[noNaRow,t,s1,s2,,]))) < epsilon) {
            jP2E[noNaRow,t,s1,s2,,] <- torch_clone(jP2[noNaRow,t,s1,s2,,]) + epsD * torch_eye(Nf) } # add a small constant to ensure p.s.d.
          if (as.numeric(torch_det(torch_clone(jP2[noNaRow,t,s1,s2,,]))) >= epsilon) {
            jP2E[noNaRow,t,s1,s2,,] <- torch_clone(jP2[noNaRow,t,s1,s2,,]) } } 
        
        for (naRow in naRows[[t]]) {
          jEta2[naRow,t,s1,s2,] <- torch_clone(jEta[naRow,t,s1,s2,]) # Eq.7 (for missing entries)
          jP2E[naRow,t,s1,s2,,] <- torch_clone(jPE[naRow,t,s1,s2,,]) } # Eq.8 (for missing entries)
        
        # step 8: joint likelihood function f(eta_{t}|s,s',eta_{t-1})
        # is likelihood function different because I am dealing with latent variables instead of observed variables?
        for (noNaRow in noNaRows[[t]]) {
          jLik[noNaRow,t,s1,s2] <- 
            (-.5*pi)**(-Nf/2) * torch_prod(torch_diag(torch_clone(jPChol[noNaRow,t,s1,s2,,])))**(-1) * 
            torch_exp(-.5*torch_matmul(torch_matmul(torch_clone(jDelta[noNaRow,t,s1,s2,]), torch_cholesky_inverse(torch_clone(jPChol[noNaRow,t,s1,s2,,]), upper=FALSE)), torch_clone(jDelta[noNaRow,t,s1,s2,]))) } } # Eq.12
      
      # step 9: transition probability P(s|s',eta_{t-1})  
      if (t == 1) {
        tPr[,t,1] <- torch_sigmoid(alpha[[1]])
        tPr[,t,2] <- torch_sigmoid(alpha[[2]]) 
        jPr[,t,2,2] <- torch_clone(tPr[,t,2]) * torch_clone(mPr[,t])
        jPr[,t,2,1] <- torch_clone(tPr[,t,1]) * (1-torch_clone(mPr[,t]))
        jPr[,t,1,2] <- (1-torch_clone(tPr[,t,2])) * torch_clone(mPr[,t])
        jPr[,t,1,1] <- (1-torch_clone(tPr[,t,1])) * (1-torch_clone(mPr[,t])) }
      else {
        for (noNaRow in noNaRows[[t-1]]) {
          tPr[noNaRow,t,1] <- torch_sigmoid(alpha[[1]] + torch_matmul(eta[noNaRow,t-1,], beta[[1]]))
          tPr[noNaRow,t,2] <- torch_sigmoid(alpha[[2]] + torch_matmul(eta[noNaRow,t-1,], beta[[2]])) 
          
          # step 10: Hamilton Filter
          # joint probability P(s,s'|eta_{t-1})
          jPr[noNaRow,t,2,2] <- torch_clone(tPr[noNaRow,t,2]) * torch_clone(mPr[noNaRow,t])
          jPr[noNaRow,t,2,1] <- torch_clone(tPr[noNaRow,t,1]) * (1-torch_clone(mPr[noNaRow,t]))
          jPr[noNaRow,t,1,2] <- (1-torch_clone(tPr[noNaRow,t,2])) * torch_clone(mPr[noNaRow,t])
          jPr[noNaRow,t,1,1] <- (1-torch_clone(tPr[noNaRow,t,1])) * (1-torch_clone(mPr[noNaRow,t])) }
        for (naRow in naRows[[t-1]]) {jPr[naRow,t,,] <- torch_clone(jPr[naRow,t-1,,])} }
      
      # marginal likelihood function f(eta_{t}|eta_{t-1})
      if (length(noNaRows[[t]]) > 0) {
        for (noNaRow in noNaRows[[t]]) {
          mLik[noNaRow,t] <- torch_sum(torch_clone(jLik[noNaRow,t,,]) * torch_clone(jPr[noNaRow,t,,]))
          # (updated) joint probability P(s,s'|eta_{t})
          jPr2[noNaRow,t,,] <- torch_clone(jLik[noNaRow,t,,]) * torch_clone(jPr[noNaRow,t,,]) / max(torch_clone(mLik[noNaRow,t]), epsilon)
          if (as.numeric(torch_sum(torch_clone(jPr2[noNaRow,t,,]))) == 0) {jPr2[noNaRow,t,,] <- torch_clone(jPr[noNaRow,t,,])} } }
      for (naRow in naRows[[t]]) {jPr2[naRow,t,,] <- torch_clone(jPr[naRow,t,,])}
      mPr[,t+1] <- torch_sum(torch_clone(jPr2[,t,2,]), dim=2)
      
      # step 11: collapsing procedure
      print('      Collapsing')
      for (s2 in 1:2) { 
        denom1 <- 1 - torch_clone(mPr[,t+1])
        denom12 <- torch_full_like(torch_clone(denom1), NaN)
        denom12[torch_clone(denom1)<=epsilon] <- torch_clone(denom1)[torch_clone(denom1)<=epsilon] + epsilon
        denom12[torch_clone(denom1)>epsilon] <- torch_clone(denom1)[torch_clone(denom1)>epsilon] 
        W[,t,1,s2] <- torch_clone(jPr2[,t,1,s2]) / torch_clone(denom12)
        denom2 <- torch_clone(mPr[,t+1])
        denom22 <- torch_full_like(torch_clone(denom2), NaN)
        denom22[torch_clone(denom2)<=epsilon] <- torch_clone(denom2)[torch_clone(denom2)<=epsilon] + epsilon
        denom22[torch_clone(denom2)>epsilon] <- torch_clone(denom2)[torch_clone(denom2)>epsilon]
        W[,t,2,s2] <- torch_clone(jPr2[,t,2,s2]) / torch_clone(denom22) }
      
      for (f in 1:Nf) {mEta[,t+1,,f] <- torch_sum(torch_clone(W[,t,,]) * torch_clone(jEta2[,t,,,f]), dim=3)}
      
      subEta <- torch_full_like(torch_clone(jEta2[,t,,,]), NaN)
      for (s2 in 1:2) {subEta[,,s2,] <- torch_clone(mEta[,t+1,,]) - torch_clone(jEta2[,t,,s2,])}
      subEta1 <- torch_unsqueeze(torch_clone(subEta), dim=-1)
      subEta2 <- torch_unsqueeze(torch_clone(subEta), dim=-2)
      subEtaSq <- torch_matmul(torch_clone(subEta1), torch_clone(subEta2))
      subEtaSqsym <- (torch_clone(subEtaSq) + torch_transpose(torch_clone(subEtaSq), 4, 5)) / 2 # ensure symmetry
      subEtaSqE <- torch_full_like(torch_clone(subEtaSqsym), NaN)
      
      for (js in 1:nrow(jS)) {
        s1 <- jS$s1[js]
        s2 <- jS$s2[js]
        
        if (sum(as.numeric(torch_det(torch_clone(subEtaSqsym[,s1,s2,,]))) < epsilon) > 0) {
          subEtaSqInd <- which(as.numeric(torch_det(torch_clone(subEtaSqsym[,s1,s2,,]))) < epsilon)
          for (ind in subEtaSqInd) {subEtaSqE[ind,s1,s2,,] <- torch_clone(subEtaSqsym[ind,s1,s2,,]) + epsD * torch_eye(Nf)} }
        if (sum(as.numeric(torch_det(torch_clone(subEtaSqsym[,s1,s2,,]))) >= epsilon) > 0) {
          subEtaSqInd <- which(as.numeric(torch_det(torch_clone(subEtaSqsym[,s1,s2,,]))) >= epsilon)
          for (ind in subEtaSqInd) {subEtaSqE[ind,s1,s2,,] <- torch_clone(subEtaSqsym[ind,s1,s2,,])} } } 
      
      jNf <- expand.grid(f1=1:Nf, f2=1:Nf)
      for (jnf in 1:nrow(jNf)) {
        f1 <- jNf$f1[jnf]
        f2 <- jNf$f2[jnf] 
        
        mP[,t+1,,f1,f2] <- torch_sum(torch_clone(W[,t,,]) * (torch_clone(jP2E[,t,,,,]) + torch_clone(subEtaSqE))[,,,f1,f2], dim=3) }
      mPsym[,t+1,,,] <- (torch_clone(mP[,t+1,,,]) + torch_transpose(torch_clone(mP[,t+1,,,]), 3, 4)) / 2 # ensure symmetry
      
      for (s1 in 1:2) {
        if (sum(as.numeric(torch_det(torch_clone(mPsym[,t+1,s1,,]))) < epsilon) > 0) {
          mPInd <- which(as.numeric(torch_det(torch_clone(mPsym[,t+1,s1,,]))) < epsilon)
          for (ind in mPInd) {mPE[ind,t+1,s1,,] <- torch_clone(mPsym[ind,t+1,s1,,]) + epsD * torch_eye(Nf)} } # add a small constant to ensure p.s.d.
        if (sum(as.numeric(torch_det(torch_clone(mPsym[,t+1,s1,,]))) >= epsilon) > 0) {
          mPInd <- which(as.numeric(torch_det(torch_clone(mPsym[,t+1,s1,,]))) >= epsilon)
          for (ind in mPInd) {mPE[ind,t+1,s1,,] <- torch_clone(mPsym[ind,t+1,s1,,])} } } } 
    
    # aggregated (summed) likelihood at each optimization step
    sumLik[iter] <- as.numeric(torch_nansum(torch_clone(mLik)))
    
    # stopping criterion
    if (abs(sumLik[iter][[1]] - sumLik[1][[1]]) > epsilon) {
      crit <- (sumLik[iter][[1]] - sumLik[1][[1]]) / (sumLik[iter][[1]] - sumLik[1][[1]]) }
    else {crit <- 0}
    
    # add count if the new sumLik does not beat the best score
    if (crit < epsD) {count <- count + 1}
    else {count <- 0}
    
    if (count==3) {print('   stopping criterion is met'); break}
    print(paste0('   sum likelihood = ', sumLik[iter]))
    
    # backward propagation
    torch_nansum(torch_clone(mLik))$backward(retain_graph=TRUE)
    
    # store gradients
    grad <- torch_cat(list(a1$grad, a2$grad, B1d$grad, B2d$grad, k1$grad, k2$grad, Lmd1v$grad, Lmd2v$grad, alpha1$grad, alpha2$grad, beta1$grad, beta2$grad, Q1d$grad, Q2d$grad, R1d$grad, R2d$grad))
    
  } # continue to numerical re-optimization
} # continue to re-initialization of parameters
