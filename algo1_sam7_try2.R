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
nInit <- 1
# a very small number
epsilon <- 1e-6
# a very large number
ceil <- 1e6
###################################
# s=1: non-drop out state 
# s=2: drop out state      
###################################

dropout <- y3D[,,dim(y3D)[3]]
y <- y3D[,,1:(dim(y3D)[3]-1)]
N <- dim(y)[1] 
Nt <- dim(y)[2]
No <- dim(y)[3]
y1Mean <- colMeans(y[,1,], na.rm=TRUE)
ySd <- sqrt(diag(var(y[,1,], na.rm=TRUE)))
for (o in 1:No) {
  if (ySd[o] == 0) {y[,,o] <- y[,,o] - y1Mean[o]}
  else {y[,,o] <- (y[,,o] - y1Mean[o]) / ySd[o]} } 


eta <- eta3D
Nf <- dim(eta)[3]
eta1Mean <- colMeans(eta[,1,], na.rm=TRUE)
etaSd <- sqrt(diag(var(eta[,1,], na.rm=TRUE)))
for (f in 1:Nf) {
  if (etaSd[f] == 0) {eta[,,f] <- eta[,,f] - eta1Mean[f]}
  else {eta[,,f] <- (eta[,,f] - eta1Mean[f]) / etaSd[f]} } 

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
  a1 <- rnorm(Nf)
  a2 <- rnorm(Nf)
  B1d <- runif(Nf, min=.5, max=1)
  B2d <- runif(Nf, min=.5, max=1)
  k1 <- rnorm(No)
  k2 <- rnorm(No)
  Lmd1v <- runif(No, min=.5, max=1)
  Lmd2v <- runif(No, min=.5, max=1)
  alpha1 <- rnorm(1)
  alpha2 <- rnorm(1)
  beta1 <- rnorm(Nf)
  beta2 <- rnorm(Nf)
  Q1d <- rchisq(Nf, df=1)
  Q2d <- rchisq(Nf, df=1)
  R1d <- rchisq(No, df=1)
  R2d <- rchisq(No, df=1)
  
  # rows that have non-NA values 
  noNaRows <- list()
  # rows that have NA values
  naRows <- list()
  # initialize moment estimates
  m <- v <- NULL
  
  try(silent = FALSE, {
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
      k1 <- torch_tensor(k1, requires_grad=TRUE)
      k2 <- torch_tensor(k2, requires_grad=TRUE)
      k <- list(k1, k2)
      Lmd1v <- torch_tensor(Lmd1v, requires_grad=TRUE)
      Lmd2v <- torch_tensor(Lmd2v, requires_grad=TRUE)
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
      alpha1 <- torch_tensor(alpha1, requires_grad=TRUE)
      alpha2 <- torch_tensor(alpha2, requires_grad=TRUE)
      alpha <- list(alpha1, alpha2)
      beta1 <- torch_tensor(beta1, requires_grad=TRUE)
      beta2 <- torch_tensor(beta2, requires_grad=TRUE)
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
      
      theta <- list(a1=a1, a2=a2, B1d=B1d, B2d=B2d, k1=k1, k2=k2, Lmd1v=Lmd1v, Lmd2v=Lmd2v, alpha1=alpha1, alpha2=alpha2, beta1=beta1, beta2=beta2, Q1d=Q1d, Q2d=Q2d, R1d=R1d, R2d=R2d)
      
      # define variables
      jEta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.2 (LHS)
      jDelta <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.3 (LHS)
      jP <- jPChol <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Eq.4 (LHS)
      jV <- torch_full(c(N,Nt,2,2,No), NaN) # Eq.5 (LHS)
      jF <- jFChol <- torch_full(c(N,Nt,2,2,No,No), NaN) # Eq.6 (LHS)
      jEta2 <- torch_full(c(N,Nt,2,2,Nf), NaN) # Eq.7 (LHS)
      jP2 <- torch_full(c(N,Nt,2,2,Nf,Nf), NaN) # Eq.8 (LHS)
      mEta <- torch_full(c(N,Nt+1,2,Nf), NaN) # Eq.9-1 (LHS)
      mP <- torch_full(c(N,Nt+1,2,Nf,Nf), NaN) # Eq.9-2 (LHS)
      W <- torch_full(c(N,Nt,2,2), NaN) # Eq.9-3 (LHS)
      jPr <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-1 (LHS)
      mLik <- torch_full(c(N,Nt), NaN) # Eq.10-2 (LHS)
      jPr2 <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-3 (LHS)
      mPr <- torch_full(c(N,Nt+1), NaN) # Eq.10-4 (LHS)
      jLik <- torch_full(c(N,Nt,2,2), NaN) # Eq.11 (LHS)
      tPr <- torch_full(c(N,Nt,2), NaN) # Eq.12 (LHS)
      subEta <- torch_full(c(N,2,2,Nf), NaN)
      
      # step 4: initialize latent variables
      mEta[,1,,] <- 0
      mP[,1,,,] <- 0 
      for (f in 1:Nf) {mP[,1,,f,f] <- 1e1}
      
      # step 5: initialize P(s'|eta_0)
      mPr[,1] <- epsilon 
      
      # store the pair (s,s') as data frame 
      jS <- expand.grid(s1=c(1,2), s2=c(1,2))
      # step 6
      for (t in 1:Nt) { 
        cat('   t=', t, '\n')
         if (t %% 30 == 0) {cat('   t=', t, '\n')}
        # rows that does not have NA values 
        noNaRows[[t]] <- which(rowSums(is.na(y[,t,])) == 0)
        # rows that have NA values
        naRows[[t]] <- which(rowSums(is.na(y[,t,])) > 0)
        
        # step 7: Kalman Filter
        for (js in 1:nrow(jS)) {
          s1 <- jS$s1[js]
          s2 <- jS$s2[js]
          jEta[,t,s1,s2,] <- torch_unsqueeze(a[[s1]], dim=1) + torch_matmul(torch_clone(mEta[,t,s2,]), B[[s1]]) # Eq.2
          jDelta[,t,s1,s2,] <- torch_tensor(eta[,t,]) - torch_clone(jEta[,t,s1,s2,]) # Eq.3
          jP[,t,s1,s2,,] <- torch_matmul(torch_matmul(B[[s1]], torch_clone(mP[,t,s2,,])), B[[s1]]) + Q[[s1]] # Eq.4
          with_no_grad ({
            jP[,t,s1,s2,,] <- (jP[,t,s1,s2,,] + torch_transpose(jP[,t,s1,s2,,], 2, 3)) / 2 # ensure symmetry
            jPEig <- linalg_eigh(jP[,t,s1,s2,,])
            jPEig[[1]]$real[jPEig[[1]]$real < epsilon] <- epsilon
            jPEig[[1]]$real[jPEig[[1]]$real > ceil] <- ceil
            for (row in 1:N) {jP[row,t,s1,s2,,] <- torch_matmul(torch_matmul(jPEig[[2]]$real[row,,], torch_diag(jPEig[[1]]$real[row,])), torch_transpose(jPEig[[2]]$real[row,,], 1, 2))} 
            if (sum(as.numeric(torch_det(jP[,t,s1,s2,,])) < epsilon) > 0) {
              jPInd <- which(as.numeric(torch_det(jP[,t,s1,s2,,])) < epsilon)
              for (ind in jPInd) {jP[ind,t,s1,s2,,]$add_(1e-1 * torch_eye(Nf))} } }) # add a small constant to ensure p.s.d.
          jPChol[,t,s1,s2,,] <- linalg_cholesky_ex(torch_clone(jP[,t,s1,s2,,]))$L # Cholesky decomposition
          # why does R skip this jV sometimes?
          jV[,t,s1,s2] <- torch_tensor(y[,t,]) - (torch_unsqueeze(k[[s1]], dim=1) + torch_matmul(torch_clone(jEta[,t,s1,s2,]), Lmd[[s1]]))
          jF[,t,s1,s2,,] <- torch_matmul(torch_matmul(torch_transpose(Lmd[[s1]], 1, 2), torch_clone(jP[,t,s1,s2,,])), Lmd[[s1]]) + R[[s1]] # Eq.6
          with_no_grad ({
            jF[,t,s1,s2,,] <- (jF[,t,s1,s2,,] + torch_transpose(jF[,t,s1,s2,,], 2, 3)) / 2 # ensure symmetry
            jFEig <- linalg_eigh(jF[,t,s1,s2,,])
            jFEig[[1]]$real[jFEig[[1]]$real < epsilon] <- epsilon
            jFEig[[1]]$real[jFEig[[1]]$real > ceil] <- ceil
            for (row in 1:N) {jF[row,t,s1,s2,,] <- torch_matmul(torch_matmul(jFEig[[2]]$real[row,,], torch_diag(jFEig[[1]]$real[row,])), torch_transpose(jFEig[[2]]$real[row,,], 1, 2))} 
            if (sum(as.numeric(torch_det(jF[,t,s1,s2,,])) < epsilon) > 0) {
              jFInd <- which(as.numeric(torch_det(jF[,t,s1,s2,,])) < epsilon)
              for (ind in jFInd) {jF[ind,t,s1,s2,,]$add_(5e-1 * torch_eye(No))} } }) # add a small constant to ensure p.s.d.
          jFChol[,t,s1,s2,,] <- linalg_cholesky_ex(torch_clone(jF[,t,s1,s2,,]))$L # Cholesky decomposition
          
          if (length(naRows[[t]]) == N) {
            jEta2[,t,s1,s2,] <- torch_clone(jEta[,t,s1,s2,]) 
            jP2[,t,s1,s2,,] <- torch_clone(jP[,t,s1,s2,,]) 
          } else {
            if (length(naRows[[t]]) > 0) {
              for (naRow in naRows[[t]]) {
                jEta2[naRow,t,s1,s2,] <- torch_clone(jEta[naRow,t,s1,s2,]) # Eq.7 (for missing entries)
                jP2[naRow,t,s1,s2,,] <- torch_clone(jP[naRow,t,s1,s2,,]) } } # Eq.8 (for missing entries)
            # kalman gain function
            KG <- torch_matmul(linalg_inv_ex(torch_clone(jP[,t,s1,s2,,]))$inverse, torch_matmul(torch_clone(jP[,t,s1,s2,,]), Lmd[[s1]]))
            # KG <- torch_matmul(torch_matmul(torch_clone(jP[,t,s1,s2,,]), Lmd[[s1]]), torch_cholesky_inverse(torch_clone(jFChol[,t,s1,s2,,]), upper=FALSE))
            for (noNaRow in noNaRows[[t]]) {
              jEta2[noNaRow,t,s1,s2,] <- torch_clone(jEta[noNaRow,t,s1,s2,]) + torch_matmul(torch_clone(KG[noNaRow,,]), torch_clone(jV[noNaRow,t,s1,s2,])) # Eq.7
              I_KGLmd <- torch_eye(Nf) - torch_matmul(torch_clone(KG[noNaRow,,]), torch_transpose(Lmd[[s1]], 1, 2))
              # jP2[noNaRow,t,s1,s2,,] <- torch_matmul(I_KGLmd, jP[noNaRow,t,s1,s2,,])} # Eq.8 
              jP2[noNaRow,t,s1,s2,,] <- torch_matmul(torch_matmul(torch_clone(I_KGLmd), torch_clone(jP[noNaRow,t,s1,s2,,])), torch_transpose(torch_clone(I_KGLmd), 1, 2)) + torch_matmul(torch_matmul(torch_clone(KG[noNaRow,,]), R[[s1]]), torch_transpose(torch_clone(KG[noNaRow,,]), 1, 2)) # Eq.9
              
              with_no_grad ({
                jP2Eig <- linalg_eigh(jP2[noNaRow,t,s1,s2,,])
                jP2Eig[[1]]$real[jP2Eig[[1]]$real < epsilon] <- epsilon
                jP2Eig[[1]]$real[jP2Eig[[1]]$real > ceil] <- ceil
                jP2[noNaRow,t,s1,s2,,] <- torch_matmul(torch_matmul(jP2Eig[[2]]$real, torch_diag(jP2Eig[[1]]$real)), torch_transpose(jP2Eig[[2]]$real, 1, 2)) 
                if (as.numeric(torch_det(jP2[noNaRow,t,s1,s2,,])) < epsilon) {jP2[noNaRow,t,s1,s2,,]$add_(1e-1 * torch_eye(Nf)) } }) } } # add a small constant to ensure p.s.d.
          
          # step 8: joint likelihood function f(eta_{t}|s,s',eta_{t-1})
          # is likelihood function different because I am dealing with latent variables instead of observed variables?
          for (noNaRow in noNaRows[[t]]) {
            jLik[noNaRow,t,s1,s2] <- torch_squeeze((-.5*pi)**(-Nf/2) * torch_det(torch_clone(jP[noNaRow,t,s1,s2,,]))**(-1) * torch_exp(-.5*torch_matmul(torch_matmul(torch_clone(jDelta[noNaRow,t,s1,s2,]), linalg_inv_ex(torch_clone(jP[noNaRow,t,s1,s2,,]))$inverse), torch_clone(jDelta[noNaRow,t,s1,s2,])))) 
            with_no_grad (jLik[noNaRow,t,s1,s2] <- min(jLik[noNaRow,t,s1,s2], ceil)) } } # Eq.12
        # torch_squeeze((-.5*pi)**(-Nf/2) * torch_prod(torch_diag(torch_clone(jPChol[noNaRow,t,s1,s2,,])))**(-1) * torch_exp(-.5*torch_matmul(torch_matmul(torch_clone(jDelta[noNaRow,t,s1,s2,]), torch_cholesky_inverse(torch_clone(jPChol[noNaRow,t,s1,s2,,]), upper=FALSE)), torch_clone(jDelta[noNaRow,t,s1,s2,])))) } } # Eq.12
        
        # step 9: transition probability P(s|s',eta_{t-1})  
        if (t == 1) {
          tPr[,t,1] <- torch_sigmoid(torch_squeeze(alpha[[1]]))
          tPr[,t,2] <- torch_sigmoid(torch_squeeze(alpha[[2]]))
          jPr[,t,2,2] <- torch_clone(tPr[,t,2]) * torch_clone(mPr[,t])
          jPr[,t,2,1] <- torch_clone(tPr[,t,1]) * (1-torch_clone(mPr[,t]))
          jPr[,t,1,2] <- (1-torch_clone(tPr[,t,2])) * torch_clone(mPr[,t])
          jPr[,t,1,1] <- (1-torch_clone(tPr[,t,1])) * (1-torch_clone(mPr[,t])) 
        } else {
          if (length(noNaRows[[t-1]]) == N) {
            tPr[,t,1] <- torch_sigmoid(torch_squeeze(alpha[[1]]) + torch_matmul(torch_tensor(eta[,t-1,]), beta[[1]]))
            tPr[,t,2] <- torch_sigmoid(torch_squeeze(alpha[[2]]) + torch_matmul(torch_tensor(eta[,t-1,]), beta[[2]])) 
            
            # step 10: Hamilton Filter
            # joint probability P(s,s'|eta_{t-1})
            jPr[,t,2,2] <- torch_clone(tPr[,t,2]) * torch_clone(mPr[,t])
            jPr[,t,2,1] <- torch_clone(tPr[,t,1]) * (1-torch_clone(mPr[,t]))
            jPr[,t,1,2] <- (1-torch_clone(tPr[,t,2])) * torch_clone(mPr[,t])
            jPr[,t,1,1] <- (1-torch_clone(tPr[,t,1])) * (1-torch_clone(mPr[,t])) 
          } else if (length(naRows[[t-1]]) == N) {jPr[,t,,] <- torch_clone(jPr[,t-1,,])
          } else { 
            for (noNaRow in noNaRows[[t-1]]) {
              tPr[noNaRow,t,1] <- torch_sigmoid(torch_squeeze(alpha[[1]]) + torch_matmul(torch_tensor(eta[noNaRow,t-1,]), beta[[1]]))
              tPr[noNaRow,t,2] <- torch_sigmoid(torch_squeeze(alpha[[2]]) + torch_matmul(torch_tensor(eta[noNaRow,t-1,]), beta[[2]])) 
              
              # step 10: Hamilton Filter
              # joint probability P(s,s'|eta_{t-1})
              jPr[noNaRow,t,2,2] <- torch_clone(tPr[noNaRow,t,2]) * torch_clone(mPr[noNaRow,t])
              jPr[noNaRow,t,2,1] <- torch_clone(tPr[noNaRow,t,1]) * (1-torch_clone(mPr[noNaRow,t]))
              jPr[noNaRow,t,1,2] <- (1-torch_clone(tPr[noNaRow,t,2])) * torch_clone(mPr[noNaRow,t])
              jPr[noNaRow,t,1,1] <- (1-torch_clone(tPr[noNaRow,t,1])) * (1-torch_clone(mPr[noNaRow,t])) } 
            for (naRow in naRows[[t-1]]) {jPr[naRow,t,,] <- torch_clone(jPr[naRow,t-1,,])} } }
        
        # marginal likelihood function f(eta_{t}|eta_{t-1})
        if (length(naRows[[t]]) == N) {jPr2[,t,,] <- torch_clone(jPr[,t,,])
        } else if (length(noNaRows[[t]]) == N) {
          mLik[,t] <- torch_sum(torch_clone(jLik[,t,,]) * torch_clone(jPr[,t,,]), dim=c(2,3))
          # (updated) joint probability P(s,s'|eta_{t})
          jPr2[,t,,] <- torch_clone(jLik[,t,,]) * torch_clone(jPr[,t,,]) / max(torch_clone(mLik[,t]), epsilon)
          for (row in 1:N) {
            with_no_grad (if (as.numeric(torch_sum(jPr2[row,t,,])) < 5e-2) {jPr2[row,t,,] <- jPr[row,t,,]} ) } 
        } else {
          for (naRow in naRows[[t]]) {jPr2[naRow,t,,] <- torch_clone(jPr[naRow,t,,])} 
          for (noNaRow in noNaRows[[t]]) {
            mLik[noNaRow,t] <- torch_sum(torch_clone(jLik[noNaRow,t,,]) * torch_clone(jPr[noNaRow,t,,]))
            # (updated) joint probability P(s,s'|eta_{t})
            jPr2[noNaRow,t,,] <- torch_clone(jLik[noNaRow,t,,]) * torch_clone(jPr[noNaRow,t,,]) / max(torch_clone(mLik[noNaRow,t]), epsilon)
            with_no_grad (if (as.numeric(torch_sum(jPr2[noNaRow,t,,])) < 5e-2) {jPr2[noNaRow,t,,] <- jPr[noNaRow,t,,]} ) } }
        mPr[,t+1] <- torch_sum(torch_clone(jPr2[,t,2,]), dim=2)
        
        # step 11: collapsing procedure
        for (s2 in 1:2) { 
          denom1 <- 1 - torch_clone(mPr[,t+1])
          with_no_grad({
            dInd <- which(as.numeric(denom1) < epsilon) 
            for (ind in dInd) {denom1[ind]$add_(epsilon)} })
          W[,t,1,s2] <- torch_clone(jPr2[,t,1,s2]) / torch_clone(denom1)
          denom2 <- torch_clone(mPr[,t+1])
          with_no_grad({
            dInd <- which(as.numeric(denom2) < epsilon) 
            for (ind in dInd) {denom2[ind]$add_(epsilon)} })
          W[,t,2,s2] <- torch_clone(jPr2[,t,2,s2]) / torch_clone(denom2) 
          with_no_grad({
            for (s1 in 1:2) {
              if (sum(as.numeric(W[,t,s1,s2]) < epsilon) > 0) {
                WInd <- which(as.numeric(W[,t,s1,s2]) < epsilon)
                for (ind in WInd) {W[ind,t,s1,s2] <- epsilon} } 
              if (sum(as.numeric(W[,t,s1,s2]) > 1) > 0) {
                WInd <- which(as.numeric(W[,t,s1,s2]) >= 1)
                for (ind in WInd) {W[ind,t,s1,s2] <- 1 - epsilon} } } }) } # add a small constant to ensure p.s.d.
        
        mEta[,t+1,,] <- torch_sum(torch_unsqueeze(torch_clone(W[,t,,]), dim=-1) * torch_clone(jEta2[,t,,,]), dim=3)
        subEta <- torch_unsqueeze(torch_clone(mEta[,t+1,,]), dim=-2) - torch_clone(jEta2[,t,,,])
        subEtaSq <- torch_matmul(torch_unsqueeze(torch_clone(subEta), dim=-1), torch_unsqueeze(torch_clone(subEta), dim=-2))
        with_no_grad({ 
          subEtaSq <- (subEtaSq + torch_transpose(subEtaSq, 4, 5)) / 2 # ensure symmetry
          subEtaSqEig <- linalg_eigh(subEtaSq)
          subEtaSqEig[[1]]$real[subEtaSqEig[[1]]$real > ceil] <- ceil
          subEtaSqEig[[1]]$real[subEtaSqEig[[1]]$real < epsilon] <- epsilon
          
          for (js in 1:nrow(jS)) {
            s1 <- jS$s1[js]
            s2 <- jS$s2[js]
            
            for (row in 1:N) {
              subEtaSq[row,s1,s2,,] <- 
                torch_matmul(torch_matmul(subEtaSqEig[[2]]$real[row,s1,s2,,], torch_diag(subEtaSqEig[[1]]$real[row,s1,s2,])), torch_transpose(subEtaSqEig[[2]]$real[row,s1,s2,,], 1, 2)) }
            
            if (sum(as.numeric(torch_det(subEtaSq[,s1,s2,,])) < epsilon) > 0) {
              subEtaSqInd <- which(as.numeric(torch_det(subEtaSq[,s1,s2,,])) < epsilon)
              for (ind in subEtaSqInd) {subEtaSq[ind,s1,s2,,]$add_(1e-1 * torch_eye(Nf))} } } })  
        
        mP[,t+1,,,] <- torch_sum(torch_unsqueeze(torch_unsqueeze(torch_clone(W[,t,,]), dim=-1), dim=-1) * (torch_clone(jP2[,t,,,,]) + torch_clone(subEtaSq)), dim=3) 
        with_no_grad({
          mP[,t+1,,,] <- (mP[,t+1,,,] + torch_transpose(mP[,t+1,,,], 3, 4)) / 2 # ensure symmetry
          for (s1 in 1:2) {
            mPEig <- linalg_eigh(mP[,t+1,s1,,])
            mPEig[[1]]$real[mPEig[[1]]$real > ceil] <- ceil
            mPEig[[1]]$real[mPEig[[1]]$real < epsilon] <- epsilon
            for (row in 1:N) {mP[row,t+1,s1,,] <- torch_matmul(torch_matmul(mPEig[[2]]$real[row,,], torch_diag(mPEig[[1]]$real[row,])), torch_transpose(mPEig[[2]]$real[row,,], 1, 2))}
            if (sum(as.numeric(torch_det(mP[,t+1,s1,,])) < epsilon) > 0) {
              mPInd <- which(as.numeric(torch_det(mP[,t+1,s1,,])) < epsilon)
              for (ind in mPInd) {mP[ind,t+1,s1,,]$add_(1e-1 * torch_eye(Nf))} } } }) }  # add a small constant to ensure p.s.d.
      
      # aggregated (summed) likelihood at each optimization step
      loss <- torch_nansum(-torch_clone(mLik))
      with_no_grad(sumLik[iter] <- as.numeric(-loss))
      
      # stopping criterion
      ifelse(abs(sumLik[iter][[1]] - sumLik[1][[1]]) > epsilon, crit <- (sumLik[iter][[1]] - sumLik[iter-1][[1]]) / (sumLik[iter][[1]] - sumLik[1][[1]]), crit <- 0)
      
      # add count if the new sumLik does not beat the best score
      ifelse(crit < 5e-2, count <- count + 1, count <- 0)
      
      cat('   sum likelihood = ', sumLik[iter][[1]], '\n')
      plot(unlist(sumLik), xlab='optimization step', ylab='sum likelihood', type='b')
      
      sumLikBest <- max(sumLikBest, sumLik[iter][[1]])
      if (sumLikBest < sumLik[iter][[1]]) {with_no_grad({thetaBest <- torch_clone(theta)})}
      # run adam function defined above
      with_no_grad({
        result <- adam(loss=loss, theta=theta, m=m, v=v)
        theta <- result$theta
        m <- result$m 
        v <- result$v 
        
        # switch off the gradient tracking
        a1 <- torch_tensor(theta$a1, requires_grad=FALSE)
        a2 <- torch_tensor(theta$a2, requires_grad=FALSE)
        B1d <- torch_tensor(theta$B1d, requires_grad=FALSE)
        B2d <- torch_tensor(theta$B2d, requires_grad=FALSE)
        k1 <- torch_tensor(theta$k1, requires_grad=FALSE)
        k2 <- torch_tensor(theta$k2, requires_grad=FALSE)
        Lmd1v <- torch_tensor(theta$Lmd1v, requires_grad=FALSE)
        Lmd2v <- torch_tensor(theta$Lmd2v, requires_grad=FALSE)
        alpha1 <- torch_tensor(theta$alpha1, requires_grad=FALSE)
        alpha2 <- torch_tensor(theta$alpha2, requires_grad=FALSE)
        beta1 <- torch_tensor(theta$beta1, requires_grad=FALSE)
        beta2 <- torch_tensor(theta$beta2, requires_grad=FALSE)
        Q1d <- torch_tensor(theta$Q1d, requires_grad=FALSE)
        Q2d <- torch_tensor(theta$Q2d, requires_grad=FALSE)
        R1d <- torch_tensor(theta$R1d, requires_grad=FALSE)
        R2d <- torch_tensor(theta$R2d, requires_grad=FALSE) })
      print('ok')
      
      if (count==3 || iter > 100) {print('   stopping criterion is met'); break}
      iter <- iter + 1
    } }) # continue to numerical re-optimization 
} # continue to re-initialization of parameters