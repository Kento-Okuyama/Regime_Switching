theta <- thetaBest

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
# alpha2 <- torch_tensor(alpha2, requires_grad=TRUE)
alpha <- list(alpha1) # alpha <- list(alpha1, alpha2)
beta1 <- torch_tensor(beta1, requires_grad=TRUE)
# beta2 <- torch_tensor(beta2, requires_grad=TRUE)
beta <- list(beta1) # beta <- list(beta1, beta2)
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

theta <- list(a1=a1, a2=a2, B1d=B1d, B2d=B2d, C1d=C1d, C2d=C2d, D1=D1, D2=D2, k1=k1, k2=k2, Lmd1v=Lmd1v, Lmd2v=Lmd2v, Omega1v=Omega1v, Omega2v=Omega2v, A1=A1, A2=A2, alpha1=alpha1, beta1=beta1, Q1d=Q1d, Q2d=Q2d, R1d=R1d, R2d=R2d)

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
        tPr[,t,2] <- .99 # torch_sigmoid(torch_squeeze(alpha[[2]] + torch_tensor(eta2) * beta[[2]][(Nf1+1):Nf]))
        
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
          tPr[,t,2] <- .99 # torch_sigmoid(torch_squeeze(alpha[[2]]) + torch_matmul(torch_tensor(eta12[,t-1,]), beta[[2]])) 
          
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
            tPr[noNaRow,t,2] <- .99 # torch_sigmoid(torch_squeeze(alpha[[2]]) + torch_matmul(torch_tensor(eta12[noNaRow,t-1,]), beta[[2]])) 
            
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
              WInd <- which(as.numeric(W[,t,s1,s2]) < 0) # cap from below
              for (ind in WInd) {W[ind,t,s1,s2] <- epsilon} }
            
            while (sum(as.numeric(W[,t,s1,s2]) > 1) > 0) {
              WInd <- which(as.numeric(W[,t,s1,s2]) >= 1) # cap from above
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

colors <- rainbow(N)
c <- brewer.pal(8, "Dark2")

i <- 6 # person in {1, ... , N}
plot(x[i,], lwd=1.5, ylim=c(0,1), type="l")
lines(mPr[i,2:(Nt+1)], lwd=1.5, col=c[i%%8]) 
cat('data', '\n', x[i,], '\n'); cat('prediction (hard clustering)', '\n', as.numeric(mPr[i,2:(Nt+1)] > .5), '\n')

for (t in 1:Nt) {
  cat('\n', 't=', t, '\n')
  print(table(as.numeric((x - mPr[i,2:(Nt+1)] > .5)[,t]))) }
