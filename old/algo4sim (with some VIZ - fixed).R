# install.packages('torch')
library(torch)
# install.packages('reticulate')
library(reticulate)
# install.packages("cowplot")
library(cowplot)

nInit <- 30
maxIter <- 300
sEpsilon <- 1e-6
ceil <- 1e6
lr <- 1e-3
betas <- c(.9, .999)
H <- 25

y1 <- df$y1
y2 <- df$y2
eta1 <- df$eta1
eta2 <- df$eta2
N <- df$N
Nt <- df$Nt
No1 <- df$No1 
No2 <- df$No2 
Nf1 <- df$Nf1
Nf2 <- df$Nf2

y1 <- torch_tensor(y1)
eta1 <- torch_tensor(eta1)
eta2 <- torch_tensor(eta2)

set.seed(42)
sumLikBest <- 0

init <- 0
while (init < nInit) {
  cat('Init step ', init+1, '\n')
  iter <- 1
  count <- 0
  sumLik <- list()
  m <- v <- m_hat <- v_hat <- list()
  
  # initialize parameters
  B11 <- torch_tensor(df$B1[,1])
  B12 <- torch_tensor(df$B1[,2])
  B21 <- torch_tensor(df$B2[,,1])
  B22 <- torch_tensor(df$B2[,,2])
  B31 <- torch_tensor(df$B3[,,1])
  B32 <- torch_tensor(df$B3[,,2])
  d <- torch_tensor(df$d)
  Lmd <- torch_tensor(df$Lmd)
  Q1 <- torch_tensor(diag(df$Q[,1]))
  Q2 <- torch_tensor(diag(df$Q[,2]))
  R1 <- torch_tensor(diag(df$R[,1]))
  R2 <- torch_tensor(diag(df$R[,2]))
  gamma11 <- torch_tensor(runif(1, -3, 0))
  gamma21 <- torch_tensor(rep(-abs(rnorm(1, 0, .5)), Nf1))
  
  # with_detect_anomaly ({
  try (silent=FALSE, {
    while (count <=3 && iter <= maxIter) {
      cat('   optim step ', iter, '\n')
      
      # B11$requires_grad_()
      # B12$requires_grad_()
      # B21$requires_grad_()
      # B22$requires_grad_()
      # B31$requires_grad_()
      # B32$requires_grad_()
      # Q1$requires_grad_()
      # Q2$requires_grad_()
      # R1$requires_grad_()
      # R2$requires_grad_()
      gamma11$requires_grad_()
      gamma21$requires_grad_()
      
      theta <- list(#B11=B11, B12=B12, B21=B21, B22=B22, B31=B31, B32=B32,
        #Q1=Q1, Q2=Q2, R1=R1, R2=R2, 
        gamma11=gamma11, gamma21=gamma21)
      
      jEta <- torch_full(c(N,Nt+H,2,2,Nf1), NaN) # Eq.2 (LHS)
      jDelta <- torch_full(c(N,Nt,2,2,Nf1), NaN) # Eq.3 (LHS)
      jP <- torch_full(c(N,Nt+H,2,2,Nf1,Nf1), NaN) # Eq.4 (LHS)
      jV <- torch_full(c(N,Nt,2,2,No1), NaN) # Eq.5 (LHS)
      jF <- torch_full(c(N,Nt,2,2,No1,No1), NaN) # Eq.6 (LHS)
      jEta2 <- torch_full(c(N,Nt,2,2,Nf1), NaN) # Eq.7 (LHS)
      jP2 <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN) # Eq.8 (LHS)
      mEta <- torch_full(c(N,Nt+H+1,2,Nf1), NaN) # Eq.9-1 (LHS)
      mP <- torch_full(c(N,Nt+H+1,2,Nf1,Nf1), NaN) # Eq.9-2 (LHS)
      W <- torch_full(c(N,Nt,2,2), NaN) # Eq.9-3 (LHS)
      jPr <- torch_full(c(N,Nt+H,2,2), NaN) # Eq.10-1 (LHS)
      mLik <- torch_full(c(N,Nt), NaN) # Eq.10-2 (LHS)
      jPr2 <- torch_full(c(N,Nt,2,2), NaN) # Eq.10-3 (LHS)
      mPr <- torch_full(c(N,Nt+H+1), NaN) # Eq.10-4 (LHS)
      jLik <- torch_full(c(N,Nt,2,2), NaN) # Eq.11 (LHS)
      tPr <- torch_full(c(N,Nt+H,2), NaN) # Eq.12 (LHS)
      KG <- torch_full(c(N,Nt,2,2,Nf1,No1), NaN) # Kalman gain function
      I_KGLmd <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN) 
      denom1 <- torch_full(c(N,Nt), NaN)
      denom2 <- torch_full(c(N,Nt), NaN)
      subEta <- torch_full(c(N,Nt,2,2,Nf1), NaN) 
      subEtaSq <- torch_full(c(N,Nt,2,2,Nf1,Nf1), NaN)
      
      mEta[,1,,] <- 0
      mP[,1,,,] <- torch_eye(Nf1)
      mPr[,1] <- sEpsilon
      
      for (t in 1:Nt) {
        # if (t%%10==0) {cat('   t=', t, '\n')}
        
        jEta[,t,1,1,] <- B11 + mEta[,t,1,]$clone()$matmul(B21) + (eta2$clone() * B31)
        jEta[,t,2,1,] <- B12 + mEta[,t,1,]$clone()$matmul(B22) + (eta2$clone() * B32)
        jEta[,t,2,2,] <- B12 + mEta[,t,2,]$clone()$matmul(B22) + (eta2$clone() * B32)
        
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
        } else {tPr[,t,1] <- (gamma11 + eta1[,t-1,]$clone()$matmul(gamma21))$sigmoid() }
        
        jPr[,t,1,1] <- (1-tPr[,t,1]$clone()) * (1-mPr[,t]$clone())
        jPr[,t,2,1] <- tPr[,t,1]$clone() * (1-mPr[,t]$clone()) 
        jPr[,t,2,2] <- mPr[,t]$clone() 
        
        mLik[,t] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() +
          jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() +
          jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone()
        
        jPr2[,t,1,1] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() / mLik[,t]$clone()
        jPr2[,t,2,1] <- jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() / mLik[,t]$clone() 
        jPr2[,t,2,2] <- jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone() / mLik[,t]$clone() 
        
        mPr[,t+1] <- jPr2[,t,2,]$clone()$sum(dim=2)
        
        W[,t,1,1] <- jPr2[,t,1,1]$clone()$clip(min=sEpsilon) / (1 - mPr[,t+1]$clone())$clip(min=sEpsilon)
        W[,t,2,1] <- jPr2[,t,2,1]$clone()$clip(min=sEpsilon) / mPr[,t+1]$clone()$clip(min=sEpsilon)
        W[,t,2,2] <- jPr2[,t,2,2]$clone()$clip(min=sEpsilon) / mPr[,t+1]$clone()$clip(min=sEpsilon)
        
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
      
      loss <- -mLik[,1:Nt]$sum()
      sumLik[iter] <- -as.numeric(loss)
      
      if (is.infinite(sumLik[iter][[1]])) {
        print('   sum likelihood overflow')
        # switch off the gradient tracking
        with_no_grad ({
          for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
        break }
      
      crit <- ifelse(abs(sumLik[iter][[1]] - sumLik[1][[1]]) > sEpsilon, (sumLik[iter][[1]] - sumLik[iter-1][[1]]) / abs(sumLik[iter][[1]] - sumLik[1][[1]]), 0)
      count <- ifelse(crit < 1e-3, count + 1, 0)
      
      cat('   sum likelihood = ', sumLik[iter][[1]], '\n')
      plot(unlist(sumLik), xlab='optimization step', ylab='sum likelihood', type='b')
      
      if (count == 3 || iter == maxIter) {
        print('   stopping criterion is met')
        # switch off the gradient tracking
        with_no_grad ({
          for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
        break
      } else if (sumLikBest < sumLik[iter][[1]]) {
        thetaBest <- as.list(theta)
        sumLikBest <- sumLik[iter][[1]] }
      
      loss$backward() 
      
      grad <- list()
      
      with_no_grad ({
        for (var in 1:length(theta)) {
          grad[[var]] <- theta[[var]]$grad
          
          if (iter == 1) {m[[var]] <- v[[var]] <- torch_zeros_like(grad[[var]])} 
          
          # update moment estimates
          m[[var]] <- betas[1] * m[[var]] + (1 - betas[1]) * grad[[var]]
          v[[var]] <- betas[2] * v[[var]] + (1 - betas[2]) * grad[[var]]**2 
          
          # update bias corrected moment estimates
          m_hat[[var]] <- m[[var]] / (1 - betas[1]**iter)
          v_hat[[var]] <- v[[var]] / (1 - betas[2]**iter) 
          
          theta[[var]]$requires_grad_(FALSE)
          theta[[var]]$sub_(lr * m_hat[[var]] / (sqrt(v_hat[[var]]) + sEpsilon)) } })
      
      # B11 <- torch_tensor(theta$B11)
      # B12 <- torch_tensor(theta$B12)
      # B21 <- torch_tensor(theta$B21)
      # B22 <- torch_tensor(theta$B22)
      # B31 <- torch_tensor(theta$B31)
      # B32 <- torch_tensor(theta$B32)
      # Q1 <- torch_tensor(theta$Q1)
      # Q2 <- torch_tensor(theta$Q2)
      # R1 <- torch_tensor(theta$R1)
      # R2 <- torch_tensor(theta$R2)
      gamma11 <- torch_tensor(theta$gamma11)
      gamma21 <- torch_tensor(theta$gamma21)
      
      iter <- iter + 1 } 
    init <- init + 1 }) }

plot_ly(z=(as.array(mPr[,1:(Nt+1)]) > .5) + 1, colorscale='Grays', type='heatmap')
plot_ly(z=as.array(mPr[,1:(Nt+1)]), colorscale='Grays', type='heatmap')
plot_ly(z=S, colorscale='Grays', type='heatmap')
plot_ly(z=as.array(cbind((as.array(mPr[,Nt+1])>.5) + 1, S[,Nt])), colorscale='Grays', type='heatmap')

table(as.array(mPr[,Nt+1])>.5)
table(S[,Nt])

df1 <- melt(as.array(mPr[,2:(Nt+1)])); colnames(df1) <- c('ID', 'time', 'PrS')
plot1 <- ggplot(data=df1, aes(time, ID, fill=PrS)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df2 <- melt(as.array(mPr[,2:(Nt+1)]) > .5) + 1; colnames(df2) <- c('ID', 'time', 'PrS')
plot2 <- ggplot(data=df2, aes(time, ID, fill=PrS)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df3 <- melt(S); colnames(df3) <- c('ID', 'time', 'S')
plot3 <- ggplot(data=df3, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
plot_grid(plot1, plot2, plot3, labels = "AUTO")

# m1 <- cbind(melt(apply(as.array(mEta[20:N,2:(Nt+1),,1]), c(2,3), mean)), melt(apply(as.array(eta1[20:N,,1]), 2, mean)))
# colnames(m1)[4] <- 'value2'
# plot4 <- ggplot(data=m1, aes(X1)) + geom_line(aes(y=value, group=X2, color=X2)) + geom_line(aes(y=value2), color = 'darkred', linetype='twodash') + theme(legend.position='none')
# 
# m2 <- cbind(melt(apply(as.array(mEta[20:N,2:(Nt+1),,2]), c(2,3), mean)), melt(apply(as.array(eta1[20:N,,2]), 2, mean)))
# colnames(m2)[4] <- 'value2'
# plot5 <- ggplot(data=m2, aes(X1)) + geom_line(aes(y=value, group=X2, color=X2)) + geom_line(aes(y=value2), color = 'darkred', linetype='twodash') + theme(legend.position='none')
# 
# plot_grid(plot4, plot5)
# 
# m3 <- cbind(melt(apply(as.array(mEta[20:N,2:(Nt+1),1,]), c(2,3), mean))[,c(1,3)], melt(apply(as.array(eta1[20:N,,]), c(2,3), mean))[,2:3])
# colnames(m3)[4] <- 'value2'
# plot6 <- ggplot(data=m3, aes(X1)) + geom_line(aes(y=value, group=X2, color=X2)) + geom_line(aes(y=value2, group=X2, color=X2), linetype='twodash', size=.75) + theme(legend.position='none')
# 
# m4 <- cbind(melt(apply(as.array(mEta[20:N,2:(Nt+1),2,]), c(2,3), mean))[,c(1,3)], melt(apply(as.array(eta1[20:N,,]), c(2,3), mean))[,2:3])
# colnames(m4)[4] <- 'value2'
# plot7 <- ggplot(data=m4, aes(X1)) + geom_line(aes(y=value, group=X2, color=X2)) + geom_line(aes(y=value2, group=X2, color=X2), linetype='twodash', size=.75) + theme(legend.position='none')
# 
# plot_grid(plot6, plot7)
# 
# plot8 <- ggplot(data=melt(apply(as.array(mP[20:N,2:(Nt+1),,1,1]), c(2,3), mean)), aes(X1, value, group=X2, color=X2)) + geom_line() + theme(legend.position='none')
# plot9 <- ggplot(data=melt(apply(as.array(mP[20:N,2:(Nt+1),,2,2]), c(2,3), mean)), aes(X1, value, group=X2, color=X2)) + geom_line() + theme(legend.position='none')
# 
# plot_grid(plot8, plot9)

# for (h in 1:H) {
#   
#   if (h == 1) {eta1_pred <- eta1[,t-1,]
#   } else {eta1_pred <- jEta[,Nt+h-1,1,1,] * jPr[,Nt+h-1,1,1]$unsqueeze(-1) + jEta[,Nt+h-1,2,1,] * jPr[,Nt+h-1,2,1]$unsqueeze(-1) + jEta[,Nt+h-1,2,2,] * jPr[,Nt+h-1,2,2]$unsqueeze(-1)}
#   tPr[,Nt+h,1] <- (gamma11 + eta1_pred$matmul(gamma21))$sigmoid()
#   
#   jPr[,Nt+h,1,1] <- (1-tPr[,Nt+h,1]) * (1-mPr[,Nt+h])
#   jPr[,Nt+h,2,1] <- tPr[,Nt+h,1] * (1-mPr[,Nt+h]) 
#   jPr[,Nt+h,2,2] <- mPr[,Nt+h] 
#   
#   mPr[,Nt+h+1] <- jPr[,Nt+h,2,]$sum(dim=2)
#   
#   jEta[,Nt+h,1,1,] <- B11 + jEta[,Nt+h-1,1,1,]$clone()$matmul(B21) + (eta2$clone() * B31)
#   jEta[,Nt+h,2,1,] <- B12 + mEta[,Nt+h,1,]$clone()$matmul(B22) + (eta2$clone() * B32)
#   jEta[,Nt+h,2,2,] <- B12 + jEta[,Nt+h-1,2,2,]$clone()$matmul(B22) + (eta2$clone() * B32)
#   
#   jP[,Nt+h,1,1,,] <- B21$matmul(mP[,Nt+h,1,,]$clone())$matmul(B21$transpose(1, 2)) + Q1
#   jP[,Nt+h,2,1,,] <- B22$matmul(mP[,Nt+h,1,,]$clone())$matmul(B22$transpose(1, 2)) + Q2
#   jP[,Nt+h,2,2,,] <- B22$matmul(mP[,Nt+h,2,,]$clone())$matmul(B22$transpose(1, 2)) + Q2
#   
#   mEta[,Nt+h+1,1,] <- jEta[,Nt+h,1,1,] * jPr[,Nt+h,1,1]$unsqueeze(-1)
#   mEta[,Nt+h+1,2,] <- (jEta[,Nt+h,2,,] * jPr[,Nt+h,2,]$unsqueeze(-1))$sum(2)
#   
#   mP[,Nt+h+1,1,,] <- jP[,Nt+h,1,1,,] * jPr[,Nt+h,1,1]$unsqueeze(-1)$unsqueeze(-1)
#   mP[,Nt+h+1,2,,] <- (jP[,Nt+h,2,,,] * jPr[,Nt+h,2,]$unsqueeze(-1)$unsqueeze(-1))$sum(2) }

for (h in 1:H) {
  if (h > 1) {
    mEta[,Nt+h,1,] <- jEta[,Nt+h-1,1,1,]
    mEta[,Nt+h,2,] <- jEta[,Nt+h-1,2,2,] 
    mP[,Nt+h,1,,] <- jP[,Nt+h-1,1,1,,]
    mP[,Nt+h,2,,] <- jP[,Nt+h-1,2,2,,] }
  jEta[,Nt+h,1,1,] <- B11 + mEta[,Nt+h,1,]$clone()$matmul(B21) + (eta2$clone() * B31)
  jEta[,Nt+h,2,2,] <- B12 + mEta[,Nt+h,2,]$clone()$matmul(B22) + (eta2$clone() * B32)
  jP[,Nt+h,1,1,,] <- B21$matmul(mP[,Nt+h,1,,]$clone())$matmul(B21$transpose(1, 2)) + Q1
  jP[,Nt+h,2,2,,] <- B22$matmul(mP[,Nt+h,2,,]$clone())$matmul(B22$transpose(1, 2)) + Q2 }

df4 <- melt(apply(as.array(mEta[,(Nt+1):(Nt+H),,1]), c(2,3), mean)); colnames(df4) <- c('time', 'regime', 'mEta')
plot4 <- ggplot(data=df4, aes(time, mEta, group=regime, color=regime)) + geom_line() + theme(legend.position='none')
df5 <- melt(apply(as.array(mEta[,(Nt+1):(Nt+H),,2]), c(2,3), mean)); colnames(df5) <- c('time', 'regime', 'mEta')
plot5 <- ggplot(data=df5, aes(time, mEta, group=regime, color=regime)) + geom_line() + theme(legend.position='none')
df6 <- melt(apply(as.array(mP[,(Nt+1):(Nt+H),,1,1]), c(2,3), mean)); colnames(df6) <- c('time', 'regime', 'mP')
plot6 <- ggplot(data=df6, aes(time, mP, group=regime, color=regime)) + geom_line() + theme(legend.position='none')
df7 <- melt(apply(as.array(mP[,(Nt+1):(Nt+H),,2,2]), c(2,3), mean)); colnames(df7) <- c('time', 'regime', 'mP')
plot7 <- ggplot(data=df7, aes(time, mP, group=regime, color=regime)) + geom_line() + theme(legend.position='none')
plot_grid(plot4, plot5, plot6, plot7, labels = "AUTO")

df8 <- melt(as.array(mEta[,(Nt+1):(Nt+H),1,1])); colnames(df8) <- c('ID', 'time', 'mEta')
plot8 <- ggplot(data=df8, aes(time, mEta, group=ID, color=as.factor(ID))) + geom_line() + theme(legend.position='none')
df9 <- melt(as.array(mEta[,(Nt+1):(Nt+H),1,2])); colnames(df9) <- c('ID', 'time', 'mEta')
plot9 <- ggplot(data=df9, aes(time, mEta, group=ID, color=as.factor(ID))) + geom_line() + theme(legend.position='none')
df10 <- melt(as.array(mEta[,(Nt+1):(Nt+H),2,1])); colnames(df10) <- c('ID', 'time', 'mEta')
plot10 <- ggplot(data=df10, aes(time, mEta, group=ID, color=as.factor(ID))) + geom_line() + theme(legend.position='none')
df11 <- melt(as.array(mEta[,(Nt+1):(Nt+H),2,2])); colnames(df11) <- c('ID', 'time', 'mEta')
plot11 <- ggplot(data=df11, aes(time, mEta, group=ID, color=as.factor(ID))) + geom_line() + theme(legend.position='none')
plot_grid(plot8, plot9, plot10, plot11, labels = "AUTO")

# plot12 <- ggplot(data=melt(as.array(mP[,(Nt+1):(Nt+H),1,1,1])), aes(X2, value, group=X1, color=X1)) + geom_line() + theme(legend.position='none')
# plot13 <- ggplot(data=melt(as.array(mP[,(Nt+1):(Nt+H),1,2,2])), aes(X2, value, group=X1, color=X1)) + geom_line() + theme(legend.position='none')
# plot14 <- ggplot(data=melt(as.array(mP[,(Nt+1):(Nt+H),2,1,1])), aes(X2, value, group=X1, color=X1)) + geom_line() + theme(legend.position='none')
# plot15 <- ggplot(data=melt(as.array(mP[,(Nt+1):(Nt+H),2,2,2])), aes(X2, value, group=X1, color=X1)) + geom_line() + theme(legend.position='none')



