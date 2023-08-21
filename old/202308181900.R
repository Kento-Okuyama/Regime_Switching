# install.packages('lavaan')
library(lavaan)
# install.packages('torch')
library(torch)
# install.packages('reticulate')
library(reticulate)
# install.packages('cowplot')
library(cowplot)
nInit <- 3
maxIter <- 100
sEpsilon <- 1e-8
lr <- 1e-2
stopCrit <- 1e-4
betas <- c(.9, .999)
H <- 25

y1 <- df$y1
y2 <- df$y2
N <- df$N
Nt <- df$Nt
O1 <- df$O1
O2 <- df$O2
L1 <- df$L1

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

set.seed(42)
sumLikBest <- 0

init <- 0
while (init < nInit) {
  cat('Init step ', init+1, '\n')
  iter <- 1
  count <- 0
  m <- v <- m_hat <- v_hat <- list()

  # initialize parameters
  B11 <- torch_tensor(alpha21[,1])
  B12 <- torch_tensor(alpha21[,2])
  B21d <- torch_tensor(B1[,1])
  B22d <- torch_tensor(B1[,2])
  B31 <- torch_tensor(beta2[,1])
  B32 <- torch_tensor(beta2[,2])
  Lmd1 <- torch_tensor(Lmd10)
  Lmd2 <- torch_tensor(Lmd10)
  Qd <- torch_tensor(zeta1_var + zeta2_var)
  Rd <- torch_tensor(eps1_var)
  gamma1 <- torch_tensor(abs(rnorm(1, 0, 3)))
  gamma2 <- torch_tensor(abs(rnorm(2, 0, 1)))
  gamma3 <- torch_tensor(0)
  gamma4 <- torch_tensor(rep(0,2))

  # with_detect_anomaly ({
  try (silent=FALSE, {
    while (count <=3 && iter <= maxIter) {
      cat('   optim step ', iter, '\n')

      B11$requires_grad_()
      B12$requires_grad_()
      B21d$requires_grad_()
      B22d$requires_grad_()
      B31$requires_grad_()
      B32$requires_grad_()
      Qd$requires_grad_()
      Rd$requires_grad_()
      gamma1$requires_grad_()
      gamma2$requires_grad_()

      theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
                    Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2)

      jEta <- torch_full(c(N,Nt+H,2,2,L1), NaN)
      jP <- torch_full(c(N,Nt+H,2,2,L1,L1), NaN)
      jV <- torch_full(c(N,Nt,2,2,O1), NaN)
      jF <- torch_full(c(N,Nt,2,2,O1,O1), NaN)
      jEta2 <- torch_full(c(N,Nt,2,2,L1), NaN)
      jP2 <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
      mEta <- torch_full(c(N,Nt+H+1,2,L1), NaN)
      mP <- torch_full(c(N,Nt+H+1,2,L1,L1), NaN)
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
      mPr[,1] <- 0
      W[,,1,1] <- 1

      B21 <- B21d$clone()$diag()
      B22 <- B22d$clone()$diag()
      Q1 <- Qd$clone()$diag()
      Q2 <- Qd$clone()$diag()
      R1 <- Rd$clone()$diag()
      R2 <- Rd$clone()$diag()

      for (t in 1:Nt) {
        # if (t%%10==0) {cat('   t=', t, '\n')}

        jEta[,t,1,1,] <- B11 + mEta[,t,1,]$clone()$matmul(B21$clone()) + eta2$clone()$outer(B31)
        jEta[,t,2,1,] <- B12 + mEta[,t,1,]$clone()$matmul(B22$clone()) + eta2$clone()$outer(B32)
        jEta[,t,2,2,] <- B12 + mEta[,t,2,]$clone()$matmul(B22$clone()) + eta2$clone()$outer(B32)

        jP[,t,1,1,,] <- B21$clone()$matmul(mP[,t,1,,]$clone())$matmul(B21$clone()) + Q1$clone()
        jP[,t,2,1,,] <- B22$clone()$matmul(mP[,t,1,,]$clone())$matmul(B22$clone()) + Q2$clone()
        jP[,t,2,2,,] <- B22$clone()$matmul(mP[,t,2,,]$clone())$matmul(B22$clone()) + Q2$clone()

        jV[,t,1,1,] <- y1[,t,]$clone() - jEta[,t,1,1,]$clone()$matmul(Lmd1$transpose(1, 2))
        jV[,t,2,1,] <- y1[,t,]$clone() - jEta[,t,2,1,]$clone()$matmul(Lmd2$transpose(1, 2))
        jV[,t,2,2,] <- y1[,t,]$clone() - jEta[,t,2,2,]$clone()$matmul(Lmd2$transpose(1, 2))

        jF[,t,1,1,,] <- Lmd1$matmul(jP[,t,1,1,,]$clone())$matmul(Lmd1$transpose(1, 2)) + R1$clone()
        jF[,t,2,1,,] <- Lmd2$matmul(jP[,t,2,1,,]$clone())$matmul(Lmd1$transpose(1, 2)) + R2$clone()
        jF[,t,2,2,,] <- Lmd2$matmul(jP[,t,2,2,,]$clone())$matmul(Lmd2$transpose(1, 2)) + R2$clone()

        KG[,t,1,1,,] <- jP[,t,1,1,,]$clone()$matmul(Lmd1$transpose(1, 2))$matmul(jF[,t,1,1,,]$clone()$cholesky_inverse())
        KG[,t,2,1,,] <- jP[,t,2,1,,]$clone()$matmul(Lmd2$transpose(1, 2))$matmul(jF[,t,2,1,,]$clone()$cholesky_inverse())
        KG[,t,2,2,,] <- jP[,t,2,2,,]$clone()$matmul(Lmd2$transpose(1, 2))$matmul(jF[,t,2,2,,]$clone()$cholesky_inverse())

        jEta2[,t,1,1,] <- jEta[,t,1,1,]$clone() + KG[,t,1,1,,]$clone()$matmul(jV[,t,1,1,]$clone()$unsqueeze(-1))$squeeze()
        jEta2[,t,2,1,] <- jEta[,t,2,1,]$clone() + KG[,t,2,1,,]$clone()$matmul(jV[,t,2,1,]$clone()$unsqueeze(-1))$squeeze()
        jEta2[,t,2,2,] <- jEta[,t,2,2,]$clone() + KG[,t,2,2,,]$clone()$matmul(jV[,t,2,2,]$clone()$unsqueeze(-1))$squeeze()

        I_KGLmd[,t,1,1,,] <- torch_eye(L1) - KG[,t,1,1,,]$clone()$matmul(Lmd1)
        I_KGLmd[,t,2,1,,] <- torch_eye(L1) - KG[,t,2,1,,]$clone()$matmul(Lmd2)
        I_KGLmd[,t,2,2,,] <- torch_eye(L1) - KG[,t,2,2,,]$clone()$matmul(Lmd2)

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

        if (t == 1) {tPr[,t,1] <- 1 - (gamma1 + gamma3 * eta2$clone())$sigmoid()
        } else {
          eta1_pred[,t-1,] <- (1 - mPr[,t]$clone()$unsqueeze(-1)) * mEta[,t,1,]$clone() + mPr[,t]$clone()$unsqueeze(-1) * mEta[,t,2,]$clone()
          tPr[,t,1] <- 1 - (gamma1 + eta1_pred[,t-1,]$clone()$matmul(gamma2) + gamma3 * eta2$clone() + eta1_pred[,t-1,]$clone()$matmul(gamma4) * eta2$clone())$sigmoid() }

        jPr[,t,1,1] <- (1-tPr[,t,1]$clone())$clip(min=sEpsilon)  * (1-mPr[,t]$clone())$clip(min=sEpsilon)
        jPr[,t,2,1] <- tPr[,t,1]$clone() * (1-mPr[,t]$clone())$clip(min=sEpsilon)
        jPr[,t,2,2] <- mPr[,t]$clone()

        mLik[,t] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() +
          jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() +
          jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone()

        jPr2[,t,1,1] <- jLik[,t,1,1]$clone() * jPr[,t,1,1]$clone() / mLik[,t]$clone()
        jPr2[,t,2,1] <- jLik[,t,2,1]$clone() * jPr[,t,2,1]$clone() / mLik[,t]$clone()
        jPr2[,t,2,2] <- jLik[,t,2,2]$clone() * jPr[,t,2,2]$clone() / mLik[,t]$clone()

        mPr[,t+1] <- jPr2[,t,2,]$clone()$sum(dim=2)

        W[,t,2,1] <- jPr2[,t,2,1]$clone() / mPr[,t+1]$clone()
        W[,t,2,2] <- jPr2[,t,2,2]$clone() / mPr[,t+1]$clone()

        mEta[,t+1,1,] <- jEta2[,t,1,1,]$clone()
        mEta[,t+1,2,] <- (W[,t,2,]$clone()$unsqueeze(-1) * jEta2[,t,2,,]$clone())$sum(2)

        subEta[,t,1,1,] <- mEta[,t+1,1,]$clone() - jEta2[,t,1,1,]$clone()
        subEta[,t,2,1,] <- mEta[,t+1,2,]$clone() - jEta2[,t,2,1,]$clone()
        subEta[,t,2,2,] <- mEta[,t+1,2,]$clone() - jEta2[,t,2,2,]$clone()

        subEtaSq[,t,1,1,,] <- subEta[,t,1,1,]$clone()$unsqueeze(-1)$matmul(subEta[,t,1,1,]$clone()$unsqueeze(-2))
        subEtaSq[,t,2,1,,] <- subEta[,t,2,1,]$clone()$unsqueeze(-1)$matmul(subEta[,t,2,1,]$clone()$unsqueeze(-2))
        subEtaSq[,t,2,2,,] <- subEta[,t,2,2,]$clone()$unsqueeze(-1)$matmul(subEta[,t,2,2,]$clone()$unsqueeze(-2))

        mP[,t+1,1,,] <- jP2[,t,1,1,,]$clone() + subEtaSq[,t,1,1,,]$clone()
        mP[,t+1,2,,] <- (W[,t,2,]$clone()$unsqueeze(dim=-1)$unsqueeze(-1) * (jP2[,t,2,,,]$clone() + subEtaSq[,t,2,,,]$clone()))$sum(2) }

      loss <- -mLik[,1:Nt]$sum()

      if (is.infinite(-as.numeric(loss))) {
        print('   sum likelihood overflow')
        with_no_grad ({
          for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
        break }

      if (init == 0 && iter == 1) {
        gamma1_list  <- melt(as.matrix(gamma1), nrow=1); gamma1_list$X1 <- as.factor(init+1); gamma1_list$X2 <- iter
        gamma21_list  <- melt(as.matrix(gamma2), nrow=1)[melt(as.matrix(gamma2), nrow=1)$X1==1,]; gamma21_list$X1 <- as.factor(init+1); gamma21_list$X2 <- iter
        gamma22_list  <- melt(as.matrix(gamma2), nrow=1)[melt(as.matrix(gamma2), nrow=1)$X1==2,]; gamma22_list$X1 <- as.factor(init+1); gamma22_list$X2 <- iter
        sumLik_list <- sumLik_init <- sumLik_new <- melt(as.matrix(-loss), nrow=1); sumLik_list$X1 <- as.factor(init+1); sumLik_list$X2 <- iter

      } else {
        gamma1_new <- melt(as.matrix(gamma1)); gamma1_new$X1 <- as.factor(init+1); gamma1_new$X2 <- iter
        gamma1_list  <- rbind(gamma1_list, gamma1_new)
        gamma21_new <- melt(as.matrix(gamma2))[melt(as.matrix(gamma2))$X1==1,]; gamma21_new$X1 <- as.factor(init+1); gamma21_new$X2 <- iter
        gamma21_list  <- rbind(gamma21_list, gamma21_new)
        gamma22_new <- melt(as.matrix(gamma2))[melt(as.matrix(gamma2))$X1==2,]; gamma22_new$X1 <- as.factor(init+1); gamma22_new$X2 <- iter
        gamma22_list  <- rbind(gamma22_list, gamma22_new)
        sumLik_prev <- sumLik_new
        sumLik_new <- melt(as.matrix(-loss)); sumLik_new$X1 <- as.factor(init+1); sumLik_new$X2 <- iter
        if (iter == 1) {sumLik_init <- sumLik_new}
        sumLik_list <- rbind(sumLik_list, sumLik_new)

        crit <- ifelse(abs(sumLik_new$value - sumLik_init$value) > sEpsilon, (sumLik_new$value - sumLik_prev$value) / abs(sumLik_new$value - sumLik_init$value), 0)
        count <- ifelse(crit < stopCrit, count + 1, 0)

        if (count == 3 || iter == maxIter) {
          print('   stopping criterion is met')
          with_no_grad ({
            for (var in 1:length(theta)) {theta[[var]]$requires_grad_(FALSE)} })
          break
        } else if (sumLikBest < sumLik_new$value) {
          thetaBest <- as.list(theta)
          sumLikBest <- sumLik_new$value }

        df_sumLik <- sumLik_list; colnames(df_sumLik) <- c('initialization', 'iteration', 'sumLik')
        df_gamma1 <- gamma1_list; colnames(df_gamma1) <- c('initialization', 'iteration', 'gamma1')
        df_gamma21 <- gamma21_list; colnames(df_gamma21) <- c('initialization', 'iteration', 'gamma21')
        df_gamma22 <- gamma22_list; colnames(df_gamma22) <- c('initialization', 'iteration', 'gamma22')

        if (iter %% 5 == 0) {
          plot_sumLik <- ggplot(data=df_sumLik, aes(iteration, sumLik, group=initialization, color=initialization)) + geom_line() + theme(legend.position='none')
          plot_gamma1 <- ggplot(data=df_gamma1, aes(iteration, gamma1, group=initialization, color=initialization)) + geom_line() + theme(legend.position='none')
          plot_gamma21 <- ggplot(data=df_gamma21, aes(iteration, gamma21, group=initialization, color=initialization)) + geom_line() + theme(legend.position='none')
          plot_gamma22 <- ggplot(data=df_gamma22, aes(iteration, gamma22, group=initialization, color=initialization)) + geom_line() + theme(legend.position='none')
          print(plot_grid(plot_sumLik, plot_gamma1, plot_gamma21, plot_gamma22), labels='AUTO') } }

      cat('   sumLik = ', sumLik_new$value, '\n')
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

      B11 <- torch_tensor(theta$B11)
      B12 <- torch_tensor(theta$B12)
      B21d <- torch_tensor(theta$B21d)
      B22d <- torch_tensor(theta$B22d)
      B31 <- torch_tensor(theta$B31)
      B32 <- torch_tensor(theta$B32)
      Qd <- torch_tensor(theta$Qd)
      Rd <- torch_tensor(theta$Rd)
      gamma1 <- torch_tensor(theta$gamma1)
      gamma2 <- torch_tensor(theta$gamma2)

      iter <- iter + 1 }
    init <- init + 1 }) }


df_S <- melt(S); colnames(df_S) <- c('ID', 'time', 'S')
plot_S <- ggplot(data=df_S, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')

df_Pr <- melt(as.array(mPr[,2:(Nt+1)])); colnames(df_Pr) <- c('ID', 'time', 'S')
plot_Pr <- ggplot(data=df_Pr, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')

df_diff <- melt(as.array(mPr[,2:(Nt+1)])); colnames(df_diff) <- c('ID', 'time', 'diff'); df_diff$diff <- abs(df_diff$diff - df_S$S)
plot_diff <- ggplot(data=df_diff, aes(time, ID, fill=diff)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')

plot_grid(plot_S, plot_Pr, plot_diff)
