set.seed(42)

# information criterion
q <- length(torch_cat(thetaBest)) - 8
sumLikBest <- as.numeric(torch_sum(mLik[,t]))
AIC <- -2 * log(sumLikBest) + 2 * q
BIC <- -2 * log(sumLikBest) + q * log(N * Nt)
sumLikBest_NT <- sumLikBest / (N*Nt) 

#####################
# regime prediction #
#####################

# plot the regime-prediction
DO <- 2 - mPr[,2:(Nt+1)]
DO2 <- 2 - mPr2[,2:(Nt+1)]
df_S <- melt(S); colnames(df_S) <- c('ID', 'time', 'S')
plot_S <- ggplot(data=df_S, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df_Pr <- melt(as.array(DO)); colnames(df_Pr) <- c('ID', 'time', 'S')
plot_Pr <- ggplot(data=df_Pr, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df_Pr2 <- melt(as.array(DO2)); colnames(df_Pr2) <- c('ID', 'time', 'S')
plot_Pr2 <- ggplot(data=df_Pr2, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df_Pr_diff <- melt(as.array(DO2  > 1.5) + 1); colnames(df_Pr_diff) <- c('ID', 'time', 'S')
df_Pr_diff$S <- abs(df_Pr_diff$S - df_S$S)
plot_Pr_diff <- ggplot(data=df_Pr_diff, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
plot_grid(plot_S, plot_Pr, plot_Pr2, plot_Pr_diff, labels = "AUTO")

# contingency table
cTable1 <- table(S[,Nt], round(as.array(DO[,Nt])))
sensitivity1 <- cTable1[1,1] / sum(cTable1[1,])
specificity1 <- cTable1[2,2] / sum(cTable1[2,])
cTable2 <- table(S[,Nt], round(as.array(DO2[,Nt])))
sensitivity2 <- cTable2[1,1] / sum(cTable2[1,])
specificity2 <- cTable2[2,2] / sum(cTable2[2,])

#####################
#####################

######################################################
# plot the intra-individual latent score predictions #
######################################################

# mean score function
delta <- as.numeric(torch_sum((eta1_pred[,Nt,] - df$eta1_true[,Nt,])**2))
delta_N <- delta / N

# filtered results
obs <- sample(1:N, 2)
df_eta1_1 <- melt(df$eta1[obs,,1]); colnames(df_eta1_1) <- c('ID', 'time', 'eta1_1')
df_eta1_2 <- melt(df$eta1[obs,,2]); colnames(df_eta1_2) <- c('ID', 'time', 'eta1_2')
plot_eta1_1 <- ggplot(data=df_eta1_1, aes(time, eta1_1, group=ID, color=as.factor(ID))) + geom_line(size=.5) + theme(legend.position='none')
plot_eta1_2 <- ggplot(data=df_eta1_2, aes(time, eta1_2, group=ID, color=as.factor(ID))) + geom_line(size=.5) + theme(legend.position='none')

df_eta1_pred_1 <- melt(as.array(eta1_pred[obs,,1])); colnames(df_eta1_pred_1) <- c('ID', 'time', 'eta1_pred_1')
df_eta1_pred_2 <- melt(as.array(eta1_pred[obs,,2])); colnames(df_eta1_pred_2) <- c('ID', 'time', 'eta1_pred_2')
df_P_pred_1 <- melt(as.array(P_pred[obs,,1,1])); colnames(df_P_pred_1) <- c('ID', 'time', 'P_pred_1')
df_P_pred_2 <- melt(as.array(P_pred[obs,,2,2])); colnames(df_P_pred_2) <- c('ID', 'time', 'P_pred_2')
df_eta1_pred_1_band <- cbind(df_eta1_pred_1, low=df_eta1_pred_1$eta1_pred_1 - 2 * sqrt(df_P_pred_1$P_pred_1 / Nt), high=df_eta1_pred_1$eta1_pred_1 + 2 * sqrt(df_P_pred_1$P_pred_1 / Nt))
df_eta1_pred_2_band <- cbind(df_eta1_pred_2, low=df_eta1_pred_2$eta1_pred_2 - 2 * sqrt(df_P_pred_2$P_pred_2 / Nt), high=df_eta1_pred_2$eta1_pred_2 + 2 * sqrt(df_P_pred_2$P_pred_2 / Nt))
plot_eta1_pred_1_band <- ggplot(data=df_eta1_pred_1_band, aes(time, eta1_pred_1, group=ID, color=as.factor(ID))) + geom_ribbon(aes(ymin=low, ymax=high), color='grey70', fill='grey70') + geom_line(size=.5) + theme(legend.position='none')
plot_eta1_pred_2_band <- ggplot(data=df_eta1_pred_2_band, aes(time, eta1_pred_2, group=ID, color=as.factor(ID))) + geom_ribbon(aes(ymin=low, ymax=high), color='grey70', fill='grey70') + geom_line(size=.5) + theme(legend.position='none')

df_eta1_diff_1_band <- df_eta1_pred_1_band; colnames(df_eta1_diff_1_band)[3] <- c('eta1_diff_1')
df_eta1_diff_1_band$eta1_diff_1 <- df_eta1_diff_1_band$eta1_diff_1 - df_eta1_1$eta1_1
df_eta1_diff_1_band$low <- df_eta1_diff_1_band$low - df_eta1_1$eta1_1
df_eta1_diff_1_band$high <- df_eta1_diff_1_band$high - df_eta1_1$eta1_1
df_eta1_diff_2_band <- df_eta1_pred_2_band; colnames(df_eta1_diff_2_band)[3] <- c('eta1_diff_2')
df_eta1_diff_2_band$eta1_diff_2 <- df_eta1_diff_2_band$eta1_diff_2 - df_eta1_2$eta1_2
df_eta1_diff_2_band$low <- df_eta1_diff_2_band$low - df_eta1_2$eta1_2
df_eta1_diff_2_band$high <- df_eta1_diff_2_band$high - df_eta1_2$eta1_2
plot_eta1_diff_1_band <- ggplot(data=df_eta1_diff_1_band, aes(time, eta1_diff_1, group=ID, color=as.factor(ID))) + geom_ribbon(aes(ymin=low, ymax=high), color='grey70', fill='grey70') +  geom_line(size=.5) + theme(legend.position='none')
plot_eta1_diff_2_band <- ggplot(data=df_eta1_diff_2_band, aes(time, eta1_diff_2, group=ID, color=as.factor(ID))) + geom_ribbon(aes(ymin=low, ymax=high), color='grey70', fill='grey70') + geom_line(size=.5) + theme(legend.position='none')
plot_grid(plot_eta1_1, plot_eta1_pred_1_band, plot_eta1_diff_1_band, plot_eta1_2, plot_eta1_pred_2_band, plot_eta1_diff_2_band, labels='AUTO')

df_eta1_diff2_1 <- melt(as.numeric(as.array(eta1_pred[,,1]) - df$eta1[,,1])); colnames(df_eta1_diff2_1) <- c('eta1_diff2_1')
plot_eta1_diff2_1 <- ggplot(data=df_eta1_diff2_1, aes(x=eta1_diff2_1)) + geom_histogram(binwidth=.05)
df_eta1_diff2_2 <- melt(as.numeric(as.array(eta1_pred[,,2]) - df$eta1[,,2])); colnames(df_eta1_diff2_2) <- c('eta1_diff2_2')
plot_eta1_diff2_2 <- ggplot(data=df_eta1_diff2_2, aes(x=eta1_diff2_2)) + geom_histogram(binwidth=.05)
plot_grid(plot_eta1_diff2_1, plot_eta1_diff2_2, labels='AUTO')

# smoothed results
df_eta1 <- melt(as.array(eta1_sm[,(Nt-3):(Nt-1),1])); colnames(df_eta1) <- c('ID', 'time', 'eta1')
plot_eta1 <- ggplot(data=df_eta1, aes(time, eta1, group=ID, color=as.factor(ID))) + geom_line() + ylim(-3, 3) + theme(legend.position='none')
plot_grid(plot_eta1, labels = "AUTO")

######################################################
######################################################

######################################################
# plot the inter-individual latent score predictions #
######################################################

df_eta2_pred <- melt(as.array(eta2)); colnames(df_eta2_pred) <- c('eta2_pred')
plot_eta2_pred <- ggplot(data=df_eta2_pred, aes(x=eta2_pred)) + geom_histogram(binwidth=.05)
plot_grid(plot_eta2, plot_eta2_pred, labels='AUTO')

######################################################
######################################################