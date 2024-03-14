set.seed(42)

##########
# regime #
##########

table(S)
table(S[,Nt])

if (N == 25 && Nt == 25) {S_25x25 <- S
} else if (N == 25 && Nt == 50) {S_25x50 <- S
} else if (N == 50 && Nt == 25) {S_50x25 <- S
} else if (N == 50 && Nt == 50) {S_50x50 <- S
} else if (N == 100 && Nt == 25) {S_100x25 <- S
} else if (N == 100 && Nt == 50) {S_100x50 <- S}

table(S_25x25[,25])[2] / sum(table(S_25x25[,25])) 
table(S_25x50[,50])[2] / sum(table(S_25x50[,50]))
table(S_50x25[,25])[2] / sum(table(S_50x25[,25]))
table(S_50x50[,50])[2] / sum(table(S_50x50[,50]))
table(S_100x25[,25])[2] / sum(table(S_100x25[,25]))
table(S_100x50[,50])[2] / sum(table(S_100x50[,50]))

# plot the regime-prediction
df_S_25x25 <- melt(S_25x25); colnames(df_S_25x25) <- c('ID', 'time', 'S')
plot_S_25x25 <- ggplot(data=df_S_25x25, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df_S_25x50 <- melt(S_25x50); colnames(df_S_25x50) <- c('ID', 'time', 'S')
plot_S_25x50 <- ggplot(data=df_S_25x50, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df_S_50x25 <- melt(S_50x25); colnames(df_S_50x25) <- c('ID', 'time', 'S')
plot_S_50x25 <- ggplot(data=df_S_50x25, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df_S_50x50 <- melt(S_50x50); colnames(df_S_50x50) <- c('ID', 'time', 'S')
plot_S_50x50 <- ggplot(data=df_S_50x50, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df_S_100x25 <- melt(S_100x25); colnames(df_S_100x25) <- c('ID', 'time', 'S')
plot_S_100x25 <- ggplot(data=df_S_100x25, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
df_S_100x50 <- melt(S_100x50); colnames(df_S_100x50) <- c('ID', 'time', 'S')
plot_S_100x50 <- ggplot(data=df_S_100x50, aes(time, ID, fill=S)) + geom_tile(color='grey') + scale_fill_gradient(low='white',high='red') + theme(legend.position='none')
plot_grid(plot_S_25x25, plot_S_25x50, plot_S_50x25, plot_S_50x50, plot_S_100x25, plot_S_100x50, labels = "AUTO")

obs <- sample(1:N, 2)
df_S_sampled <- df_S[df_S$ID %in% obs,]
df_S$ID <- as.factor(df_S$ID)
ggplot(data=df_S_sampled, aes(time, S, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position=c(0.2, 0.85))

#########################################
# intra-individual latent factor scores #
#########################################

# plot(eta1[1,,1])
# hist(eta1[,50,1])

df_eta1_1 <- melt(df$eta1_true[obs,,1]); colnames(df_eta1_1) <- c('ID', 'time', 'eta1_1')
df_eta1_2 <- melt(df$eta1_true[obs,,2]); colnames(df_eta1_2) <- c('ID', 'time', 'eta1_2')
df_eta1_1$ID[df_eta1_1$ID==1] <- obs[1]
df_eta1_1$ID[df_eta1_1$ID==2] <- obs[2]
df_eta1_2$ID[df_eta1_2$ID==1] <- obs[1]
df_eta1_2$ID[df_eta1_2$ID==2] <- obs[2]
df_eta1_1$ID <- as.factor(df_eta1_1$ID)
df_eta1_2$ID <- as.factor(df_eta1_2$ID)
plot_eta1_1 <- ggplot(data=df_eta1_1, aes(time, eta1_1, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position=c(0.75, 0.45))
plot_eta1_2 <- ggplot(data=df_eta1_2, aes(time, eta1_2, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position=c(0.75, 0.45))
plot_grid(plot_eta1_1, plot_eta1_2, labels='AUTO')

###########################################
# intra-individual observed factor scores #
###########################################

df_y1_1 <- melt(df$y1[obs,,1]); colnames(df_y1_1) <- c('ID', 'time', 'y1_1')
df_y1_2 <- melt(df$y1[obs,,2]); colnames(df_y1_2) <- c('ID', 'time', 'y1_2')
df_y1_3 <- melt(df$y1[obs,,3]); colnames(df_y1_3) <- c('ID', 'time', 'y1_3')
df_y1_4 <- melt(df$y1[obs,,4]); colnames(df_y1_4) <- c('ID', 'time', 'y1_4')
df_y1_5 <- melt(df$y1[obs,,5]); colnames(df_y1_5) <- c('ID', 'time', 'y1_5')
df_y1_6 <- melt(df$y1[obs,,6]); colnames(df_y1_6) <- c('ID', 'time', 'y1_6')
df_y1_1$ID[df_y1_1$ID==1] <- obs[1]
df_y1_1$ID[df_y1_1$ID==2] <- obs[2]
df_y1_2$ID[df_y1_2$ID==1] <- obs[1]
df_y1_2$ID[df_y1_2$ID==2] <- obs[2]
df_y1_3$ID[df_y1_3$ID==1] <- obs[1]
df_y1_3$ID[df_y1_3$ID==2] <- obs[2]
df_y1_4$ID[df_y1_4$ID==1] <- obs[1]
df_y1_4$ID[df_y1_4$ID==2] <- obs[2]
df_y1_5$ID[df_y1_5$ID==1] <- obs[1]
df_y1_5$ID[df_y1_5$ID==2] <- obs[2]
df_y1_6$ID[df_y1_6$ID==1] <- obs[1]
df_y1_6$ID[df_y1_6$ID==2] <- obs[2]
df_y1_1$ID <- as.factor(df_y1_1$ID)
df_y1_2$ID <- as.factor(df_y1_2$ID)
df_y1_3$ID <- as.factor(df_y1_3$ID)
df_y1_4$ID <- as.factor(df_y1_4$ID)
df_y1_5$ID <- as.factor(df_y1_5$ID)
df_y1_6$ID <- as.factor(df_y1_6$ID)
plot_y1_1 <- ggplot(data=df_y1_1, aes(time, y1_1, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position='none')
plot_y1_2 <- ggplot(data=df_y1_2, aes(time, y1_2, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position='none')
plot_y1_3 <- ggplot(data=df_y1_3, aes(time, y1_3, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position='none')
plot_y1_4 <- ggplot(data=df_y1_4, aes(time, y1_4, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position='none')
plot_y1_5 <- ggplot(data=df_y1_5, aes(time, y1_5, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position='none')
plot_y1_6 <- ggplot(data=df_y1_6, aes(time, y1_6, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position='none')
plot_grid(plot_y1_1, plot_y1_2, plot_y1_3, plot_y1_4, plot_y1_5, plot_y1_6, labels='AUTO')

#########################################
# inter-individual latent factor scores #
#########################################

df_eta2 <- melt(df$eta2_true); colnames(df_eta2) <- c('ID', 'eta2')
plot_eta2 <- ggplot(data=df_eta2, aes(x=eta2)) + geom_histogram(binwidth=.03)

###########################################
# inter-individual observed factor scores #
###########################################

df_y2_1 <- melt(df$y2[,1]); colnames(df_y2_1) <- c('y2_1')
df_y2_2 <- melt(df$y2[,2]); colnames(df_y2_2) <- c('y2_2')
df_y2_3 <- melt(df$y2[,3]); colnames(df_y2_3) <- c('y2_3')
plot_y2_1 <- ggplot(data=df_y2_1, aes(x=y2_1)) + geom_histogram(binwidth=.06)
plot_y2_2 <- ggplot(data=df_y2_2, aes(x=y2_2)) + geom_histogram(binwidth=.06)
plot_y2_3 <- ggplot(data=df_y2_3, aes(x=y2_3)) + geom_histogram(binwidth=.057)
plot_grid(plot_y2_1, plot_y2_2, plot_y2_3, labels='AUTO')
