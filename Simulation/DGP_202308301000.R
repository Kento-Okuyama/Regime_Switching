set.seed(42)
# install.packages("reshape")
library(reshape)
# install.packages("ggplot2")
library(ggplot2)
# install.packages("plotly")
library(plotly)
# install.packages("sigmoid")
library(sigmoid)
# install.packages("Rlab")
library(Rlab)
# install.packages('cowplot')
library(cowplot)

setwd("C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation")

N <- 25 # try for 25, 50, 100
Nt <- 25 # try for 25, 50
O1 <- 6
O2 <- 3
L1 <- 2

y1 <- eps1 <- array(NA, c(N, Nt, O1))
y2 <- eps2 <- array(NA, c(N, O2))
eta1 <- zeta1 <- array(NA, c(N, Nt, L1))
eta2 <- zeta2 <- array(NA, N)
S <- array(NA, c(N, Nt))

alpha21 <- array(NA, c(L1, 2))
beta2 <- array(NA, c(L1,2))
B1 <- array(NA, c(L1, 2))
Lmd10 <- array(NA, c(O1, L1))
Lmd2 <- array(NA, O2)

S[,1] <- 1
eps1_var <- rep(.3, O1) 
eps2_var <- rep(.3, O2)
zeta1_var <- rep(.1, L1)
zeta2_var <- .1
eta1_var <- rep(.1, L1)
eta2_var <- .1

alpha21[,1] <- c(.2, .3)
alpha21[,2] <- c(-.1, -.2)
beta2[,1] <- c(.1, .1)
beta2[,2] <- c(-.1 -.1)
B1[,1] <- rep(0.8, L1)
B1[,2] <- rep(0.4, L1)
Lmd10[,1] <- c(1, .4, .8, 0, 0, 0)
Lmd10[,2] <- c(0, 0, 0, 1, .5, 1.2)
Lmd2[] <- c(1, 1.1, .8)
gamma1 <- 3.5
gamma2 <- rep(1, 2)
gamma3 <- 0
gamma4 <- rep(0, 2)
P12 = 0

for (l1 in 1:L1) {eta1[, 1, l1] <- rnorm(N, 0, eta1_var[l1])}
eta2[] <- rnorm(N, 0, eta2_var)
for (l1 in 1:L1) {zeta1[, , l1] <- rnorm(N*Nt, 0, zeta1_var[l1])}
zeta2[] <- rnorm(N, 0, zeta2_var)

for (o1 in 1:O1) {eps1[, , o1] <- rnorm(N, 0, eps1_var[o1])}
for (o2 in 1:O2) {eps2[, o2] <- rnorm(N, 0, eps2_var[o2])}
for (i in 1:N) {
  y2[i,] <- Lmd2 * eta2[i] + eps2[i,] 
  y1[i,1,] <- Lmd10 %*% eta1[i,1,] + eps1[i,1,] }


for (t in 2:Nt) {
  for (i in 1:N) {
    # Markov switching model
    if (S[i,t-1] == 1) {pr <- sigmoid(gamma1 + gamma2 %*% eta1[i,t-1,] + gamma3 * eta2[i] + gamma4 %*% eta1[i,t-1,] * eta2[i])
    } else {pr <- P12}
    S[i,t] <- 2 - rbern(1, pr)
    # Structural model
    eta1[i,t,] <- alpha21[,S[i,t]] + beta2[,S[i,t]] * eta2[i] + diag(B1[,S[i,t]]) %*% eta1[i,t-1,] + zeta1[i,t,] + zeta2[i]
    # Measurement model
    y1[i,t,] <- Lmd10 %*% eta1[i,t,] + eps1[i,t,] } }

table(S)
table(S[,Nt])
# plot(eta1[1,,1])
# hist(eta1[,50,1])

set.seed(5)
obs <- sample(1:N, 2)
df_S_sampled <- df_S[df_S$ID %in% obs,]
df_S$ID <- as.factor(df_S$ID)
ggplot(data=df_S_sampled, aes(time, S, group=ID, color=ID)) + geom_line(size=.5) + theme(legend.position=c(0.2, 0.85))

df <- list(y1=y1, y2=y2, N=N, Nt=Nt, O1=O1, O2=O2, L1=L1,
           eta1_true=eta1, eta2_true=eta2,
           B11_true=alpha21[,1], B12_true=alpha21[,2], 
           B21d_true=B1[,1], B22d_true=B1[,2], 
           B31_true=beta2[,1], B32_true=beta2[,2], 
           Lmdd_true=Lmd10, Qd_true=zeta1_var+zeta2_var, 
           Rd_true=eps1_var, gamma1_true=gamma1, gamma2_true=gamma2, 
           gamma3_true=gamma3, gamma4_true=gamma4)

# some visualizations of the simulated data
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

df_eta2 <- melt(df$eta2_true); colnames(df_eta2) <- c('ID', 'eta2')
plot_eta2 <- ggplot(data=df_eta2, aes(x=eta2)) + geom_histogram(binwidth=.03)

df_y2_1 <- melt(df$y2[,1]); colnames(df_y2_1) <- c('y2_1')
df_y2_2 <- melt(df$y2[,2]); colnames(df_y2_2) <- c('y2_2')
df_y2_3 <- melt(df$y2[,3]); colnames(df_y2_3) <- c('y2_3')
plot_y2_1 <- ggplot(data=df_y2_1, aes(x=y2_1)) + geom_histogram(binwidth=.06)
plot_y2_2 <- ggplot(data=df_y2_2, aes(x=y2_2)) + geom_histogram(binwidth=.06)
plot_y2_3 <- ggplot(data=df_y2_3, aes(x=y2_3)) + geom_histogram(binwidth=.057)
plot_grid(plot_y2_1, plot_y2_2, plot_y2_3, labels='AUTO')
