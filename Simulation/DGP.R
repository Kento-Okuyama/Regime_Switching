set.seed(42)
# install.packages("reshape")
library(reshape)
# install.packages("ggplot2")
library(ggplot2)
# install.packages("plotly")
library(plotly)
# install.packages("sigmoid")
library(sigmoid)

N <- 100
Nt <- 50
O1 <- 6
O2 <- 3
O <- 9
L1 <- 2
L2 <- 1
L <- 3

eta1 <- array(NA, c(N, Nt, L1))
eta2 <- array(NA, c(N, L2))
S <- array(NA, c(N, Nt))
B1 <- array(NA, c(L1, 2))
B2 <- array(NA, c(L1, L1, 2))
B3 <- array(NA, c(L1, L2, 2))
d_1 <- array(NA, O1)
d_2 <- array(NA, O2)
Lmd_1 <- array(0, c(O1, L1))
Lmd_2 <- array(0, c(O2, L2))
y1 <- array(NA, c(N, Nt, O1))
y2 <- array(NA, c(N, O2))
Q <- array(NA, c(L1, 2))
R <- array(NA, c(O1, 2))

S[,1] <- 1
B1[,1] <- c(.01, .02)
B1[,2] <- c(-.03, -.04)
B2[,,1] <- diag(rep(1, L1))
B2[,,2] <- diag(rep(1, L1))
B3[,,1] <- c(.03, .025)
B3[,,2] <- c(-.03, -.02)
theta <- -1
Q[,1] <- c(.075, .06)
Q[,2] <- c(.06, .05)
d_1[] <- 0
Lmd_1[1,1] <- .5
Lmd_1[2,1] <- .4
Lmd_1[3,1] <- .3
Lmd_1[4,2] <- .7
Lmd_1[5,2] <- .2
Lmd_1[6,2] <- .1
d_2[] <- 0
Lmd_2[1,1] <- 1.5
Lmd_2[2,1] <- 1.1
Lmd_2[3,1] <- .8
R[,1] <- c(.5, .45)
R[,2] <- c(.3, .35)

for (l1 in 1:L1) {eta1[, 1, l1] <- rnorm(N, 0, .4)}
for (l2 in 1:L2) {eta2[, l2] <- rnorm(N, 0, .4)}

# structural model
for (i in 1:N) {
  for (t in 2:Nt) {
    s <- S[i,t-1]
    if (s == 1) {s <- 1 + (.5 * runif(1, -.5, .0) + .5 * (1 / (1 + exp(-sum(eta1[i,t-1,])))) < .0)}
    # if (s == 1) {s <- 1 + ((1 / (1 + exp(-sum(eta1[i,t-1,])))) < runif(1, 0, .3))}
    eta1[i,t,] <- B1[,s] + as.numeric(B2[,,s] %*% eta1[i,t-1,]) + as.numeric(B3[,,s] * eta2[i,]) + rnorm(L1, 0, Q[,s]) 
    S[i,t] <- ifelse(S[i,t-1]==2, 2, (1 / (1 + exp(-sum(eta1[i,t,]))) < runif(1, min=-.65, max=.35)) + 1) } }


# measurement model
for (i in 1:N) {
  for (t in 1:Nt) {y1[i,t,] <- d_1 + as.numeric(Lmd_1 %*% eta1[i,t,]) + rnorm(O1, 0, R[,s]) }}
for (i in 1:N) {y2[i,] <- d_2 + as.numeric(Lmd_2 %*% eta2[i]) + rnorm(O2, 0, 1)}

table(S[,Nt])

# heatmap(S[,], Colv=NA, Rowv=NA, scale='none', revC=TRUE, xlab='time', ylab='person', main='regime switch')
# ggplot(melt(S), aes(X2, X1)) + geom_tile(aes(fill = value))
plot_ly(z=S, colorscale='Grays', type='heatmap')

# heatmap(apply(eta1, c(1,2), sum), Colv=NA, Rowv=NA, scale='none', revC=TRUE)
# ggplot(melt(apply(eta1, c(1,2), sum)), aes(X2, X1)) + geom_tile(aes(fill = value))

plot_ly(z=apply(eta1[1:30,,], c(1,2), sum), type='heatmap')

df <- list(eta1=eta1, eta2=eta2, y1=y1, y2=y2, 
           B1=B1, B2=B2, B3=B3, S=S, d=d_1, Lmd=Lmd_1, Q=Q, R=R,
           N=N, Nt=Nt, Nf1=L1, Nf2=L2, No1=O1, No2=O2)

# store <-  array(NA, dim=c(N,Nt))
# for (i in 1:N) {
#   for (t in 1:Nt) {
#     store[i,t] <- 1 / (1 + exp(-sum(eta1[i,t,]))) 
#   }
# }
#   
# plot_ly(z=store, type='heatmap')
# plot(store[1,])


