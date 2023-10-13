set.seed(42)

N <- 50 # try for 25, 50, 100
Nt <- 50 # try for 25, 50
O1 <- 6
O2 <- 3
L1 <- 2

y1 <- eps1 <- array(NA, c(N, Nt+1, O1))
y2 <- eps2 <- array(NA, c(N, O2))
eta1 <- zeta1 <- array(NA, c(N, Nt+1, L1))
eta2 <- zeta2 <- array(NA, N)
S <- array(NA, c(N, Nt+1))

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
B1[,1] <- rep(0.8, L1)
B1[,2] <- rep(0.4, L1)
beta2[,1] <- c(.1, .1)
beta2[,2] <- c(-.1 -.1)
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
for (l1 in 1:L1) {zeta1[, , l1] <- rnorm(N*(Nt+1), 0, zeta1_var[l1])}
zeta2[] <- rnorm(N, 0, zeta2_var)

for (o1 in 1:O1) {eps1[, , o1] <- rnorm(N, 0, eps1_var[o1])}
for (o2 in 1:O2) {eps2[, o2] <- rnorm(N, 0, eps2_var[o2])}
for (i in 1:N) {
  y2[i,] <- Lmd2 * eta2[i] + eps2[i,] 
  y1[i,1,] <- Lmd10 %*% eta1[i,1,] + eps1[i,1,] }


for (t in 2:(Nt+1)) {
  for (i in 1:N) {
    # Markov switching model
    if (S[i,t-1] == 1) {pr <- sigmoid(gamma1 + gamma2 %*% eta1[i,t-1,] + gamma3 * eta2[i] + gamma4 %*% eta1[i,t-1,] * eta2[i])
    } else {pr <- P12}
    S[i,t] <- 2 - rbern(1, pr)
    # Structural model
    eta1[i,t,] <- alpha21[,S[i,t]] + beta2[,S[i,t]] * eta2[i] + diag(B1[,S[i,t]]) %*% eta1[i,t-1,] + zeta1[i,t,] + zeta2[i]
    # Measurement model
    y1[i,t,] <- Lmd10 %*% eta1[i,t,] + eps1[i,t,] } }

df <- list(y1=y1, y2=y2, N=N, Nt=Nt, O1=O1, O2=O2, L1=L1,
           eta1_true=eta1, eta2_true=eta2,
           B11_true=alpha21[,1], B12_true=alpha21[,2], 
           B21d_true=B1[,1], B22d_true=B1[,2], 
           B31_true=beta2[,1], B32_true=beta2[,2], 
           Lmdd_true=Lmd10, Qd_true=zeta1_var+zeta2_var, 
           Rd_true=eps1_var, gamma1_true=gamma1, gamma2_true=gamma2, 
           gamma3_true=gamma3, gamma4_true=gamma4)
