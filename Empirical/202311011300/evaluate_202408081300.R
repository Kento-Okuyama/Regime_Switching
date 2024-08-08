seed <- 42
init <- 3
set.seed(seed + init)

# Set working directory
setwd('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408051300')

# Read the filtered data
filtered <- readRDS(paste0('output/filter__emp_42_N_80_T_51_O1_9_O2_3_L1_4_init_', init, '.RDS', sep=''))

# Call the preprocessing function and load the resulting variables into the global environment
source('library_202408051300.R')
source('preprocessing_202408051300.R')
library_load()
df <- readRDS('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408051300/output/df__emp_42_N_80_T_51_O1_9_O2_3_L1_4.RDS')
list2env(as.list(df), envir=.GlobalEnv)

# Extract the best theta parameters from the filtered data
theta <- filtered$theta_best

# Extract specific parameters from the theta vector
B11 <- theta[1:L1]
B12 <- theta[(L1+1):(2*L1)]
B21d <- theta[(2*L1+1):(3*L1)]
B22d <- theta[(3*L1+1):(4*L1)]
B31 <- theta[(4*L1+1):(5*L1)]
B32 <- theta[(5*L1+1):(6*L1)]
Lmdd1 <- theta[6*L1+1]
Lmdd2 <- theta[6*L1+2]
Lmdd3 <- theta[6*L1+3]
Lmdd4 <- theta[6*L1+4]
Lmdd5 <- theta[6*L1+5]
gamma1 <- theta[6*(L1+1)]
gamma2 <- theta[(6*(L1+1)+1):(6*(L1+1)+L1)]
Qd <- theta[(6*(L1+1)+L1+1):(6*(L1+1)+2*L1)]
Rd <- theta[(6*(L1+1)+2*L1+1):(6*(L1+1)+2*L1+O1)]
mP_DO <- theta[6*(L1+1)+2*L1+O1+1]
tP_SB <- theta[6*(L1+1)+2*L1+O1+2]

theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
              Lmdd1=Lmdd1, Lmdd2=Lmdd2, Lmdd3=Lmdd3, Lmdd4=Lmdd4, Lmdd5=Lmdd5,
              Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2, mP_DO=mP_DO, tP_SB=tP_SB)

lEpsilon <- 1e-3
ceil <- 1e15
sEpsilon <- 1e-15
epsilon <- 1e-8
betas <- c(.9, .999)
const <- (2*pi)**(-O1/2)

#####################
# Measurement model #
#####################
model_cfa <- '
# latent variables
IQ =~ abiMath + TIMMS + totIQ'

y2_df <- as.data.frame(y2)

colnames(y2_df) <- c('abiMath', 'TIMMS', 'totIQ')
fit_cfa <- cfa(model_cfa, data=y2_df)
eta2_score <- lavPredict(fit_cfa, method='Bartlett')
eta2 <- as.array(eta2_score[,1])

y1 <- torch_tensor(y1[,,1:O1])
eta2 <- torch_tensor(eta2)

sumLik_best <- -99
output_best <- NULL

# cat('Init step ', init, '\n')
iter <- 1
count <- 0
m <- v <- m_hat <- v_hat <- list()

# Initialize parameters
B11 <- torch_tensor(B11)
B12 <- torch_tensor(B12)
B21d <- torch_tensor(B21d)
B22d <- torch_tensor(B22d)
B31 <- torch_tensor(B31)
B32 <- torch_tensor(B32)
Lmdd1 <- torch_tensor(Lmdd1)
Lmdd2 <- torch_tensor(Lmdd2)
Lmdd3 <- torch_tensor(Lmdd3)
Lmdd4 <- torch_tensor(Lmdd4)
Lmdd5 <- torch_tensor(Lmdd5)
gamma1 <- torch_tensor(gamma1)
gamma2 <- torch_tensor(gamma2)
Qd <- torch_tensor(Qd)
Rd <- torch_tensor(Rd)
mP_DO <- torch_tensor(mP_DO)
tP_SB <- torch_tensor(tP_SB)

jEta <- torch_full(c(N,Nt+1,2,2,L1), 0)
jP <- torch_full(c(N,Nt+1,2,2,L1,L1), 0)
jV <- torch_full(c(N,Nt,2,2,O1), NaN)
jF <- torch_full(c(N,Nt,2,2,O1,O1), NaN)
jEta2 <- torch_full(c(N,Nt,2,2,L1), 0)
jP2 <- torch_full(c(N,Nt,2,2,L1,L1), 0)
mEta <- torch_full(c(N,Nt+1,2,L1), 0)
mP <- torch_full(c(N,Nt+1,2,L1,L1), NaN)
W <- torch_full(c(N,Nt,2,2), NaN)
jPr <- torch_full(c(N,Nt+1,2,2), 0)
mLik <- torch_full(c(N,Nt), NaN)
jPr2 <- torch_full(c(N,Nt,2,2), 0)
mPr <- torch_full(c(N,Nt+2,2), NaN)
jLik <- torch_full(c(N,Nt,2,2), 0)
tPr <- torch_full(c(N,Nt+1,2,2), NaN)
KG <- torch_full(c(N,Nt,2,2,L1,O1), 0)
I_KGLmd <- torch_full(c(N,Nt,2,2,L1,L1), NaN)
subEta <- torch_full(c(N,Nt,2,2,L1), NaN)
eta1_pred <- torch_full(c(N,Nt+2,L1), NaN)
P_pred <- torch_full(c(N,Nt+2,L1,L1), NaN)

mP[,1,,,] <- torch_eye(L1)
mPr[,1,1] <- 1 - mP_DO
mPr[,1,2] <- mP_DO
tPr[,,1,2] <- tP_SB
tPr[,,2,2] <- 1 - tPr[,,1,2]   

Lmd <- torch_full(c(O1, L1), 0)
Lmd[1,1] <- Lmd[3,2] <- Lmd[5,3] <- Lmd[7,4] <- 1 
Lmd[2,1] <- Lmdd1
Lmd[4,2] <- Lmdd2
Lmd[6,3] <- Lmdd3
Lmd[8,4] <- Lmdd4
Lmd[9,4] <- Lmdd5
B21 <- B21d$diag()
B22 <- B22d$diag()
LmdT <- Lmd$transpose(1, 2)
Q <- Qd$diag()
R <- Rd$diag() 
B1 <- torch_cat(c(B11, B12))$reshape(c(2, L1))
B2 <- torch_cat(c(B21, B22))$reshape(c(2, L1, L1))
B3 <- torch_cat(c(B31, B32))$reshape(c(2, L1))

for (t in 1:Nt) {
  for (i in 1:N) {
    
    #################
    # Kalman filter #
    #################
    
    if (as.logical(sum(torch_isnan(y1[i,t,]))) <= 0) {
      
      jEta[i,t,,,] <- B1$unsqueeze(-2) + mEta[i,t,,]$clone()$unsqueeze(2)$matmul(B2) + eta2[i]$clone()$unsqueeze(-1)$unsqueeze(-1)$unsqueeze(-1) * B3$unsqueeze(-2)
      jP[i,t,,,,] <- mP[i,t,,,]$clone()$unsqueeze(2)$matmul(B2[2,,]**2) + Q$expand(c(2, 2, -1, -1))
      jV[i,t,,,] <- y1[i,t,]$clone()$unsqueeze(-2)$unsqueeze(-2) - jEta[i,t,,,]$clone()$matmul(LmdT) # possible missingness
      jF[i,t,,,,] <- Lmd$matmul(jP[i,t,,,,]$clone()$matmul(LmdT)) + R$expand(c(2, 2, -1, -1))
      KG[i,t,,,,] <- jP[i,t,,,,]$clone()$matmul(LmdT)$matmul(jF[i,t,,,,]$clone()$cholesky_inverse())
      jEta2[i,t,,,] <- jEta[i,t,,,] + KG[i,t,,,,]$clone()$matmul(jV[i,t,,,]$clone()$unsqueeze(-1))$squeeze()
      I_KGLmd[i,t,,,,] <- torch_eye(L1)$expand(c(2,2,-1,-1)) - KG[i,t,,,,]$clone()$matmul(Lmd)
      jP2[i,t,,,,] <- I_KGLmd[i,t,,,,]$clone()$matmul(jP[i,t,,,,]$clone())$matmul(I_KGLmd[i,t,,,,]$clone()$transpose(3, 4)) +
        KG[i,t,,,,]$clone()$matmul(R)$matmul(KG[i,t,,,,]$clone()$transpose(3, 4))
      
      log_det_jF <- jF[i,t,,,,]$clone()$det()$clip(min=-ceil, max=ceil)$log()
      quadratic_term <- -.5 * jF[i,t,,,,]$clone()$cholesky_inverse()$matmul(jV[i,t,,,]$clone()$unsqueeze(-1))$squeeze()$unsqueeze(-2)$matmul(jV[i,t,,,]$clone()$unsqueeze(-1))$squeeze()$squeeze()
      jLik[i,t,,] <- log(sEpsilon + const) - log_det_jF + quadratic_term
      
      ###################
      # Hamilton filter #
      ###################
      
      eta1_pred[i,t,] <- mPr[i,t,1]$clone()$unsqueeze(-1) * mEta[i,t,1,]$clone() + mPr[i,t,2]$clone()$unsqueeze(-1) * mEta[i,t,2,]$clone()
      P_pred[i,t,,] <- mPr[i,t,1]$clone()$unsqueeze(-1)$unsqueeze(-1) * mP[i,t,1,,]$clone() + mPr[i,t,2]$clone()$unsqueeze(-1)$unsqueeze(-1) * mP[i,t,2,,]$clone()
      tPr[i,t,1,1] <- (gamma1 + eta1_pred[i,t,]$clone()$matmul(gamma2))$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
      tPr[i,t,2,1] <- 1 - tPr[i,t,1,1]
      jPr[i,t,,] <- tPr[i,t,,]$clone() * mPr[i,t,]$clone()$unsqueeze(-1)
      mLik[i,t] <- (jLik[i,t,,]$clone() * jPr[i,t,,]$clone())$sum() 
      jPr2[i,t,,] <- jLik[i,t,,]$clone() * jPr[i,t,,]$clone() / mLik[i,t]$clone()$unsqueeze(-1)$unsqueeze(-1) # possible missingness
      mPr[i,t+1,] <- jPr2[i,t,,]$sum(2)$clip(min=sEpsilon, max=1-sEpsilon)
      W[i,t,,] <- jPr2[i,t,,]$clone() / mPr[i,t+1,]$clone()$unsqueeze(-1)
      mEta[i,t+1,,] <- (W[i,t,,]$clone()$unsqueeze(-1) * jEta2[i,t,,,]$clone())$sum(2) 
      subEta[i,t,,,] <- mEta[i,t+1,,]$unsqueeze(2) - jEta2[i,t,,,]
      mP[i,t+1,,,] <- (W[i,t,,]$clone()$unsqueeze(-1)$unsqueeze(-1) * (jP2[i,t,,,,] + subEta[i,t,,,]$clone()$unsqueeze(-1)$matmul(subEta[i,t,,,]$clone()$unsqueeze(-2))))$sum(2) 
    }
    
    if (as.logical(sum(torch_isnan(y1[i,t,]))) > 0) {
      jEta2[i,t,,,] <- jEta2[i,t-1,,,]$clone()
      jP2[i,t,,,,] <- jP2[i,t-1,,,,]$clone()
      mEta[i,t+1,,] <- mEta[i,t,,]$clone()
      mP[i,t+1,,,] <- mP[i,t,,,]$clone()
      jPr2[i,t,,] <- jPr2[i,t-1,,]$clone()  
      mPr[i,t+1,] <- mPr[i,t,]$clone()
    }
  }
}

eta1_pred[,Nt+1,] <- mPr[,Nt+1,1]$unsqueeze(-1) * mEta[,Nt+1,1,] + mPr[,Nt+1,2]$unsqueeze(-1) * mEta[,Nt+1,2,]
P_pred[,Nt+1,,] <- mPr[,Nt+1,1]$unsqueeze(-1)$unsqueeze(-1) * mP[,Nt+1,1,,] + mPr[,Nt+1,2]$unsqueeze(-1)$unsqueeze(-1) * mP[,Nt+1,2,,]

jEta[,Nt+1,1,1,] <- B11 + mEta[,Nt+1,1,]$matmul(B21) + eta2$outer(B31)
jEta[,Nt+1,1,2,] <- B11 + mEta[,Nt+1,2,]$matmul(B21) + eta2$outer(B31)
jEta[,Nt+1,2,1,] <- B12 + mEta[,Nt+1,1,]$matmul(B22) + eta2$outer(B32)
jEta[,Nt+1,2,2,] <- B12 + mEta[,Nt+1,2,]$matmul(B22) + eta2$outer(B32)

jP[,Nt+1,1,1,,] <- B21$matmul(mP[,Nt+1,1,,])$matmul(B21) + Q
jP[,Nt+1,1,2,,] <- B21$matmul(mP[,Nt+1,2,,])$matmul(B21) + Q
jP[,Nt+1,2,1,,] <- B22$matmul(mP[,Nt+1,1,,])$matmul(B22) + Q
jP[,Nt+1,2,2,,] <- B22$matmul(mP[,Nt+1,2,,])$matmul(B22) + Q

tPr[,Nt+1,1,1] <- (gamma1 + eta1_pred[,Nt+1,]$matmul(gamma2))$sigmoid()$clip(min=sEpsilon, max=1-sEpsilon)
tPr[,Nt+1,2,1] <- 1 - tPr[,Nt+1,1,1]

jPr[,Nt+1,1,1] <- tPr[,Nt+1,1,1] * mPr[,Nt+1,1]
jPr[,Nt+1,2,1] <- tPr[,Nt+1,2,1] * mPr[,Nt+1,1]
jPr[,Nt+1,1,2] <- tPr[,Nt+1,1,2] * mPr[,Nt+1,2]
jPr[,Nt+1,2,2] <- tPr[,Nt+1,2,2] * mPr[,Nt+1,2]

eta1_pred[,Nt+2,] <- jEta[,Nt+1,1,1,] * jPr[,Nt+1,1,1]$unsqueeze(-1) + jEta[,Nt+1,2,1,] * jPr[,Nt+1,2,1]$unsqueeze(-1) + jEta[,Nt+1,2,2,] * jPr[,Nt+1,2,2]$unsqueeze(-1)
P_pred[,Nt+2,,] <- jP[,Nt+1,1,1,,] * jPr[,Nt+1,1,1]$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,1,,] * jPr[,Nt+1,2,1]$unsqueeze(-1)$unsqueeze(-1) + jP[,Nt+1,2,2,,] * jPr[,Nt+1,2,2]$unsqueeze(-1)$unsqueeze(-1)

mPr[,Nt+2,1] <- jPr[,Nt+1,1,]$sum(2)
mPr[,Nt+2,2] <- jPr[,Nt+1,2,]$sum(2)

############################################
n_persons <- 80
n_timepoints <- 51
data <- as.array(mPr[,3:53,2])
dim(data)
summary(data)

# Convert the data into a data frame and reshape it for plotting
df <- as.data.frame(data)
df_long <- df %>%
  mutate(ID = row_number()) %>%
  pivot_longer(cols = -ID, names_to = "Time", values_to = "Value") %>%
  mutate(Time = as.numeric(gsub("V", "", Time)))  # Convert Time to numeric

# Plot histograms and KDEs for each time point
ggplot(df_long, aes(x = Value, fill = as.factor(Time))) +
  geom_histogram(aes(y = after_stat(density)), alpha = 0.4, position = "identity", bins = 30) +
  geom_density(alpha = 0.6) +
  labs(title = "Distribution of Values at Each Time Point",
       x = "Value",
       y = "Density") +
  facet_wrap(~ Time, scales = "fixed") +  # Ensure all plots have the same scale
  coord_cartesian(xlim = c(0, 1), ylim = c(0, 3)) +       # Set the x-axis limits from 0 to 1
  scale_x_continuous(breaks = seq(0, 1, by = 0.2)) +  # Set x-axis ticks at 0.2 intervals
  theme_minimal() +
  theme(legend.position = "none")  # Hide legend for clarity
