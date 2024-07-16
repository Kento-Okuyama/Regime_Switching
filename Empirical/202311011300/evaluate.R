# Set working directory
setwd('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Empirical/202311011300')

# Read the filtered data
filtered <- readRDS(paste('output/filter__emp_42_N_80_T_51_O1_17_O2_3_L1_7_init_3.RDS', sep=''))

# Call the preprocessing function and load the resulting variables into the global environment
source('library_202311011300.R')
source('preprocessing_202311011300.R')
library_load()
df <- readRDS('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Empirical/202311011300/output/df__emp_42_N_80_T_51_O1_17_O2_3_L1_7.RDS')
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
Lmdd1 <- theta[(6*L1+1)]
Lmdd2 <- theta[(6*L1+2)]
Lmdd3 <- theta[(6*L1+3)]
Lmdd4 <- theta[(6*L1+4)]
Lmdd5 <- theta[(6*L1+5)]
Lmdd6 <- theta[(6*L1+6)]
Lmdd7 <- theta[(7*L1)]
gamma1 <- theta[(7*L1+1)]
gamma2 <- theta[(7*L1+1+1):(7*L1+1+L1)]
Qd <- theta[(7*L1+1+L1+1):(7*L1+1+2*L1)]
Rd <- theta[(7*L1+1+2*L1+1):(7*L1+1+2*L1+O1)]

theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
              Lmdd1=Lmdd1, Lmdd2=Lmdd2, Lmdd3=Lmdd3, Lmdd4=Lmdd4, Lmdd5=Lmdd5, Lmdd6=Lmdd6, Lmdd7=Lmdd7,
              Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2)
