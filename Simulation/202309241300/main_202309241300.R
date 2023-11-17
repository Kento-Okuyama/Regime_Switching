setwd('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309241300')

if (!requireNamespace('doParallel', quietly = TRUE)) {
  install.packages('doParallel') }

library(doParallel)  

cl <- makeCluster(7)
registerDoParallel(cl)

source('library_202309241300.R')
source('DGP_202309241300.R')
source('filtering_202309241300.R')

library_load()

N_vec <- c(25, 25, 50, 50, 100, 100)
Nt_vec <- c(25, 50, 25, 50, 25, 50)
O1 <- 6
O2 <- 3
L1 <- 2

nInit <- 30
maxIter <- 300

nNT <- 6/6
nSim <- 200/200*7

foreach(ind=1:nNT, .packages=c('reshape', 'ggplot2', 'plotly', 'sigmoid', 'Rlab', 'Rmpfr', 'cowplot', 'lavaan', 'torch', 'reticulate', 'cowplot', 'data.table', 'dplyr', 'tidyverse')) %:%
  foreach(seed=1:nSim) %dopar% {
    
    print(c(ind, seed))
    N <- N_vec[ind]
    Nt <- Nt_vec[ind]
    df <- DGP(seed, N, Nt, O1, O2, L1)
    #df <- readRDS(paste('C:/Users/Methodenzentrum/Desktop/Kento/202309241300/output/df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    filter <- filtering(seed, N, Nt, O1, O2, L1, df$y1, df$y2, df$S, df$eta1, nInit, maxIter)
    #filter <- readRDS(paste('C:/Users/Methodenzentrum/Desktop/Kento/202309241300/output/params__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    saveRDS(df, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309241300/output/df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    saveRDS(filter, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309241300/output/filter__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))
  }

stopCluster(cl)

(filter <- readRDS('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309241300/output/filter__sim_7_N_25_T_25_O1_6_O2_3_L1_2.RDS'))
