setwd('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300')

if (!requireNamespace('doParallel', quietly = TRUE)) {
  install.packages('doParallel') }

library(doParallel)  

cl <- makeCluster(7)
registerDoParallel(cl)
# stopCluster(cl)

source('library_202309221300.R')
source('DGP_202309221300.R')
source('filtering_202309221300.R')
source('result_202309221300.R')

library_load()

N_vec <- c(100) #c(25, 25, 50, 50, 100, 100)
Nt_vec <- c(50) #c(25, 50, 25, 50, 25, 50)
O1 <- 6
O2 <- 3
L1 <- 2

nInit <- 30/30
maxIter <- 300/100

nNT <- 6/6
nSim <- 200

foreach(ind=1:nNT, .packages=c('reshape', 'ggplot2', 'plotly', 'sigmoid', 'Rlab', 'cowplot', 'lavaan', 'torch', 'reticulate', 'cowplot')) %:%
  foreach(seed=1:nSim) %dopar% {
    
    print(c(ind, seed))
    N <- N_vec[ind]
    Nt <- Nt_vec[ind]
    df <- DGP(seed, N, Nt, O1, O2, L1)
    #df <- readRDS(paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300/output/df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    filter <- filtering(seed, N, Nt, O1, O2, L1, df$y1, df$y2, nInit, maxIter)
    #filter <- readRDS(paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300/output/params__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    output <- result(N, Nt, O1, O2, L1, df$y1, df$y2, filter$params, df$S, df$eta1)
    saveRDS(df, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300/output/df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    saveRDS(params, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300/output/params__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))
    saveRDS(output, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300/output/output__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))
  } 
 
# (df <- readRDS('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300/output/df__sim_16_N_25_T_25_O1_6_O2_3_L1_2.RDS'))
# (filter <- readRDS('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300/output/params__sim_13_N_25_T_25_O1_6_O2_3_L1_2.RDS'))
# (output <- readRDS('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309221300/output/output__sim_13_N_25_T_25_O1_6_O2_3_L1_2.RDS'))
