setwd('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202309121900')

if (!requireNamespace('doParallel', quietly = TRUE)) {
  install.packages('doParallel') }

library(doParallel)  

cl <- makeCluster(6)
registerDoParallel(cl)
# stopCluster(cl)

source('library_202309121900.R')
source('DGP_202309121900.R')
source('filtering_202309121900.R')
source('smoothing_202309121900.R')

library_load()

N_vec <- c(25, 25, 50, 50, 100, 100)
Nt_vec <- c(25, 50, 25, 50, 25, 50)
O1 <- 6
O2 <- 3
L1 <- 2

nInit <- 30/15
maxIter <- 300/100

nNT <- 6/6
nSim <- 200/200

foreach(ind=1:nNT, .packages=c('reshape', 'ggplot2', 'plotly', 'sigmoid', 'Rlab', 'cowplot', 'lavaan', 'torch', 'reticulate', 'cowplot')) %:%
  foreach(seed=1:nSim) %dopar% {
    
    print(c(ind, seed))
    N <- N_vec[ind]
    Nt <- Nt_vec[ind]
    df <- DGP(seed, N, Nt, O1, O2, L1)
    params <- filtering(seed, N, Nt, O1, O2, L1, df$y1, df$y2, nInit, maxIter)
    output <- smoothing(N, Nt, O1, O2, L1, df$y1, df$y2, params, df$S, df$eta1)
    saveRDS(df, paste('df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    saveRDS(params, paste('params__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))
    saveRDS(output, paste('output__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))
  } 

