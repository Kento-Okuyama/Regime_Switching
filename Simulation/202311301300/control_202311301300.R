control <- function() {
  
  setwd('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311301300')
  
  source('library_202311301300.R')
  source('DGP_202311301300.R')
  source('filtering_202311301300.R')
  
  library_load()
  
  N_vec <- c(100)
  Nt_vec <- c(25)
  O1 <- 6
  O2 <- 3
  L1 <- 2
  
  nInit <- 30
  maxIter <- 300
  
  nNT <- 6
  nSim <- 200
  
  ind <- seed <- 1
  seed <- 100
  
  # Define a function to process each combination of ind and seed
  process_combination <- function(ind, seed) {
    print(c(ind, seed))
    df <- DGP(seed, N, Nt, O1, O2, L1)
    saveRDS(df, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311301300/output/df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    filter <- filtering(seed, N, Nt, O1, O2, L1, df$y1, df$y2, df$S, df$eta1, nInit, maxIter)
    saveRDS(filter, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311301300/output/filter__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))
    rm(list = ls())
    gc()
  }
  
  N <- N_vec[ind]
  Nt <- Nt_vec[ind]
  
  while (file.exists(paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311301300/output/df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))) {
    if (seed == nSim) {seed <- 1; if (ind == nNT) {restartSession(command="print('finished!')")}; ind <- ind + 1
    } else {seed <- seed + 1}
  }
  
  if (seed == nSim && ind == nNT) {restartSession(command="print('finished')")}
  process_combination(ind, seed)
  
  restartSession(command="source('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/202311301300/control_202311301300.R'); control()")
}