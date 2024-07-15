control <- function() {
  
  setwd('~/RS/20240602')
  
  source('library_20240602.R')
  source('DGP_20240602.R')
  source('filtering_20240602.R')
  
  library_load()
  
  N <- 100
  Nt <- 25
  O1 <- 6
  O2 <- 3
  L1 <- 2
  
  nInit <- 30
  maxIter <- 300
  nSim <- 201
  seed <- 1
  
  # Define a function to process each combination of ind and seed
  process_combination <- function(seed) {
    print(seed)
    df <- DGP(seed, N, Nt, O1, O2, L1)
    saveRDS(df, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/20240602/output/df__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1,'.RDS', sep=''))
    filter <- filtering(seed, N, Nt, O1, O2, L1, df$y1, df$y2, df$S, df$eta1, nInit, maxIter)
    saveRDS(filter, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/20240602/output/filter__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))
    rm(list = ls())
    gc()
  }
  
  while (seed < nSim) {
    if (!file.exists(paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Simulation/20240602/output/filter__sim_', seed, '_N_', N, '_T_', Nt, '_O1_', O1, '_O2_', O2, '_L1_', L1, '.RDS', sep=''))) {
      process_combination(seed) 
      seed <- seed + 1
      source('control_20240602_2.R')
      control()
    } else {seed <- seed + 1}
  }
  print('finished!')
}

