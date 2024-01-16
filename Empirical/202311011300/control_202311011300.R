control <- function() {
  
  setwd('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Empirical/202311011300')
  
  source('library_202311011300.R')
  source('preprocessing_202311011300.R')
  source('filtering_202401101300.R')
  
  library_load()
  
  nInit <- 30/30
  maxIter <- 300/150
  seed <- 42
  
  # Define a function to process each combination of ind and seed
  process <- function() {
    df <- preprocessing()
    saveRDS(df, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Empirical/202311011300/output/df__emp_', seed, '_N_', df$N, '_T_', df$Nt, '_O1_', df$O1, '_O2_', df$O2, '_L1_', df$L1,'.RDS', sep=''))
    N <- df$N; Nt <- df$Nt; O1 <- df$O1; O2 <- df$O2; L1 <- df$L1; y1 <- df$y1; y2 <- df$y2
    filter <- filtering(seed, df$N, df$Nt, df$O1, df$O2, df$L1, df$y1, df$y2, nInit, maxIter)
    saveRDS(filter, paste('C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/Regime_Switching/Empirical/202311011300/output/filter__emp_', seed, '_N_', df$N, '_T_', df$Nt, '_O1_', df$O1, '_O2_', df$O2, '_L1_', df$L1, '.RDS', sep=''))
    rm(list = ls())
    gc()
  }

  process()
  restartSession(command="print('finished')")
}

control()