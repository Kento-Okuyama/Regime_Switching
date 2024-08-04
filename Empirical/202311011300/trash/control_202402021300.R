# install.packages('rstudioapi')
rm(list=ls())
library(rstudioapi)

control <- function() {
  setwd('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202311011300')
  
  source('library_202311011300.R')
  source('preprocessing_202407161300.R')
  source('filtering_202407161300.R')
  
  library_load()
  
  # nInit <- 1
  init <- 6
  maxIter <- 300
  seed <- 42
     
  # Define a function to process each combination of ind and seed
  process <- function() {
    df <- preprocessing()
    saveRDS(df, paste0('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202311011300/output/df__emp_', seed, '_N_', df$N, '_T_', df$Nt, '_O1_', df$O1, '_O2_', df$O2, '_L1_', df$L1,'.RDS'))
    # for (init in 1:nInit) {
    cat('init step ', init, '\n')
    filter <- filtering(seed, df$N, df$Nt, df$O1, df$O2, df$L1, df$y1, df$y2, init, maxIter)
    saveRDS(filter, paste0('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202311011300/output/filter__emp_', seed, '_N_', df$N, '_T_', df$Nt, '_O1_', df$O1, '_O2_', df$O2, '_L1_', df$L1, '_init_', init, '.RDS'))
    rm(filter) 
    gc() 
    # }
    rm(df)
    gc()
  }
  process()
} 

# df <- readRDS('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202311011300/output/df__emp_42_N_80_T_51_O1_17_O2_3_L1_7.RDS')
control()
restartSession()