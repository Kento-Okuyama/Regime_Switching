# install.packages('rstudioapi')
rm(list=ls())
library(rstudioapi)

control <- function(init) {
  setwd('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408261300')
  
  source('library_202408261300.R')
  source('preprocessing_202408261300.R')
  source('filtering_202408261300.R')
  
  library_load()
  
  maxIter <- 300
  seed <- 42
  
  # Define a function to process each combination of ind and seed
  process <- function() {
    df <- preprocessing()
    saveRDS(df, paste0('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408261300/output/df__emp_', seed, '_N_', df$N, '_T_', df$Nt, '_O1_', df$O1, '_O2_', df$O2, '_L1_', df$L1,'.RDS'))
    cat('init step ', init, '\n')
    filter <- filtering(seed, df$N, df$Nt, df$O1, df$O2, df$L1, df$y1, df$y2, init, maxIter)
    saveRDS(filter, paste0('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408261300/output/filter__emp_', seed, '_N_', df$N, '_T_', df$Nt, '_O1_', df$O1, '_O2_', df$O2, '_L1_', df$L1, '_init_', init, '.RDS'))
    rm(filter) 
    rm(df)
    gc()
  }
  process()
} 

for (init in 1:18) {
  control(init)
}

# restartSession()