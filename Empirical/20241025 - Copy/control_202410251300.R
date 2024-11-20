# Install required packages if not already installed
# install.packages('rstudioapi')

# Clear all objects from the workspace
rm(list=ls())

# Load necessary libraries
library(rstudioapi)

# Define the control function
control <- function(init, m, seed) {
  
  # Set the working directory
  setwd('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/20241025 - Copy')
  
  # Source external R scripts
  source('library_202409021300.R')
  source('preprocessing_202409021300.R')
  source('filtering_202410251300.R')
  
  # Load additional libraries or custom functions
  library_load()
  
  # Initial settings
  maxIter <- 150
  m <- 1
  seed <- 42
  
  # Define a function to process each combination of ind and seed
  process <- function(m, seed) {
    # Preprocess the data
    df <- preprocessing(m, seed)
    
    # Save the preprocessed data
    saveRDS(df, paste0('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/20241025 - Copy/output/df__emp_', 
                       seed, '_N_', df$N, '_T_', df$Nt, '_O1_', df$O1, '_O2_', df$O2, '_L1_', df$L1, '_m_', m,'.RDS'))
    
    # Display the current initialization
    cat('init =', init, '\n')
    
    # Execute the filtering process
    filter <- filtering(seed, df$N, df$Nt, df$O1, df$O2, df$L1, df$y1, df$y2, df$DO, init, maxIter)
    
    # Save the filtering result
    saveRDS(filter, paste0('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/20241025 - Copy/output/filter__emp_', 
                           seed, '_N_', df$N, '_T_', df$Nt, '_O1_', df$O1, '_O2_', df$O2, '_L1_', df$L1, '_m_', m, '_init_', init, '.RDS'))
    
    # Free up memory
    rm(filter) 
    rm(df)
    gc()
  }
  
  # Execute the process function
  process(m, seed)
}

# call the control function
for (init in 16:16) {
  control(init, m, seed)
}

# Code to restart the session (commented out)
# restartSession()
