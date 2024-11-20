# Set the working directory
setwd('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202409021300')

# Source external R scripts
source('library_202409021300.R')
source('preprocessing_202409021300.R')
source('filtering_202409021300.R')

# Load additional libraries or custom functions
library_load()

# Initial settings
maxIter <- 150
m <- 1 
seed <- 42
init <-  18

# Preprocess the data
df <- preprocessing(m, seed)

# Display the current initialization
cat('init =', init, '\n')

N <- df$N 
Nt <- df$Nt
O1 <- df$O1
O2 <- df$O2
L1 <- df$L1
y1 <- df$y1
y2 <- df$y2
DO <- df$DO

