# Clear the environment
rm(list = ls())
seed <- 42

# Set working directory
setwd('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408261300')

# Function to read and filter data with error handling
read_filtered_data <- function(init, seed, file_path) {
  set.seed(seed + init)
  file <- paste0(file_path, 'output/filter__emp_42_N_80_T_51_O1_9_O2_3_L1_4_init_', init, '.RDS')
  
  if (file.exists(file)) {
    return(readRDS(file))
  } else {
    stop("File not found: ", file)
  }
}

# Function to load necessary files and preprocess
load_and_preprocess <- function() {
  source('library_202408261300.R')
  source('preprocessing_202408261300.R')
  library_load()
  
  df <- readRDS('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408261300/output/df__emp_42_N_80_T_51_O1_9_O2_3_L1_4.RDS')
  data_true <- readRDS('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408261300/output/df__emp_42_N_80_T_51_O1_9_O2_3_L1_4.RDS')$DO
  list2env(as.list(df), envir = .GlobalEnv)
  
  return(list(df = df, data_true = data_true))
}

# Initialize and read filtered data
init <- 19
filtered <- read_filtered_data(init, seed, 'C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202408261300/')

# Load preprocessing and data
data_loaded <- load_and_preprocess()
df <- data_loaded$df
data_true <- data_loaded$data_true

# Extract best theta parameters
theta <- filtered$theta_best

# Extract specific parameters from the theta vector
B11 <- theta[1:L1]
B12 <- theta[(L1+1):(2*L1)]
B21d <- theta[(2*L1+1):(3*L1)]
B22d <- theta[(3*L1+1):(4*L1)]
B31 <- theta[(4*L1+1):(5*L1)]
B32 <- theta[(5*L1+1):(6*L1)]
Lmdd1 <- theta[6*L1+1]
Lmdd2 <- theta[6*L1+2]
Lmdd3 <- theta[6*L1+3]
Lmdd4 <- theta[6*L1+4]
Lmdd5 <- theta[6*L1+5]
gamma1 <- theta[6*(L1+1)]
gamma2 <- theta[(6*(L1+1)+1):(6*(L1+1)+L1)]
Qd <- theta[(6*(L1+1)+L1+1):(6*(L1+1)+2*L1)]
Rd <- theta[(6*(L1+1)+2*L1+1):(6*(L1+1)+2*L1+O1)]
mP_DO <- theta[6*(L1+1)+2*L1+O1+1]
tP_SB <- theta[6*(L1+1)+2*L1+O1+2]

theta <- list(B11=B11, B12=B12, B21d=B21d, B22d=B22d, B31=B31, B32=B32,
              Lmdd1=Lmdd1, Lmdd2=Lmdd2, Lmdd3=Lmdd3, Lmdd4=Lmdd4, Lmdd5=Lmdd5,
              Qd=Qd, Rd=Rd, gamma1=gamma1, gamma2=gamma2, mP_DO=mP_DO, tP_SB=tP_SB)

############################################

# Visualization part: Improved data visualization
library(ggplot2)
library(dplyr)
library(tidyr)

# Prepare data for all individuals
df_pred <- as.data.frame(filtered$mPr_best)  # Predicted data
df_pred <- df_pred %>%
  mutate(ID = row_number()) %>%
  pivot_longer(cols = -ID, names_to = "Time", values_to = "Predicted_Value") %>%
  mutate(Time = as.numeric(gsub("V", "", Time)))

df_true <- as.data.frame(data_true)  # True data
df_true_long <- df_true %>%
  mutate(ID = row_number()) %>%
  pivot_longer(cols = -ID, names_to = "Time", values_to = "True_Value") %>%
  mutate(Time = as.numeric(gsub("V", "", Time)))

# Combine predicted and true data
df_combined <- df_pred %>%
  inner_join(df_true_long, by = c("ID", "Time"))

# Plot comparison for all individuals
ggplot(df_combined, aes(x = Time)) +
  geom_line(aes(y = Predicted_Value, color = "Predicted"), size = 1, alpha = 0.7) +
  geom_line(aes(y = True_Value, color = "True"), size = 1, alpha = 0.7) +
  facet_wrap(~ ID, ncol = 5) +  # Display data for all individuals in 5 columns
  labs(title = "Comparison of Predicted and True Values for All Individuals",
       x = "Time",
       y = "Value",
       color = "Legend") +
  scale_color_manual(values = c("Predicted" = "blue", "True" = "red")) +
  theme_minimal(base_size = 15) +  # Set base theme to minimal with adjusted font size
  coord_cartesian(ylim = c(0, 1)) +  # Limit the Y-axis range to 0-1
  theme(
    strip.text = element_text(size = 12),  # Font size for facet titles
    panel.grid = element_blank(),  # Hide grid lines
    legend.position = "bottom",  # Place legend at the bottom
    plot.title = element_text(hjust = 0.5, face = "bold"),  # Center-align and bold the title
    axis.text.y = element_blank(),  # Hide Y-axis text
    axis.ticks.y = element_blank()  # Hide Y-axis ticks
  )
