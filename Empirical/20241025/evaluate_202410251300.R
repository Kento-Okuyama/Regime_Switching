# Clear the environment
rm(list = ls())

# Set seed value
seed <- 42
m <- 1

# Set working directory
setwd('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/20241025')

# Function to read and filter data with error handling
read_filtered_data <- function(init, seed, file_path) {
  # Set seed for reproducibility
  set.seed(seed + init)
  
  # Construct the file path dynamically
  file <- paste0(file_path, 'output/filter__emp_', seed, '_N_80_T_51_O1_9_O2_3_L1_4_m_', m, '_init_', init, '.RDS')
  
  # Check if the file exists and read it, otherwise throw an error
  if (file.exists(file)) {
    return(readRDS(file))
  } else {
    stop("File not found: ", file)
  }
}

# Function to load necessary files and preprocess data
load_and_preprocess <- function() {
  # Load required R scripts
  source('library_202409021300.R')
  source('preprocessing_202409021300.R')
  
  # Call a custom function from the loaded scripts (assumed to load required libraries)
  library_load()
  
  # Load the main data
  df <- readRDS('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/20241025/output/df__emp_42_N_80_T_51_O1_9_O2_3_L1_4_m_1.RDS')
  
  # Extract the true data from the loaded data
  data_true <- df$DO
  
  # Assign variables from the dataframe to the global environment
  list2env(as.list(df), envir = .GlobalEnv)
  
  # Return the data as a list for further use
  return(list(df = df, data_true = data_true))
}

# Load preprocessing and data
data_loaded <- load_and_preprocess()
df <- data_loaded$df
data_true <- data_loaded$data_true

# Initialize the best log-likelihood value as negative infinity
LL_best <- -Inf

# Iterate through 30 initializations to find the best log-likelihood
for (init in 1:30) {
  # Read filtered data for each initialization
  filtered <- read_filtered_data(init, seed, 'C:/Users/Methodenzentrum/Desktop/Kento/Empirical/20241025/')
  
  # If the sumLik_best field exists, check if it's better than the current best
  if (!is.null(filtered$sumLik_best)) {
    LL_new <- filtered$sumLik_best
    cat('init =', init, ', LL =', LL_new, '\n')
    
    # Update the best log-likelihood and initialization index if a better one is found
    if (LL_best < LL_new) {
      filtered_best <-  filtered
      LL_best <- LL_new
      init_best <- init
    }
  }
}

# Print the best initialization and corresponding log-likelihood
cat('init_best =', init_best, '\n')
cat('LL_best =', LL_best)

############################################

# Visualization part: Improved data visualization
library(ggplot2)
library(dplyr)
library(tidyr)

# Prepare data for all individuals
# df_pred <- as.data.frame(filtered_best$mPr_filtered_best)  # Predicted data
df_pred <- as.data.frame(filtered_best$mPr_pred_best)  # Predicted data
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
  geom_line(aes(y = True_Value, color = "True", linetype = "True"), size = 1, alpha = 0.7) +
  facet_wrap(~ ID, ncol = 5) +  # Display data for all individuals in 5 columns
  labs(title = "Comparison of Predicted and True Values for All Individuals",
       x = "Time",
       y = "Value",
       color = "Legend") +
  scale_color_manual(values = c("Predicted" = "blue", "True" = "red")) +
  scale_linetype_manual(values = c("True" = "dashed")) +
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

# Convert mPr_pred_best in filtered_best to a matrix for easier calculations
data_predP <- as.matrix(filtered_best$mPr_pred_best)

# Create a binary matrix based on the condition that elements in mPr_pred_best are greater than 0.5
# Any element greater than 0.5 is set to 1, otherwise set to 0
data_pred <- as.matrix(filtered_best$mPr_pred_best > 0.5) * 1

# Calculate the difference in the means of the predicted and true state at time 51
mean_diff <- mean(data_pred[,51]) - mean(data_true[,51])

# Compute switch time by calculating the time to the first 1 in each row of data_pred
switch_time <- 51 - rowSums(data_pred) + 1

# Calculate the average switch time, only considering cases where the switch time is within 52
avg_switch_time <- mean(switch_time[switch_time < 52])

# Extract the true dropout indicator at time 51
true_DO <- data_true[,51]

# Calculate switch times specifically for dropout cases (where true_DO is 1)
DO_switch_time <- switch_time[true_DO == 1]

# Calculate the average switch time for dropout cases within 52
avg_DO_switch_time <- mean(DO_switch_time[DO_switch_time < 52])

# Compute dropout time from data_true matrix
DO_time <- 51 - rowSums(data_true) + 1

# Calculate the average dropout time for cases where it is within 52
avg_DO_time <- mean(DO_time[DO_time < 52])

filtered_best$mPr_pred_best
filtered_best$theta_best
filtered_best$eta1_best

