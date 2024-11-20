# Create a time index for 51 time points
time_index <- 1:51

# Define the indices of the individuals to include in the plot
person_index <- c(20, 24, 38)

# Attach the filtered_best object to access its variables directly
attach(filtered_best)

# Initialize an empty data frame to hold the data for plotting
data <- data.frame()

# Loop through each individual and extract their data
for (person in person_index) {
  # Combine data for each individual into a single data frame
  data <- rbind(data, data.frame(
    Time = rep(time_index, times = 4), # Repeat time indices for all 4 variables (Series)
    Value = c(eta1_best[person, , 1], eta1_best[person, , 2], eta1_best[person, , 3], eta1_best[person, , 4]), # Extract values for each variable
    StdDev = sqrt(c(P_best[person, , 1, 1], P_best[person, , 2, 2], P_best[person, , 3, 3], P_best[person, , 4, 4])), # Compute standard deviations for error bands
    Series = factor(rep(1:4, each = 51)), # Label each variable as Series 1 to 4
    Person = factor(rep(person, times = 51 * 4)) # Label each individual by their ID
  ))
}

# Detach the filtered_best object after extracting all necessary data
detach(filtered_best)

# Update the Series labels with descriptive names
series_labels <- c("Cost", "Not Understanding", "Afraid to Fail", "Negative Affect")

# Plot the data using ggplot
ggplot(data, aes(x = Time, y = Value, color = Person, fill = Person)) +
  geom_line(size = 1) + # Plot lines for each individual
  geom_ribbon(aes(ymin = Value - StdDev, ymax = Value + StdDev), alpha = 0.2) + # Add error bands using standard deviation
  facet_wrap(~ Series, scales = "free_y", labeller = labeller(Series = setNames(series_labels, 1:4))) + # Separate plots for each variable with descriptive labels
  labs(
    title = "Time Series with Variance Bands for Multiple Individuals", # Title of the plot
    x = "Time Index", # Label for the x-axis
    y = "Value" # Label for the y-axis
  ) +
  theme_minimal() + # Use a minimal theme for a clean look
  theme(
    legend.title = element_blank(), # Remove the title of the legend
    strip.text = element_text(size = 12) # Adjust the size of facet labels
  )

