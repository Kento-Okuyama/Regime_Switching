library(ggplot2)
library(dplyr)
library(tidyr)

# Data Preparation
df_pred <- as.data.frame(filtered_best$mPr_pred_best)[c(20, 24, 38),]  # Predicted data
df_pred <- df_pred %>%
  mutate(ID = c(20, 24, 38)) %>%  # Specify ID
  pivot_longer(cols = -ID, names_to = "Time", values_to = "Predicted_Value") %>%
  mutate(Time = as.numeric(gsub("V", "", Time)))

df_true <- as.data.frame(data_true)[c(20, 24, 38),]  # True data
df_true_long <- df_true %>%
  mutate(ID = c(20, 24, 38)) %>%  # Specify ID
  pivot_longer(cols = -ID, names_to = "Time", values_to = "True_Value") %>%
  mutate(Time = as.numeric(gsub("V", "", Time)))

# Combine data
df_combined <- df_pred %>%
  inner_join(df_true_long, by = c("ID", "Time"))

# Plot creation
ggplot(df_combined, aes(x = Time)) +
  # Line for true and predicted values with increased thickness
  geom_line(aes(y = Predicted_Value, color = "Predicted"), size = 1.5, alpha = 0.8) +
  geom_line(aes(y = True_Value, color = "True"), linetype = "dashed", size = 1.5) +
  # Dotted line at y = 0.5
  geom_hline(yintercept = 0.5, linetype = "dotted", color = "black") +
  # Separate by ID in facets
  facet_wrap(~ ID, ncol = 3, labeller = labeller(ID = function(x) paste0("#", x))) +
  # Labels and theme settings
  labs(title = "Comparison of Predicted and True Values for Selected Individuals",
       x = "Time point",
       y = "Intention to Dropout",
       color = "Legend") +
  scale_color_manual(values = c("Predicted" = "black", "True" = "red")) +
  guides(linetype = "none") +  # Hide linetype in legend
  theme_minimal(base_size = 15) +
  coord_cartesian(ylim = c(0, 1)) +
  scale_y_continuous(breaks = c(0, 0.5, 1), limits = c(0, 1), expand = c(0, 0)) +
  theme(
    strip.text = element_text(size = 12),  # Set font size for facet titles
    panel.grid = element_blank(),  # Hide grid lines
    legend.position = "bottom",  # Place legend at the bottom
    plot.title = element_text(hjust = 0.5, face = "bold")  # Center-align and bold the title
  )
