
# New data for boxplot
data3 <- c(0.6250,  0.8231,  0.9167,  1.0000,  1.0000)  # 75x25_sensitivity
data4 <- c(0.3684,  0.8333,  0.8824,  1.0000,  1.0000)  # 100x25_sensitivity
data5 <- c(0.9375,  0.9841,  0.9844,  1.0000,  1.0000)  # 75x25_specificity
data6 <- c(0.9012,  0.9884,  1.0000,  1.0000,  1.0000)  # 100x25_specificity
data1 <- c(0.8933,  0.9600,  0.9733,    0.9867,  1.0000)  # 75x25_accuracy
data2 <- c(0.8000,  0.9600,  0.9800,    0.9900,  1.0000)  # 100x25_accuracy

par(mfrow = c(1,3), mar=c(5.1, 2.6, 4.1, 1.1))
# Create boxplot
boxplot(list(data3, data4),
        horizontal=FALSE,
        col = "lightblue",
        names = c("N=75", "N=100"),
        main = "Sensitivity",
        ylim = c(min(c(min(data3), min(data4))), 1)
)

boxplot(list(data5, data6),
        horizontal=FALSE,
        col = "lightblue",
        names = c("N=75", "N=100"),
        main = "Specificity",
        ylim = c(min(c(min(data3), min(data4))), 1),
        ylab = "",
        yaxt = "n"
)

boxplot(list(data1, data2),
        horizontal=FALSE,
        col = "lightblue",
        names = c("N=75", "N=100"),
        main = "Accuracy",
        ylim = c(min(c(min(data3), min(data4))), 1),
        ylab = "",
        yaxt = "n"
        
)

# New mean values
mean1 <- 0.9694  # Mean of 75x25_accuracy
mean2 <- 0.9713  # Mean of 100x25_accuracy
mean3 <- 0.8837  # Mean of 75x25_sensitivity
mean4 <- 0.8759  # Mean of 100x25_sensitivity
mean5 <- 0.9902  # Mean of 75x25_specificity
mean6 <- 0.9940  # Mean of 100x25_specificity 

# Add points for mean values
# points(c(1, 2, 3, 4, 5, 6), c(mean1, mean2, mean3, mean4, mean5, mean6), pch=18, col="red")

library(gridExtra)

df_filter_75x25 <- data.frame(sensitivity_75x25_group)
df_filter_75x25$Source <- "N=75"
colnames(df_filter_75x25)[1] <- "Values"
df_filter_100x25 <- data.frame(sensitivity_100x25_group)
df_filter_100x25$Source <- "N=100"
colnames(df_filter_100x25)[1] <- "Values"
combined_df <- rbind(df_filter_75x25[, c("Values", "Source")], df_filter_100x25[, c("Values", "Source")])
combined_df$Source <- factor(combined_df$Source, levels = c("N=75", "N=100"))

# Create the violin plot
p1 <- ggplot(combined_df, aes(x = Source, y = Values, fill = Source)) + 
  geom_violin(trim = FALSE) +
  labs(title = "Sensitivity",
       x = "",
       y = "Value") +
  theme_minimal() + scale_fill_brewer(palette="Blues") + theme_classic() + geom_jitter(shape=16, position=position_jitter(0.2)) + scale_y_continuous(limits = c(min(combined_df$Values) - 0.01, max(combined_df$Values) + 0.01)) + theme(legend.position = "none") 

df_filter_75x25 <- data.frame(specificity_75x25_group)
df_filter_75x25$Source <- "N=75"
colnames(df_filter_75x25)[1] <- "Values"
df_filter_100x25 <- data.frame(specificity_100x25_group)
df_filter_100x25$Source <- "N=100"
colnames(df_filter_100x25)[1] <- "Values"
combined_df <- rbind(df_filter_75x25[, c("Values", "Source")], df_filter_100x25[, c("Values", "Source")])
combined_df$Source <- factor(combined_df$Source, levels = c("N=75", "N=100"))

# Create the violin plot
p2 <- ggplot(combined_df, aes(x = Source, y = Values, fill = Source)) + 
  geom_violin(trim = FALSE) +
  labs(title = "Specificity",
       x = "",
       y = "") +
  theme_minimal() + scale_fill_brewer(palette="Blues") + theme_classic() + geom_jitter(shape=16, position=position_jitter(0.2)) + scale_y_continuous(limits = c(min(combined_df$Values) - 0.01, max(combined_df$Values) + 0.01)) + theme(legend.position = "none", # axis.text.y = element_blank(),
                                                                                                                                        # axis.ticks.y = element_blank(),
                                                                                                                                        axis.title.y = element_blank())
df_filter_75x25 <- data.frame(accuracy_75x25_group)
df_filter_75x25$Source <- "N=75"
colnames(df_filter_75x25)[1] <- "Values"
df_filter_100x25 <- data.frame(accuracy_100x25_group)
df_filter_100x25$Source <- "N=100"
colnames(df_filter_100x25)[1] <- "Values"
combined_df <- rbind(df_filter_75x25[, c("Values", "Source")], df_filter_100x25[, c("Values", "Source")])
combined_df$Source <- factor(combined_df$Source, levels = c("N=75", "N=100"))

# Create the violin plot
p3 <- ggplot(combined_df, aes(x = Source, y = Values, fill = Source)) + 
  geom_violin(trim = FALSE) +
  labs(title = "accuracy",
       x = "",
       y = "") +
  theme_minimal() + scale_fill_brewer(palette="Blues") + theme_classic() + geom_jitter(shape=16, position=position_jitter(0.2)) + scale_y_continuous(limits = c(min(combined_df$Values) - 0.01, max(combined_df$Values) + 0.01)) + theme(legend.position = "none", # axis.text.y = element_blank(),
                                                                                                                                        # axis.ticks.y = element_blank(),
                                                                                                                                        axis.title.y = element_blank()) 

library(patchwork)
p1 + p2 + p3 + plot_layout(ncol = 3)
grid.arrange(p1, p2, p3, ncol = 3)

