# New data for boxplot
data1 <- c(0.1467,  0.1467,  0.1733,  0.1867,  0.2400)  # 75x25
data2 <- c(0.1400,  0.1500,  0.1600,  0.1700,  0.2600)  # 100x25

# Create boxplot
boxplot(list(data1, data2),
        horizontal=FALSE,
        col = "lightblue",
        names=c("N=75", "N=100")
)

# New mean values
# mean1 <- 0.1719  # Mean of 75x25
# mean2 <- 0.1666  # Mean of 100x25

# Add points for mean values
# points(c(1, 2), c(mean1, mean2), pch=18, col="red")

library(ggplot2)
library(dplyr)

df_S_75x25 <- data.frame(S_75x25_group)
df_S_75x25$Source <- "N=75"
colnames(df_S_75x25)[1] <- "Values"
df_S_100x25 <- data.frame(S_100x25_group)
df_S_100x25$Source <- "N=100"
colnames(df_S_100x25)[1] <- "Values"
combined_df <- rbind(df_S_75x25[, c("Values", "Source")], df_S_100x25[, c("Values", "Source")])
combined_df$Source <- factor(combined_df$Source, levels = c("N=75", "N=100"))

# Create the violin plot
ggplot(combined_df, aes(x = Source, y = Values, fill = Source)) + 
  geom_violin(trim = FALSE) +
  labs(title = "",
       x = "",
       y = "Value") +
  theme_minimal() + scale_fill_brewer(palette="Blues") + theme_classic() + geom_jitter(shape=16, position=position_jitter(0.2)) + theme(legend.position = "none") 
