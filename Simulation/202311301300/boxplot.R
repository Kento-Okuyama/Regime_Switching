# # sensitivity
# # specificity
# 
# # Example data (replace with your actual summary statistics)
# summary_data <- data.frame(
#   variable = c("(75,25)_sensitivity", "(100,25)_sensitivity", "(75,25)_specificity", "(100,25)_specificity"),
#   min = c(0.0000, 0.3684, 0.8209, 0.8876),
#   q1 = c(0.8571, 0.8824, 0.9545, 0.9659),
#   median = c(0.9231, 0.9375, 0.9692, 0.9882),
#   mean = c(0.9080, 0.9133, 0.9706, 0.9801),
#   q3 = c(1.0000, 1.0000, 0.9961, 1.0000),
#   max = c(1.0000, 1.0000, 1.0000, 1.0000)
# )
# 
# # Function to plot a boxplot from summary data
# plot_custom_boxplot <- function(data, var_name) {
#   var_data <- data[data$variable == var_name,]
#   boxplot.stats <- list(
#     stats = matrix(c(var_data$min, var_data$q1, var_data$median, var_data$q3, var_data$max), ncol = 1)
#   )
#   boxplot(boxplot.stats, main = var_name, ylab = "Value")
#   points(1, var_data$mean, col = "red", pch = 18)  # Add mean value as a point
# }
# 
# # Plotting for each variable
# par(mfrow = c(1,1))
# plot_custom_boxplot(summary_data, "(75,25)_sensitivity")
# plot_custom_boxplot(summary_data, "(100,25)_sensitivity")
# plot_custom_boxplot(summary_data, "(75,25)_specificity")
# plot_custom_boxplot(summary_data, "(100,25)_specificity")


# 各箱ひげ図のデータをリストとして準備
data1 <- c(0.0000, 0.8571, 0.9231, 1.0000, 1.0000)
data2 <- c(0.3684, 0.8824, 0.9375, 1.0000, 1.0000)
data3 <- c(0.8209, 0.9545, 0.9692, 0.9961, 1.0000)
data4 <- c(0.8876, 0.9659, 0.9882, 1.0000, 1.0000)

# 箱ひげ図を描画
boxplot(list(data1, data2, data3, data4),
        horizontal=FALSE, # 箱ひげ図を横に表示
        names=c("75_sensitivity", "100_sensitivity", "75_specificity", "100_specificity") # 各箱ひげ図の名前
)

# 各データセットの平均値
mean1 <- 0.9080
mean2 <- 0.9133
mean3 <- 0.9706
mean4 <- 0.9801

# 平均値を赤色の点として追加
points(c(1, 2, 3, 4), c(mean1, mean2, mean3, mean4), pch=18, col="red")

