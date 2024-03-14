
# New data for boxplot
data3 <- c(0.6250,  0.8231,  0.9167,  1.0000,  1.0000)  # 75x25_sensitivity
data4 <- c(0.3684,  0.8333,  0.8824,  1.0000,  1.0000)  # 100x25_sensitivity
data5 <- c(0.9375,  0.9841,  0.9844,  1.0000,  1.0000)  # 75x25_specificity
data6 <- c(0.9012,  0.9884,  1.0000,  1.0000,  1.0000)  # 100x25_specificity
data1 <- c(0.8933,  0.9600,  0.9733,    0.9867,  1.0000)  # 75x25_accuracy
data2 <- c(0.8000,  0.9600,  0.9800,    0.9900,  1.0000)  # 100x25_accuracy

# Create boxplot
boxplot(list(data3, data4, data5, data6, data1, data2),
        horizontal=FALSE,
        col = "lightblue",
        names=c("sensitivity_75", "_100", "specificity_75", "_100", "accuracy_75", "_100")
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
