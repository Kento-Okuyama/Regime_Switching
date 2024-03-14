# 
# data1 <- c(0.0000, 0.8571, 0.9231, 1.0000, 1.0000)
# data2 <- c(0.3684, 0.8824, 0.9375, 1.0000, 1.0000)
# data3 <- c(0.8209, 0.9545, 0.9692, 0.9961, 1.0000)
# data4 <- c(0.8876, 0.9659, 0.9882, 1.0000, 1.0000)
# 
#  
# boxplot(list(data1, data2, data3, data4),
#         horizontal=FALSE, 
#         names=c("75_sensitivity", "100_sensitivity", "75_specificity", "100_specificity") 
# )
# 
# 
# mean1 <- 0.9080
# mean2 <- 0.9133
# mean3 <- 0.9706
# mean4 <- 0.9801
# 
# 
# points(c(1, 2, 3, 4), c(mean1, mean2, mean3, mean4), pch=18, col="red")
# 



# New data for boxplot
data1 <- c(0.6250,  0.8231,  0.9167,  1.0000,  1.0000)  # 75x25_sensitivity
data2 <- c(0.3684,  0.8333,  0.8824,  1.0000,  1.0000)  # 100x25_sensitivity
data3 <- c(0.9375,  0.9841,  0.9844,  1.0000,  1.0000)  # 75x25_specificity
data4 <- c(0.9012,  0.9884,  1.0000,  1.0000,  1.0000)  # 100x25_specificity

# Create boxplot
boxplot(list(data1, data2, data3, data4),
        horizontal=FALSE,
        col = "lightblue",
        names=c("sensitivity_75", "_100", "specificity_75", "_100")
)

# New mean values
mean1 <- 0.8837  # Mean of 75x25_sensitivity
mean2 <- 0.8759  # Mean of 100x25_sensitivity
mean3 <- 0.9902  # Mean of 75x25_specificity
mean4 <- 0.9940  # Mean of 100x25_specificity 

# Add points for mean values
points(c(1, 2, 3, 4), c(mean1, mean2, mean3, mean4), pch=18, col="red")
