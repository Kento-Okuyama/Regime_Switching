# Adjust margins: Bottom, Left, Top, Right
par(mar=c(5, 4, 4, 2) + 0.1)  # Default is c(5, 4, 4, 2) + 0.1


######
# 75 #
######
# # Prepare data for boxplot from summary statistics
# data_list <- list(
#   d1 = c(-0.005404, 0.088620, 0.130745, 0.178547, 0.782399),
#   d2 = c(-0.001439, 0.123471, 0.194896, 0.273371, 0.628895),
#   d3 = c(-0.77264, -0.22303, -0.14432, -0.08936, 0.10002),
#   d4 = c(-0.63017, -0.25532, -0.16502, -0.08518, -0.06142),
#   d5 = c(0.4511, 0.7611, 0.8350, 0.8915, 1.0193),
#   d6 = c(0.5352, 0.7764, 0.8587, 0.9006, 1.0612),
#   d7 = c(-0.01166, 0.15190, 0.24603, 0.33260, 0.48309),
#   d8 = c(-0.01113, 0.10519, 0.20684, 0.30655, 0.47765),
#   d9 = c(-0.43691, -0.01182, 0.04465, 0.13606, 0.75606),
#   d10 = c(-0.47400, -0.02473, 0.04960, 0.11153, 0.38749),
#   d11 = c(-0.62467, -0.14621, -0.06821, -0.01513, 0.37832),
#   d12 = c(-0.57264, -0.12753, -0.05786, -0.02062, 0.22710),
#   d13 = c(0.3374, 0.6014, 0.8548, 1.0819, 1.3765),
#   d14 = c(0.5086, 0.7652, 0.9095, 1.1424, 1.3784),
#   d15 = c(0.3322, 0.5933, 0.7883, 1.1356, 1.3402),
#   d16 = c(0.6283, 0.9323, 1.1235, 1.3007, 1.6244),
#   d17 = c(0.001000, 0.003945, 0.172705, 0.174482, 0.177272),
#   d18 = c(0.001000, 0.003696, 0.172365, 0.174088, 0.177934),
#   d19 = c(0.04092, 0.13954, 0.37419, 0.37567, 0.37870),
#   d20 = c(0.02091, 0.10841, 0.37334, 0.37459, 0.37686),
#   d21 = c(0.01627, 0.12495, 0.37282, 0.37431, 0.37806),
#   d22 = c(0.03871, 0.12670, 0.37372, 0.37525, 0.37895),
#   d23 = c(0.01726, 0.10861, 0.37310, 0.37484, 0.37777),
#   d24 = c(0.04888, 0.13006, 0.37234, 0.37432, 0.37775),
#   d25 = c(4.097, 4.122, 4.124, 4.350, 4.637),
#   d26 = c(-0.08532, 0.52421, 0.88572, 1.29703, 2.19050),
#   d27 = c(0.1315, 0.5084, 0.8170, 1.2589, 2.8608)
# )
# 
# # ture values
# true_values <- c(.2, .3, -.1, -.2, .8, .8, .4, .4, .1, .1, -.1, -.1, .4, .8, .5, 1.2, .2, .2, .3, .3, .3, .3, .3, .3, 3.5, 1, 1)
# 
# # Plot the boxplot
# boxplot(data_list, 
#         names = c("B11_1", "B11_2", "B12_1", "B12_2", "B21d_1", "B21d_2", "B22d_1", "B22d_2", "B31_1", "B31_2", "B32_1", "B32_2", "Lmdd_3", "Lmdd_5", "Lmdd_10", "Lmdd_12", "Qd_1", "Qd_2", "Rd_1", "Rd_2", "Rd_3", "Rd_4", "Rd_5", "Rd_6", "gamma1", "gamma2_1", "gamma2_2"),
#         # main="Custom Boxplot with Rotated Labels and Mean Points",
#         xaxt='n'
# )
# points(1:27, true_values, col="red", pch=18)  # Add mean values
# 
# # Rotate and add x-axis labels
# axis(1, at=1:27, labels=FALSE)
# text(1:27, par("usr")[3] - 0.1, srt=90, adj=1, labels=c("B11_1", "B11_2", "B12_1", "B12_2", "B21d_1", "B21d_2", "B22d_1", "B22d_2", "B31_1", "B31_2", "B32_1", "B32_2", "Lmdd_3", "Lmdd_5", "Lmdd_10", "Lmdd_12", "Qd_1", "Qd_2", "Rd_1", "Rd_2", "Rd_3", "Rd_4", "Rd_5", "Rd_6", "gamma1", "gamma2_1", "gamma2_2"), xpd=TRUE)
 

# Prepare data for boxplot from summary statistics
data_list <- list(
  B11_1 = c(Min = -0.005404, `1st Qu.` = 0.081249, Median = 0.129955, Mean = 0.152380, `3rd Qu.` = 0.178977, Max = 0.496356),
  B11_2 = c(Min = -0.001439, `1st Qu.` = 0.111375, Median = 0.176406, Mean = 0.204756, `3rd Qu.` = 0.277566, Max = 0.628895),
  B12_1 = c(Min = -0.53371, `1st Qu.` = -0.22518, Median = -0.13869, Mean = -0.16693, `3rd Qu.` = -0.09048, Max = 0.06342),
  B12_2 = c(Min = -0.53115, `1st Qu.` = -0.25584, Median = -0.17840, Mean = -0.18115, `3rd Qu.` = -0.08317, Max = 0.05582),
  B21d_1 = c(Min = 0.4511, `1st Qu.` = 0.7612, Median = 0.8243, Mean = 0.8181, `3rd Qu.` = 0.8883, Max = 1.0103),
  B21d_2 = c(Min = 0.5352, `1st Qu.` = 0.7679, Median = 0.8668, Mean = 0.8388, `3rd Qu.` = 0.9109, Max = 1.0612),
  B22d_1 = c(Min = -0.01166, `1st Qu.` = 0.15302, Median = 0.22621, Mean = 0.23627, `3rd Qu.` = 0.33108, Max = 0.48309),
  B22d_2 = c(Min = -0.01113, `1st Qu.` = 0.13462, Median = 0.23602, Mean = 0.22365, `3rd Qu.` = 0.29995, Max = 0.47765),
  B31_1 = c(Min = -0.347024, `1st Qu.` = -0.007436, Median = 0.050005, Mean = 0.086535, `3rd Qu.` = 0.153132, Max = 0.756058),
  B31_2 = c(Min = -0.33124, `1st Qu.` = 0.00218, Median = 0.06176, Mean = 0.06069, `3rd Qu.` = 0.12392, Max = 0.38749),
  B32_1 = c(Min = -0.62467, `1st Qu.` = -0.12406, Median = -0.06386, Mean = -0.06899, `3rd Qu.` = -0.01433, Max = 0.37832),
  B32_2 = c(Min = -0.57264, `1st Qu.` = -0.13436, Median = -0.07592, Mean = -0.09483, `3rd Qu.` = -0.03222, Max = 0.22710),
  Lmdd_3 = c(Min = 0.3374, `1st Qu.` = 0.6161, Median = 0.8874, Mean = 0.8695, `3rd Qu.` = 1.0833, Max = 1.3765),
  Lmdd_5 = c(Min = 0.5086, `1st Qu.` = 0.7685, Median = 0.9561, Mean = 0.9502, `3rd Qu.` = 1.1513, Max = 1.3701),
  Lmdd_10 = c(Min = 0.3559, `1st Qu.` = 0.5964, Median = 0.8009, Mean = 0.8579, `3rd Qu.` = 1.1773, Max = 1.3402),
  Lmdd_12 = c(Min = 0.6283, `1st Qu.` = 0.9183, Median = 1.1226, Mean = 1.1000, `3rd Qu.` = 1.2923, Max = 1.6244),
  Qd_1 = c(Min = 0.001000, `1st Qu.` = 0.004685, Median = 0.172533, Mean = 0.114878, `3rd Qu.` = 0.174634, Max = 0.176980),
  Qd_2 = c(Min = 0.001000, `1st Qu.` = 0.004092, Median = 0.172400, Mean = 0.114631, `3rd Qu.` = 0.174352, Max = 0.177934),
  Rd_1 = c(Min = 0.04706, `1st Qu.` = 0.13104, Median = 0.37410, Mean = 0.28308, `3rd Qu.` = 0.37563, Max = 0.37870),
  Rd_2 = c(Min = 0.02224, `1st Qu.` = 0.10105, Median = 0.37318, Mean = 0.26956, `3rd Qu.` = 0.37458, Max = 0.37686),
  Rd_3 = c(Min = 0.01627, `1st Qu.` = 0.11922, Median = 0.37250, Mean = 0.27913, `3rd Qu.` = 0.37430, Max = 0.37806),
  Rd_4 = c(Min = 0.04403, `1st Qu.` = 0.15281, Median = 0.37363, Mean = 0.28354, `3rd Qu.` = 0.37545, Max = 0.37895),
  Rd_5 = c(Min = 0.01943, `1st Qu.` = 0.10601, Median = 0.37305, Mean = 0.27256, `3rd Qu.` = 0.37489, Max = 0.37777),
  Rd_6 = c(Min = 0.05158, `1st Qu.` = 0.13486, Median = 0.37229, Mean = 0.28216, `3rd Qu.` = 0.37463, Max = 0.37775),
  gamma1 = c(Min = 4.097, `1st Qu.` = 4.122, Median = 4.124, Mean = 4.222, `3rd Qu.` = 4.341, Max = 4.618),
  gamma2_1 = c(Min = -0.08532, `1st Qu.` = 0.52889, Median = 0.88986, Mean = 0.90578, `3rd Qu.` = 1.24353, Max = 2.12982),
  gamma2_2 = c(Min = 0.1315, `1st Qu.` = 0.4904, Median = 0.8262, Mean = 0.9665, `3rd Qu.` = 1.2945, Max = 2.8608)
)


# True values for each parameter
true_values <- c(.2, .3, -.1, -.2, .8, .8, .4, .4, .1, .1, -.1, -.1, .4, .8, .5, 1.2, .2, .2, .3, .3, .3, .3, .3, .3, 3.5, 1, 1)

# # Plot the boxplot
# boxplot(data_list, 
#         names = names(data_list),
#         # main="Parameter Estimates with True Values",
#         # xlab="Parameters",
#         # ylab="Estimates",
#         las=2, # Rotate axis labels
#         cex.axis=0.7, # Adjust size of axis labels
#         col="lightblue"
# )
# 
# 
# # Add points for true values
# points(1:27, true_values, col="red", pch=18)  # Add mean values


# Define a vector of colors corresponding to each parameter category
colors <- c(rep("lightblue", length(grep("^B", names(data_list)))),  # Structural model parameters
            rep("lightgreen", length(grep("^Lmdd", names(data_list)))), # Measurement model parameters
            rep("grey", length(grep("^[RQ]d", names(data_list)))), # Residual variances
            rep("orange", length(grep("^gamma", names(data_list)))) # Parameters w.r.t. Markov switching models
)

# Plot the updated boxplot with specified colors
boxplot(data_list, 
        names = names(data_list),
        las = 2, # Rotate axis labels
        cex.axis = 0.7, # Adjust size of axis labels
        col = colors # Use the colors vector here
)

# Add points for true values, assuming 'true_values' vector is defined
points(1:length(data_list), true_values, col = "red", pch = 18) # Add true values


#######
# 100 #
#######


# # Prepare data for boxplot from summary statistics
# data_list <- list(
#   d1 <- c(-0.01528, 0.08216, 0.13185, 0.18528, 0.77047),
#   d2 <- c(-0.01651, 0.12862, 0.19659, 0.28144, 0.62777),
#   d3 <- c(-0.6261, -0.2302, -0.1460, -0.0907, 0.0535),
#   d4 <- c(-0.53039, -0.26381, -0.17652, -0.09429, 0.12856),
#   d5 <- c(0.4929, 0.7523, 0.8323, 0.8868, 1.0005),
#   d6 <- c(0.5361, 0.7925, 0.8576, 0.8993, 1.0932),
#   d7 <- c(-0.01191, 0.14234, 0.25695, 0.35225, 0.48244),
#   d8 <- c(-0.02454, 0.10872, 0.22395, 0.33784, 0.47801),
#   d9 <- c(-0.488528, 0.002254, 0.046325, 0.129591, 0.485016),
#   d10 <- c(-0.38338, -0.01410, 0.05034, 0.13530, 0.41882),
#   d11 <- c(-0.49202, -0.13249, -0.07914, -0.01719, 0.25562),
#   d12 <- c(-0.51070, -0.13905, -0.06332, -0.01847, 0.26331),
#   d13 <- c(0.1945, 0.5704, 0.8371, 1.0814, 1.3765),
#   d14 <- c(0.4931, 0.7564, 0.9162, 1.1404, 1.3704),
#   d15 <- c(0.3869, 0.5831, 0.8007, 1.1183, 1.3755),
#   d16 <- c(0.6467, 0.9379, 1.1234, 1.2814, 1.6282),
#   d17 <- c(0.001000, 0.004055, 0.172877, 0.174630, 0.177859),
#   d18 <- c(0.001000, 0.003547, 0.172353, 0.174377, 0.177937),
#   d19 <- c(0.03815, 0.13121, 0.37461, 0.37606, 0.37842),
#   d20 <- c(0.01084, 0.11671, 0.37353, 0.37489, 0.37703),
#   d21 <- c(0.02748, 0.11579, 0.37305, 0.37455, 0.37739),
#   d22 <- c(0.04015, 0.15458, 0.37395, 0.37542, 0.39179),
#   d23 <- c(0.01649, 0.11801, 0.37362, 0.37489, 0.37764),
#   d24 <- c(0.04643, 0.13999, 0.37296, 0.37424, 0.37711),
#   d25 <- c(3.660, 4.121, 4.124, 4.361, 4.630),
#   d26 <- c(-0.04767, 0.49255, 0.85947, 1.29935, 2.85594),
#   d27 <- c(0.1312, 0.5100, 0.8118, 1.2781, 2.8608)
# )
# 
# # true values
# true_values <- c(.2, .3, -.1, -.2, .8, .8, .4, .4, .1, .1, -.1, -.1, .4, .8, .5, 1.2, .2, .2, .3, .3, .3, .3, .3, .3, 3.5, 1, 1)
# 
# 
# # Plot the boxplot
# boxplot(data_list, 
#         names = c("B11_1", "B11_2", "B12_1", "B12_2", "B21d_1", "B21d_2", "B22d_1", "B22d_2", "B31_1", "B31_2", "B32_1", "B32_2", "Lmdd_3", "Lmdd_5", "Lmdd_10", "Lmdd_12", "Qd_1", "Qd_2", "Rd_1", "Rd_2", "Rd_3", "Rd_4", "Rd_5", "Rd_6", "gamma1", "gamma2_1", "gamma2_2"),
#         # main="Custom Boxplot with Rotated Labels and Mean Points",
#         xaxt='n'
# )
# points(1:27, true_values, col="red", pch=18)  # Add mean values
# 
# # Rotate and add x-axis labels
# axis(1, at=1:27, labels=FALSE)
# text(1:27, par("usr")[3] - 0.1, srt=90, adj=1, labels=c("B11_1", "B11_2", "B12_1", "B12_2", "B21d_1", "B21d_2", "B22d_1", "B22d_2", "B31_1", "B31_2", "B32_1", "B32_2", "Lmdd_3", "Lmdd_5", "Lmdd_10", "Lmdd_12", "Qd_1", "Qd_2", "Rd_1", "Rd_2", "Rd_3", "Rd_4", "Rd_5", "Rd_6", "gamma1", "gamma2_1", "gamma2_2"), xpd=TRUE)

# Updated data for boxplot from summary statistics
data_list_updated <- list(
  B11_1 = c(Min = -0.01527, `1st Qu.` = 0.08021, Median = 0.12350, Mean = 0.15640, `3rd Qu.` = 0.18299, Max = 0.77047),
  B11_2 = c(Min = -0.01651, `1st Qu.` = 0.11257, Median = 0.16132, Mean = 0.20556, `3rd Qu.` = 0.25412, Max = 0.62777),
  B12_1 = c(Min = -0.62611, `1st Qu.` = -0.23011, Median = -0.15324, Mean = -0.17338, `3rd Qu.` = -0.09935, Max = -0.01233),
  B12_2 = c(Min = -0.5304, `1st Qu.` = -0.2738, Median = -0.1804, Mean = -0.1900, `3rd Qu.` = -0.1137, Max = 0.1286),
  B21d_1 = c(Min = 0.4929, `1st Qu.` = 0.7513, Median = 0.8354, Mean = 0.8166, `3rd Qu.` = 0.8904, Max = 1.0005),
  B21d_2 = c(Min = 0.5361, `1st Qu.` = 0.7950, Median = 0.8702, Mean = 0.8410, `3rd Qu.` = 0.9114, Max = 1.0932),
  B22d_1 = c(Min = -0.01191, `1st Qu.` = 0.14382, Median = 0.26324, Mean = 0.24916, `3rd Qu.` = 0.35560, Max = 0.48244),
  B22d_2 = c(Min = -0.02454, `1st Qu.` = 0.11142, Median = 0.21907, Mean = 0.22554, `3rd Qu.` = 0.34282, Max = 0.47801),
  B31_1 = c(Min = -0.338132, `1st Qu.` = 0.007845, Median = 0.062482, Mean = 0.079100, `3rd Qu.` = 0.136237, Max = 0.485016),
  B31_2 = c(Min = -0.32823, `1st Qu.` = -0.01042, Median = 0.05185, Mean = 0.06652, `3rd Qu.` = 0.15688, Max = 0.41882),
  B32_1 = c(Min = -0.26274, `1st Qu.` = -0.12349, Median = -0.08359, Mean = -0.07326, `3rd Qu.` = -0.02174, Max = 0.25562),
  B32_2 = c(Min = -0.51070, `1st Qu.` = -0.14563, Median = -0.06320, Mean = -0.08262, `3rd Qu.` = -0.01776, Max = 0.21095),
  Lmdd_3 = c(Min = 0.1945, `1st Qu.` = 0.6074, Median = 0.8861, Mean = 0.8564, `3rd Qu.` = 1.0626, Max = 1.3731),
  Lmdd_5 = c(Min = 0.4931, `1st Qu.` = 0.7772, Median = 0.9361, Mean = 0.9617, `3rd Qu.` = 1.1415, Max = 1.3654),
  Lmdd_10 = c(Min = 0.3895, `1st Qu.` = 0.6216, Median = 0.8633, Mean = 0.8649, `3rd Qu.` = 1.1224, Max = 1.3755),
  Lmdd_12 = c(Min = 0.6467, `1st Qu.` = 0.9177, Median = 1.0843, Mean = 1.0836, `3rd Qu.` = 1.2407, Max = 1.6208),
  Qd_1 = c(Min = 0.001000, `1st Qu.` = 0.004273, Median = 0.173008, Mean = 0.118793, `3rd Qu.` = 0.174648, Max = 0.177056),
  Qd_2 = c(Min = 0.001000, `1st Qu.` = 0.006755, Median = 0.172499, Mean = 0.118754, `3rd Qu.` = 0.174423, Max = 0.177937),
  Rd_1 = c(Min = 0.03963, `1st Qu.` = 0.14156, Median = 0.37466, Mean = 0.28832, `3rd Qu.` = 0.37610, Max = 0.37842),
  Rd_2 = c(Min = 0.01084, `1st Qu.` = 0.12866, Median = 0.37355, Mean = 0.27844, `3rd Qu.` = 0.37491, Max = 0.37652),
  Rd_3 = c(Min = 0.03354, `1st Qu.` = 0.12948, Median = 0.37318, Mean = 0.28092, `3rd Qu.` = 0.37461, Max = 0.37739),
  Rd_4 = c(Min = 0.04015, `1st Qu.` = 0.16215, Median = 0.37399, Mean = 0.29225, `3rd Qu.` = 0.37565, Max = 0.39179),
  Rd_5 = c(Min = 0.02062, `1st Qu.` = 0.12741, Median = 0.37341, Mean = 0.27959, `3rd Qu.` = 0.37509, Max = 0.37764),
  Rd_6 = c(Min = 0.04769, `1st Qu.` = 0.14167, Median = 0.37300, Mean = 0.28627, `3rd Qu.` = 0.37419, Max = 0.37711),
  gamma1 = c(Min = 4.049, `1st Qu.` = 4.122, Median = 4.124, Mean = 4.224, `3rd Qu.` = 4.364, Max = 4.630),
  gamma2_1 = c(Min = -0.04767, `1st Qu.` = 0.48931, Median = 0.85377, Mean = 0.91322, `3rd Qu.` = 1.17154, Max = 2.19817),
  gamma2_2 = c(Min = 0.1312, `1st Qu.` = 0.5173, Median = 0.8391, Mean = 0.9650, `3rd Qu.` = 1.2404, Max = 2.8608)
)

# # Plot the updated boxplot
# boxplot(data_list_updated, 
#         names = names(data_list_updated),
#         las = 2, # Rotate axis labels
#         cex.axis = 0.7, # Adjust size of axis labels
#         col = "lightblue",
#         # main = "Updated Parameter Estimates with True Values",
#         # xlab = "Parameters",
#         # ylab = "Estimates"
# )
# 
# # Add points for true values
# points(1:27, true_values, col = "red", pch = 18) # Add true values


# Define a vector of colors corresponding to each parameter category
colors <- c(rep("lightblue", length(grep("^B", names(data_list_updated)))),  # Structural model parameters
            rep("lightgreen", length(grep("^Lmdd", names(data_list_updated)))), # Measurement model parameters
            rep("grey", length(grep("^[RQ]d", names(data_list_updated)))), # Residual variances
            rep("orange", length(grep("^gamma", names(data_list_updated)))) # Parameters w.r.t. Markov switching models
)

# Plot the updated boxplot with specified colors
boxplot(data_list_updated, 
        names = names(data_list_updated),
        las = 2, # Rotate axis labels
        cex.axis = 0.7, # Adjust size of axis labels
        col = colors # Use the colors vector here
)

# Add points for true values, assuming 'true_values' vector is defined
points(1:length(data_list_updated), true_values, col = "red", pch = 18) # Add true values
