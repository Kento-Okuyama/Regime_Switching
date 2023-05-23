# install.packages("RColorBrewer")
library(RColorBrewer)

setwd("C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/WS23/Regime-Switching")

# import data
data <- read.table("sam_1718_t_C.csv", header=TRUE, sep=";", dec=",", na="-99")

# change the first column name from Code to ID
colnames(data)[1] <- "ID"
# change the second column name from T to day
colnames(data)[2] <- "day"
# change the last column name from event to dropout
colnames(data)[92] <- "dropout"

# select intra-individual observed variables
cols_w <- c("Av1_state", "Iv1_state", "Uv1_state", "Co1_state", "Co2_state", "Leist_verstehen_state", "Leist_bearbeiten_state", 
          "Leist_stress_state", "Leist_ueberfordert_state", "Angst_abbruch_state", "Angst_scheitern_state", "PANP01_state", 
          "PANP05_state", "PANP08_state", "PANN01_state", "PANN05_state", "PANN09_state")
# select inter-individual observed variables
cols_b <- c("gesamt_iq")

cols <- c(cols_w, cols_b)
  
# number of observed variables that are used
No <- length(cols)

############################
# get 2D longitudinal data #
############################
# Code : participant's ID 
# day 
# cols: intra-individual and inter-individual observed variables
# dropout : whether drop-out occurred (binary) 
y2D <- data[, c("ID", "day", cols, "dropout")]

str(data)
# unique ID
uID <- unique(y2D$ID)
# unique day
uDay <- min(y2D$day):max(y2D$day)

# number of days
Nt <- length(uDay)
# number of persons
N <- length(uID)
# number of columns
nC2D <- ncol(y2D)

# rename within-variables
cols_w <- c("Av", "Iv", "Uv", "Co1", "Co2", "understand1", "understand2",
          "stress1", "stress2", "AtF1", "AtF2", "PA1", "PA5", "PA8",
          "NA1", "NA5", "NA9")
# rename between-variables
cols_b <- c("totIQ")

cols <- c(cols_w, cols_b)

######################################################################
## within ##
# Av: attainment value
# Iv: intrinsic value
# Uv: utility value
# Co1, Co2: cost
# understand1, understand2: understanding
# stress1, stress2: stress
# AtF1, AtF2: afraid to fail
# PA1, PA5, PA8: positive affect scale (careful, active, excited)
# NA1, NA5, NA9: negative affect scale (nervous, afraid, distressed)
## between ##
# totIQ: cognitive basic ability (total score)
######################################################################

colnames(y2D)[3:(nC2D-1)] <- cols

# unexpected input: dropout = -1
# table(yt2$dropout)

#########################################################
# reshape the longitudinal data into 3D (N x Nt x nC3D) #
#########################################################
# col1-17 : cols: intra-individual observed variables
# col18 : dropout : whether drop-out occurred (binary) 

# number of columns
nC3D <- nC2D - 2

# 1: crate an array of NAs 
y3D <- array(NA, c(N, Nt, nC3D)) 
# dim(y3D)

# 2: fill the entries of (y3D) based on (y2D) 
for (i in 1:N) {
  # take ith person's data from y2D
  y2D_i <- y2D[y2D$ID==uID[i],]
  
  for (t in 1:Nt) {
    # if more than one response for a day, average the responses
    if (nrow(y2D_i[y2D_i$day==uDay[t],]) > 0) {
      y3D[i,t,] <- 
        colMeans(y2D_i[y2D_i$day==uDay[t],3:nC2D], na.rm=TRUE) } } }

# replace NA dropout values with 1e30
y3D[,,nC3D][is.na(y3D[,,nC3D])] <- 1e30
# remove the persons with dropout = -1
remID <- rowSums(y3D[,,nC3D]==-1, na.rm=TRUE) > 0
y3D <- y3D[remID==FALSE,,]
# unique ID (updated)
uID <- uID[remID==FALSE]

# number of persons (updated)
N <- dim(y3D)[1]

# there are dropout = 0.5 
# table(y3D)

# substitute dropout = 0.5 with 1
y3D[,,nC3D] <- ceiling(y3D[,,nC3D])

# there is no dropout = 0.5
# table(y3D)

# side note: there is no negative entries in y3D
# sum(y3D < 0, na.rm=TRUE)

###########################################################################
# as a placeholder, assign a negative values to cells with missing values #
###########################################################################

# replace dropout = 1e30 with -1e30
y3D[,,nC3D][y3D[,,nC3D] == 1e30] <- -1e30
# fill NA cells with -1e30
y3D[is.na(y3D)] <- -1e30

######################
# delete switch back #
######################

for (i in 1:N) {
  for (t in 2:Nt) {
    if (y3D[i,t-1,nC3D] == 1) {
      y3D[i,t:Nt,nC3D] <- 1
      break } } }

# plot persons' drop out occurrence
c <- brewer.pal(8, "Dark2")
plot(y3D[1,,nC3D], type="l", ylim=c(0,1), lwd=1.5, xlab="day", ylab="dropout", main="dropout occurrence", yaxt="n")
for (i in 2:N){
  lines(y3D[i,,nC3D], col=c[i%%8], lwd=1.5)
}

# replace -1e30 with NA
y3D[y3D==-1e30] <- NA

# 117 people with no response at t=10
summary(y3D[,10,])
# 33 people with no response at t=11
summary(y3D[,11,])
# 117 people with no response at t=15
summary(y3D[,15,])
# 47 people with no response at t=16
summary(y3D[,16,])

# select data only after 11th days
y3D <- y3D[,11:Nt,]

# unique day (updated)
uDay <- uDay[11:Nt]

# number of days (updated)
Nt <- dim(y3D)[2]

# unique ID (updated)
uID <- uID[complete.cases(y3D[,1,])]

# remove rows with missing values at t=11
y3D <- y3D[complete.cases(y3D[,1,]),,] 
# dim(y3D)

# number of persons (updated)
N <- dim(y3D)[1]

# doNull <- is.na(y3D[,,nC3D])
# otherNullMat <- is.na(y3D[,,1:(nC3D-1)])
# otherNullMax <- apply(otherNull, 1:2, max)

# dropout does not have a null value when all 17 variables are not null
# sum((doNull - otherNullMax) > 0)

###############################################
# reshape the 3D data into 2D again (for CFA) #
###############################################

y2D2 <- matrix(data=NA, nrow=N*Nt, ncol=nC2D)
y2D2 <- data.frame(y2D2)
colnames(y2D2) <- c("ID", "day", cols, "dropout")
# N*Nt
NxNt <- nrow(y2D2)
# dim(y2D2)

count <- 1
for (i in 1:N) {
  for (t in 1:Nt) {
    y2D2[count,1] <- uID[i]
    y2D2[count,2] <- uDay[t]
    y2D2[count,3:nC2D] <- y3D[i,t,]
    count <- count + 1
  }
}

# save data as a list
df <- list(y2D=y2D2, y3D=y3D)
y2D <- df$y2D
y3D <- df$y3D
