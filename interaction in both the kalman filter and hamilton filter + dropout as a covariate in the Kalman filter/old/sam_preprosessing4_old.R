# install.packages("RColorBrewer")
library(RColorBrewer)
# install.packages("lavaan")
library(lavaan)
# install.packages("abind")
library(abind)

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
No1 <- length(cols_w)

# select inter-individual observed variables
cols_b <- c("abi_m_note", "fw_pkt", "gesamt_iq")
No2 <- length(cols_b)

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
uDay <- sort(unique(data$day))

# number of days
Nt <- length(uDay)
# number of persons
N <- length(uID)
# number of columns
nC2D <- ncol(y2D)

dUDay <-  list()
for (t in 2:Nt) {dUDay[t-1] <- uDay[t] - uDay[t-1]}
table(unlist(dUDay))

# rename within-variables
cols_w <- c("Av", "Iv", "Uv", "Co1", "Co2", "understand1", "understand2",
            "stress1", "stress2", "AtF1", "AtF2", "PA1", "PA5", "PA8",
            "NA1", "NA5", "NA9")
# rename between-variables
cols_b <- c("abiMath", "TIMMS", "totIQ")
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
# abiMath: math abinote between 1 and 6 (the smaller the better)
# TIMMS: prior math performance (Sum score of the 20 TIMMS items)
# totIQ: cognitive basic ability (total score)
######################################################################

colnames(y2D)[3:(nC2D-1)] <- cols

y2D$abiMath[y2D$abiMath<=1.1 & is.na(y2D$abiMath)==FALSE] <- 15
y2D$abiMath[y2D$abiMath>=1.2 & y2D$abiMath<=1.5&is.na(y2D$abiMath)==FALSE] <- 14
y2D$abiMath[y2D$abiMath>=1.6 & y2D$abiMath<=1.8&is.na(y2D$abiMath)==FALSE] <- 13
y2D$abiMath[y2D$abiMath>=1.9 & y2D$abiMath<=2.1&is.na(y2D$abiMath)==FALSE] <- 12
y2D$abiMath[y2D$abiMath>=2.2 & y2D$abiMath<=2.5&is.na(y2D$abiMath)==FALSE] <- 11
y2D$abiMath[y2D$abiMath>=2.6 & y2D$abiMath<=2.8&is.na(y2D$abiMath)==FALSE] <- 10
y2D$abiMath[y2D$abiMath>=2.9 & y2D$abiMath<=3.1&is.na(y2D$abiMath)==FALSE] <- 9
y2D$abiMath[y2D$abiMath>=3.2 & y2D$abiMath<=3.5&is.na(y2D$abiMath)==FALSE] <- 8

#########################################################
# reshape the longitudinal data into 3D (N x Nt x nC3D) #
#########################################################
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

###########################################################
# impute missing data using a most recent available entry #
###########################################################

# NA values are temporarily filled with negative values
# table(y3D)

# for (t in 2:Nt) {
#   for (i in 1:N) {
#     for (col in 1:nC3D) {
#       if (y3D[i,t-1,col] < 0) {next}
#       if (y3D[i,t,col] < 0) {y3D[i,t,col] <- y3D[i,t-1,col] } } } }

# impute dropout entries that are missing
for (i in 1:N) {
  for (t in 2:Nt) {
    if (y3D[i,t-1,nC3D] < 0) {next}
    if (y3D[i,t,nC3D] < 0) {y3D[i,t,nC3D] <- y3D[i,t-1,nC3D] } } } 

# there are less negative entries
# table(y3D)

######################
# delete switch back #
######################

for (i in 1:N) {
  for (t in 2:Nt) {
    if (y3D[i,t-1,nC3D] == 1) {y3D[i,t:Nt,nC3D] <- 1; break} } }

# plot persons' drop out occurrence
# c <- brewer.pal(8, "Dark2")
# plot(y3D[1,,nC3D], type="l", ylim=c(0,1), lwd=1.5, xlab="day", ylab="dropout", main="dropout occurrence", yaxt="n")
# for (i in 2:N){lines(y3D[i,,nC3D], col=c[i%%8], lwd=1.5)}

# replace -1e30 with NA
y3D[y3D==-1e30] <- NA

# 117 people with no response at t=1
summary(y3D[,1,])
# 36 people with no response at t=2
summary(y3D[,2,])
# 33 people with no response at t=15
summary(y3D[,3,])
# 42 people with no response at t=16
summary(y3D[,4,])

# select data only after the second day
y3D <- y3D[,2:Nt,]

# unique day (updated)
uDay <- uDay[2:Nt]

# number of days (updated)
Nt <- dim(y3D)[2]

# unique ID (updated)
uID <- uID[complete.cases(y3D[,1,])]

# remove rows with missing values at t=1
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

y3D1 <- y3D[,,1:No1] # take only intra-individual observed variables
y3D2 <- y3D[,,(No1+1):No] # take only inter-individual observed variables

# save data as a list
df <- list(y2D=y2D2, y3D=y3D, y3D1=y3D1, y3D2=y3D2)
y2D <- df$y2D
y3D <- df$y3D

y3D1 <- df$y3D1 # only intra-individual factors 
y3D2 <- df$y3D2 # only inter-individual factors

############################################
# Confirmatory Factor Analysis with lavaan #
############################################

model_cfa <- '
# latent variables
subImp =~ Av + Iv + Uv
cost =~ Co1 + Co2
understand =~ understand1 + understand2
stress =~ stress1 + stress2
AtF =~ AtF1 + AtF2
PAS =~ PA1 + PA5 + PA8
NAS =~ NA1 + NA5 + NA9
IQ =~ abiMath + TIMMS + totIQ
'

fit_cfas <- list()
eta <- list()

# apply CFA to intra- and inter-individual variables in order to obtain factors  
fit_cfas <- cfa(model_cfa, data=y2D[,3:(nC2D-1)])
eta2D <- lavPredict(fit_cfas, method='Bartlett')
eta2D <- data.frame(eta2D)

dim(eta2D) # number of valid observations (<< NxNt)
# number of latent factor 
Nf <- dim(eta2D)[2]

# create NxNt array to store factore scores
eta2DFull <- matrix(data=NA, nrow=NxNt, ncol=Nf+2)
eta2DFull <- data.frame(eta2DFull)
colnames(eta2DFull) <- c("ID", "day", colnames(eta2D))
nC2D_eta <- ncol(eta2DFull)

count <- 1
for (i in 1:NxNt) {
  if (rowSums(is.na(y2D))[i] == 0) {
    eta2DFull[i,(1:Nf)+2] <- eta2D[count,]
    count <- count + 1 } }

eta2DFull[,c(1,2)] <- y2D[,c(1,2)]

# valid observations (same as eta2D)
sum(rowSums(is.na(eta2DFull)) == 0)

#######################################################
# reshape the longitudinal data into 3D (N x Nt x Nf) #
#######################################################
# cols: intra- and inter-individual latent factors
# dropout : whether drop-out occurred (binary) 

# number of columns
nC3D_eta <- nC2D_eta - 2

# 1: crate an array of NAs 
eta3D <- array(NA, c(N, Nt, nC3D_eta))
# dim(eta3D) 

# 2: fill the entries of (eta3D) based on (eta2D) 
for (i in 1:N) {
  # take ith person's data from eta2D
  eta2DFull_i <- eta2DFull[eta2DFull$ID==uID[i],]
  for (t in 1:Nt) {eta3D[i,t,] <- as.numeric(eta2DFull_i[eta2DFull_i$day==uDay[t],3:nC2D_eta]) } } 

# impute IQ entries that are missing
for (i in 1:N) {
  for (t in 2:Nt) {
    if (is.na(eta3D[i,t-1,nC3D_eta])) {next}
    if (is.na(eta3D[i,t,nC3D_eta])) {eta3D[i,t,nC3D_eta] <- eta3D[i,t-1,nC3D_eta] } } } 

######################################
# add interaction of latent factors  #
######################################

# number of latent intra-individual factors
Nf1 <- 7
# number of latent inter-individual factors
Nf2 <- 1

# store the pair (Nf1,Nf2) as data frame 
jNf12 <- expand.grid(f1=1:Nf1, f2=1:Nf2)
eta3DInt <- array(NA, c(N, Nt, nrow(jNf12)))
for (row in 1:nrow(jNf12)) {
  f1 <- jNf12$f1[row]; f2 <- jNf12$f2[row]
  eta3DInt[,,row] <- eta3D[,,f1] * eta3D[,,Nf1+f2] }
# merge main effects and interaction effects
eta3D12 <- abind(eta3D, eta3DInt, along=3)
# number of total latent factors (including interaction)
Nf12 <- dim(eta3D12)[3]

eta3D1 <- eta3D[,,1:Nf1] # take only intra-individual factors
eta3D2 <- eta3D[,1,(Nf1+1):Nf] # take only inter-individual factors
  
# save data as a list
df <- list(eta3D=eta3D, eta3D12=eta3D12, eta3D1=eta3D1, eta3D2=eta3D2)
eta3D <- df$eta3D # only main effect
eta3D12 <- df$eta3D12 # main effect and interaction effect
eta3D1 <- df$eta3D1 # only intra-individual factors 
eta3D2 <- df$eta3D2 # only inter-individual factors

# sanity check: if eta2D and eta3D has same number of valid (non-NA) entries
sum(is.na(apply(eta3D, 1:2, sum)) == 0)
sum(is.na(apply(eta3D12, 1:2, sum)) == 0)
