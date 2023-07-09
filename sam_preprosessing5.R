# install.packages("RColorBrewer")
library(RColorBrewer)
# install.packages("lavaan")
library(lavaan)
# install.packages("abind")
library(abind)

setwd("C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/WS23/Regime-Switching")

###############
# import data #
###############
data <- read.table("sam_1718_t_C.csv", header=TRUE, sep=";", dec=",", na="-99")
colnames(data)[c(1,2,92)] <- c("ID", "day", "dropout")

############################
# get 2D longitudinal data #
############################
# ID : participant's ID 
# day: day 
# cols_b: inter-individual observed variables
# dropout : whether drop-out occurred (binary) 

# select inter-individual observed variables
cols_b <- c("abi_m_note", "fw_pkt", "gesamt_iq")
y2D_b <- data[, c("ID", "day", cols_b)]

x <- data[, c("ID", "day", "dropout")]

######################################################################
## between ##
# abiMath: math abinote between 1 and 6 (the smaller the better)
# TIMMS: prior math performance (Sum score of the 20 TIMMS items)
# totIQ: cognitive basic ability (total score)
######################################################################

# rename between-variables
cols_b <- c("abiMath", "TIMMS", "totIQ")
colnames(y2D_b) <-  c("ID", "day", cols_b)

# correct abiMath columns
y2D_b$abiMath[y2D_b$abiMath<=1.1 & is.na(y2D_b$abiMath)==FALSE] <- 15
y2D_b$abiMath[y2D_b$abiMath>=1.2 & y2D_b$abiMath<=1.5&is.na(y2D_b$abiMath)==FALSE] <- 14
y2D_b$abiMath[y2D_b$abiMath>=1.6 & y2D_b$abiMath<=1.8&is.na(y2D_b$abiMath)==FALSE] <- 13
y2D_b$abiMath[y2D_b$abiMath>=1.9 & y2D_b$abiMath<=2.1&is.na(y2D_b$abiMath)==FALSE] <- 12
y2D_b$abiMath[y2D_b$abiMath>=2.2 & y2D_b$abiMath<=2.5&is.na(y2D_b$abiMath)==FALSE] <- 11
y2D_b$abiMath[y2D_b$abiMath>=2.6 & y2D_b$abiMath<=2.8&is.na(y2D_b$abiMath)==FALSE] <- 10
y2D_b$abiMath[y2D_b$abiMath>=2.9 & y2D_b$abiMath<=3.1&is.na(y2D_b$abiMath)==FALSE] <- 9
y2D_b$abiMath[y2D_b$abiMath>=3.2 & y2D_b$abiMath<=3.5&is.na(y2D_b$abiMath)==FALSE] <- 8

# rewrite ID as numeric 
index <- 1
for (id in unique(y2D_b$ID)) {y2D_b$ID[which(y2D_b$ID==id)] <- x$ID[which(x$ID==id)] <- index; index <- index + 1}
y2D_b$ID <- as.numeric(y2D_b$ID); x$ID <- as.numeric(x$ID) 

# days are mostly equally spaced
dUDay <-  list()
for (t in 2:length(unique(y2D_b$day))) {dUDay[t-1] <- sort(unique(y2D_b$day))[t] - sort(unique(y2D_b$day))[t-1]}
table(unlist(dUDay))

# rewrite day as a sequence 
index <- 1
for (day in sort(unique(y2D_b$day))) {
  y2D_b$day[which(y2D_b$day==day)] <- x$day[which(x$day==day)] <- index; index <- index + 1 }

#########################################
# reshape the longitudinal data into 3D #
#########################################
y3D_b <- array(NA, c(length(unique(y2D_b$ID)), length(unique(y2D_b$day)), length(cols_b)))
x_new <- array(NA, c(dim(y3D_b)[1], dim(y3D_b)[2])) 
# fill the entries of (y3D_b) based on (y2D_b) 
for (i in 1:length(unique(y2D_b$ID))) {
  # take ith person's data
  x_i <- x[x$ID==i,]; y2D_b_i <- y2D_b[y2D_b$ID==i,]
  for (t in 3:length(unique(y2D_b$day))) {
    # if more than one response for a day, average the responses
    if (nrow(x_i[x_i$day==t,]) > 0) {
      y3D_b[i,t,] <- colMeans(y2D_b_i[y2D_b_i$day==t, 3:ncol(y2D_b)], na.rm=TRUE) 
      x_new[i,t] <- mean(x_i$dropout[x_i$day==t], na.rm=TRUE) } } }

# there are persons with dropout = -1 and dropout = 0.5
x_new <- abs(ceiling(x_new))

##########################
# impute missing entries #
##########################
for (i in 1:nrow(x_new)) {
  for (t in 2:ncol(x_new)) {
    if (is.na(x_new[i,t-1])) {next}
    if (is.na(x_new[i,t])) {x_new[i,t] <- x_new[i,t-1] } } } 

for (i in 1:nrow(x_new)) {
  for (t in 1:(ncol(x_new)-1)) {
    if (is.na(x_new[i,ncol(x_new)-t+1])) {next}
    if (is.na(x_new[i,ncol(x_new)-t])) {x_new[i,ncol(x_new)-t] <- x_new[i,ncol(x_new)-t+1] } } } 

for (i in 1:nrow(x_new)) {
  for (t in 2:ncol(x_new)) {
    for (col in 1:dim(y3D_b)[3]) {
      if (is.na(y3D_b[i,t-1,col])) {next}
      if (is.na(y3D_b[i,t,col])) {y3D_b[i,t,col] <- y3D_b[i,t-1,col] } } } }

for (i in 1:nrow(x_new)) {
  for (t in 1:(ncol(x_new)-1)) {
    for (col in 1:dim(y3D_b)[3]) {
      if (is.na(y3D_b[i,ncol(x_new)-t+1,col])) {next}
      if (is.na(y3D_b[i,ncol(x_new)-t,col])) {y3D_b[i,ncol(x_new)-t,col] <- y3D_b[i,ncol(x_new)-t+1,col] } } } } 

######################
# delete switch back #
######################
for (i in 1:nrow(x_new)) {
  for (t in 2:ncol(x_new)) {
    if (x_new[i,t-1] == 1) {x_new[i,t:ncol(x_new)] <- 1; break} } }

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
#######################################################################

###############
# import data #
###############
y2D_w <- read.table("sam_1718_t_C_Impt.csv", header=TRUE, sep=";", dec=",", na="-99")
cols_w <- c("Av", "Iv", "Uv", "Co1", "Co2", "understand1", "understand2",
"stress1", "stress2", "AtF1", "AtF2", "PA1", "PA5", "PA8", "NA1", "NA5", "NA9")
colnames(y2D_w) <- c("ID", "day", cols_w)
index <- 0
for (row in 1:length(y2D_w$ID)) {index <- ifelse(y2D_w$day[row]==1, index+1, index); y2D_w$ID[row] <- index}

# drop first two entries 
y3D_b <- y3D_b[,3:ncol(y3D_b),] 
x_new <- x_new[,3:ncol(x_new)]

###############################################
# reshape the 3D data into 2D again (for CFA) #
###############################################
y2D_b_new <- data.frame(matrix(data=NA, nrow=nrow(y3D_b)*ncol(y3D_b), ncol=ncol(y2D_b)))
colnames(y2D_b_new) <- colnames(y2D_b)
y2D_b_new[,1:2] <- y2D_w[,1:2]
index <- 0
for (i in 1:nrow(y3D_b)) {
  y2D_b_new[(index+1):(index+ncol(y3D_b)),3:ncol(y2D_b_new)] <- y3D_b[i,,]; index <- index + ncol(y3D_b) }

# drop incomplete entries
keepID <- which(!is.na(apply(y3D_b, 1, sum)))
y3D_b <- y3D_b[keepID,,]
x_new <- x_new[keepID,] 
y2D_b_new <- y2D_b_new[y2D_b_new$ID %in% keepID,] 
y2D_w <- y2D_w[y2D_w$ID %in% keepID,] 

# adjust ID for removed raws
index <- 0
for (row in 1:length(y2D_w$ID)) {
  index <- ifelse(y2D_w$day[row]==1, index+1, index)
  y2D_b_new$ID[row] <- y2D_w$ID[row] <- index }

#########################################
# reshape the longitudinal data into 3D #
#########################################
y3D_w <- array(NA, c(nrow(x_new), ncol(x_new), length(cols_w)))
# fill the entries of (y3D_w) based on (y2D_w) 
for (i in 1:length(unique(y2D_w$ID))) {
  # take ith person's data
  y2D_w_i <- y2D_w[y2D_w$ID==i,]
  for (t in 1:length(unique(y2D_w$day))) {
    # if more than one response for a day, average the responses
    y3D_w[i,t,] <- colMeans(y2D_w_i[y2D_w_i$day==t, 3:ncol(y2D_w)]) } }

y3D <- abind(y3D_w, y3D_b, along=3)

############################################
# Confirmatory Factor Analysis with lavaan #
############################################

model_cfa_w <- '
# latent variables
subImp =~ Av + Iv + Uv
cost =~ Co1 + Co2
understand =~ understand1 + understand2
stress =~ stress1 + stress2
AtF =~ AtF1 + AtF2
PAS =~ PA1 + PA5 + PA8
NAS =~ NA1 + NA5 + NA9 '

model_cfa_b <- '
# latent variables
IQ =~ abiMath + TIMMS + totIQ '

fit_cfas <- list()
eta <- list()

# apply CFA to intra- and inter-individual variables in order to obtain factors  
fit_cfas_w <- cfa(model_cfa_w, data=y2D_w[,3:ncol(y2D_w)])
eta2D_w <- lavPredict(fit_cfas_w, method='Bartlett')
eta2D_w <- data.frame(eta2D_w)
colnames(eta2D_w) <- c('subImp', 'cost', 'understand', 'stress', 'AtF', 'PAS', 'NAS')

fit_cfas_b <- cfa(model_cfa_b, data=y2D_b_new[,3:ncol(y2D_b_new)])
eta2D_b <- lavPredict(fit_cfas_b, method='Bartlett')
eta2D_b <- data.frame(eta2D_b)
colnames(eta2D_b) <- c('IQ')

eta2D <- cbind(ID=y2D_w$ID, day=y2D_w$day, eta2D_w, eta2D_b)

#########################################
# reshape the longitudinal data into 3D #
#########################################

# crate an array of NAs 
eta3D <- array(NA, c(nrow(x_new), ncol(x_new), ncol(eta2D_w)+ncol(eta2D_b)))
# fill the entries of (eta3D) based on (eta2D) 
for (i in 1:nrow(x_new)) {
  # take ith person's data from eta2D
  eta2D_i <- eta2D[eta2D$ID==i,]
  for (t in 1:ncol(x_new)) {eta3D[i,t,] <- colMeans(eta2D_i[eta2D_i$day==t,3:dim(eta2D)[2]]) } } 

eta3D_w <- eta3D[,,1:ncol(eta2D_w)] # take only intra-individual factors
eta3D_b <- eta3D[,1,(ncol(eta2D_w)+1):(ncol(eta2D_w)+ncol(eta2D_b))] # take only inter-individual factors

######################################
# add interaction of latent factors  #
######################################
# number of latent intra-individual factors
# number of latent inter-individual factors

# store the pair (Nf1,Nf2) as data frame 
jNf12 <- expand.grid(f1=1:ncol(eta2D_w), f2=1:ncol(eta2D_b))
eta3DInt <- array(NA, c(nrow(x_new), ncol(x_new), nrow(jNf12)))
for (row in 1:nrow(jNf12)) {
  f1 <- jNf12$f1[row]; f2 <- jNf12$f2[row]
  eta3DInt[,,row] <- eta3D[,,f1] * eta3D[,,ncol(eta2D_w)+f2] }
# merge main effects and interaction effects
eta3D_wb <- abind(eta3D, eta3DInt, along=3)
# number of total latent factors (including interaction)
Nf12 <- dim(eta3D_wb)[3]

#######################
# save data as a list #
#######################
df <- list(eta3D=eta3D, eta3D1=eta3D_w, eta3D2=eta3D_b, eta3D12=eta3D_wb, 
           y3D=y3D, y3D1=y3D_w, y3D2=y3D_b, x=x_new, N=nrow(x_new), Nt=ncol(x_new),
           Nf1=ncol(eta2D_w), Nf2=ncol(eta2D_b), No1=length(cols_w), No2=length(cols_b))


