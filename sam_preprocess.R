setwd("C:/Users/kento/OneDrive - UT Cloud/Tuebingen/Research/Methods Center/WS23/Regime-Switching")

# import data
data <- read.table("sam_1718_t_C.csv", header=TRUE, sep=";", dec=",", na="-99")

# change the first column name from Code to ID
colnames(data)[1] <- "ID"
# change the second column name from T to day
colnames(data)[2] <- "day"
# change the last column name from event to dropout
colnames(data)[92] <- "dropout"

# number of observed variables
No <- 17

# select 17 intra-individual variables
cols_w <- c("Av1_state", "Iv1_state", "Uv1_state", "Co1_state", "Co2_state", "Leist_verstehen_state", "Leist_bearbeiten_state", 
            "Leist_stress_state", "Leist_ueberfordert_state", "Angst_abbruch_state", "Angst_scheitern_state", "PANP01_state", 
            "PANP05_state", "PANP08_state", "PANN01_state", "PANN05_state", "PANN09_state")

#####################################
# get longitudinal data (4063 x 20) #
#####################################
# col1 : Code : participant's ID 
# col2 : tage.num : number of days 
# col3-19 : cols_w (intra-individual variables)
# col20 : event : whether drop-out occurred (binary) 
y_w <- data[, c("ID", "day", cols_w, "dropout")]

# dropout contains -1
# table(y_w$dropout)

# unique ID
uID <- unique(y_w$ID)
# unique day
uDay <- min(y_w$day):max(y_w$day)

# number of days : 132
Nt <- length(uDay)
# number of persons : 122
N <- length(uID)
# number of columns : 20
nCol <- dim(y_w)[2]

###################################################
# reshape the longitudinal data into 3 dimensions #
###################################################

# 1: crate an array of NAs (yw)
yw <- array(NA, c(N, Nt, nCol-1))
dim(yw) # 122 x 132 x 19

# 2: fill the entries of (yw) based on (y_w) 
for (i in 1:N) {
  # take ith person's data from y_w
  yw_i <- y_w[y_w$ID==uID[i],]
  yw[i,,1] <- uDay # day
  
  for (t in 1:Nt) {
    # if more than one response for a day, average the responses
    if (nrow(yw_i[yw_i$day==uDay[t],]) > 0) {
      yw[i,t,2:(nCol-1)] <- colMeans(yw_i[yw_i$day==uDay[t],3:nCol], na.rm=TRUE) }
  }
}

# number of columns : 19
nCol <- dim(yw)[3]

# replace NA dropout values with 1e30
yw[,,nCol][is.na(yw[,,nCol])] <- 1e30
# remove those with dropout = -1
yw[,,nCol][yw[,,nCol]==-1] <- NA
yw <- yw[complete.cases(yw[,,nCol]),,]


# unique ID
uID <- unique(y_w$ID)
# unique day
uDay <- min(y_w$day):max(y_w$day)

# number of persons : 117
N <- dim(yw)[1]
# number of days : 132
# Nt <- dim(yw)[2]

# there is a dropout = 0.5 
# table(yw)
# substitute event = 0.5 with event = 1
yw[,,nCol] <- ceiling(yw[,,nCol])
# there is no dropout = 0.5
# table(yw)

# no negative yw entries 
# sum(yw < 0)

# replace dropout value 1e30 with -1e30
yw[,,nCol][yw[,,nCol] == 1e30] <- -1e30
# fill NA values with -1e30
yw[is.na(yw)] <- -1e30

###########################################################
# impute missing data using a most recent available entry #
###########################################################

# dropout = -1e30
# table(yw)

for (t in 2:Nt) {
  for (i in 1:N) {
    for (col in 2:nCol) {
      if (yw[i,t-1,col] < 0) {next}
      if (yw[i,t,col] < 0) {
        yw[i,t,col] <- yw[i,t-1,col] }
    }
  }
}

# there are less dropout = -1e30
# table(yw)

#####################
# erase switch back #
#####################

for (i in 1:N) {
  for (t in 2:Nt) {
    if (yw[i,t-1,nCol]==1) {
      yw[i,t:Nt,nCol] <- 1
      break
    }
  }
} 

# plot each person's drop out occurrence
# library(RColorBrewer)
# c <- brewer.pal(8, "Dark2")
# plot(yw[i,,nCol], type="l", ylim=c(0,1), lwd=1.5, xlab = "day", ylab = "drop out", main = 'drop out occurrences, yaxt='n')
# for (i in 2:N){
#   lines(yw[i,,nCol], col=c[i%%8], lwd=1.5)
# }

# replace -1e30 with NA
yw[yw == -1e30] <- NA

############################################
# Confirmatory Factor Analysis with lavaan #
############################################

library(lavaan)

#######
# CFA #
#######

model_cfa <- '
# latent variables
subImp =~ Av + Iv + Uv
cost =~ Co1 + Co2
understand =~ understand1 + understand2
stress =~ stress1 + stress2
AtF =~ AtF1 + AtF2
PAS =~ PA1 + PA5 + PA8
NAS =~ NA1 + NA5 + NA9
'

# number of latent factor 
Nf <- 7

# # 36 people with no response at t=10
# summary(yw[,10,])
# # 17 people with no response at t=11
# summary(yw[,11,])
# # 17 people with no response at t=15
# summary(yw[,15,])
# # 9 people with no response at t=16
# summary(yw[,16,])

# select data only after 11th days
yw <- yw[,11:Nt,]

# number of days : 122
Nt <- dim(yw)[2]

# fit_cfas <- list()
# Yt <- list()

# remove rows with missing values at t=11
yw <- yw[complete.cases(yw[,1,]),,2:nCol] 
# dim(yw)


# # select 17 intra-individual variables
# cols_w <- c("Av", "Iv", "Uv", "Co1", "Co2", "understand1", "understand2", 
#             "stress1", "stress2", "AtF1", "AtF2", "PA1", "PA5", "PA8", 
#             "NA1", "NA5", "NA9")

# Av: attainment value
# Iv: intrinsic value
# Uv: utility value
# Co1, Co2: cost
# understand1, understand2: understanding
# stress1, stress2: stress
# AtF1, AtF2: afraid to fail
# PA1, PA5, PA8: positive affect scale (careful, active, excited)
# NA1, NA5, NA9: negative affect scale (nervous, afraid, distressed)

# for (t in 1:Nt) {
#   y <- yw[,t,1:(nCol-2)]
#   colnames(y) <- cols_w
#   fit_cfas[[t]] <- cfa(model_cfa, data=y)
#   Yt[[t]] <- lavPredict(fit_cfas[[t]], method = "Bartlett")
# }

# save data as list
(df <- list(state=yw[,,nCol-1], Yt=yw[,,1:(nCol-2)]))
