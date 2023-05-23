# install.packages("lavaan")
library(lavaan)

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
IQ =~ totIQ
'

# number of latent factor 
Nf <- 8

fit_cfas <- list()
eta <- list()

# apply CFA to intra- and inter-individual variables in order to obtain factors  
fit_cfas <- cfa(model_cfa, data=y2D[,3:(nC2D-1)])
eta2D <- lavPredict(fit_cfas, method='Bartlett')
eta2D <- data.frame(eta2D)

dim(eta2D) # number of valid observations (<< NxNt)

# create NxNt array to store factore scores
eta2DFull <- matrix(data=NA, nrow=NxNt, ncol=Nf+3)
eta2DFull <- data.frame(eta2DFull)
colnames(eta2DFull) <- c("ID", "day", colnames(eta2D), "dropout")
nC2D_eta <- ncol(eta2DFull)

count <- 1
for (i in 1:NxNt) {
  if (rowSums(is.na(y2D))[i] == 0) {
    eta2DFull[i,(1:Nf)+2] <- eta2D[count,]
    count <- count + 1 } }

eta2DFull[,c(1,2,nC2D_eta)] <- y2D[,c(1,2,nC2D)]

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
  for (t in 1:Nt) {eta3D[i,t,] <- as.logical(eta2DFull_i[eta2DFull_i$day==uDay[t],3:nC2D_eta]) } } 

# save data as a list
df <- list(eta2D=eta2DFull, eta3D=eta3D)
eta2D <- df$eta2D
eta3D <- df$eta3D

# sanity check: if eta2D and eta3D has same number of valid (non-NA) entries
sum(rowSums(is.na(eta2D)) == 0)
sum(is.na(apply(eta3D, 1:2, sum)) == 0)
# sum(apply(is.na(eta3D) == 0, 1:2, min))
