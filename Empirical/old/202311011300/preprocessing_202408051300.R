preprocessing <- function() {
  # import data
  data <- read.table('sam_1718_t_C.csv', header=TRUE, sep=';', dec=',', na='-99')
  colnames(data)[c(1,2,92)] <- c('ID', 'day', 'dropout')
  
  # select intra-individual observed variables  
  cols_w <- c('Co1_state', 'Co2_state', 'Leist_verstehen_state', 'Leist_bearbeiten_state', 'Angst_abbruch_state', 'Angst_scheitern_state', 'PANN01_state', 'PANN05_state', 'PANN09_state')
  # select inter-individual observed variables
  cols_b <- c('abi_m_note', 'fw_pkt', 'gesamt_iq')
  
  ############################
  # get 2D longitudinal data #
  ############################
  y1_2D <- data[, c('ID', 'day', cols_w, 'dropout')]
  y2 <- unique(data[, c('ID', cols_b)])
  
  # rename within-variables
  colnames(y1_2D) <- c('ID', 'day', 'Co1', 'Co2', 'understand1', 'understand2', 'AtF1', 'AtF2', 'NA1', 'NA5', 'NA9', 'dropout')
  # rename between-variables
  colnames(y2) <- c('ID', 'abiMath', 'TIMMS', 'totIQ')
  
  # reverse scales (these are positive items; others are negative)
  y1_2D$understand1 <- -y1_2D$understand1
  y1_2D$understand2 <- -y1_2D$understand2
  
  # unique ID
  uID <- unique(y2$ID)
  # unique day
  uDay <- sort(unique(y1_2D$day))
  
  # number of intra-individual observed factors
  O1 <- length(cols_w)
  # number of inter-individual observed factors
  O2 <- length(cols_b)
  
  dUDay <-  list()
  for (t in 2:length(uDay)) {dUDay[t-1] <- uDay[t] - uDay[t-1]}
  table(unlist(dUDay))
  
  ######################################################################
  ## within ##
  # Co1, Co2: cost
  # understand1, understand2: understanding
  # AtF1, AtF2: afraid to fail
  # NA1, NA5, NA9: negative affect scale (nervous, afraid, distressed)
  ## between ##
  # abiMath: math abinote between 1 and 6 (the smaller the better)
  # TIMMS: prior math performance (Sum score of the 20 TIMMS items)
  # totIQ: cognitive basic ability (total score)
  ######################################################################
  
  #########################################################
  # reshape the longitudinal data into 3D (N x Nt x nC3D) #
  #########################################################
  
  # 1: crate an array of NAs 
  y1 <- array(NA, c(length(uID), length(uDay), O1+1)) 
  
  # 2: fill the entries of (y1) based on (y1_2D) 
  for (i in 1:length(uID)) {
    # take ith person's data from y2D
    y1_2D_i <- y1_2D[y1_2D$ID==uID[i],]
    for (t in 1:length(uDay)) {
      # if more than one response for a day, average the responses
      if (nrow(y1_2D_i[y1_2D_i$day==uDay[t],]) > 0) {
        y1[i,t,] <- 
          colMeans(y1_2D_i[y1_2D_i$day==uDay[t],3:(O1+3)], na.rm=TRUE) } } }
  
  # some negative entries
  # sum(is.na(y1))
  
  # replace NA dropout values with 1e30
  y1[,,O1+1][is.na(y1[,,O1+1])] <- 1e30
  # remove the persons with dropout = -1
  remID <- rowSums(y1[,,O1+1]==-1, na.rm=TRUE) > 0
  y1 <- y1[remID==FALSE,,]
  y2 <- y2[remID==FALSE,,]
  # unique ID (updated)
  uID <- uID[remID==FALSE]
  
  # there are dropout = 0.5 
  # table(y1[,,O1+1])
  
  # substitute dropout = 0.5 with 1
  y1[,,O1+1] <- ceiling(y1[,,O1+1])
  
  # there is no dropout = 0.5
  # table(y1[,,O1+1])
  
  # side note: there is no negative entries in y1
  # sum(y1 < 0, na.rm=TRUE)
  
  ###########################################################################
  # as a placeholder, assign a negative values to cells with missing values #
  ###########################################################################
  
  # replace dropout = 1e30 with -1e30
  y1[,,O1+1][y1[,,O1+1] == 1e30] <- -1e30
  # fill NA cells with -1e30
  y1[is.na(y1)] <- -1e30
  
  ###########################################################
  # impute missing data using a most recent available entry #
  ###########################################################
  
  # fill dropout entries that are missing
  for (i in 1:dim(y1)[1]) {
    for (t in 2:length(uDay)) {
      if (y1[i,t-1,O1+1] < 0) {next}
      if (y1[i,t,O1+1] < 0) {y1[i,t,O1+1] <- y1[i,t-1,O1+1] } } } 
  
  ######################
  # delete switch back #
  ######################
  
  for (i in 1:dim(y1)[1]) {
    for (t in 2:length(uDay)) {
      if (y1[i,t-1,O1+1] == 1) {y1[i,t:length(uDay),O1+1] <- 1; break} } }
  
  # plot persons' drop out occurrence
  # c <- brewer.pal(8, 'Dark2')
  # plot(y1[1,,O1+1], type='l', ylim=c(0,1), lwd=1.5, xlab='day', ylab='dropout', main='dropout occurrence', yaxt='n')
  # for (i in 2:N){lines(y1[i,,O1+1], col=c[i%%8], lwd=1.5)}
  
  # replace -1e30 with NA
  y1[y1==-1e30] <- NA
  
  # there are less negative entries
  # sum(is.na(y1))
  
  # 117 people with no response at t=1
  summary(y1[,1,])
  # 36 people with no response at t=2
  summary(y1[,2,])
  # 33 people with no response at t=15
  summary(y1[,3,])
  # 42 people with no response at t=16
  summary(y1[,4,])
  
  # select data only after the second day
  y1 <- y1[,2:length(uDay),]
  
  # unique day (updated)
  uDay <- uDay[2:length(uDay)]
  
  # unique ID (updated)
  uID <- uID[complete.cases(y1[,1,])]
  
  # remove rows with missing values at t=1
  valid_ID <- complete.cases(y1[,1,])
  y1 <- y1[valid_ID,,]
  y2 <- y2[valid_ID,]
  # dim(y1)
  # dim(y2)
  
  # number of persons (updated)
  N <- dim(y1)[1]
  
  # number of days (updated)
  Nt <- dim(y1)[2]
  
  # impute NA value for y2
  summary(y2)
  y2[which(is.na(y2$abiMath)),]
  y2[y2$TIMMS==13,][y2[y2$TIMMS==13,]$totIQ==28,]
  y2[which(is.na(y2$abiMath)),]$abiMath <- mean(y2[y2$TIMMS==13,][y2[y2$TIMMS==13,]$totIQ==28,]$abiMath, na.rm=TRUE)
  
  DO <- y1[,,O1+1]
  
  # standardize
  y1_mean <- apply(y1[,1,1:O1], 2, mean)
  y1_sd <- apply(y1[,1,1:O1], 2, sd)
  y1_std <- array(NA, dim=dim(y1[,,1:O1]))
  for (var in 1:O1) {
    y1_std[,,var] <- (y1[,,var] - y1_mean[var]) / y1_sd[var]
  }
  y2_std <- apply(y2[,2:4], 2, scale)
  
  # save data as a list
  df <- list(y1=y1_std, y2=y2_std, O1=O1, O2=O2, L1=4, N=N, Nt=Nt, DO=DO)
  
  setwd('C:/Users/Methodenzentrum/Desktop/Kento/Empirical/202311011300')
  
  return(df)
}