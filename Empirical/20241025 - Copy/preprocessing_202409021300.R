preprocessing <- function(m, seed) {
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
  y1_2D <- data[, c('ID', 'day', cols_w)]
  DO_2D <- data[, c('ID', 'day', 'dropout')]
  y2 <- unique(data[, c('ID', cols_b)])
  
  # rename within-variables
  colnames(y1_2D) <- c('ID', 'day', 'Co1', 'Co2', 'understand1', 'understand2', 'AtF1', 'AtF2', 'NA1', 'NA5', 'NA9')
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
  y1 <- array(NA, c(length(uID), length(uDay), O1)) 
  DO <- array(NA, c(length(uID), length(uDay)))
  # 2: fill the entries of (y1) based on (y1_2D) 
  for (i in 1:length(uID)) {
    # take ith person's data from y2D
    y1_2D_i <- y1_2D[y1_2D$ID==uID[i],]
    DO_i <- DO_2D[DO_2D$ID==uID[i],]
    
    for (t in 1:length(uDay)) {
      # if more than one response for a day, average the responses
      if (nrow(y1_2D_i[y1_2D_i$day==uDay[t],]) > 0) {
        y1[i,t,] <- 
          colMeans(y1_2D_i[y1_2D_i$day==uDay[t],3:(O1+2)], na.rm=TRUE)
        DO[i,t] <- 
          mean(DO_i[DO_i$day==uDay[t],3], na.rm=TRUE) } } }
  
  # remove the persons with dropout = -1
  remID <- rowSums(DO==-1, na.rm=TRUE) > 0
  y1 <- y1[remID==FALSE,,]
  y2 <- y2[remID==FALSE,,]
  DO <- DO[remID==FALSE,]
  
  # unique ID (updated)
  uID <- uID[remID==FALSE]
  
  # there are dropout = 0.5 
  # table(DO)
  
  # substitute dropout = 0.5 with 1
  DO <- ceiling(DO)
  
  # there is no dropout = 0.5
  # table(DO)
  
  ###########################################################
  # impute missing data using a most recent available entry #
  ###########################################################
  
  # fill dropout entries that are missing
  for (i in 1:dim(DO)[1]) {
    for (t in 2:length(uDay)) {
      if (is.na(DO[i,t-1])) {next}
      if (is.na(DO[i,t])) {DO[i,t] <- DO[i,t-1] } } } 
  
  ######################
  # delete switch back #
  ######################
  
  for (i in 1:dim(DO)[1]) {
    for (t in 2:length(uDay)) {
      if (DO[i,t-1] == 1) {DO[i,t:length(uDay)] <- 1; break} } }
  
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
  DO <- DO[,2:length(uDay)]
  
  # unique day (updated)
  uDay <- uDay[2:length(uDay)]
  
  # unique ID (updated)
  uID <- uID[complete.cases(y1[,1,])]
  
  # remove rows with missing values at t=1
  valid_ID <- complete.cases(y1[,1,])
  y1 <- y1[valid_ID,,]
  y2 <- y2[valid_ID,]
  DO <- DO[valid_ID,]
  
  # number of persons (updated)
  N <- dim(y1)[1]
  
  # number of days (updated)
  Nt <- dim(y1)[2]
  
  # impute NA value for y2
  y2[which(is.na(y2$abiMath)),]
  y2[y2$TIMMS==13,][y2[y2$TIMMS==13,]$totIQ==28,]
  y2[which(is.na(y2$abiMath)),]$abiMath <- mean(y2[y2$TIMMS==13,][y2[y2$TIMMS==13,]$totIQ==28,]$abiMath, na.rm=TRUE)
  
  y1_2D_again <- matrix(NA, nrow=N*Nt, ncol=O1+2)
  colnames(y1_2D_again) <- colnames(y1_2D)
  
  start <- 1
  for (i in 1:N) {
    y1_2D_again[start:(start-1+Nt),] <- cbind(i, 1:Nt, y1[i,,])
    start <- start + Nt
  }
  
  y1_2D_again_mice <- mice(y1_2D_again, method='rf', 5, seed=seed)
  y1_2D_again_complete_m <- complete(y1_2D_again_mice, m)
  y1_2D_again <- y1_2D_again_complete_m
  
  # 1: crate an array of NAs 
  y1_again <- array(NA, c(N, Nt, O1)) 
  # 2: fill the entries of (y1_again) based on (y1_2D_again) 
  for (i in 1:N) {
    # take ith person's data from y1_2D_again
    y1_2D_again_i <- y1_2D_again[y1_2D_again$ID==i,]
    
    for (t in 1:Nt) {
      y1_again[i,t,] <- 
        as.numeric(y1_2D_again_i[y1_2D_again_i$day==t,(1:O1)+2]) } } 
  
  # standardize
  y1_again_mean <- apply(y1_again[,1,], 2, mean)
  y1_again_sd <- apply(y1_again[,1,], 2, sd)
  y1_again_std <- array(NA, dim=dim(y1_again))
  for (var in 1:O1) {
    y1_again_std[,,var] <- (y1_again[,,var] - y1_again_mean[var]) / y1_again_sd[var]
  }
  y2_std <- apply(y2[,2:4], 2, scale)
  
  # save data as a list
  df <- list(y1=y1_again_std, y2=y2_std, O1=O1, O2=O2, L1=4, N=N, Nt=Nt, DO=DO)
  return(df)
}
