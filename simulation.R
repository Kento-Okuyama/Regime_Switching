# for reproducibility
set.seed(42)

# number of subjects
N <- 100
# number of time responses
Nt <- 150

state <- array(NA, c(N,Nt))

for (i in 1:N){
  # start from state = 1
  state[i,1] <- 1
  for (t in 2:Nt){
    # no switch back from state = 2 to state = 1
    if (state[i,t-1] == 2) state[i,t] <- state[i,t-1]
    # P(S_{i,t} = 2 | S_{i,t-1} = 1) = 0.01
    else state[i,t] <- rbinom(n=1, size=1, prob=0.01) + 1
  }
}

ytMean1 <- 1
ytMean2 <- 5

ytSd1 <- 1e-2
ytSd2 <- 1e-2

# covariates
yt <- matrix(data=NA, nrow=N, ncol=Nt)
for (i in 1:N){
  # (Y_{i,t} | S_{i,t} = 1) ~ N(ytMean1, ytSd1)
  yt[i,1] <- rnorm(n=1, mean=ytMean1, sd=ytSd1)
  for (t in 2:Nt){
    # (Y_{i,t} | S_{i,t} = S_{i,t-1}) = Y_{i,t-1} 
    if (state[i,t] == state[i,t-1]) yt[i,t] <- yt[i,t-1] * log(t+1) / log(t)
    # (Y_{i,t} | S_{i,t} = 2) ~ N(ytMean2, ytSd2)
    else yt[i,t] <- rnorm(n=1, mean=ytMean2, sd=ytSd2) * log(t+1)
  }
}

# save data as list
(df <- list(state=state, yt=yt))

plot(yt[6,], type="l")
