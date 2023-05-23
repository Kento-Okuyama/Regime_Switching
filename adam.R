###################################
# function for Adam optimization 
###################################
# input: initial parameter values and gradients 
# output: updated parameter values
adam <- function(theta, grad, iter, m, v, lr=3e-2, beta1=0.9, beta2=0.999, epsilon=1e-30) {
  # theta: parameters values
  # grad: gradient of the objective function with respect to the parameters at the current iteration
  # lr: learning rate
  # beta1, beta2: hyper-parameters controlling the exponential decay rates for the moment estimates
  # epsilon: small constant added to the denominator to avoid division by zero
  # iter: current iteration number
  # m, v: first and second moment estimates
  
  # initialize moment estimates
  if (is.null(m) || is.null(v)) {
    m <- rep(0, length(theta))
    v <- rep(0, length(theta))
  }
  if (as.numeric(torch_sum(torch_isnan(m))) > 0) {
    m[torch_isnan(m)] <- 0 }
  
  if (as.numeric(torch_sum(torch_isnan(v))) > 0) {
    v[torch_isnan(v)] <- 0 }
  
  # update moment estimates
  m <- beta1 * m + (1 - beta1) * grad
  v <- beta2 * v + (1 - beta2) * grad**2
  
  # update bias corrected moment estimates
  m_hat <- m / (1 - beta1**iter)
  v_hat <- v / (1 - beta2**iter)
  
  # Update parameters using Adam update rule
  
  compared <- (sqrt(v_hat) + epsilon) > epsilon
  denom <- torch_full(nparams, epsilon)
  if (as.numeric(torch_sum(compared)) > 0) {
    denom[compared] <- (sqrt(v_hat) + epsilon)[compared] }
  theta <- theta + lr * m_hat / denom
  
  return(list(theta=theta, m=m, v=v))
}