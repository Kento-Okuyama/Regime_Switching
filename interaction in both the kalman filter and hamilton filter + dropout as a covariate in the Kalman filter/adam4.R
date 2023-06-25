###################################
# function for Adam optimization 
###################################
# input: initial parameter values and gradients 
# output: updated parameter values
adam4 <- function(loss, theta, m, v, lr=1e-2, beta1=.9, beta2=.999, epsilon=1e-8) {
  
  # initialize moment estimates
  if (is.null(m) || is.null(v)) {m <- v <- rep(0, length(torch_cat(theta)))}
  
  # backward propagation
  loss$backward() 
  
  with_no_grad({
    # store gradients
    grad <- torch_cat(list(theta$a1$grad, theta$a2$grad, theta$B1d$grad, theta$B2d$grad, theta$C1d$grad, theta$C2d$grad, theta$D1$grad, theta$D2$grad, theta$k1$grad, theta$k2$grad, theta$Lmd1v$grad, theta$Lmd2v$grad, theta$Omega1v$grad, theta$Omega2v$grad, theta$M1$grad, theta$M2$grad, theta$alpha1$grad, theta$alpha2$grad, theta$beta1$grad, theta$beta2$grad, theta$gamma1$grad, theta$gamma2$grad, theta$rho1$grad, theta$rho2$grad, theta$Q1d$grad, theta$Q2d$grad, theta$R1d$grad, theta$R2d$grad))
    
    # update moment estimates
    m <- beta1 * m + (1 - beta1) * grad
    v <- beta2 * v + (1 - beta2) * grad**2
    
    # update bias corrected moment estimates
    m_hat <- m / (1 - beta1**iter)
    v_hat <- v / (1 - beta2**iter)
    
    # Update parameters using Adam update rule
    denom <- sqrt(v_hat) + epsilon
    denom[denom < epsilon]$add_(epsilon)
    index <- 0
    theta$a1$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$a2$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$B1d$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$B2d$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$C1d$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$C2d$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$D1$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$D2$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$k1$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$k2$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$Lmd1v$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$Lmd2v$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$Omega1v$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$Omega2v$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$M1$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$M2$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$alpha1$sub_(lr * m_hat[(index+1):(index+1)] / denom[(index+1):(index+1)])
    index <- index + 1
    theta$alpha2$sub_(lr * m_hat[(index+1):(index+1)] / denom[(index+1):(index+1)])
    index <- index + 1
    theta$beta1$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$beta2$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$gamma1$sub_(lr * m_hat[(index+1):(index+1)] / denom[(index+1):(index+1)])
    index <- index + 1
    theta$gamma2$sub_(lr * m_hat[(index+1):(index+1)] / denom[(index+1):(index+1)])
    index <- index + 1
    theta$rho1$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$rho2$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$Q1d$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$Q2d$sub_(lr * m_hat[(index+1):(index+Nf1)] / denom[(index+1):(index+Nf1)])
    index <- index + Nf1
    theta$R1d$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1
    theta$R2d$sub_(lr * m_hat[(index+1):(index+No1)] / denom[(index+1):(index+No1)])
    index <- index + No1 })
  
  return(list(theta=theta, m=m, v=v))
}