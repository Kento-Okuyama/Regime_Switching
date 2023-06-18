###################################
# function for Adam optimization 
###################################
# input: initial parameter values and gradients 
# output: updated parameter values
adam3 <- function(loss, theta, m, v, lr=1e-3, beta1=.9, beta2=.999, epsilon=1e-8) {
  
  # initialize moment estimates
  if (is.null(m) || is.null(v)) {m <- v <- rep(0, length(torch_cat(theta)))}
  
  # backward propagation
  loss$backward() 
  
  with_no_grad({
    # store gradients
    grad <- torch_cat(list(theta$a1$grad, theta$a2$grad, theta$B1d$grad, theta$B2d$grad, theta$c1$grad, theta$c2$grad, theta$k1$grad, theta$k2$grad, theta$Lmd1v$grad, theta$Lmd2v$grad, theta$A1$grad, theta$A2$grad, theta$alpha1$grad, theta$beta1$grad, theta$Q1d$grad, theta$Q2d$grad, theta$R1d$grad, theta$R2d$grad))
    
    # update moment estimates
    m <- beta1 * m + (1 - beta1) * grad
    v <- beta2 * v + (1 - beta2) * grad**2
    
    # update bias corrected moment estimates
    m_hat <- m / (1 - beta1**iter)
    v_hat <- v / (1 - beta2**iter)
    
    # Update parameters using Adam update rule
    denom <- sqrt(v_hat) + epsilon
    denom[denom<epsilon] <- denom[denom<epsilon] + epsilon
    index <- 0
    theta$a1$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$a2$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$B1d$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$B2d$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$c1$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$c2$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$k1$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$k2$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$Lmd1v$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$Lmd2v$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$A1$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$A2$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$alpha1$sub_(lr * m_hat[(index+1):(index+1)] / denom[(index+1):(index+1)])
    index <- index + 1
    # theta$alpha2$sub_(lr * m_hat[(index+1):(index+1)] / denom[(index+1):(index+1)])
    # index <- index + 1
    theta$beta1$sub_(lr * m_hat[(index+1):(index+Nf12)] / denom[(index+1):(index+Nf12)])
    index <- index + Nf12
    # theta$beta2$sub_(lr * m_hat[(index+1):(index+Nf12)] / denom[(index+1):(index+Nf12)])
    # index <- index + Nf
    theta$Q1d$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$Q2d$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$R1d$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$R2d$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No })
  
  return(list(theta=theta, m=m, v=v))
}