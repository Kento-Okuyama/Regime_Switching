###################################
# function for Adam optimization 
###################################
# input: initial parameter values and gradients 
# output: updated parameter values
adam <- function(loss, theta, m, v, lr=1e-3, beta1=.9, beta2=.999, epsilon=1e-8) {

  # initialize moment estimates
  if (is.null(m) || is.null(v)) {m <- v <- rep(0, length(torch_cat(theta)))}
  
  # backward propagation
  loss$backward(retain_graph=TRUE) # not working

  with_no_grad({
    # store gradients
    grad <- torch_cat(list(theta$a1$grad, theta$a2$grad, theta$B1d$grad, theta$B2d$grad, theta$k1$grad, theta$k2$grad, theta$Lmd1v$grad, theta$Lmd2v$grad, theta$alpha1$grad, theta$alpha2$grad, theta$beta1$grad, theta$beta2$grad, theta$Q1d$grad, theta$Q2d$grad, theta$R1d$grad, theta$R2d$grad))
    
    # update moment estimates
    m <- beta1 * m + (1 - beta1) * grad
    v <- beta2 * v + (1 - beta2) * grad**2
    
    # update bias corrected moment estimates
    m_hat <- m / (1 - beta1**iter)
    v_hat <- v / (1 - beta2**iter)
    
    # zero the gradients
    theta$a1$grad$zero_()
    theta$a2$grad$zero_() 
    theta$B1d$grad$zero_()
    theta$B2d$grad$zero_()
    theta$k1$grad$zero_()
    theta$k2$grad$zero_()
    theta$Lmd1v$grad$zero_()
    theta$Lmd2v$grad$zero_()
    theta$alpha1$grad$zero_()
    theta$alpha2$grad$zero_()
    theta$beta1$grad$zero_()
    theta$beta2$grad$zero_()
    theta$Q1d$grad$zero_()
    theta$Q2d$grad$zero_()
    theta$R1d$grad$zero_()
    theta$R2d$grad$zero_()
    
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
    theta$k1$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$k2$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$Lmd1v$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$Lmd2v$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$alpha1$sub_(lr * m_hat[(index+1):(index+1)] / denom[(index+1):(index+1)])
    index <- index + 1
    theta$alpha2$sub_(lr * m_hat[(index+1):(index+1)] / denom[(index+1):(index+1)])
    index <- index + 1
    theta$beta1$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$beta2$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$Q1d$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$Q2d$sub_(lr * m_hat[(index+1):(index+Nf)] / denom[(index+1):(index+Nf)])
    index <- index + Nf
    theta$R1d$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
    theta$R2d$sub_(lr * m_hat[(index+1):(index+No)] / denom[(index+1):(index+No)])
    index <- index + No
  })

  return(list(m=m, v=v))
}