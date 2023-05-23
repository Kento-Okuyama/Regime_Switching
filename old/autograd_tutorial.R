library(torch)
x <- torch_ones(2,2, requires_grad = TRUE)
y <- x$mean()
y$grad_fn
y$backward()
x$grad

x1 <- torch_ones(2,2, requires_grad = TRUE)
x2 <- torch_tensor(1.1, requires_grad = TRUE)
y <- x1 * (x2 + 2)
y$retain_grad()
z <- y$pow(2) * 3
z$retain_grad()
out <- z$mean()

# how to compute the gradient for mean, the last operation executed
out$grad_fn
# how to compute the gradient for the multiplication by 3 in z = y$pow(2) * 3
out$grad_fn$next_functions
# how to compute the gradient for pow in z = y.pow(2) * 3
out$grad_fn$next_functions[[1]]$next_functions
# how to compute the gradient for the multiplication in y = x * (x + 2)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions
# how to compute the gradient for the two branches of y = x * (x + 2),
# where the left branch is a leaf node (AccumulateGrad for x1)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions[[1]]$next_functions
# here we arrive at the other leaf node (AccumulateGrad for x2)
out$grad_fn$next_functions[[1]]$next_functions[[1]]$next_functions[[1]]$next_functions[[2]]$next_functions

out$backward()
z$grad
y$grad
x2$grad
x1$grad
