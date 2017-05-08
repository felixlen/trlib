Hess = diag(sparse(linspace(-1, 100, 100)));
grad = ones(100, 1);
[x, flag] = trlib(Hess, grad, 0.1);
norm(x)
