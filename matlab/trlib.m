function [x, flag, prob, TR] = trlib(Hess, grad, radius, M, itmax)
% [x, flag, prob, TR] = trlib(Hess, grad, radius, M, itmax)
%   Solves trust region subproblem.
%   Inputs:
%     Hess: Hessian
%     grad: gradient
%     radius: trust-region radius
%     M: trust-region scalar product matrix
%     itmax: maximum number of iterations
%   Outputs:
%     x: solution vector
%     flag: return value, < 0 on failure (consult trlib C API)
%     prob: structure with problem data for reuse with hotstarts
%     TR: structure with options and iteration data for reuse with hotstarts
%
%   Example:
%   >> Hess = diag(sparse(linspace(-1, 100, 10000)));
%   >> grad = ones(10000, 1);
%   >> [x, flag] = trlib(Hess, grad, 0.1);
%   >> norm(x)
%
%   This function is a convenience layer function.
%   See also: trlib_options, trlib_problem, trlib_solve, trlib_set_hotstart

% Authors: F. Lenders, A. Potschka
% Date: Mar 17, 2017

n = size(grad, 1);

if nargin < 5
    itmax = n;
end

if nargin < 4
    M = [];
end

TR = trlib_options(itmax);
prob = trlib_problem(Hess, grad, M);
[x, flag, prob, TR] = trlib_solve(prob, radius, TR);

