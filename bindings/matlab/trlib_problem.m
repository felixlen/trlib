function prob = trlib_problem(Hess, grad, M)
% prob = trlib_problem(Hess, grad, M)
%   Prepare problem structure of trust-region subproblem.
%   Inputs:
%     Hess: Hessian
%     grad: gradient
%     M: trust-region scalar product matrix
%   Output:
%     prob: Problem structure
%
%   This function is not needed if the convenience layer function trlib is used.
%   See also: trlib, trlib_options, trlib_solve, trlib_set_hotstart

% Authors: F. Lenders, A. Potschka
% Date: Mar 17, 2017

n = size(grad, 1);

if nargin < 3 || isempty(M)
    prob.solve_M = @(v) v;
elseif isa(M, 'function_handle')
    prob.solve_M = M;
else
    prob.solve_M = @(v) M \ v;
end

if isa(Hess, 'function_handle')
    prob.apply_H = Hess;
else
    prob.apply_H = @(v) Hess * v;
end

prob.s = zeros(n, 1);
prob.g = grad;
prob.gm = zeros(n, 1);
prob.v = zeros(n, 1);
prob.p = zeros(n, 1);
prob.Hp = zeros(n, 1);
%prob.Q = zeros(n, itmax + 1);

