% MIT License
%
% Copyright (c) 2016--2017 Andreas Potschka, Felix Lenders
%
% Permission is hereby granted, free of charge, to any person obtaining a copy
% of this software and associated documentation files (the "Software"), to deal
% in the Software without restriction, including without limitation the rights
% to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
% copies of the Software, and to permit persons to whom the Software is
% furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included in all
% copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
% AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
% SOFTWARE.

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
    prob.apply_M = @(v) v;
elseif isa(M, 'function_handle')
    prob.solve_M = M;
    prob.apply_M = @(v) cgs(M,v);
else
    prob.solve_M = @(v) M \ v;
    prob.apply_M = @(v) v;
end

if isa(Hess, 'function_handle')
    prob.apply_H = Hess;
else
    prob.apply_H = @(v) Hess * v;
end

prob.s = zeros(n, 1);
prob.g = grad;
prob.grad = grad;
prob.gm = zeros(n, 1);
prob.v = zeros(n, 1);
prob.p = zeros(n, 1);
prob.Hp = zeros(n, 1);
%prob.Q = zeros(n, itmax + 1);

