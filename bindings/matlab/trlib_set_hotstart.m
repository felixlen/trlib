function TR = trlib_set_hotstart(TR)
% TR = trlib_set_hotstart(TR)
%   Prepare trlib for hotstarts to reuse previous computations when only the
%   trust-region radius changes.
%
%  Example:
%   >> Hess = diag(sparse(linspace(-1, 100, 10000)));
%   >> grad = ones(10000, 1);
%   >> prob = trlib_problem(Hess, grad);
%   >> TR = trlib_options(1000);
%   >> [x, flag, prob, TR] = trlib_solve(prob, 1, TR);
%   >> norm(x)
%   >> TR = trlib_set_hotstart(TR);
%   >> [x, flag, prob, TR] = trlib_solve(prob, 0.1, TR);
%   >> norm(x)
%
%   See also: trlib, trlib_options, trlib_problem, trlib_solve

% Authors: F. Lenders, A. Potschka
% Date: Mar 17, 2017

TR.init = int64(2); % TRLIB_CLS_HOTSTART

