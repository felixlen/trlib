function TR = trlib_options(itmax)
% TR = trlib_options(itmax)
%   Prepare options structure for solution of trust-region subproblem.
%   Input: 
%     itmax: Maximum number of iterations (required for memory allocation)
%   Output:
%     TR: Structure whose fields can be adjusted before calling trlib_solve
%         (consult trlib C API)
%
%   This function is not needed if the convenience layer function trlib is used.
%   See also: trlib, trlib_problem, trlib_solve, trlib_set_hotstart

% Authors: F. Lenders, A. Potschka
% Date: Mar 17, 2017

TR = mex_trlib('i', int64(itmax));

