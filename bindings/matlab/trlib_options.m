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

t = mex_trlib('t');

if t == 8
    TR = mex_trlib('i', int64(itmax));
else
    TR = mex_trlib('i', int32(itmax));
end

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

