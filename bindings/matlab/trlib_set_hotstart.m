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

