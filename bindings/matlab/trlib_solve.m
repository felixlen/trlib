function [x, flag, prob, TR] = trlib_solve(prob, radius, TR)
% [x, flag, prob, TR] = trlib_solve(prob, radius, TR)
%   Solves trust region subproblem.
%   Inputs:
%     prob: structure with problem data
%     radius: trust-region radius
%     TR: structure with options and iteration data
%   Outputs:
%     x: solution vector
%     flag: return value, < 0 on failure (consult trlib C API)
%     prob: structure with problem data for reuse with hotstarts
%     TR: structure with options and iteration data for reuse with hotstarts
%
%   This function is not needed if the convenience layer function trlib is used.
%   See also: trlib, trlib_options, trlib_problem, trlib_set_hotstart

% Authors: F. Lenders, A. Potschka
% Date: Mar 17, 2017

n = size(prob.g, 1);

if ~isfield(prob, 'Q')
    prob.Q = zeros(n, TR.itmax + 1);
end

TR.radius = radius;
TR.g_dot_g = prob.g' * prob.g;

% reverse communication loop
while true
    TR = mex_trlib('s', TR);

    switch TR.action
    case 0 % TRLIB_CLA_TRIVIAL
    case 1 % TRLIB_CLA_INIT
        prob.v = prob.solve_M(prob.g);
        prob.p = -prob.v;
        prob.Hp = prob.apply_H(prob.p);
        TR.g_dot_g = prob.g' * prob.g;
        TR.v_dot_g = prob.v' * prob.g;
        TR.p_dot_Hp = prob.p' * prob.Hp;
        prob.Q(:,1) = (1/sqrt(TR.v_dot_g)) * prob.v;
    case 2 % TRLIB_CLA_RETRANSF
        h = TR.fwork(TR.h_pointer:TR.h_pointer+TR.iter);
        prob.s = prob.Q(:, 1:TR.iter+1) * h;
    case 3 % TRLIB_CLA_UPDATE_STATIO
        if TR.ityp == 1 % TRLIB_CLT_CG
            prob.s = prob.s + TR.flt(1) * prob.p;
        end
    case 4 % TRLIB_CLA_UPDATE_GRAD
        if TR.ityp == 1 % TRLIB_CLT_CG
            prob.Q(:, TR.iter+1) = TR.flt(2) * prob.v;
            prob.gm = prob.g;
            prob.g = prob.g + TR.flt(1) * prob.Hp;
            prob.v = prob.solve_M(prob.g);
            TR.g_dot_g = prob.g' * prob.g;
            TR.v_dot_g = prob.v' * prob.g;
        else
            prob.s = prob.Hp + [prob.g, prob.gm] * TR.flt(1:2);
            prob.gm = TR.flt(3) * prob.g;
            prob.g = prob.s;
            prob.v = prob.solve_M(prob.g);
            TR.g_dot_g = prob.g' * prob.g;
            TR.v_dot_g = prob.v' * prob.g;
        end
    case 5 % TRLIB_CLA_UPDATE_DIR
        if TR.ityp == 1 % TRLIB_CLT_CG
            prob.p = -prob.v + TR.flt(2) * prob.p;
            prob.Hp = prob.apply_H(prob.p);
            TR.p_dot_Hp = prob.p' * prob.Hp;
        else
            prob.p = TR.flt(1) * prob.v;
            lanczos_start_iter = TR.iwork(9);
            if lanczos_start_iter >= 0 && TR.iter - lanczos_start_iter > 100
                warning('trlib:warn_reorth', ...
                    'Many Lanczos iterations. Reorthogonalization possibly needed.')
            end
            prob.Hp = prob.apply_H(prob.p);
            TR.p_dot_Hp = prob.p' * prob.Hp;
            prob.Q(:, TR.iter+1) = prob.p;
        end
    case 6 % TRLIB_CLA_NEW_KRYLOV
        error('Invariant Krylov space detected. Not implemented yet.')
    case 7 % TRLIB_CLA_CONV_HARD
        Hsg = prob.apply_H(prob.s) + prob.grad;
        TR.v_dot_g = (Hsg+TR.flt(1)*prob.apply_M(prob.s))' * (prob.solve_M(Hsg) + TR.flt(1)*prob.s);
    otherwise
        error('Internal error')
    end
    
    if TR.krylov_min_retval < 10
        break
    end
end

TR.init = int64(1); % do not hotstart, unless trlib_set_hotstart is called
x = prob.s;
flag = TR.krylov_min_retval;

