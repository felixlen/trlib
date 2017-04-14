module trlib
    type trlib_data
        H 
        invM
        grad          :: Array{Float64, 1}
        itmax         :: Int64
        equality      :: Int64
        itmax_lanczos :: Int64
        tol_rel_i     :: Float64
        tol_abs_i     :: Float64
        tol_rel_b     :: Float64
        tol_abs_b     :: Float64
        zero          :: Float64
        obj_lb        :: Float64
        ctl_invariant :: Int64
        convexify     :: Int64
        earlyterm     :: Int64
        refine        :: Int64
        verbose       :: Int64
        h_pointer     :: Int64
        iwork         :: Array{Int64, 1}
        fwork         :: Array{Float64, 1}
        timing        :: Array{Int64, 1}
        init          :: Int64
        sol           :: Array{Float64, 1}
        g             :: Array{Float64, 1}
        v             :: Array{Float64, 1}
        gm            :: Array{Float64, 1}
        p             :: Array{Float64, 1}
        Hp            :: Array{Float64, 1}
        Q             :: Array{Float64, 2}
        ret           :: Int64
        obj           :: Float64
        lam           :: Float64
        trlib_data(H, invM, grad) = (itmax = min(round(Int, 2e8/size(grad,1)), 2*size(grad, 1));
                               iwork_size = Array{Clong}(1);
                               fwork_size = Array{Clong}(1);
                               h_pointer = Array{Clong}(1); 
                               ccall(
                                     (:trlib_krylov_memory_size, "libtrlib"),
                                     Clong,
                                     (Clong, Ptr{Clong}, Ptr{Clong}, Ptr{Clong}),
                                     itmax, iwork_size, fwork_size, h_pointer);
                               fwork = Array{Cdouble}(fwork_size[1]);
                               ccall(
                                     (:trlib_krylov_prepare_memory, "libtrlib"),
                                     Clong,
                                     (Clong, Ptr{Cdouble}),
                                     itmax, fwork);
                               new(H, invM, grad, itmax,
                                   0, 100, -2.0, 0.0, -3.0, 0.0, 2e-16, -1e20, 0, 1, 1, 1, 0,
                                   h_pointer[1], Array{Clong}(iwork_size[1]), fwork,
                                   Array{Clong}(ccall((:trlib_krylov_timing_size, "libtrlib"), Clong, ())),
                                   1, similar(grad), similar(grad), similar(grad),
                                   similar(grad), similar(grad), similar(grad),
                                   Array{Float64}(size(grad,1), itmax+1), 0, 0.0, 0.0 ) )
        trlib_data(H, grad) = (itmax = min(round(Int, 2e8/size(grad,1)), 2*size(grad, 1));
                               iwork_size = Array{Clong}(1);
                               fwork_size = Array{Clong}(1);
                               h_pointer = Array{Clong}(1); 
                               ccall(
                                     (:trlib_krylov_memory_size, "libtrlib"),
                                     Clong,
                                     (Clong, Ptr{Clong}, Ptr{Clong}, Ptr{Clong}),
                                     itmax, iwork_size, fwork_size, h_pointer);
                               fwork = Array{Cdouble}(fwork_size[1]);
                               ccall(
                                     (:trlib_krylov_prepare_memory, "libtrlib"),
                                     Clong,
                                     (Clong, Ptr{Cdouble}),
                                     itmax, fwork);
                               new(H, speye(size(grad,1)), grad, itmax,
                                   0, 100, -2.0, 0.0, -3.0, 0.0, 2e-16, -1e20, 0, 1, 1, 1, 0,
                                   h_pointer[1], Array{Clong}(iwork_size[1]), fwork,
                                   Array{Clong}(ccall((:trlib_krylov_timing_size, "libtrlib"), Clong, ())),
                                   1, similar(grad), similar(grad), similar(grad),
                                   similar(grad), similar(grad), similar(grad),
                                   Array{Float64}(size(grad,1), itmax+1), 0, 0.0, 0.0 ) )
    end
    
    function trlib_solve(TR::trlib_data, radius::Float64)
        g_dot_g = 0.0
        v_dot_g = 0.0
        p_dot_Hp = 0.0
        
        action = Array{Clong}(1)
        it     = Array{Clong}(1)
        ityp   = Array{Clong}(1)
        flt1   = Array{Cdouble}(1)
        flt2   = Array{Cdouble}(1)
        flt3   = Array{Cdouble}(1)
        
        while true
            TR.ret = ccall(
                          (:trlib_krylov_min, "libtrlib"),
                          Clong,
                          (Clong, Cdouble, Clong, Clong, Clong, Cdouble, Cdouble,
                           Cdouble, Cdouble, Cdouble, Cdouble, Clong, Clong, Clong,
                           Cdouble, Cdouble, Cdouble, Ptr{Clong}, Ptr{Cdouble},
                           Clong, Clong, Clong, Cstring, Ptr{Clong}, Ptr{Clong},
                           Ptr{Clong}, Ptr{Clong}, Ptr{Clong}, Ptr{Cdouble},
                           Ptr{Cdouble}, Ptr{Cdouble}),
                          TR.init, radius, TR.equality, TR.itmax, TR.itmax_lanczos,
                          TR.tol_rel_i, TR.tol_abs_i, TR.tol_rel_b, TR.tol_abs_b,
                          TR.zero, TR.obj_lb, TR.ctl_invariant, TR.convexify,
                          TR.earlyterm, g_dot_g, v_dot_g, p_dot_Hp, TR.iwork,
                          TR.fwork, TR.refine, 0, 0, "", C_NULL,
                          TR.timing, action, it, ityp, flt1, flt2, flt3)
            TR.init = 0
            if action[1] == 1
                TR.sol[:] = 0.0
                TR.gm[:] = 0.0
                TR.g = TR.grad
                TR.v = TR.invM * TR.g
                g_dot_g = dot(TR.g, TR.g)
                v_dot_g = dot(TR.v, TR.g)
                TR.p = -TR.v
                TR.Hp = TR.H*TR.p
                p_dot_Hp = dot(TR.p, TR.Hp)
                TR.Q[:,1] = TR.v/sqrt(v_dot_g)
            end
            if action[1] == 2
                TR.sol = TR.Q[:,1:1+it[1]] * TR.fwork[1+TR.h_pointer:1+TR.h_pointer+it[1]]
            end
            if action[1] == 3
                if ityp[1] == 1
                    TR.sol += flt1[1] * TR.p
                end
            end
            if action[1] == 4
                if ityp[1] == 1
                    TR.Q[:,1+it] = flt2[1] * TR.v
                    TR.gm = TR.g
                    TR.g += flt1[1]*TR.Hp
                end
                if ityp[1] == 2
                    TR.sol = TR.Hp + flt1[1] * TR.g + flt2[1] * TR.gm
                    TR.gm = flt3[1] * TR.g
                    TR.g = TR.sol
                end
                TR.v = TR.invM * TR.g
                g_dot_g = dot(TR.g, TR.g)
                v_dot_g = dot(TR.v, TR.g)
            end
            if action[1] == 5
                TR.p = flt1[1]*TR.v + flt2[1]*TR.p
                TR.Hp = TR.H*TR.p
                p_dot_Hp = dot(TR.p, TR.Hp)
                if ityp[1] == 2
                    TR.Q[:,1+it] = TR.p
                end
            end
            if action[1] == 8
                g_dot_g = .5 * dot(TR.sol, TR.H*TR.sol) + dot(TR.sol, TR.grad)
            end
            if TR.ret < 10
                break
            end
        end
        TR.obj = TR.fwork[9]
        TR.lam = TR.fwork[8]
        TR.init = 2
    end
end
