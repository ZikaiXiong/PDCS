
function oracle_h_cpu(r::rpdhg_float, s::rpdhg_float, t::rpdhg_float, rho::rpdhg_float)
    exprho = exp(rho);
    expnegrho = exp(-rho);
    f  = ((rho-1)*r+s)*exprho -     (r-rho*s)*expnegrho - (rho*(rho-1)+1)*t;
    df =     (rho*r+s)*exprho + (r-(rho-1)*s)*expnegrho -       (2*rho-1)*t;
    return f, df
end

function projsol_primalexpcone_cpu(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rho::rpdhg_float)
    linrho = ((rho-1)*r0+s0)
    exprho = exp(rho)
    if (linrho>0) && isfinite(exprho)
        quadrho=rho*(rho-1)+1
        temp = linrho/quadrho
        vpr = rho * temp
        vps = temp
        vpt = exprho * temp

        dist = sqrt((vpt-t0)^2 + (vps-s0)^2 + (vpr-r0)^2)
    else
        vpr = 0
        vps = 0
        vpt = Inf
        dist = Inf
    end
    return vpr, vps, vpt, dist
end

function oracle_f_cpu(r::rpdhg_float, s::rpdhg_float, t::rpdhg_float, rho::rpdhg_float)
    exprho = exp(rho);
    expnegrho = exp(-rho);
    f  = ((rho-1)*r+s)*exprho -     (r-rho*s)*expnegrho - (rho*(rho-1)+1)*t;
    return f
end

function rootsearch_bn_cpu(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rhol::rpdhg_float, rhoh::rpdhg_float, rho0::rpdhg_float)
    rho = 0.0;
    @assert rhol < rhoh
    while true
        f = oracle_f_cpu(r0, s0, t0, rho0);
        if( f < 0.0 )
            rhol = rho0;
        else
            rhoh = rho0;
        end

        rho = 0.5*(rhol + rhoh)

        if( abs(rho - rho0) <= positive_zero*max(1.,abs(rho)) || rho==rhol || rho==rhoh )
            break;
        end

        rho0 = rho;
    end

    return rho;
end

function newton_rootsearch_cpu(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rhol::rpdhg_float, rhoh::rpdhg_float, rho0::rpdhg_float, max_iter::Integer = 20, tol = 1e-10)
    converged = false;
    rho = rho0;
    LODAMP = 0.05
    HIDAMP = 0.95
    for i = 1:max_iter
        f, df = oracle_h_cpu(r0, s0, t0, rho0)
        if( f < 0.0 )
            rhol = rho0;
        else
            rhoh = rho0;
        end
        if (rhoh <= rhol)
            converged = true;
            break;
        end
        if (isfinite(f) && df > tol)
            rho = rho0 - f/df;
        else
            break;
        end
        if( abs(rho - rho0) <= positive_zero*max(1., abs(rho)) )
            converged = true;
            break;
        end

        if( rho >= rhoh )
            rho0 = min(LODAMP*rho0 + HIDAMP*rhoh, rhoh);
        elseif ( rho <= rhol )
            rho0 = max(LODAMP*rho0 + HIDAMP*rhol, rhol);
        else
            rho0 = rho;
        end
    end # end for
    if (converged)
        return max(rhol, min(rhoh, rho));
    else
        return rootsearch_bn_cpu(r0, s0, t0, rhol, rhoh, rho0);
    end
end

function rho_bound_cpu(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, pdist::rpdhg_float, ddist::rpdhg_float)

    baselow, baseupr = real(-Inf), real(Inf)
    low, upr = real(-Inf), real(Inf)
    Delta_p = sqrt(pdist^2 - min(s0, 0)^2)
    Delta_d = sqrt(ddist^2 - min(r0, 0)^2)

    if t0 > 0
        curbnd = log(t0 / ppsi(r0, s0, t0))
        low = max(low, curbnd)
    end

    if t0 < 0
        curbnd = -log(-t0 / dpsi(r0, s0))
        upr = min(upr, curbnd)
    end
    if (r0 > 0)
        baselow = 1 - s0 / r0
        low = max(low, baselow)

        tpu = max(1e-12, min(Delta_d, Delta_p + t0))
        palpha = low
        curbnd = max(palpha, baselow + tpu / r0 / pomega(palpha))
        upr = min(upr, curbnd)
    end
    if (s0 > 0)
        baseupr = r0 / s0
        upr     = min(upr, baseupr)

        tdl    = -max(1e-12, min(Delta_p, Delta_d-t0))
        dalpha = upr
        curbnd = min(dalpha, baseupr - tdl/s0/domega(dalpha))
        low    = max(low, curbnd)
    end

    @assert baselow <= baseupr
    @assert isfinite(low)
    @assert isfinite(upr)

    low,upr = min(low, upr),max(low, upr)
    low,upr = clamp(low, baselow, baseupr),clamp(upr, baselow, baseupr)
    if low != upr
        fl = oracle_f_cpu(r0, s0, t0, low)
        fu = oracle_f_cpu(r0, s0, t0, upr)

        if !(fl * fu < 0)
            if (abs(fl) < abs(fu) || isnan(fl))
                upr = low;
            else
                low = upr;
            end
        end
    end
    return low, upr
end

function dual_heuristic_cpu(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float )
    vd1, vd2, vd3 =  0.0, min(s0,0), min(t0,0)
    dist = sqrt((vd1-r0)^2 + (vd2-s0)^2 + (vd3-t0)^2)

    # perspective interior
    if r0 > 0.0
        td = min(t0, -r0*exp(s0/r0-1))
        newdist = t0-td
        if newdist < dist
            vd1, vd2, vd3 = r0, s0, td
            dist  = newdist
        end
    end
    return vd1, vd2, vd3, dist
end

function primal_heuristic_cpu(r0::rpdhg_float, s0::rpdhg_float,  t0::rpdhg_float)
    # perspective boundary
    vpr, vps, vpt = min(r0,0), 0.0, max(t0,0)
    dist = sqrt((vpt-t0)^2 + (vps-s0)^2 + (vpr-r0)^2)

    # perspective interior
    if s0 > 0.0
        tp = max(t0, s0*exp(r0/s0))
        newdist = tp - t0
        if newdist < dist
            vpr, vps, vpt = r0, s0, tp
            dist = newdist
        end
    end
    return vpr, vps, vpt, dist
end

function exponent_proj_cpu!(v::AbstractVector{rpdhg_float}, tol = 1e-10)
    """
    exponent_proj!: projects the primal solution onto the exponential cone.
        min ||v - v0||_2 s.t. v in Kexp
    """
    # heuristic solution
    r0 = v[1]; s0 = v[2]; t0 = v[3];
    vpr, vps, vpt, pdist = primal_heuristic_cpu(r0, s0, t0)
    vdr, vds, vdt, ddist = dual_heuristic_cpu(r0, s0, t0)

    min_dist = min(pdist, ddist)
    inf_norm_vp_vd = -1;
    inf_norm_vp_vd = max(abs(vpr + vdr - r0), inf_norm_vp_vd)
    inf_norm_vp_vd = max(abs(vps + vds - s0), inf_norm_vp_vd)
    inf_norm_vp_vd = max(abs(vpt + vdt - t0), inf_norm_vp_vd)
    dot_vp_vd = vpr*vdr + vps*vds + vpt*vdt

    if !((s0<=0 && r0 <= 0) || min_dist <= tol || (inf_norm_vp_vd <= tol && dot_vp_vd <= tol))
        rho_l, rho_h = rho_bound_cpu(r0, s0, t0, pdist, ddist)
        rho = newton_rootsearch_cpu(r0, s0, t0, rho_l, rho_h, 0.5*(rho_l+rho_h))
        rtmp, stmp, ttmp, pdist1 = projsol_primalexpcone_cpu(r0, s0, t0, rho)
        if (pdist1 <= pdist)
            vpr, vps, vpt = rtmp, stmp, ttmp
        end
    end # not three special cases and not heuristic solution
    v[1], v[2], v[3] = vpr, vps, vpt
    return
end

function __precompile_gpu()
    basedim = Int64(3)
    n = 2 * basedim # number of variables
    m = 5 * basedim # number of constraints
    m_zero = 1 * basedim # number of zero constraints
    m_nonnegative = 1 * basedim # number of nonnegative constraints
    m_exp = m - m_zero - m_nonnegative # number of second-order cone constraints
    c = ones(n) * 10
    # sparse matrix with 10% nonzeros
    density = 1.0
    A = sprand(m, n, density)
    x_fea = rand(n)
    x_fea .= max.(x_fea, 0.0)
    b = A * x_fea
    bCopy = deepcopy(b)
    bCopy[1:m_zero] .= 0.0
    bCopy[m_zero+1:m_zero+m_nonnegative] .= max.(b[m_zero+1:m_zero+m_nonnegative], 0.0)
    for i in 0:(Int(m_exp / 3) - 1)
        exponent_proj_cpu!(@view(bCopy[m_zero+m_nonnegative + i * 3 + 1:m_zero+m_nonnegative+(i+1) * 3]))
    end
    b .-= bCopy

    model = Model(PDCS_GPU.Optimizer)
    set_optimizer_attribute(model, "time_limit_secs", 1000.0)
    set_optimizer_attribute(model, "verbose", 2)
    @variable(model, x[1:n] >= 0)
    @objective(model, Min, c' * x)
    # (A * x - b)[1:m_zero] == 0
    @constraint(model, (A * x - b)[1:m_zero] .== 0)
    # (A * x - b)[m_zero+1:m_zero+m_nonnegative] >= 0
    @constraint(model, (A * x - b)[m_zero+1:m_zero+m_nonnegative] .>= 0)
    # (A * x - b)[m_zero+m_nonnegative+1:end] in SOC
    for i in 0:(Int(m_exp / 3) - 1)
        @constraint(model, (A * x - b)[m_zero+m_nonnegative + i * 3 + 1:m_zero+m_nonnegative+(i+1) * 3] in MOI.ExponentialCone())
    end
    optimize!(model)

    G_gpu = CUDA.CUSPARSE.CuSparseMatrixCSR(A)
    sol_res = PDCS_GPU.rpdhg_gpu_solve_input_gpu_data(
        n = n,
        m = m,
        nb = n,
        c = CuArray(c),
        G = G_gpu,
        h = CuArray(b),
        mGzero = m_zero,
        mGnonnegative = m_nonnegative,
        socG = Vector{Integer}([]),
        rsocG = Vector{Integer}([]),
        expG = Int(m_exp / 3),
        dual_expG = 0,
        bl = CuArray(zeros(n)),
        bu = CuArray(ones(n) * Inf),
        soc_x = Vector{Integer}([]),
        rsoc_x = Vector{Integer}([]),
        exp_x = 0,
        dual_exp_x = 0,
        use_preconditioner = true,
        method = :average,
        print_freq = 2000,
        time_limit = 1000.0,
        use_adaptive_restart = true,
        use_adaptive_step_size_weight = true,
        use_resolving = true,
        use_accelerated = false,
        use_aggressive = true,
        verbose = 2,
        rel_tol = 1e-6,
        abs_tol = 1e-6,
        kkt_restart_freq = 2000,
        duality_gap_restart_freq = 2000,
        use_kkt_restart = false,
        use_duality_gap_restart = true,
        logfile_name = nothing,
        # max_outer_iter = 3,
        # max_inner_iter = 10,
    )
end


function __precompile_gpu_clean_pointer()
    _massive_block_proj_mod[] = nothing
    _massive_block_proj_kernel[] = nothing
    _moderate_block_proj_mod[] = nothing
    _moderate_block_proj_kernel[] = nothing
    _sufficient_block_proj_mod[] = nothing
    _sufficient_block_proj_kernel[] = nothing
    _reflection_update_mod[] = nothing
    _reflection_update_kernel[] = nothing
    _primal_update_mod[] = nothing
    _primal_update_kernel[] = nothing
    _dual_update_mod[] = nothing
    _dual_update_kernel[] = nothing
    _extrapolation_update_mod[] = nothing
    _extrapolation_update_kernel[] = nothing
    _calculate_diff_mod[] = nothing
    _calculate_diff_kernel[] = nothing
    _axpyz_mod[] = nothing
    _axpyz_kernel[] = nothing
    _average_seq_mod[] = nothing
    _average_seq_kernel[] = nothing
    _rescale_csr_mod[] = nothing
    _rescale_csr_kernel[] = nothing
    _replace_inf_mod[]    = nothing
    _replace_inf_kernel[] = nothing
    _max_abs_row_mod[] = nothing
    _max_abs_row_kernel[] = nothing
    _max_abs_col_mod[] = nothing
    _max_abs_col_kernel[] = nothing
    _alpha_norm_row_mod[] = nothing
    _alpha_norm_row_kernel[] = nothing
    _alpha_norm_col_mod[] = nothing
    _alpha_norm_col_kernel[] = nothing
    _get_row_index_mod[] = nothing
    _get_row_index_kernel[] = nothing
    _rescale_coo_mod[] = nothing
    _rescale_coo_kernel[] = nothing
    _max_abs_row_elementwise_mod[] = nothing
    _max_abs_row_elementwise_kernel[] = nothing
    _max_abs_col_elementwise_mod[] = nothing
    _max_abs_col_elementwise_kernel[] = nothing
    _alpha_norm_col_elementwise_mod[] = nothing
    _alpha_norm_col_elementwise_kernel[] = nothing
end