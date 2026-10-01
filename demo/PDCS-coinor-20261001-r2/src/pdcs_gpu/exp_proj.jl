function oracle_h(r::rpdhg_float, s::rpdhg_float, t::rpdhg_float, rho::rpdhg_float)
    exprho = exp(rho);
    expnegrho = exp(-rho);
    f  = ((rho-1)*r+s)*exprho -     (r-rho*s)*expnegrho - (rho*(rho-1)+1)*t;
    df =     (rho*r+s)*exprho + (r-(rho-1)*s)*expnegrho -       (2*rho-1)*t;
    return f, df
end

function oracle_h_diagonal(r::rpdhg_float, s::rpdhg_float, t::rpdhg_float, dr::rpdhg_float, dt::rpdhg_float, ds_squared::rpdhg_float, ds_div_dr::rpdhg_float, dsdr::rpdhg_float, rho::rpdhg_float)
    dtexprho = dt * exp(rho);
    dt_inv_expnegrho = exp(-rho) / dt;
    rho_minus_one = rho - 1
    ds_squared_r = ds_squared * r

    f = dtexprho * (s * ds_div_dr + r * rho_minus_one) - dt_inv_expnegrho * (ds_squared_r - dsdr * s * rho) - t * (ds_squared / dr + rho * rho_minus_one * dr)
    df = dtexprho * (r * rho + s * ds_div_dr) + dt_inv_expnegrho * (ds_squared_r - dsdr * s * rho_minus_one) - t * dr * (rho_minus_one + rho)
    return f, df
end

function oracle_f(r::rpdhg_float, s::rpdhg_float, t::rpdhg_float, rho::rpdhg_float)
    exprho = exp(rho);
    expnegrho = exp(-rho);
    f  = ((rho-1)*r+s)*exprho -     (r-rho*s)*expnegrho - (rho*(rho-1)+1)*t;
    return f
end

function oracle_f_diagonal(r::rpdhg_float, s::rpdhg_float, t::rpdhg_float, dr::rpdhg_float, dt::rpdhg_float, ds_squared::rpdhg_float, ds_div_dr::rpdhg_float, dsdr::rpdhg_float, rho::rpdhg_float)
    dtexprho = dt * exp(rho);
    dt_inv_expnegrho = exp(-rho) / dt;
    rho_minus_one = rho - 1
    ds_squared_r = ds_squared * r
    f = dtexprho * (s * ds_div_dr + r * rho_minus_one) - dt_inv_expnegrho * (ds_squared_r - dsdr * s * rho) - t * (ds_squared / dr + rho * rho_minus_one * dr)
    return f
end

function primal_heuristic(r0::rpdhg_float, s0::rpdhg_float,  t0::rpdhg_float)
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

function primal_heuristic_diagonal(r0::rpdhg_float, s0::rpdhg_float,  t0::rpdhg_float, a3::rpdhg_float,  dt_div_ds::rpdhg_float)
    vpr, vps, vpt = min(r0,0), 0.0, max(t0,0)
    dist = sqrt((vpt-t0)^2 + (vps-s0)^2 + (vpr-r0)^2)
    if s0 > 0.0
        tp = max(t0, dt_div_ds * s0 * exp(a3))
        newdist = tp - t0
        if newdist < dist
            vpr, vps, vpt = r0, s0, tp
            dist = newdist
        end
    end
    return vpr, vps, vpt, dist
end

function dual_heuristic(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float )
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

function dual_heuristic_diagonal(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, a4::rpdhg_float, dt_div_dr::rpdhg_float)
    # perspective boundary
    vdr, vds, vdt = 0.0, min(s0,0), min(t0,0)
    dist = sqrt((vdt-t0)^2 + (vds-s0)^2 + (vdr-r0)^2)

    # perspective interior
    if r0 > 0.0
        td = min(t0, -r0 * exp(-a4) / dt_div_dr)
        newdist = t0 - td
        if newdist < dist
            vdr, vds, vdt = r0, s0, td
            dist  = newdist
        end
    end
    return vdr, vds, vdt, dist
end



function ppsi(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float)
    # two expressions for the same to avoid catastrophic cancellation
    if (r0 > s0)
        psi = (r0-s0 + sqrt(r0^2 + s0^2 - r0*s0)) / r0
    else
        psi = -s0 / (r0-s0 - sqrt(r0^2 + s0^2 - r0*s0))
    end
    return ((psi-1)*r0 + s0)/(psi*(psi-1) + 1)
end

function ppsi_diagonal(r0::rpdhg_float, s0::rpdhg_float, dr::rpdhg_float, ds::rpdhg_float, dt::rpdhg_float, ds_squared::rpdhg_float, discriminant::rpdhg_float)
    if (abs(r0) < positive_zero)
        psi = 0.5
    else
        dr3r0 = dr^3 * r0
        temp1 = dr3r0 - dr^2 * ds * s0
        psi = (temp1 + dr^2 * sqrt(discriminant)) / dr3r0
    end
    temp2 = (s0 * ds + dr * r0 * (psi - 1)) / (ds_squared + dr^2 * (psi * (psi - 1)))
    return max(temp2, positive_zero) * dt
end

function dpsi(r0::rpdhg_float, s0::rpdhg_float)
    # two expressions for the same to avoid catastrophic cancellation
    if( s0 > r0 )
        psi = (r0 - sqrt(r0^2 + s0^2 - r0*s0)) / s0
    else
        psi = (r0 - s0) / (r0 + sqrt(r0^2 + s0^2 - r0*s0))
    end
    
    res = (r0 - psi*s0)/(psi*(psi-1) + 1)
    return res
end

function dpsi_diagonal(r0::rpdhg_float, s0::rpdhg_float, dr::rpdhg_float, ds::rpdhg_float, dt::rpdhg_float, ds_squared::rpdhg_float, discriminant::rpdhg_float, dsr0::rpdhg_float, drs0::rpdhg_float)
    if (abs(s0) < positive_zero)
        psi = 0.5
    else
        psi = (dsr0 - sqrt(discriminant)) / drs0
    end
    temp2 = (r0 - dr * psi * s0 / ds) / (1/dr + psi * (psi - 1) * dr / ds_squared)
    res = max(temp2, positive_zero) / dt
    return res
end

function pomega(rho::rpdhg_float)
    val = exp(rho)/(rho*(rho-1)+1)
    if rho < 2.0
        val = min(val, exp(2)/3)
    end
    return val
end

function pomega_diagonal(rho::rpdhg_float, dr::rpdhg_float, ds_squared::rpdhg_float, ds_div_dr_squared::rpdhg_float, dt::rpdhg_float)
    rho2 = (3 + sqrt(5 - 4 * ds_div_dr_squared)) / 2
    val = exp(rho) / (ds_squared + dr^2 * (rho * (rho - 1)))
    if rho < rho2
        val_temp = exp(rho2) / (ds_squared + dr^2 * (rho2 * (rho2 - 1)))
        val = min(val, val_temp)
    end
    val *= dt
    return val
end

function domega(rho::rpdhg_float)
    val = -exp(-rho)/(rho*(rho-1)+1)
    if rho > -1.0
        val = max(val, -exp(1)/3)
    end
   
    return val
end

function domega_diagonal(rho::rpdhg_float, dr::rpdhg_float, dr_inv::rpdhg_float, ds_squared::rpdhg_float, ds_div_dr_squared::rpdhg_float, dt::rpdhg_float)
    rho1 = (-1 - sqrt(5 - 4 * ds_div_dr_squared)) / 2
    dr_div_ds_squared = dr / ds_squared
    val = -exp(-rho) / (dr_inv + rho * (rho - 1) * dr_div_ds_squared)
    if rho > rho1
        val_temp = -exp(-rho1) / (dr_inv + rho1 * (rho1 - 1) * dr_div_ds_squared)
        val = max(val, val_temp)
    end
    val /= dt
    return val
end

function rho_bound(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, pdist::rpdhg_float, ddist::rpdhg_float)

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
        fl = oracle_f(r0, s0, t0, low)
        fu = oracle_f(r0, s0, t0, upr)

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

function rho_bound_diagonal(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, pdist::rpdhg_float, ddist::rpdhg_float, dr::rpdhg_float, ds::rpdhg_float, dt::rpdhg_float, ds_squared::rpdhg_float, c1::rpdhg_float, c2::rpdhg_float, a3::rpdhg_float, a4::rpdhg_float, ds_div_dr::rpdhg_float, ds_div_dr_squared::rpdhg_float, dr_inv::rpdhg_float, dsdr::rpdhg_float)
    baselow, baseupr = real(-Inf), real(Inf)
    low, upr = real(-Inf), real(Inf)

    Delta_p = sqrt(pdist^2 - min(s0, 0)^2)
    Delta_d = sqrt(ddist^2 - min(r0, 0)^2)

    if dr < 2 * ds
        if r0 > 0 && s0 == 0
            low = max(low, 1- c1 * c2)
        end
        if r0 == 0 && s0 > 0
            upr = min(upr, c2 / c1)
        end
        if r0 > 0 && s0 > 0
            low = max(low, a3)
            upr = min(upr, a4)
        end
        if r0 > 0 && s0 < 0
            low = max(low, a3)
            low = max(low, a4)
        end
        if r0 < 0 && s0 > 0
            upr = min(upr, a3)
            upr = min(upr, a4)
        end
    end

    if (dr > 2 * ds)
        discriminant = sqrt(1 - 4 * ds_div_dr_squared)
        a1 = (1 - discriminant) / 2
        a2 = (1 + discriminant) / 2
        if r0 == 0 && s0 > 0
            upr = min(upr, a1)
            if upr < low
                low = real(-Inf)
            end
        end
        if r0 == 0 && s0 < 0
            low = max(low, a1)
            upr = min(upr, a2)
            upr = min(upr, a3)
        end
        if s0 == 0 && r0 < 0 
            low = max(low, a1)
            low = max(low, a4)
            upr = min(upr, a2)
        end
        if s0 == 0 && r0 > 0
            low = max(low, a2)
        end
        if r0 > 0 && s0 > 0
            low = max(low, a1)
            low = max(low, a3)
            upr = min(upr, a2)
            upr = min(upr, a4)
        end

        if r0 < 0 && s0 > 0
            if a1 < a3 && a3 > a4
                fa1 = oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, a1)
                if fa1 < 0
                    upr = min(upr, a2)
                    low = max(low, a3)
                else
                    upr = min(upr, a1)
                end
            end
            if (a1 < a3 && a3 < a4) || (a1 > a3 && a3 < a4 && a1 < a4)
                fa1 = oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, a1)
                if a1 < 0
                    upr = min(upr, a2)
                    low = max(low, a4)
                else
                    upr = min(upr, a1)
                end
            end
            if (a1 > a3 && a3 > a4) || (a1 > a3 && a3 < a4 && a1 > a4)
                upr = min(upr, a2)
            end
        end

        if r0 > 0 && s0 < 0
            if a4 < a2 && a2 < a3
                fa2 = oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, a2)
                if fa2 > 0
                    upr = min(upr, a4)
                    low = max(low, a1)
                else
                    low = max(low, a2)
                end
            end
            if a2 < a3 && a3 < a4 || a2 < a4 && a4 < a3
                low = max(low, a1)
            end
            if a3 < a2 && a2 < a4 || (a3 < a4 && a4 < a2)
                fa2 = oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, a2)
                if fa2 > 0
                    upr = min(upr, a3)
                    low = max(low, a1)
                else
                    low = max(low, a2)
                    if upr < low
                        upr = real(Inf)
                    end
                end 
            end
            if a4 < a3 && a3 < a2
                fa2 = oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, a2)
                if fa2 > 0
                    upr = min(upr, a4)
                    low = max(low, a1)
                else
                    low = max(low, a2)
                    if upr < low
                        upr = real(Inf)
                    end
                end
            end
        end
        
    end
    
    if t0 > 0
        sds = s0 * ds
        rds = r0 * ds
        rdr = r0 * dr
        discriminant = (sds^2 + rds^2 - sds * rdr)
        if discriminant > 0 
            c = ppsi_diagonal(r0, s0, dr, ds, dt, ds_squared, discriminant)
            curbnd = log(t0 / c)
            # @info ("c: ", c, " curbnd: ", curbnd)
            if curbnd < upr
                low = max(low, curbnd)
            end
        end
    end

    if t0 < 0
        rds = r0 * ds
        sdr = s0 * dr
        sds = s0 * ds
        discriminant = (rds^2 + sds^2 - rds * sdr)
        if discriminant > 0
            c = dpsi_diagonal(r0, s0, dr, ds, dt, ds_squared, discriminant, rds, sdr)
            curbnd = -log(-t0 / c)
            if curbnd > low
                upr = min(upr, curbnd)
            end
        end
    end
    if (r0 > 0) && (dr < 2 * ds) && (dr > 2 * ds / sqrt(5))
        baselow = a4
        low = max(low, baselow)

        tpu = max(1e-12, min(Delta_d, Delta_p + t0))
        palpha = low
        # @info ("ds_div_dr_squared: ", ds_div_dr_squared)
        c = pomega_diagonal(palpha, dr, ds_squared, ds_div_dr_squared, dt)
        curbnd = max(palpha, baselow + tpu / (r0 * c * dr) )
        if curbnd > low
            upr = min(upr, curbnd)
        end
        # @info ("2upr: ", upr)
    end

    if (s0 > 0) && (dr < 2 * ds) && (dr > 2 * ds / sqrt(5))
        baseupr = a3
        # @info ("baseupr: ", baseupr)
        # @info ("upr: ", upr)
        upr = min(upr, baseupr)

        tdl = -max(1e-12, min(Delta_p, Delta_d - t0))
        dalpha = upr
        # @info ("dalpha: ", dalpha)
        c = domega_diagonal(dalpha, dr, dr_inv, ds_squared, ds_div_dr_squared, dt)
        # @info ("c: ", c)
        curbnd = min(dalpha, baseupr - tdl * c2 / (c * s0))
        # @info ("2low: ", low)
        # @info ("2curbnd: ", curbnd)
        if curbnd < upr
            low = max(low, curbnd)
        end
    end


    # Guarantee valid bracket
    low, upr = min(low, upr), max(low, upr)
    if !isfinite(low)
        low = -1.0
        while oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, low) > 0
            upr = min(upr, low)
            low *= 2.0
        end
    end
    if !isfinite(upr)
        upr = 1.0
        while oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, upr) < 0
            low = max(low, upr)
            upr *= 2.0
        end
    end
    if low != upr
        fl = oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, low)
        fu = oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, upr)
        if !(fl * fu < 0)
            if (abs(fl) < abs(fu) || isnan(fl))
                upr = low
            else
                low = upr
            end
        end
    end
    return low, upr
end

function rootsearch_bn(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rhol::rpdhg_float, rhoh::rpdhg_float, rho0::rpdhg_float)
    rho = 0.0;
    @assert rhol < rhoh
    while true
        f = oracle_f(r0, s0, t0, rho0);
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

function newton_rootsearch(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rhol::rpdhg_float, rhoh::rpdhg_float, rho0::rpdhg_float, max_iter::Integer = 20, tol = 1e-10)
    converged = false;
    rho = rho0;
    LODAMP = 0.05
    HIDAMP = 0.95
    for i = 1:max_iter
        f, df = oracle_h(r0, s0, t0, rho0)
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
        return rootsearch_bn(r0, s0, t0, rhol, rhoh, rho0);
    end
end

function rootsearch_bn_diagonal(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rhol::rpdhg_float, rhoh::rpdhg_float, rho0::rpdhg_float, dr::rpdhg_float, dt::rpdhg_float, ds_squared::rpdhg_float, ds_div_dr::rpdhg_float, dsdr::rpdhg_float)
    rho = 0.5*(rhol + rhoh)
    @assert rhol < rhoh
    while true
        f = oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, rho0);
        
        if( f < 0.0 )
            rhol = rho0;
        else
            rhoh = rho0;
        end

        rho = 0.5*(rhol + rhoh)

        if( abs(rho - rho0) <= positive_zero*max(1.,abs(rho)) || rho==rhol || rho==rhoh || abs(f) <= 1e-10)
            break;
        end

        rho0 = rho;
    end

    return rho;
end

function newton_rootsearch_diagonal(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rhol::rpdhg_float, rhoh::rpdhg_float, rho0::rpdhg_float, dr::rpdhg_float, dt::rpdhg_float, ds_squared::rpdhg_float, ds_div_dr::rpdhg_float, dsdr::rpdhg_float, max_iter::Integer = 20, tol = 1e-10)
    converged = false;
    rho = rho0;
    LODAMP = 0.05
    HIDAMP = 0.95
    for i = 1:max_iter
        f, df = oracle_h_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, rho0)
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
        if(abs(f) <= tol)
            converged = true;
            break;
        end

        if( rho >= rhoh )
            rho0 = min(LODAMP*rho0+HIDAMP*rhoh, rhoh);
        elseif ( rho<=rhol )
            rho0 = max(LODAMP*rho0+HIDAMP*rhol, rhol);
        else
            rho0 = rho;
        end
    end # end for
    if (converged)
        return max(rhol, min(rhoh, rho));
    else
        return rootsearch_bn_diagonal(r0, s0, t0, rhol, rhoh, rho0, dr, dt, ds_squared, ds_div_dr, dsdr);
    end
end

function projsol_primalexpcone(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rho::rpdhg_float)
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

function projsol_primalexpcone_diagonal(r0::rpdhg_float, s0::rpdhg_float, t0::rpdhg_float, rho::rpdhg_float, ds::rpdhg_float, dr::rpdhg_float, dt::rpdhg_float)
    linrho = ((rho-1) * r0 + s0 * ds / dr)
    exprho = exp(rho)
    if isfinite(exprho)
        quadrho=rho*(rho-1) * dr + ds^2 / dr
        temp = linrho/quadrho
        vpr = dr * rho * temp
        vps = ds * temp
        vpt = dt * exprho * temp
        dist = sqrt((vpt-t0)^2 + (vps-s0)^2 + (vpr-r0)^2)
    else
        vpr = 0
        vps = 0
        vpt = Inf
        dist = Inf
    end
    return vpr, vps, vpt, dist
end

function projsol_dualexpcone(r0::rpdhg_float, s0::rpdhg_float, rho::rpdhg_float, dr::rpdhg_float, ds::rpdhg_float, dt::rpdhg_float)
    rd = (ds^2 * r0 - ds * dr * rho * s0) / (ds^2 / dr + rho * (rho-1) * dr)
    rdstar = rd / dr
    sd = (1 - rho) * rd / ds
    td = -1 / dt * exp(-rho) * rd
    return rdstar, sd, td
end

function exponent_proj!(v::CuArray, tol = 1e-10)
    """
    exponent_proj!: projects the primal solution onto the exponential cone.
        min ||v - v0||_2 s.t. v in Kexp
    """
    # heuristic solution
    r0 = v[1]; s0 = v[2]; t0 = v[3];
    vpr, vps, vpt, pdist = primal_heuristic(r0, s0, t0)
    vdr, vds, vdt, ddist = dual_heuristic(r0, s0, t0)

    min_dist = min(pdist, ddist)
    inf_norm_vp_vd = -1;
    inf_norm_vp_vd = max(abs(vpr + vdr - r0), inf_norm_vp_vd)
    inf_norm_vp_vd = max(abs(vps + vds - s0), inf_norm_vp_vd)
    inf_norm_vp_vd = max(abs(vpt + vdt - t0), inf_norm_vp_vd)
    dot_vp_vd = vpr*vdr + vps*vds + vpt*vdt

    if !((s0<=0 && r0 <= 0) || min_dist <= tol || (inf_norm_vp_vd <= tol && dot_vp_vd <= tol))
        rho_l, rho_h = rho_bound(r0, s0, t0, pdist, ddist)
        rho = newton_rootsearch(r0, s0, t0, rho_l, rho_h, 0.5*(rho_l+rho_h))
        rtmp, stmp, ttmp, pdist1 = projsol_primalexpcone(r0, s0, t0, rho)
        if (pdist1 <= pdist)
            vpr, vps, vpt = rtmp, stmp, ttmp
        end
    end # not three special cases and not heuristic solution
    v[1], v[2], v[3] = vpr, vps, vpt
    return
end

function exponent_proj_diagonal!(v::CuArray, D::CuArray, tol = 1e-10)
    """
    exponent_proj_diagonal!: projects the primal solution onto the exponential cone.
        min ||v - v0||_2 s.t. D^{-1} v in Kexp
    """
    r0 = v[1]; s0 = v[2]; t0 = v[3];
    dr = D[1]; ds = D[2]; dt = D[3];
    ds_squared = ds^2; dr_inv = 1/dr;
    dt_div_ds = dt / ds
    ds_div_dr = ds / dr
    ds_div_dr_squared = ds_div_dr^2
    c1 = s0 / r0
    c2 = ds / dr
    a3 = c2 / c1
    a4 = 1 - c1 * c2
    ds_div_dr = ds / dr
    ds_div_dr_squared = ds_div_dr^2 
    dr_inv = 1 / dr
    dt_inv = 1 / dt
    dsdr = ds * dr
    dt_div_ds = dt / ds
    dt_div_dr = dt / dr
    # heuristic solution
    vpr, vps, vpt, pdist = primal_heuristic_diagonal(r0, s0, t0, a3, dt_div_ds)
    vdr, vds, vdt, ddist = dual_heuristic_diagonal(r0, s0, t0, a4, dt_div_dr)

    min_dist = min(pdist, ddist)
    inf_norm_vp_vd = -1;
    inf_norm_vp_vd = max(abs(vpr + vdr - r0), inf_norm_vp_vd)
    inf_norm_vp_vd = max(abs(vps + vds - s0), inf_norm_vp_vd)
    inf_norm_vp_vd = max(abs(vpt + vdt - t0), inf_norm_vp_vd)
    dot_vp_vd = vpr*vdr + vps*vds + vpt*vdt

    if !((s0<=0 && r0 <= 0) || min_dist <= tol || (inf_norm_vp_vd <= tol && dot_vp_vd <= tol))
        rho_l, rho_h = rho_bound_diagonal(r0, s0, t0, pdist, ddist, dr, ds, dt, ds_squared, c1, c2, a3, a4, ds_div_dr, ds_div_dr_squared, dr_inv, dsdr)
        rho = newton_rootsearch_diagonal(r0, s0, t0, rho_l, rho_h, 0.5*(rho_l+rho_h), dr, dt, ds_squared, ds_div_dr, dsdr)
        rtmp, stmp, ttmp, pdist1 = projsol_primalexpcone_diagonal(r0, s0, t0, rho, ds, dr, dt)
        rdstar, sdstar, tdstar = projsol_dualexpcone(r0 , s0, rho, dr, ds, dt)
        if (pdist1 <= pdist)
            vpr, vps, vpt = rtmp, stmp, ttmp
        end
    end # not three special cases and not heuristic solution
    v[1], v[2], v[3] = vpr, vps, vpt
    return
end

function dualExponent_proj_diagonal!(v::CuArray, D::CuArray, temp::CuArray)
    """
    dual_exponent_proj_diagonal!: projects the dual solution onto the exponential cone.
        min ||v - v0||_2 s.t. D v in Kexp^*
    """
    temp .= -v
    exponent_proj_diagonal!(temp, D)
    v .+= temp
    return
end

function dualExponent_proj!(v::CuArray)
    """
    dual_exponent_proj!: projects the dual solution onto the exponential cone.
        min ||v - v0||_2 s.t. v in Kexp^*
    """
    vCopy = deepcopy(v)
    vCopy .*= -1
    exponent_proj!(vCopy)
    v .+= vCopy
    return
end