#include <math_constants.h>
#define positive_inf 1e32
#define negative_inf -1e32
// #define proj_abs_tol_exp 5e-16
// #define proj_rel_tol_exp 5e-16

__device__ void oracle_h(double *r, double *s, double *t, double *rho, double *f, double *df){
    double exprho = exp(rho[0]);
    double expnegrho = exp(-rho[0]);
    f[0]  = ((rho[0]-1) * r[0] + s[0])*exprho -     (r[0]-rho[0]*s[0])*expnegrho - (rho[0]*(rho[0]-1)+1)*t[0];
    df[0] =     (rho[0]*r[0]+s[0])*exprho + (r[0]-(rho[0]-1)*s[0])*expnegrho -       (2*rho[0]-1)*t[0];
}

__device__ void oracle_h_diagonal(double *r, double *s, double *t, double *dr, double *dt, double *ds_squared, double *ds_div_dr, double *dsdr, double *rho, double *f, double *df){
    double dtexprho = dt[0] * exp(rho[0]);
    double dt_inv_expnegrho = exp(-rho[0]) / dt[0];
    double rho_minus_one = rho[0] - 1;
    double ds_squared_r = ds_squared[0] * r[0];

    f[0] = dtexprho * (s[0] * ds_div_dr[0] + r[0] * rho_minus_one) - dt_inv_expnegrho * (ds_squared_r - dsdr[0] * s[0] * rho[0]) - t[0] * (ds_squared[0] / dr[0] + rho[0] * rho_minus_one * dr[0]);
    df[0] = dtexprho * (r[0] * rho[0] + s[0] * ds_div_dr[0]) + dt_inv_expnegrho * (ds_squared_r - dsdr[0] * s[0] * rho_minus_one) - t[0] * dr[0] * (rho_minus_one + rho[0]);
}

__device__ void oracle_f(double *r, double *s, double *t, double *rho, double *f){
    double exprho = exp(rho[0]);
    double expnegrho = exp(-rho[0]);
    f[0]  = ((rho[0]-1)*r[0]+s[0])*exprho -     (r[0]-rho[0]*s[0])*expnegrho - (rho[0]*(rho[0]-1)+1)*t[0];
}

__device__ void oracle_f_diagonal(double *r, double *s, double *t, double *dr, double *dt, double *ds_squared, double *ds_div_dr, double *dsdr, double *rho, double *f){
    // printf("cuda enter oracle_f_diagonal rho: %.20e\n", rho[0]);
    double dtexprho = dt[0] * exp(rho[0]);
    // printf("cuda dtexprho: %.20e\n", dtexprho);
    double dt_inv_expnegrho = exp(-rho[0]) / dt[0];
    // printf("cuda dt_inv_expnegrho: %.20e\n", dt_inv_expnegrho);
    double rho_minus_one = rho[0] - 1.0;
    // printf("cuda rho_minus_one: %.20e\n", rho_minus_one);
    double ds_squared_r = ds_squared[0] * r[0];
    // printf("cuda ds_squared_r: %.20e\n", ds_squared_r);
    f[0] = dtexprho * (s[0] * ds_div_dr[0] + r[0] * rho_minus_one) - dt_inv_expnegrho * (ds_squared_r - dsdr[0] * s[0] * rho[0]) - t[0] * (ds_squared[0] / dr[0] + rho[0] * rho_minus_one * dr[0]);
    // printf("cuda oracle_f_diagonal: %.20e\n", f[0]);
    // printf("--------------------------------\n");
}

__device__ void primal_heuristic(double *r, double *s, double *t, double *vpr, double *vps, double *vpt, double *dist){
    vpr[0] = fmin(r[0],0.0);
    vps[0] = 0.0;
    vpt[0] = fmax(t[0],0.0);
    dist[0] = sqrt((vpt[0]-t[0])*(vpt[0]-t[0]) + (vps[0]-s[0])*(vps[0]-s[0]) + (vpr[0]-r[0])*(vpr[0]-r[0]));
    // perspective interior
    if (s[0] > 0.0){
        double tp = fmax(t[0], s[0]*exp(r[0]/s[0]));
        double newdist = tp - t[0];
        if (newdist < dist[0]){
            vpr[0] = r[0];
            vps[0] = s[0];
            vpt[0] = tp;
            dist[0] = newdist;
        }
    }
}

__device__ void primal_heuristic_diagonal(double *r, double *s, double *t, double *a3, double *dt_div_ds, double *vpr, double *vps, double *vpt, double *dist){
    vpr[0] = fmin(r[0],0.0);
    vps[0] = 0.0;
    vpt[0] = fmax(t[0],0.0);
    dist[0] = sqrt((vpt[0]-t[0]) * (vpt[0]-t[0]) + (vps[0]-s[0]) * (vps[0]-s[0]) + (vpr[0]-r[0]) * (vpr[0]-r[0]));
    if (s[0] > 0.0){
        double tp = fmax(t[0], dt_div_ds[0] * s[0] * exp(a3[0]));
        double newdist = tp - t[0];
        if (newdist < dist[0]){
            vpr[0] = r[0];
            vps[0] = s[0];
            vpt[0] = tp;
            dist[0] = newdist;
        }
    }
}

__device__ void dual_heuristic(double *r, double *s, double *t, double *vd1, double *vd2, double *vd3, double *dist){
    vd1[0] = 0.0;
    vd2[0] = fmin(s[0],0.0);
    vd3[0] = fmin(t[0],0.0);
    dist[0] = sqrt((vd1[0]-r[0])*(vd1[0]-r[0]) + (vd2[0]-s[0])*(vd2[0]-s[0]) + (vd3[0]-t[0])*(vd3[0]-t[0]));

    // perspective interior
    if (r[0] > 0.0){
        double td = fmin(t[0], -r[0]*exp(s[0]/r[0]-1));
        double newdist = t[0]-td;
        if (newdist < dist[0]){
            vd1[0] = r[0];
            vd2[0] = s[0];
            vd3[0] = td;
            dist[0] = newdist;
        }
    }
}

__device__ void dual_heuristic_diagonal(double *r, double *s, double *t, double *a4, double *dt_div_dr, double *vdr, double *vds, double *vdt, double *dist){
    // perspective boundary
    vdr[0] = 0.0;
    vds[0] = fmin(s[0],0.0);
    vdt[0] = fmin(t[0],0.0);
    dist[0] = sqrt((vdt[0]-t[0])*(vdt[0]-t[0]) + (vds[0]-s[0])*(vds[0]-s[0]) + (vdr[0]-r[0])*(vdr[0]-r[0]));

    // perspective interior
    if (r[0] > 0.0){
        double td = fmin(t[0], -r[0] * exp(-a4[0]) / dt_div_dr[0]);
        double newdist = t[0] - td;
        if (newdist < dist[0]){
            vdr[0] = r[0];
            vds[0] = s[0];
            vdt[0] = td;
            dist[0] = newdist;
        }
    }
}



__device__ void ppsi(double *r, double *s, double *t, double *psi){
    // two expressions for the same to avoid catastrophic cancellation
    if (r[0] > s[0]){
        psi[0] = (r[0]-s[0] + sqrt(r[0]*r[0] + s[0]*s[0] - r[0]*s[0])) / r[0];
    }
    else{
        psi[0] = -s[0] / (r[0]-s[0] - sqrt(r[0]*r[0] + s[0]*s[0] - r[0]*s[0]));
    }
    psi[0] = ((psi[0]-1)*r[0] + s[0])/(psi[0]*(psi[0]-1) + 1);
}

__device__ void ppsi_diagonal(double *r, double *s, double *dr, double *ds, double *dt, double *ds_squared, double *discriminant, double *res){
    double psi = 0.0;
    if (abs(r[0]) < positive_zero){
        psi = 0.5;
    }
    else{
        double dr3r0 = dr[0] * dr[0] * dr[0] * r[0];
        double temp1 = dr3r0 - dr[0] * dr[0] * ds[0] * s[0];
        psi = (temp1 + dr[0] * dr[0] * sqrt(discriminant[0])) / dr3r0;
    }
    double temp2 = (s[0] * ds[0] + dr[0] * r[0] * (psi - 1)) / (ds_squared[0] + dr[0] * dr[0] * (psi * (psi - 1)));
    res[0] = fmax(temp2, positive_zero) * dt[0];
}

__device__ void dpsi(double *r, double *s, double *psi){
    // two expressions for the same to avoid catastrophic cancellation
    if( s[0] > r[0] ){
        psi[0] = (r[0] - sqrt(r[0]*r[0] + s[0]*s[0] - r[0]*s[0])) / s[0];
    }
    else{
        psi[0] = (r[0] - s[0]) / (r[0] + sqrt(r[0]*r[0] + s[0]*s[0] - r[0]*s[0]));
    }
    psi[0] = (r[0] - psi[0]*s[0])/(psi[0]*(psi[0]-1) + 1);
}

__device__ void dpsi_diagonal(double *r0, double *s0, double *dr, double *ds, double *dt, double *ds_squared, double *discriminant, double *dsr0, double *drs0, double *res){
    double psi = 0.0;
    if (abs(s0[0]) < positive_zero){
        psi = 0.5;
    }
    else{
        psi = (dsr0[0] - sqrt(discriminant[0])) / drs0[0];
    }
    double temp2 = (r0[0] - dr[0] * psi * s0[0] / ds[0]) / (1/dr[0] + psi * (psi - 1) * dr[0] / ds_squared[0]);
    res[0] = fmax(temp2, positive_zero) / dt[0];
}

__device__ void pomega(double *rho, double *val){
    val[0] = exp(rho[0])/(rho[0]*(rho[0]-1)+1);
    if (rho[0] < 2.0){
        val[0] = fmin(val[0], exp(2.0)/3.0);
    }
}

__device__ void pomega_diagonal(double *rho, double *val, double *ds_squared, double *dr_squared, double *dt, double *ds_div_dr_squared) {
    double rho2 = (3 + sqrt(5 - 4 * ds_div_dr_squared[0])) / 2;
    val[0] = exp(rho[0]) / (ds_squared[0] + dr_squared[0] * (rho[0] * (rho[0] - 1)));
    if (rho[0] < rho2) {
        double val_temp = exp(rho2) / (ds_squared[0] + dr_squared[0] * (rho2 * (rho2 - 1)));
        val[0] = fmin(val[0], val_temp);
    }
    val[0] *= dt[0];
}

__device__ void domega(double *rho, double *val){
    val[0] = -exp(-rho[0])/(rho[0]*(rho[0]-1)+1);
    if (rho[0] > -1.0){
        val[0] = fmax(val[0], -exp(1.0)/3.0);
    }
}

__device__ void domega_diagonal(double *rho, double *val, double *dr_inv, double *ds_squared, double *ds_div_dr_squared, double *dt){
    double rho1 = (-1 - sqrt(5 - 4 * ds_div_dr_squared[0])) / 2;
    double dr_div_ds_squared = 1.0 / (dr_inv[0] * ds_squared[0]);
    val[0] = -exp(-rho[0]) / (dr_inv[0] + rho[0] * (rho[0] - 1) * dr_div_ds_squared);
    if (rho[0] > rho1){
        double val_temp = -exp(-rho1) / (dr_inv[0] + rho1 * (rho1 - 1) * dr_div_ds_squared);
        val[0] = fmax(val[0], val_temp);
    }
    val[0] /= dt[0];
}

__device__ void rho_bound(double *r0, double *s0, double *t0, double *pdist, double *ddist, double *low, double *upr) {
    // Fix variable declarations - separate each variable
    double baselow = negative_inf;
    double baseupr = positive_inf;
    *low = negative_inf;
    *upr = positive_inf;
    double psi_val = 0.0;
    double omega_val = 0.0;
    
    // Fix pointer dereferencing for calculations
    double temp_s0 = fmin(s0[0], 0.0);
    double Delta_p = sqrt((pdist[0] * pdist[0]) - (temp_s0 * temp_s0));
    double temp_r0 = fmin(r0[0], 0.0);
    double Delta_d = sqrt((ddist[0] * ddist[0]) - (temp_r0 * temp_r0));

    // Fix function calls to pass pointers correctly
    if (*t0 > 0.0) {
        ppsi(r0, s0, t0, &psi_val);
        double curbnd = log(*t0 / psi_val);
        *low = fmax(*low, curbnd);
    }
    
    if (*t0 < 0.0) {
        dpsi(r0, s0, &psi_val);
        double curbnd = -log(-*t0 / psi_val);
        *upr = fmin(*upr, curbnd);
    }

    if (*r0 > 0.0) {
        baselow = 1.0 - *s0 / *r0;
        *low = fmax(*low, baselow);

        double tpu = fmax(1e-30, fmin(Delta_d, Delta_p + *t0));
        double palpha = *low;
        pomega(&palpha, &omega_val);
        double curbnd = fmax(palpha, baselow + tpu / (*r0 * omega_val));
        *upr = fmin(*upr, curbnd);
    }

    if (*s0 > 0.0) {
        baseupr = *r0 / *s0;
        *upr = fmin(*upr, baseupr);

        double tdl = -fmax(1e-30, fmin(Delta_p, Delta_d - *t0));
        double dalpha = *upr;
        domega(&dalpha, &omega_val);
        double curbnd = fmin(dalpha, baseupr - tdl / (*s0 * omega_val));
        *low = fmax(*low, curbnd);
    }

    // Final adjustments
    double temp_low = fmin(*low, *upr);
    double temp_upr = fmax(*low, *upr);
    *low = fmin(fmax(temp_low, baselow), baseupr);
    *upr = fmax(fmin(temp_upr, baseupr), baselow);

    if (*low != *upr) {
        double fl, fu;
        oracle_f(r0, s0, t0, low, &fl);
        oracle_f(r0, s0, t0, upr, &fu);

        if (fl * fu >= 0.0) {
            if (fabs(fl) < fabs(fu)) {
                *upr = *low;
            } else {
                *low = *upr;
            }
        }
    }
}

__device__ void rho_bound_diagonal(double *r0, double *s0, double *t0,
                                 double *pdist, double *ddist, double *dr,
                                double *ds, double *dt, double *ds_squared,
                                double *c1, double *c2, double *a3, double *a4,
                                double *ds_div_dr, double *ds_div_dr_squared,
                                double *dr_inv, double *dsdr, double *rho_l, double *rho_h) {
    // Initialize bounds
    double baselow = negative_inf;
    double baseupr = positive_inf;
    double low = negative_inf;
    double upr = positive_inf;

    // Calculate deltas
    double temp_s0 = fmin(s0[0], 0.0);
    double Delta_p = sqrt((pdist[0] * pdist[0]) - (temp_s0 * temp_s0));
    double temp_r0 = fmin(r0[0], 0.0);
    double Delta_d = sqrt((ddist[0] * ddist[0]) - (temp_r0 * temp_r0));

    // First case: dr < 2ds
    if (dr[0] < (2.0 * ds[0])) {
        if (r0[0] > 0.0 && s0[0] == 0.0) {
            low = fmax(low, 1.0 - c1[0] * c2[0]);
        }
        if (r0[0] == 0.0 && s0[0] > 0.0) {
            upr = fmin(upr, c2[0] / c1[0]); 
        }
        if (r0[0] > 0.0 && s0[0] > 0.0) {
            low = fmax(low, a3[0]);
            upr = fmin(upr, a4[0]);
        }
        if (r0[0] > 0.0 && s0[0] < 0.0) {
            low = fmax(low, a3[0]);
            low = fmax(low, a4[0]);
        }
        if (r0[0] < 0.0 && s0[0] > 0.0) {
            upr = fmin(upr, a3[0]);
            upr = fmin(upr, a4[0]);
        }
    }

    // Second case: dr > 2ds
    if (dr[0] > (2.0 * ds[0])) {
        double discriminant_val = sqrt(1.0 - 4.0 * ds_div_dr_squared[0]);
        double a1 = (1.0 - discriminant_val) / 2.0;
        double a2 = (1.0 + discriminant_val) / 2.0;

        if (r0[0] == 0.0 && s0[0] > 0.0) {
            upr = fmin(upr, a1);
            if (upr < low) {
                low = negative_inf;
            }
        }
        if (r0[0] == 0.0 && s0[0] < 0.0) {
            low = fmax(low, a1);
            upr = fmin(upr, a2);
            upr = fmin(upr, a3[0]);
        }
        if (s0[0] == 0.0 && r0[0] < 0.0) {
            low = fmax(low, a1);
            low = fmax(low, a4[0]);
            upr = fmin(upr, a2);
        }
        if (s0[0] == 0.0 && r0[0] > 0.0) {
            low = fmax(low, a2);
        }
        if (r0[0] > 0.0 && s0[0] > 0.0) {
            low = fmax(low, a1);
            low = fmax(low, a3[0]);
            upr = fmin(upr, a2);
            upr = fmin(upr, a4[0]);
        }
        double fa1 = 0.0;
        if (r0[0] < 0.0 && s0[0] > 0.0) {
            if (a1 < a3[0] && a3[0] > a4[0]) {
                oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &a1, &fa1);
                if (fa1 < 0.0) {
                    upr = fmin(upr, a2);
                    low = fmax(low, a3[0]);
                } else {
                    upr = fmin(upr, a1);
                }
            }
            
            if ((a1 < a3[0] && a3[0] < a4[0]) || (a1 > a3[0] && a3[0] < a4[0] && a1 < a4[0])) {
                oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &a1, &fa1);
                if (fa1 < 0.0) {
                    upr = fmin(upr, a2);
                    low = fmax(low, a4[0]);
                } else {
                    upr = fmin(upr, a1);
                }
            }
            if ((a1 > a3[0] && a3[0] > a4[0]) || (a1 > a3[0] && a3[0] < a4[0] && a1 > a4[0])) {
                upr = fmin(upr, a2);
            }
        }
        double fa2 = 0.0;
        if (r0[0] > 0.0 && s0[0] < 0.0) {
            if (a4[0] < a2 && a2 < a3[0]) {
                oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &a2, &fa2);
                if (fa2 > 0.0) {
                    upr = fmin(upr, a4[0]);
                    low = fmax(low, a1);
                } else {
                    low = fmax(low, a2);
                }
            }
            
            if ((a2 < a3[0] && a3[0] < a4[0]) || (a2 < a4[0] && a4[0] < a3[0])) {
                low = fmax(low, a1);
            }
            
            if ((a3[0] < a2 && a2 < a4[0]) || (a3[0] < a4[0] && a4[0] < a2)) {
                oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &a2, &fa2);
                if (fa2 > 0.0) {
                    upr = fmin(upr, a3[0]);
                    low = fmax(low, a1);
                } else {
                    low = fmax(low, a2);
                    if (upr < low) {
                        upr = positive_inf;
                    }
                }
            }
            
            if (a4[0] < a3[0] && a3[0] < a2) {
                oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &a2, &fa2);
                if (fa2 > 0.0) {
                    upr = fmin(upr, a4[0]);
                    low = fmax(low, a1);
                } else {
                    low = fmax(low, a2);
                    if (upr < low) {
                        upr = positive_inf;
                    }
                }
            }
        }
    }
    // Handle t0 > 0 case
    if (t0[0] > 0.0) {
        double sds = s0[0] * ds[0];
        double rds = r0[0] * ds[0];
        double rdr = r0[0] * dr[0];
        double c = 0.0;
        double discriminant_val = (sds * sds + rds * rds - sds * rdr);
        if (discriminant_val > 0.0) {
            ppsi_diagonal(r0, s0, dr, ds, dt, ds_squared, &discriminant_val, &c);
            double curbnd = log(t0[0] / c);
            if (curbnd < upr) {
                low = fmax(low, curbnd);
            }
        }
    }
    // Handle t0 < 0 case
    if (t0[0] < 0.0) {
        double rds = r0[0] * ds[0];
        double sdr = s0[0] * dr[0];
        double sds = s0[0] * ds[0];
        double c = 0.0;
        double discriminant_val = (rds * rds + sds * sds - rds * sdr);
        if (discriminant_val > 0.0) {
            dpsi_diagonal(r0, s0, dr, ds, dt, ds_squared, &discriminant_val, &rds, &sdr, &c);
            double curbnd = -log(-t0[0] / c);
            if (curbnd > low) {
                upr = fmin(upr, curbnd);
            }
        }
    }
    // Special case for r0 > 0
    if (r0[0] > 0.0 && dr[0] < (2.0 * ds[0]) && dr[0] > (2.0 * ds[0] / sqrt(5.0))) {
        baselow = a4[0];
        low = fmax(low, baselow);
        double c = 0.0;
        double tpu = fmax(1e-30, fmin(Delta_d, Delta_p + t0[0]));
        double palpha = low;
        double dr_squared = dr[0] * dr[0];
        pomega_diagonal(&palpha, &c, ds_squared, &dr_squared, dt, ds_div_dr_squared);
        double curbnd = fmax(palpha, baselow + tpu / (r0[0] * c * dr[0]));
        if (curbnd > low) {
            upr = fmin(upr, curbnd);
        }
    }
    // Special case for s0 > 0
    if (s0[0] > 0.0 && dr[0] < (2.0 * ds[0]) && dr[0] > (2.0 * ds[0] / sqrt(5.0))) {
        baseupr = a3[0];
        upr = fmin(upr, baseupr);
        double c = 0.0;
        double tdl = -fmax(1e-30, fmin(Delta_p, Delta_d - t0[0]));
        double dalpha = upr;
        domega_diagonal(&dalpha, &c, dr_inv, ds_squared, ds_div_dr_squared, dt);
        double curbnd = fmin(dalpha, baseupr - tdl * c2[0] / (c * s0[0]));
        if (curbnd < upr) {
            low = fmax(low, curbnd);
        }
    }
    // Guarantee valid bracket
    double temp = low;
    low = fmin(temp, upr);
    upr = fmax(temp, upr);
    // Handle infinite cases
    if (!isfinite(low)) {
        low = -1.0;
        double fl = 0.0;
        oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &low, &fl);
        while (fl > 0.0) {
            upr = fmin(upr, low);
            low *= 2.0;
            oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &low, &fl);
        }
    }
    
    if (!isfinite(upr)) {
        upr = 1.0;
        double fu = 0.0;
        oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &upr, &fu);
        while (fu < 0.0) {
            low = fmax(low, upr);
            upr *= 2.0;
            oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &upr, &fu);
        }
    }
    // Final check for equal bounds
    if (!(low == upr)) {
        double fl = 0.0;
        double fu = 0.0;
        oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &low, &fl);
        oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, &upr, &fu);
        if (!(fl * fu < 0.0)) {
            if (fabs(fl) < fabs(fu) || isnan(fl)) {
                upr = low;
            } else {
                low = upr;
            }
        }
    }
    *rho_l = low;
    *rho_h = upr;
}

__device__ void rootsearch_bn(double *r0, double *s0, double *t0, double *rhol, double *rhoh, double *rho0, double *rho, double abs_tol, double rel_tol){
    *rho = 0.0;
    double f = 0.0;
    int count = 0;
    while (true){
        oracle_f(r0, s0, t0, rho0, &f);
        if( f < 0.0 ){
            *rhol = *rho0;
        } else {
            *rhoh = *rho0;
        }
        *rho = 0.5*(*rhol + *rhoh);
        // if( fabs(*rho - *rho0) <= positive_zero*max(1.,abs(*rho)) || *rho==*rhol || *rho==*rhoh ){
        //     printf("cuda rho: %.20e, rho0: %.20e, rhol: %.20e, rhoh: %.20e, f: %.20e\n", *rho, *rho0, *rhol, *rhoh, f);
        //     break;
        // }
        if (fabs(f) <= abs_tol || fabs(*rhoh - *rhol) / fabs(1 + *rhoh + *rhol) <= rel_tol){
            break;
        }
        count++;
        if (count > MAX_ITER){
            break;
        }
        *rho0 = *rho;
    }
}

__device__ void newton_rootsearch(double *r0, double *s0, double *t0, double *rhol, double *rhoh, double *rho0, double *rho, double abs_tol, double rel_tol, int max_iter = 20) {
    // bool converged = false;
    const double LODAMP = 0.05;
    const double HIDAMP = 0.95;
    double f = 0.0;
    double df = 0.0;
    for (int i = 1; i <= max_iter; ++i) {
        oracle_h(r0, s0, t0, rho0, &f, &df);
        
        if (f < 0.0) {
            *rhol = *rho0;
        } else {
            *rhoh = *rho0;
        }
        
        if (*rhoh <= *rhol) {
            // converged = true;
            break;
        }
        
        if (isfinite(f) && df > abs_tol) {
            *rho = *rho0 - f/df;
        } else {
            break;
        }
        
        if (fabs(*rho - *rho0) <= abs_tol * fmax(1.0, fabs(*rho))) {
            // converged = true;
            break;
        }

        if (*rho >= *rhoh) {
            *rho0 = fmin(LODAMP * *rho0 + HIDAMP * *rhoh, *rhoh);
        } else if (*rho <= *rhol) {
            *rho0 = fmax(LODAMP * *rho0 + HIDAMP * *rhol, *rhol);
        } else {
            *rho0 = *rho;
        }
    }

    // if (converged) {
        *rho0 = fmax(*rhol, fmin(*rhoh, *rho));
    // } else {
        rootsearch_bn(r0, s0, t0, rhol, rhoh, rho0, rho, abs_tol, rel_tol);
    // }
}

__device__ void rootsearch_bn_diagonal(double *r0, double *s0, double *t0, double *rhol, double *rhoh, double *rho0, double *dr, double *dt, double *ds_squared, double *ds_div_dr, double *dsdr, double *rho, double abs_tol, double rel_tol){
    *rho = 0.5*(*rhol + *rhoh);
    *rho0 = *rho;
    double f = 0.0;
    int count = 0;
    while (true){
        oracle_f_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, rho0, &f);
        if (f < 0.0) {
            *rhol = *rho0;
        } else {
            *rhoh = *rho0;
        }
        *rho = 0.5*(*rhol + *rhoh);
        // printf("cuda enter loop rootsearch_bn_diagonal rho: %.20e\n", *rho);
        // if (fabs(*rho - *rho0) <= positive_zero*fmax(1.0, fabs(*rho)) || *rho == *rhol || *rho == *rhoh || fabs(f) <= proj_abs_tol_exp) {
        //     printf("cuda rho: %.20e, rho0: %.20e, rhol: %.20e, rhoh: %.20e, f: %.20e\n", *rho, *rho0, *rhol, *rhoh, f);
        //     break;
        // }
        if (fabs(f) <= abs_tol || fabs(*rhoh - *rhol) / fabs(1 + *rhoh + *rhol) <= rel_tol){
            break;
        }
        count++;
        if (count > MAX_ITER){
            break;
        }
        *rho0 = *rho;
    }
    // printf("*****cuda output rho: %.20e\n", *rho);
}

__device__ void newton_rootsearch_diagonal(double *r0, double *s0, double *t0, double *rhol, double *rhoh, double *rho0, double *dr, double *dt, double *ds_squared, double *ds_div_dr, double *dsdr, double *rho, double abs_tol, double rel_tol, int max_iter = 30){
    // bool converged = false;
    double LODAMP = 0.05;
    double HIDAMP = 0.95;
    double f = 0.0;
    double df = 0.0;
    *rho = *rho0;
    for (int i = 1; i <= max_iter; ++i) {
        oracle_h_diagonal(r0, s0, t0, dr, dt, ds_squared, ds_div_dr, dsdr, rho0, &f, &df);
        if (f < 0.0) {
            *rhol = *rho0;
        } else {
            *rhoh = *rho0;
        }
        if (*rhoh <= *rhol) {
            // converged = true;
            break;
        }
        if (isfinite(f) && df > abs_tol) {
            *rho = *rho0 - f/df;
            // printf("cuda rho: %f\n", *rho);
        } else {
            break;
        }
        if (fabs(f) <= abs_tol || fabs(*rhoh - *rhol) / fabs(1 + *rhoh + *rhol) <= rel_tol) {   
            // converged = true;
            break;
        }
        if( *rho >= *rhoh ){
            *rho0 = fmin(LODAMP * *rho0 + HIDAMP * *rhoh, *rhoh);
        } else if (*rho <= *rhol) {
            *rho0 = fmax(LODAMP * *rho0 + HIDAMP * *rhol, *rhol);
        } else {
            *rho0 = *rho;
        }
    }
    // if (converged) {
    //     *rho0 = fmax(*rhol, fmin(*rhoh, *rho));
    // } else {
        // printf("cuda enter rootsearch_bn_diagonal rhol: %f, rhoh: %f, rho0: %f, rho: %f\n", *rhol, *rhoh, *rho0, *rho);
        rootsearch_bn_diagonal(r0, s0, t0, rhol, rhoh, rho0, dr, dt, ds_squared, ds_div_dr, dsdr, rho, abs_tol, rel_tol);
    // }
}

__device__ void projsol_primalexpcone(double *r0, double *s0, double *t0, double *rho, double *vpr, double *vps, double *vpt, double *dist){
    double linrho = ((rho[0]-1) * r0[0] + s0[0]);
    double exprho = exp(rho[0]);
    if (linrho>0 && isfinite(exprho)) {
        double quadrho = rho[0] * (rho[0] - 1) + 1;
        double temp = linrho/quadrho;
        *vpr = rho[0] * temp;
        *vps = temp;
        *vpt = exprho * temp;
        *dist = sqrt((*vpt-t0[0]) * (*vpt-t0[0]) + (*vps-s0[0]) * (*vps-s0[0]) + (*vpr-r0[0]) * (*vpr-r0[0]));
    } else {
        *vpr = 0;
        *vps = 0;
        *vpt = positive_inf;
        *dist = positive_inf;
    }
}

__device__ void projsol_primalexpcone_diagonal(double *r0, double *s0, double *t0, double *rho, double *ds, double *dr, double *dt, double *vpr, double *vps, double *vpt, double *dist){
    double linrho = ((rho[0]-1) * r0[0] + s0[0] * ds[0] / dr[0]);
    double exprho = exp(rho[0]);
    if (isfinite(exprho)) {
        double quadrho = rho[0] * (rho[0] - 1) * dr[0] + ds[0] * ds[0] / dr[0];
        double temp = linrho / quadrho;
        *vpr = dr[0] * rho[0] * temp;
        *vps = ds[0] * temp;
        *vpt = dt[0] * exprho * temp;
        *dist = sqrt((*vpt-t0[0]) * (*vpt-t0[0]) + (*vps-s0[0]) * (*vps-s0[0]) + (*vpr-r0[0]) * (*vpr-r0[0]));
    } else {
        *vpr = 0;
        *vps = 0;
        *vpt = positive_inf;
        *dist = positive_inf;
    }
}

__device__ void projsol_dualexpcone(double *r0, double *s0, double *rho, double *dr, double *ds, double *dt, double *rdstar, double *sd, double *td){
    double rd = (ds[0] * ds[0] * r0[0] - ds[0] * dr[0] * rho[0] * s0[0]) / (ds[0] * ds[0] / dr[0] + rho[0] * (rho[0] - 1) * dr[0]);
    *rdstar = rd / dr[0];
    *sd = (1 - rho[0]) * rd / ds[0];
    *td = -1 / dt[0] * exp(-rho[0]) * rd;
}

__device__ void exponent_proj(double *v, double *t_warm_start, double abs_tol, double rel_tol){
    // heuristic solution
    double r0 = v[0];
    double s0 = v[1];
    double t0 = v[2];
    double rho = 0.0;
    double rho0 = 0.0;
    double vpr, vps, vpt, pdist = 0.0;
    primal_heuristic(&r0, &s0, &t0, &vpr, &vps, &vpt, &pdist);
    double vdr, vds, vdt, ddist = 0.0;
    dual_heuristic(&r0, &s0, &t0, &vdr, &vds, &vdt, &ddist);
    double min_dist = fmin(pdist, ddist);
    double inf_norm_vp_vd = -1;
    inf_norm_vp_vd = fmax(fabs(vpr + vdr - r0), inf_norm_vp_vd);
    inf_norm_vp_vd = fmax(fabs(vps + vds - s0), inf_norm_vp_vd);
    inf_norm_vp_vd = fmax(fabs(vpt + vdt - t0), inf_norm_vp_vd);
    double dot_vp_vd = vpr*vdr + vps*vds + vpt*vdt;
    bool case1 = (s0 <= 0 && r0 <= 0);
    bool case2 = (min_dist <= abs_tol);
    bool case3 = (inf_norm_vp_vd <= abs_tol && dot_vp_vd <= abs_tol);
    if (!case1 && !case2 && !case3){
        double f;
        oracle_f(&r0, &s0, &t0, t_warm_start, &f);
        if (fabs(f) > abs_tol){ // warm start
            double rho_l, rho_h;
            rho_bound(&r0, &s0, &t0, &pdist, &ddist, &rho_l, &rho_h);
            if (rho_l < t_warm_start[0] && rho_h > t_warm_start[0]){
                rho0 = t_warm_start[0];
                rho = rho0;
            }else{
                rho0 = 0.5*(rho_l+rho_h);
            }
            newton_rootsearch(&r0, &s0, &t0, &rho_l, &rho_h, &rho0, &rho, abs_tol, rel_tol);
            t_warm_start[0] = rho;
        }else{ // no warm start
            rho = t_warm_start[0];
        }
        double rtmp, stmp, ttmp, pdist1;
        projsol_primalexpcone(&r0, &s0, &t0, &rho, &rtmp, &stmp, &ttmp, &pdist1);
        if (pdist1 <= pdist){
            vpr = rtmp;
            vps = stmp;
            vpt = ttmp;
        }
    } // not three special cases and not heuristic solution
    v[0] = vpr;
    v[1] = vps;
    v[2] = vpt;
    return;
}

__device__ void exponent_proj_diagonal(double *v, double *D, double *t_warm_start, double abs_tol, double rel_tol){
    //    exponent_proj_diagonal!: projects the primal solution onto the exponential cone.
        // min ||v - v0||_2 s.t. D^ v in Kexp
    double r0 = v[0];
    double s0 = v[1];
    double t0 = v[2];
    bool scale_flag = false;
    if (v[0] >= 1e+4 || v[1] >= 1e+4 || v[2] >= 1e+4 || v[0] <= -1e+4 || v[1] <= -1e+4 || v[2] <= -1e+4){
        scale_flag = true;
        r0 /= 1e+4;
        s0 /= 1e+4;
        t0 /= 1e+4;
    }
    bool scale_flag_two = false;
    if (fabs(v[0]) < 1e-10 || fabs(v[1]) < 1e-10 || fabs(v[2]) < 1e-10){
        scale_flag_two = true;
        r0 *= 1e+4;
        s0 *= 1e+4;
        t0 *= 1e+4;
    }
    double dr = 1 / D[0];
    double ds = 1 / D[1];
    double dt = 1 / D[2];
    double ds_squared = ds * ds;
    double dt_div_ds = dt / ds;
    double ds_div_dr = ds / dr;
    double ds_div_dr_squared = ds_div_dr * ds_div_dr;
    double c1 = s0 / r0;
    double c2 = ds / dr;
    double a3 = c2 / c1;
    double a4 = 1 - c1 * c2;
    double dr_inv = 1 / dr;
    double dsdr = ds * dr;
    double dt_div_dr = dt / dr;
    // heuristic solution
    double vpr, vps, vpt, pdist = 0.0;
    primal_heuristic_diagonal(&r0, &s0, &t0, &a3, &dt_div_ds, &vpr, &vps, &vpt, &pdist);
    double vdr, vds, vdt, ddist = 0.0;
    dual_heuristic_diagonal(&r0, &s0, &t0, &a4, &dt_div_dr, &vdr, &vds, &vdt, &ddist);
    double min_dist = fmin(pdist, ddist);
    // printf("cuda vpr: %f, vps: %f, vpt: %f, pdist: %f, ddist: %f, min_dist: %f\n", vpr, vps, vpt, pdist, ddist, min_dist);
    double inf_norm_vp_vd = -1.0;
    inf_norm_vp_vd = fmax(fabs(vpr + vdr - r0), inf_norm_vp_vd);
    inf_norm_vp_vd = fmax(fabs(vps + vds - s0), inf_norm_vp_vd);
    inf_norm_vp_vd = fmax(fabs(vpt + vdt - t0), inf_norm_vp_vd);
    double dot_vp_vd = vpr*vdr + vps*vds + vpt*vdt;
    double rho = 0.0;
    double rho0 = 0.0;
    bool case1 = (s0<=0 && r0 <= 0);
    bool case2 = (min_dist <= abs_tol);
    bool case3 = (inf_norm_vp_vd <= abs_tol && dot_vp_vd <= abs_tol);
    if (!case1 && !case2 && !case3){
        double f;
        oracle_f_diagonal(&r0, &s0, &t0, &dr, &dt, &ds_squared, &ds_div_dr, &dsdr, t_warm_start, &f);
        if (fabs(f) > abs_tol){ // warm start
            double rho_l, rho_h;
            rho_bound_diagonal(&r0, &s0, &t0, &pdist, &ddist, &dr, &ds, &dt, &ds_squared, &c1, &c2, &a3, &a4, &ds_div_dr, &ds_div_dr_squared, &dr_inv, &dsdr, &rho_l, &rho_h);
            // if (rho_h - rho_l < 1e-10){
            //     printf("cuda rho_l: %f, rho_h: %f, r0: %f, s0: %f, t0: %f, dr: %f, ds: %f, dt: %f\n", rho_l, rho_h, r0, s0, t0, dr, ds, dt);
            // }
            if (rho_l < t_warm_start[0] && rho_h > t_warm_start[0]){
                rho0 = t_warm_start[0];
                rho = rho0;
            }else{
                rho0 = 0.5*(rho_l+rho_h);
            }
            newton_rootsearch_diagonal(&r0, &s0, &t0, &rho_l, &rho_h, &rho0, &dr, &dt, &ds_squared, &ds_div_dr, &dsdr, &rho, abs_tol, rel_tol);
            t_warm_start[0] = rho;
        }else{ // no warm start
            rho = t_warm_start[0];
        }
        // printf("cuda output rho: %f\n", rho);
        double rtmp, stmp, ttmp, pdist1;
        projsol_primalexpcone_diagonal(&r0, &s0, &t0, &rho, &ds, &dr, &dt, &rtmp, &stmp, &ttmp, &pdist1);
        // printf("cuda rtmp: %f, stmp: %f, ttmp: %f, pdist1: %f, pdist: %f\n", rtmp, stmp, ttmp, pdist1, pdist);
        double rdstar, sdstar, tdstar;
        projsol_dualexpcone(&r0 , &s0, &rho, &dr, &ds, &dt, &rdstar, &sdstar, &tdstar);
        if (pdist1 <= pdist){
            vpr = rtmp;
            vps = stmp;
            vpt = ttmp;
        }
    } // not three special cases and not heuristic solution
    v[0] = vpr;
    v[1] = vps;
    v[2] = vpt;
    if (scale_flag){
        v[0] *= 1e+4;
        v[1] *= 1e+4;
        v[2] *= 1e+4;
    }
    if (scale_flag_two){
        v[0] /= 1e+4;
        v[1] /= 1e+4;
        v[2] /= 1e+4;
    }
    return;
}

__device__ void exponent_proj_diagonal_initial(double *v, double *D, double *t_warm_start, double abs_tol, double rel_tol){
    //    exponent_proj_diagonal!: projects the primal solution onto the exponential cone.
        // min ||v - v0||_2 s.t. D^ v in Kexp
    double r0 = v[0];
    double s0 = v[1];
    double t0 = v[2];
    bool scale_flag = false;
    if (v[0] >= 1e+4 || v[1] >= 1e+4 || v[2] >= 1e+4 || v[0] <= -1e+4 || v[1] <= -1e+4 || v[2] <= -1e+4){
        scale_flag = true;
        r0 /= 1e+4;
        s0 /= 1e+4;
        t0 /= 1e+4;
    }
    bool scale_flag_two = false;
    if (fabs(v[0]) < 1e-10 || fabs(v[1]) < 1e-10 || fabs(v[2]) < 1e-10){
        scale_flag_two = true;
        r0 *= 1e+4;
        s0 *= 1e+4;
        t0 *= 1e+4;
    }
    double dr = D[0];
    double ds = D[1];
    double dt = D[2];
    double ds_squared = ds * ds;
    double dt_div_ds = dt / ds;
    double ds_div_dr = ds / dr;
    double ds_div_dr_squared = ds_div_dr * ds_div_dr;
    double c1 = s0 / r0;
    double c2 = ds / dr;
    double a3 = c2 / c1;
    double a4 = 1 - c1 * c2;
    double dr_inv = 1 / dr;
    double dsdr = ds * dr;
    double dt_div_dr = dt / dr;
    // heuristic solution
    double vpr, vps, vpt, pdist = 0.0;
    primal_heuristic_diagonal(&r0, &s0, &t0, &a3, &dt_div_ds, &vpr, &vps, &vpt, &pdist);
    double vdr, vds, vdt, ddist = 0.0;
    dual_heuristic_diagonal(&r0, &s0, &t0, &a4, &dt_div_dr, &vdr, &vds, &vdt, &ddist);
    double min_dist = fmin(pdist, ddist);
    // printf("cuda vpr: %f, vps: %f, vpt: %f, pdist: %f, ddist: %f, min_dist: %f\n", vpr, vps, vpt, pdist, ddist, min_dist);
    double inf_norm_vp_vd = -1.0;
    inf_norm_vp_vd = fmax(fabs(vpr + vdr - r0), inf_norm_vp_vd);
    inf_norm_vp_vd = fmax(fabs(vps + vds - s0), inf_norm_vp_vd);
    inf_norm_vp_vd = fmax(fabs(vpt + vdt - t0), inf_norm_vp_vd);
    double dot_vp_vd = vpr*vdr + vps*vds + vpt*vdt;
    double rho = 0.0;
    double rho0 = 0.0;
    bool case1 = (s0<=0 && r0 <= 0);
    bool case2 = (min_dist <= abs_tol);
    bool case3 = (inf_norm_vp_vd <= abs_tol && dot_vp_vd <= abs_tol);
    if (!case1 && !case2 && !case3){
        double f;
        oracle_f_diagonal(&r0, &s0, &t0, &dr, &dt, &ds_squared, &ds_div_dr, &dsdr, t_warm_start, &f);
        if (fabs(f) > abs_tol){ // warm start
            double rho_l, rho_h;
            rho_bound_diagonal(&r0, &s0, &t0, &pdist, &ddist, &dr, &ds, &dt, &ds_squared, &c1, &c2, &a3, &a4, &ds_div_dr, &ds_div_dr_squared, &dr_inv, &dsdr, &rho_l, &rho_h);
            // if (rho_h - rho_l < 1e-10){
            //     printf("cuda rho_l: %f, rho_h: %f, r0: %f, s0: %f, t0: %f, dr: %f, ds: %f, dt: %f\n", rho_l, rho_h, r0, s0, t0, dr, ds, dt);
            // }
            if (rho_l < t_warm_start[0] && rho_h > t_warm_start[0]){
                rho0 = t_warm_start[0];
                rho = rho0;
            }else{
                rho0 = 0.5*(rho_l+rho_h);
            }
            newton_rootsearch_diagonal(&r0, &s0, &t0, &rho_l, &rho_h, &rho0, &dr, &dt, &ds_squared, &ds_div_dr, &dsdr, &rho, abs_tol, rel_tol);
            t_warm_start[0] = rho;
        }else{ // no warm start
            rho = t_warm_start[0];
        }
        // printf("cuda output rho: %f\n", rho);
        double rtmp, stmp, ttmp, pdist1;
        projsol_primalexpcone_diagonal(&r0, &s0, &t0, &rho, &ds, &dr, &dt, &rtmp, &stmp, &ttmp, &pdist1);
        // printf("cuda rtmp: %f, stmp: %f, ttmp: %f, pdist1: %f, pdist: %f\n", rtmp, stmp, ttmp, pdist1, pdist);
        double rdstar, sdstar, tdstar;
        projsol_dualexpcone(&r0 , &s0, &rho, &dr, &ds, &dt, &rdstar, &sdstar, &tdstar);
        if (pdist1 <= pdist){
            vpr = rtmp;
            vps = stmp;
            vpt = ttmp;
        }
    } // not three special cases and not heuristic solution
    v[0] = vpr;
    v[1] = vps;
    v[2] = vpt;
    if (scale_flag){
        v[0] *= 1e+4;
        v[1] *= 1e+4;
        v[2] *= 1e+4;
    }
    if (scale_flag_two){
        v[0] /= 1e+4;
        v[1] /= 1e+4;
        v[2] /= 1e+4;
    }
    return;
}

__device__ void dualExponent_proj_diagonal(double *v, double *D, double *temp, double *t_warm_start, double abs_tol, double rel_tol){
    for (int i = 0; i < 3; ++i) {
        temp[i] = -v[i];
    }
    exponent_proj_diagonal_initial(temp, D, t_warm_start, abs_tol, rel_tol);
    for (int i = 0; i < 3; ++i) {
        v[i] += temp[i];
    }
    return;
}

__device__ void dualExponent_proj(double *v, double *t_warm_start, double abs_tol, double rel_tol){
    double vCopy[3];
    for (int i = 0; i < 3; ++i) {
        vCopy[i] = -v[i];
    }
    exponent_proj(vCopy, t_warm_start, abs_tol, rel_tol);
    for (int i = 0; i < 3; ++i) {
        v[i] += vCopy[i];
    }
    return;
}