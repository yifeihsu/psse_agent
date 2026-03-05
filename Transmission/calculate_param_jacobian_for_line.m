function H_pg_line = calculate_param_jacobian_for_line(mpc_case, line_idx, x_state_vec_scan, Ybus, Yf, Yt)
    % Calculates the Jacobian of measurements with respect to the R and X parameters
    % of a single specified line (line_idx).
    %
    % Inputs:
    %   mpc_case          : MATPOWER case struct with current parameter estimates
    %   line_idx          : Index of the branch whose R and X are being considered
    %   x_state_vec_scan  : Column vector [theta_all_buses_rad; V_all_buses_pu]
    %   Ybus, Yf, Yt      : System admittance matrices (Yf, Yt for line flows)
    %
    % Outputs:
    %   H_pg_line         : Jacobian matrix (nz_actual x 2), where columns are d/dR_k and d/dX_k

    nb = size(mpc_case.bus, 1);
    nl = size(mpc_case.branch, 1);
    
    % Ensure fbus and tbus are correctly obtained. These are actual bus numbers.
    fbus_num = mpc_case.branch(line_idx, 1); % From-bus number of the target line
    tbus_num = mpc_case.branch(line_idx, 2); % To-bus number of the target line

    % Measurement vector structure: Vm (nb), Pinj (nb), Qinj (nb), Pf (nl), Qf (nl), Pt (nl), Qt (nl)
    % Total nz_actual = nb + nb + nb + nl + nl + nl + nl = 3*nb + 4*nl
    nz_actual = 3 * nb + 4 * nl; 
    H_pg_line = zeros(nz_actual, 2); % Initialize Jacobian for R_k and X_k

    % Extract states from the input vector
    theta_est_rad = x_state_vec_scan(1:nb);
    V_est_pu      = x_state_vec_scan(nb+1 : 2*nb);

    r_k = mpc_case.branch(line_idx, 3); 
    x_k = mpc_case.branch(line_idx, 4);

    denom = r_k^2 + x_k^2;
    if denom < 1e-9 % Avoid division by zero or extremely small values
        warning('Line %d parameters R=%.4e, X=%.4e lead to near-zero denominator. Jacobian may be unstable.', line_idx, r_k, x_k);
        % Returning zeros might prevent immediate crash but hides the problem.
        % Consider alternative handling if this occurs frequently.
        return;
    end

    idx_fbus = fbus_num; % Assuming bus numbers are direct indices for V_est, theta_est
    idx_tbus = tbus_num; % Assuming bus numbers are direct indices

    Vi   = V_est_pu(idx_fbus);
    Vj   = V_est_pu(idx_tbus);
    dth_rad  = theta_est_rad(idx_fbus) - theta_est_rad(idx_tbus); % theta_i - theta_j
    
    cosd = cos(dth_rad);
    sind = sin(dth_rad);

    % Derivatives of line conductance (g_k) and susceptance (b_k) w.r.t. R_k and X_k
    % g_k = R_k / (R_k^2 + X_k^2)
    % b_k = -X_k / (R_k^2 + X_k^2)
    dg_dr = (x_k^2 - r_k^2) / (denom^2);
    db_dr = (2 * r_k * x_k) / (denom^2);     % d(b_k)/d(R_k)
    dg_dx = (-2 * r_k * x_k) / (denom^2);    % d(g_k)/d(X_k)
    db_dx = (x_k^2 - r_k^2) / (denom^2);     % d(b_k)/d(X_k)

    % --- Partial derivatives of power flow P_ik, Q_ik (from i to j on line k) w.r.t. g_k, b_k ---
    % P_ik = Vi^2*g_k - Vi*Vj*(g_k*cosd + b_k*sind)
    % Q_ik = -Vi^2*b_k - Vi*Vj*(g_k*sind - b_k*cosd)
    
    dP_ik_dgk = Vi^2 - Vi*Vj*cosd;
    dP_ik_dbk = -Vi*Vj*sind;       % CORRECTED SIGN
    dQ_ik_dgk = -Vi*Vj*sind;       % CORRECTED SIGN
    dQ_ik_dbk = -Vi^2 + Vi*Vj*cosd;

    % --- Partial derivatives of power flow P_jk, Q_jk (from j to i on line k) w.r.t. g_k, b_k ---
    % Note: delta_ji = -delta_ij, so cos(delta_ji)=cos(delta_ij), sin(delta_ji)=-sin(delta_ij)
    % P_jk = Vj^2*g_k - Vi*Vj*(g_k*cosd - b_k*sind)
    % Q_jk = -Vj^2*b_k - Vi*Vj*(-g_k*sind - b_k*cosd)
    
    dP_jk_dgk = Vj^2 - Vi*Vj*cosd;
    dP_jk_dbk = Vi*Vj*sind;        % CORRECTED SIGN (due to sin(delta_ji))
    dQ_jk_dgk = Vi*Vj*sind;        % CORRECTED SIGN (due to sin(delta_ji))
    dQ_jk_dbk = -Vj^2 + Vi*Vj*cosd;
    
    % --- Apply Chain Rule ---
    % Derivatives of P and Q injections at bus i (due to line k) w.r.t R_k, X_k
    % P_i_due_to_line_k = P_ik, Q_i_due_to_line_k = Q_ik
    dPi_dr = dP_ik_dgk * dg_dr + dP_ik_dbk * db_dr;
    dPi_dx = dP_ik_dgk * dg_dx + dP_ik_dbk * db_dx;
    dQi_dr = dQ_ik_dgk * dg_dr + dQ_ik_dbk * db_dr;
    dQi_dx = dQ_ik_dgk * dg_dx + dQ_ik_dbk * db_dx;
    
    % Derivatives of P and Q injections at bus j (due to line k) w.r.t R_k, X_k
    % P_j_due_to_line_k = P_jk, Q_j_due_to_line_k = Q_jk
    dPj_dr = dP_jk_dgk * dg_dr + dP_jk_dbk * db_dr;
    dPj_dx = dP_jk_dgk * dg_dx + dP_jk_dbk * db_dx;
    dQj_dr = dQ_jk_dgk * dg_dr + dQ_jk_dbk * db_dr;
    dQj_dx = dQ_jk_dgk * dg_dx + dQ_jk_dbk * db_dx;

    % --- Populate H_pg_line (Jacobian matrix) ---
    % Measurement order: Vm (nb), Pinj (nb), Qinj (nb), Pf (nl), Qf (nl), Pt (nl), Qt (nl)
    
    % Voltage magnitude measurements (Vm_1 ... Vm_nb) are not explicitly functions of R_k, X_k
    % H_pg_line(1:nb, :) remains zeros.

    % P injections at bus idx_fbus and idx_tbus
    % Note: P_injection at bus 'idx_fbus' is affected by line k's parameters.
    % Other P_injection measurements (for buses not idx_fbus or idx_tbus) have zero derivative w.r.t R_k, X_k.
    H_pg_line(nb + idx_fbus, 1) = dPi_dr; 
    H_pg_line(nb + idx_fbus, 2) = dPi_dx;
    H_pg_line(nb + idx_tbus, 1) = dPj_dr; 
    H_pg_line(nb + idx_tbus, 2) = dPj_dx;
    
    % Q injections at bus idx_fbus and idx_tbus
    H_pg_line(2*nb + idx_fbus, 1) = dQi_dr; 
    H_pg_line(2*nb + idx_fbus, 2) = dQi_dx;
    H_pg_line(2*nb + idx_tbus, 1) = dQj_dr; 
    H_pg_line(2*nb + idx_tbus, 2) = dQj_dx;
    
    % Line flow measurements for the specific line k = line_idx
    % Pf_k (flow from i to j on line k) is P_ik
    offset_pf = 3*nb;
    H_pg_line(offset_pf + line_idx, 1) = dPi_dr; 
    H_pg_line(offset_pf + line_idx, 2) = dPi_dx;
    
    % Qf_k (flow from i to j on line k) is Q_ik
    offset_qf = 3*nb + nl;
    H_pg_line(offset_qf + line_idx, 1) = dQi_dr; 
    H_pg_line(offset_qf + line_idx, 2) = dQi_dx;
    
    % Pt_k (flow from j to i on line k) is P_jk
    offset_pt = 3*nb + 2*nl;
    H_pg_line(offset_pt + line_idx, 1) = dPj_dr; 
    H_pg_line(offset_pt + line_idx, 2) = dPj_dx;
    
    % Qt_k (flow from j to i on line k) is Q_jk
    offset_qt = 3*nb + 3*nl;
    H_pg_line(offset_qt + line_idx, 1) = dQj_dr; 
    H_pg_line(offset_qt + line_idx, 2) = dQj_dx;
    
end