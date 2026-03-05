function [corrected_params_group, success_correction] = correct_parameter_group_multi_scan(mpc_with_error, line_to_correct_idx, multi_scan_measurements_z, initial_states_multi_scan, R_variances_vec, baseMVA)
%CORRECT_PARAMETER_GROUP_MULTI_SCAN
%  Performs multi-scan augmented-state estimation to correct R and X
%  of a specific line (line_to_correct_idx).

    fprintf('    Entering correct_parameter_group_multi_scan for line %d...\n', line_to_correct_idx);

    %% 1) Basic Setup
    nb = size(mpc_with_error.bus, 1);
    nl = size(mpc_with_error.branch, 1);
    s  = size(multi_scan_measurements_z, 2);  % Number of scans
    success_correction = 0;

    % Current param guesses (start from the erroneous ones in mpc_with_error)
    p_g_current = [
        mpc_with_error.branch(line_to_correct_idx, 3);  % R
        mpc_with_error.branch(line_to_correct_idx, 4);  % X
    ];
    corrected_params_group = p_g_current;  % default return value

    % Build the per-scan measurement covariance block
    nz_single = size(multi_scan_measurements_z, 1);
    R_single_scan = spdiags(R_variances_vec, 0, nz_single, nz_single);
    R_inv_single  = inv(R_single_scan);
    
    % We remove the reference bus angle => each scan has (2*nb - 1) states.
    n_states_per_scan = 2*nb - 1;
    N = s*n_states_per_scan + 2;  % dimension of the augmented vector

    % Prepare initial guess for the augmented vector x_v:
    x_states_all_scans_current = zeros(s*n_states_per_scan, 1);

    % Identify the reference bus
    ref_bus = find(mpc_with_error.bus(:,2) == 3);
    if isempty(ref_bus)
        warning('No type-3 (slack) bus found; defaulting ref_bus=1');
        ref_bus = 1;
    end

    %% 2) Build the initial augmented state x_v_current
    for k_scan = 1:s
        v_init     = initial_states_multi_scan(1:nb, k_scan);
        a_init_deg = initial_states_multi_scan(nb+1 : 2*nb, k_scan);
        a_init_rad_full = a_init_deg * (pi/180);

        % Remove reference angle
        a_sub = a_init_rad_full;
        a_sub(ref_bus) = [];
        
        % Reduced state for this scan
        x_k_scan = [a_sub; v_init];

        % Insert into the big augmented vector
        idx_start = (k_scan-1)*n_states_per_scan + 1;
        idx_end   = k_scan*n_states_per_scan;
        x_states_all_scans_current(idx_start : idx_end) = x_k_scan;
    end

    % Append the 2 line parameters [R, X] at the end
    x_v_current = [x_states_all_scans_current; p_g_current];

    %% 3) Iterative ASE to correct R and X
    max_iter_corr = 60;
    tol_corr      = 5e-3;

    fprintf('    Starting iterative correction for line %d with %d scans...\n', line_to_correct_idx, s);

    for iter = 1:max_iter_corr

        % Build big G_v and RHS
        G_v = sparse(N, N);
        RHS = zeros(N,1);

        for k_scan = 1:s
            %% 3a) Extract sub-vector of states for this scan
            idx_start = (k_scan-1)*n_states_per_scan + 1;
            idx_end   = k_scan*n_states_per_scan;
            x_scan_k  = x_v_current(idx_start : idx_end);

            %% 3b) Expand it back to full dimension
            [theta_full, V_full] = expand_state_vector(x_scan_k, nb, ref_bus);

            %% 3c) Overwrite parameters in copy
            mpc_iter = mpc_with_error;
            mpc_iter.branch(line_to_correct_idx, 3) = x_v_current(s*n_states_per_scan + 1); % R
            mpc_iter.branch(line_to_correct_idx, 4) = x_v_current(s*n_states_per_scan + 2); % X

            %% 3d) Predicted measurements and mismatch
            hx_k = calculate_hx(mpc_iter, theta_full, V_full);
            z_k  = multi_scan_measurements_z(:, k_scan);
            delta_z_k = z_k - hx_k;

            %% 3e) H_x Jacobian
            [Ybus_iter, Yf_iter, Yt_iter] = makeYbus(mpc_iter);
            fbus_iter = mpc_iter.branch(:,1);
            tbus_iter = mpc_iter.branch(:,2);
            
            x_full_scan = [theta_full; V_full];
            Vc_full     = V_full .* exp(1j*theta_full);

            [J_full, ~, ~] = makeJaco(x_full_scan, Ybus_iter, Yf_iter, Yt_iter, ...
                                      nb, nl, fbus_iter, tbus_iter, Vc_full);
            J_full(:, ref_bus) = [];
            H_x_k = J_full;

            %% 3f) H_p Parameter Jacobian
            H_p_k = calculate_param_jacobian_for_line(mpc_iter, line_to_correct_idx, ...
                                                      [theta_full; V_full], ...
                                                      Ybus_iter, Yf_iter, Yt_iter);

            %% 3g) Accumulate G_v and RHS
            Wblock  = R_inv_single;
            Ht_W    = H_x_k' * Wblock;
            Hp_t_W  = H_p_k' * Wblock;
            
            Ht_W_dz    = Ht_W  * delta_z_k;  
            Hp_t_W_dz  = Hp_t_W * delta_z_k;

            row_s  = idx_start : idx_end;  
            col_s  = row_s;
            row_p  = s*n_states_per_scan + (1:2);
            col_p  = row_p;

            G_v(row_s, col_s) = G_v(row_s, col_s) + (Ht_W * H_x_k);
            G_v(row_s, col_p) = G_v(row_s, col_p) + (Ht_W * H_p_k);
            G_v(row_p, col_s) = G_v(row_p, col_s) + (Hp_t_W * H_x_k);
            G_v(row_p, col_p) = G_v(row_p, col_p) + (Hp_t_W * H_p_k);

            RHS(row_s) = RHS(row_s) + Ht_W_dz;
            RHS(row_p) = RHS(row_p) + Hp_t_W_dz;
        end 

        %% 3h) Solve
        if rcond(full(G_v)) < 1e-14
            fprintf('    G_v is ill-conditioned at iteration %d. Aborting correction.\n', iter);
            success_correction = 0;
            return;
        end
        
        x_prev = x_v_current;  % Store for convergence check

        delta_x_v = G_v \ RHS;
        x_v_current = x_v_current + 0.5 * delta_x_v;  % update

        % Positive Constraint
        idx_param = s*n_states_per_scan + (1:2);
        p_temp = x_v_current(idx_param);
        p_temp = max(p_temp, 1e-6); 
        x_v_current(idx_param) = p_temp;

        %% 3i) Check convergence
        max_update = max(abs(x_v_current - x_prev));
        p_g_current = x_v_current(idx_param);
        fprintf('    Iter %d: max update=%.2e, R_est=%.4f, X_est=%.4f\n', ...
                iter, max_update, p_g_current(1), p_g_current(2));

        if max_update < tol_corr
            success_correction    = 1;
            corrected_params_group = p_g_current;
            fprintf('    Correction converged in %d iterations.\n', iter);
            break;
        end

        if iter == max_iter_corr
            success_correction    = 0;
            corrected_params_group = p_g_current;
            fprintf('    Correction did NOT converge by iteration %d. Returning last estimate.\n', iter);
        end
    end

    fprintf('    Exiting correct_parameter_group_multi_scan. success=%d\n', success_correction);
end
