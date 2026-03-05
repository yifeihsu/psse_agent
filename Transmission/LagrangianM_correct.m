%% ================== State Estimation Function ====================
function [lambdaN, success, r_norm, Omega, final_resid_raw, z_corrected_info] = ...
    LagrangianM_correct(z_in_full, result, ind, bus_data, options, R_variances_full_in)
%LAGRANGIANM_PARTIAL Standard State Estimation Program with Iterative Group Correction
%
%  Inputs:
%    z_in_full    : Full measurement vector (nz_actual x 1).
%    result       : MATPOWER struct.
%    ind          : Legacy flag.
%    bus_data     : Initial bus data from OPF.
%    options      : Struct for SE options:
%                   .enable_group_correction (true/false)
%                   .correction_group_full_indices (global indices of suspect group)
%                   .max_correction_iterations (max correction cycles)
%                   .correction_error_tolerance (norm of error vector)
%    R_variances_full_in: Full vector of measurement variances.
%
%  Outputs:
%    (Same as before, z_corrected_info is enhanced)

%% 1) Basic Setup
ref = find(result.bus(:, 2)==3); 
if isempty(ref)
    ref = 1; 
    fprintf('Warning: No slack bus type 3 found. Using bus 1 as reference.\n');
end
eps_tol = 1e-4;   
maxIter_wls = 20; % Max iterations for each individual WLS run
nb   = size(result.bus, 1);
nl   = size(result.branch, 1);
fbus = result.branch(:, 1); 
tbus = result.branch(:, 2); 

% Initialize z_corrected_info
z_corrected_info.applied_any_correction = false;
z_corrected_info.iterations_performed = 0;
z_corrected_info.skipped_reason = 'Correction not enabled or no group specified by options.';
small_eps_omega = 1e-12; 

%% 2) Identify Valid Measurements and Build R for them
validMeasMask = ~isnan(z_in_full) & (z_in_full < 998) & (z_in_full > -998); 
z_active = z_in_full(validMeasMask); 

if isempty(z_active)
    error('No valid measurements found in z_in_full.');
end

R_variances_active = R_variances_full_in(validMeasMask);
R_active = spdiags(R_variances_active, 0, numel(z_active), numel(z_active));
W_active = inv(R_active); 

%% 3) Build Full Y-bus
[Ybus, Yf, Yt] = makeYbus(result.baseMVA, result.bus, result.branch);

%% 4) State Initialization (Done before each WLS run if iterative)
nstate_vars = [1:ref-1, ref+1:nb, nb+1:2*nb]'; 

%% 5) Iterative WLS and Correction Loop

% --- Get correction iteration parameters from options or set defaults ---
max_total_wls_runs = 1; % Default is 1 WLS run (no iterative correction)
if isfield(options, 'enable_group_correction') && options.enable_group_correction
    if isfield(options, 'max_correction_iterations')
        % max_correction_iterations is the number of *correction cycles*
        max_total_wls_runs = 1 + options.max_correction_iterations; % 1 initial WLS + N correction cycles
    else
        max_total_wls_runs = 2; % Default: initial WLS + 1 correction cycle (total 2 WLS runs)
    end
    if isfield(options, 'correction_error_tolerance')
        error_tol_for_correction_norm = options.correction_error_tolerance;
    else
        error_tol_for_correction_norm = 1e-4; % Default norm of error vector
    end
else % Correction not enabled
    options.enable_group_correction = false; % Ensure it's explicitly false
end


current_wls_run_number = 0;
perform_next_wls_run = true;
overall_wls_success_status = false; % Tracks if the final WLS state is from a converged run

% Initialize variables that will be updated in the loop and used afterwards
H_active = []; Gain = []; hx_active = []; x_state = []; % Ensure they are defined

while perform_next_wls_run && current_wls_run_number < max_total_wls_runs
    current_wls_run_number = current_wls_run_number + 1;
    fprintf('--- Starting WLS Run: %d of max %d potential runs ---\n', current_wls_run_number, max_total_wls_runs);

    % --- State Initialization for current WLS run ---
    initial_angles = deg2rad(bus_data(:,9)); 
    initial_angles = initial_angles - initial_angles(ref); 
    initial_voltages = bus_data(:,8);   
    x_state = [initial_angles; initial_voltages]; % Always re-initialize for each WLS run in the iterative process
    
    k_iter_wls = 0; % Iteration counter for this specific WLS run
    wls_converged_this_run = false;

    % --- Inner WLS Iteration Loop ---
    while ~wls_converged_this_run && k_iter_wls < maxIter_wls
        k_iter_wls = k_iter_wls + 1;
        
        Vc_complex = x_state(nb+1 : 2*nb) .* exp(1j * x_state(1:nb));
        hx_V = x_state(nb+1 : 2*nb);
        Ibus_complex = Ybus * Vc_complex;
        S_inj_complex = Vc_complex .* conj(Ibus_complex); 
        hx_Pinj = real(S_inj_complex); hx_Qinj = imag(S_inj_complex);
        Sf_complex = Vc_complex(fbus) .* conj(Yf * Vc_complex);
        St_complex = Vc_complex(tbus) .* conj(Yt * Vc_complex);
        hx_Pf = real(Sf_complex); hx_Qf = imag(Sf_complex);
        hx_Pt = real(St_complex); hx_Qt = imag(St_complex);
        hx_full_ordered = [hx_V; hx_Pinj; hx_Qinj; hx_Pf; hx_Qf; hx_Pt; hx_Qt];
        hx_active = hx_full_ordered(validMeasMask); 
    
        [dSbus_dVa_raw, dSbus_dVm_raw] = dSbus_dV(Ybus, Vc_complex);
        [dSf_dVa_raw, dSf_dVm_raw, dSt_dVa_raw, dSt_dVm_raw] = dSbr_dV(result.branch, Yf, Yt, Vc_complex);
        H_V_Vang = sparse(nb, nb); H_V_Vmag = speye(nb);
        H_Pinj_Vang = real(dSbus_dVa_raw); H_Pinj_Vmag = real(dSbus_dVm_raw);
        H_Qinj_Vang = imag(dSbus_dVa_raw); H_Qinj_Vmag = imag(dSbus_dVm_raw);
        H_Pf_Vang = real(dSf_dVa_raw); H_Pf_Vmag = real(dSf_dVm_raw);
        H_Qf_Vang = imag(dSf_dVa_raw); H_Qf_Vmag = imag(dSf_dVm_raw);
        H_Pt_Vang = real(dSt_dVa_raw); H_Pt_Vmag = real(dSt_dVm_raw);
        H_Qt_Vang = imag(dSt_dVa_raw); H_Qt_Vmag = imag(dSt_dVm_raw);
        H_full_ordered_Vang = [H_V_Vang; H_Pinj_Vang; H_Qinj_Vang; H_Pf_Vang; H_Qf_Vang; H_Pt_Vang; H_Qt_Vang];
        H_full_ordered_Vmag = [H_V_Vmag; H_Pinj_Vmag; H_Qinj_Vmag; H_Pf_Vmag; H_Qf_Vmag; H_Pt_Vmag; H_Qt_Vmag];
        H_full_ordered = [H_full_ordered_Vang, H_full_ordered_Vmag];
        H_active_temp = H_full_ordered(validMeasMask, :);
        H_active = H_active_temp(:, nstate_vars); 
            
        mismatch = z_active - hx_active;
        Gain = H_active' * W_active * H_active;
        
        if rcond(full(Gain)) < 1e-14 
            fprintf('Warning: Gain matrix is ill-conditioned (rcond: %e) at WLS iter %d, WLS Run %d.\n', rcond(Gain), k_iter_wls, current_wls_run_number);
            wls_converged_this_run = false; % Mark as not converged for this run
            break; % Exit inner WLS loop
        end
        
        rhs = H_active' * W_active * mismatch;
        dx = Gain \ rhs;
    
        x_state_temp_update = zeros(2*nb,1);
        x_state_temp_update(nstate_vars) = dx;
        x_state = x_state + x_state_temp_update;
        x_state(ref) = 0; 
    
        if max(abs(dx)) < eps_tol
            wls_converged_this_run = true;
            fprintf('WLS Run %d converged in %d WLS iterations.\n', current_wls_run_number, k_iter_wls);
        elseif k_iter_wls >= maxIter_wls
            fprintf('WLS Run %d did NOT converge after %d WLS iterations.\n', current_wls_run_number, maxIter_wls);
            wls_converged_this_run = false; 
            break; % Exit inner WLS loop
        end
    end % End of Inner WLS Iteration Loop (while ~wls_converged_this_run ...)

    % --- Post Inner WLS Run ---
    if ~wls_converged_this_run
        overall_wls_success_status = false; 
        perform_next_wls_run = false; % Stop all further processing
        if current_wls_run_number == 1 && options.enable_group_correction
            z_corrected_info.skipped_reason = 'Initial WLS run failed to converge, correction not attempted.';
        elseif options.enable_group_correction && z_corrected_info.applied_any_correction
            z_corrected_info.skipped_reason = sprintf('WLS failed to converge after %d correction iteration(s).', z_corrected_info.iterations_performed);
        end
    else % Current WLS run converged successfully
        overall_wls_success_status = true; 
        
        can_attempt_correction = options.enable_group_correction && ...
                                 ~isempty(options.correction_group_full_indices) && ...
                                 current_wls_run_number < max_total_wls_runs; 

        if can_attempt_correction
            fprintf('--- Performing Group Bad Data Correction Check (after WLS Run %d) ---\n', current_wls_run_number);
            
            temp_Omega = inv(W_active) - H_active * (Gain \ H_active'); % Omega from this run
            temp_S_active = temp_Omega * W_active; % S from this run

            global_indices_of_active_meas = find(validMeasMask);
            [is_member_of_group, ~] = ismember(global_indices_of_active_meas, options.correction_group_full_indices);
            group_local_indices_in_active = find(is_member_of_group);

            if ~isempty(group_local_indices_in_active)
                r_s_raw = (z_active - hx_active); % current raw residuals
                r_s_raw_group = r_s_raw(group_local_indices_in_active);
                S_ss = temp_S_active(group_local_indices_in_active, group_local_indices_in_active);

                if rcond(full(S_ss)) > 1e-12 
                    estimated_errors = S_ss \ r_s_raw_group;
                    norm_of_estimated_errors = norm(estimated_errors);

                    if norm_of_estimated_errors > error_tol_for_correction_norm
                        fprintf('Estimated error norm (%.3e) > tolerance (%.3e). Applying correction for next WLS run.\n', norm_of_estimated_errors, error_tol_for_correction_norm);
                        
                        original_group_values = z_active(group_local_indices_in_active);
                        corrected_group_values = original_group_values - estimated_errors;
                        z_active(group_local_indices_in_active) = corrected_group_values; % IMPORTANT: Update z_active
                        
                        z_corrected_info.applied_any_correction = true;
                        z_corrected_info.iterations_performed = z_corrected_info.iterations_performed + 1;
                        z_corrected_info.last_applied_error_norm = norm_of_estimated_errors;
                        z_corrected_info.last_corrected_global_indices = global_indices_of_active_meas(group_local_indices_in_active);
                        z_corrected_info.last_original_values = original_group_values;
                        z_corrected_info.last_estimated_errors = estimated_errors;
                        z_corrected_info.last_corrected_values = corrected_group_values;
                        z_corrected_info.skipped_reason = ''; % Clear any previous skipped reason
                        
                        perform_next_wls_run = true; % Continue to the next WLS run
                    else
                        fprintf('Estimated error norm (%.3e) <= tolerance (%.3e). No further correction needed.\n', norm_of_estimated_errors, error_tol_for_correction_norm);
                        perform_next_wls_run = false; % Stop WLS runs
                        if ~z_corrected_info.applied_any_correction 
                            z_corrected_info.skipped_reason = 'Estimated errors below tolerance on first check.';
                        else
                            z_corrected_info.skipped_reason = sprintf('Errors below tolerance after %d correction(s).', z_corrected_info.iterations_performed);
                        end
                    end
                else % S_ss ill-conditioned
                    fprintf('Warning: S_ss ill-conditioned (rcond %e) during correction check. Correction skipped for this iteration.\n', rcond(S_ss));
                    perform_next_wls_run = false; % Stop if S_ss is bad
                    z_corrected_info.skipped_reason = 'S_ss matrix ill-conditioned during correction attempt.';
                end
            else % Group empty in active measurements
                 fprintf('No measurements from the specified correction group found in the active set. Correction check skipped.\n');
                 perform_next_wls_run = false; % Stop
                 if ~z_corrected_info.applied_any_correction
                    z_corrected_info.skipped_reason = 'Correction group empty in active measurements.';
                 end
            end
        else % Correction not enabled, or max WLS runs for correction reached, or group empty
            perform_next_wls_run = false; % Stop WLS runs
            if options.enable_group_correction && current_wls_run_number >= max_total_wls_runs
                 fprintf('Max WLS runs reached for iterative correction.\n');
                 if ~z_corrected_info.applied_any_correction
                    z_corrected_info.skipped_reason = 'Max WLS runs reached; no effective correction applied or errors were small.';
                 elseif isempty(z_corrected_info.skipped_reason) % Only set if not already set by error condition
                    z_corrected_info.skipped_reason = 'Max WLS runs reached.';
                 end
            end
        end
    end
end % End of outer `while perform_next_wls_run` loop

%% 6) Post-Loop Calculations (using results from the LAST successful WLS run)
success = overall_wls_success_status; % Final success status of the function

if success 
    % Ensure hx_active, H_active, Gain are from the very last converged WLS iteration
    % These would have been set correctly in the last successful pass of the inner WLS loop
    final_resid_raw = z_active - hx_active; 
    Omega = inv(W_active) - H_active * (Gain \ H_active'); 
    
    diagOmega = diag(Omega);
    diagOmega(diagOmega < 0 & diagOmega > -small_eps_omega*10) = 0; 
    diagOmega(diagOmega <= 0) = small_eps_omega; 
    
    r_norm = abs(final_resid_raw) ./ sqrt(diagOmega);
else
    lambdaN = []; r_norm = []; Omega = []; final_resid_raw = [];
    % z_corrected_info.skipped_reason might already be set
    if isempty(z_corrected_info.skipped_reason) && current_wls_run_number > 0 % if loop was entered but failed
        z_corrected_info.skipped_reason = 'WLS process failed to converge.';
    end
end

lambdaN = zeros(nl,1); 

end % End of LagrangianM_partial function

