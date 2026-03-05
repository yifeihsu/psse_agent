    clc;
    clear;
    close all;
    %% --- 1) Load and Modify the Base Case ---
    mpc_orig = loadcase('case9');
    disp('Original system loaded (IEEE 118 Bus).');
    mpc = mpc_orig;
    nbus    = size(mpc.bus, 1);
    nbranch = size(mpc.branch, 1);
    %% --- 2) Run Base Case OPF ONCE ---
    disp('Running base case OPF on modified system...');
    mpopt_base = mpoption('verbose', 0, 'out.all', 1); 
    [baseMVA, bus_base, gen_base, ~, branch_base, ~, success_base, ~] = runopf(mpc, mpopt_base);
    if success_base ~= 1
        error('Base case OPF failed to converge! Cannot proceed.');
    end
    disp('Base case OPF successful.');
    result_struct.bus    = bus_base;
    result_struct.branch = branch_base;
    result_struct.gen    = gen_base;
    result_struct.baseMVA= baseMVA;
    %% --- 3) Generate a "complete" Measurement Vector (z_full) ---
    nz_actual = nbus + nbus + nbus + 4*nbranch;
    z_full = zeros(nz_actual, 1);
    % 1) Voltage magnitudes (PU)
    z_full(1:nbus) = bus_base(:, 8);  % VM
    % 2) Nodal power injections (PU on system baseMVA)
    Sbus_base = makeSbus(baseMVA, bus_base, gen_base); % Returns PU on system baseMVA
    Pinj_base = real(Sbus_base);
    Qinj_base = imag(Sbus_base);
    z_full(nbus+1 : 2*nbus)   = Pinj_base;   % P_inj
    z_full(2*nbus+1 : 3*nbus) = Qinj_base;   % Q_inj
    % 3) Branch flows (PU on system baseMVA)
    z_full(3*nbus+1 : 3*nbus+nbranch)             = branch_base(:,14)/baseMVA;  % Pf
    z_full(3*nbus+nbranch+1 : 3*nbus+2*nbranch)   = branch_base(:,15)/baseMVA;  % Qf
    z_full(3*nbus+2*nbranch+1 : 3*nbus+3*nbranch) = branch_base(:,16)/baseMVA;  % Pt
    z_full(3*nbus+3*nbranch+1 : 3*nbus+4*nbranch) = branch_base(:,17)/baseMVA;  % Qt
    %% --- 4) Remove (or De-Weight) Some Measurements ---
    remove_indices = []; % Keep all measurements for this example
    all_indices = 1:nz_actual;
    keep_mask = true(1, nz_actual);
    keep_mask(remove_indices) = false;
    z_base = z_full(keep_mask); % True values of measurements we are keeping
    kept_indices = find(keep_mask); % Global indices of measurements we are keeping
    %% --- 5) Define Measurement Variances (for R matrix) ---
    R_variances_full = zeros(nz_actual, 1);
    R_variances_full(1:nbus) = (0.01)^2;  % Voltage magnitudes
    R_variances_full(nbus+1 : 3*nbus) = (0.005)^2; % P and Q Injections
    R_variances_full(3*nbus+1 : 3*nbus+4*nbranch) = (0.002)^2; % P and Q Flows
    sigma_full = sqrt(R_variances_full);
    sigma = sigma_full(kept_indices); 
    %% --- 6) Define Bad Data Group and Inject Errors ---
    bus_of_interest = 3;
    fprintf('Introducing bad data related to bus %d.\n', bus_of_interest);
    
    % Identify all measurements associated with bus_of_interest to form the "suspect group"
    suspect_group_global_indices = [];
    % (a) Voltage at bus_of_interest
    suspect_group_global_indices = [suspect_group_global_indices, bus_of_interest];
    % (b) P and Q injection at bus_of_interest
    suspect_group_global_indices = [suspect_group_global_indices, nbus + bus_of_interest, 2*nbus + bus_of_interest];
    % (c) Flows on branches connected to bus_of_interest
    branches_connected_from = find(mpc.branch(:,1) == bus_of_interest);
    branches_connected_to   = find(mpc.branch(:,2) == bus_of_interest);
    for ln_idx = branches_connected_from(:).'
        suspect_group_global_indices = [suspect_group_global_indices, ...
                                        3*nbus + ln_idx, ...                 % Pf
                                        3*nbus + nbranch + ln_idx];         % Qf
    end
    for ln_idx = branches_connected_to(:).'
        suspect_group_global_indices = [suspect_group_global_indices, ...
                                        3*nbus + 2*nbranch + ln_idx, ...     % Pt
                                        3*nbus + 3*nbranch + ln_idx];       % Qt
    end
    suspect_group_global_indices = unique(suspect_group_global_indices);
    
    % Filter suspect_group_global_indices to only include those that are in kept_indices
    suspect_group_for_corruption = intersect(suspect_group_global_indices, kept_indices);
    
    scale_factor_min = 1.3; 
    scale_factor_max = 1.7;  
    
    max_iterations_main_loop   = 1;
    disp('========================================');
    if max_iterations_main_loop > 1
        disp('Starting loop for multiple runs (bad data detection focus)...');
    else
        disp('Starting single run (bad data correction focus)...');
    end
    
    for iter = 1:max_iterations_main_loop
        % 6a) Start from the "clean" measurement subset
        z_noisy_iter = z_base; 
        
        % 6b) Add normal measurement noise to all kept measurements
        noise = sigma .* randn(size(z_noisy_iter));
        z_noisy_iter = z_noisy_iter + noise;
        
        % 6c) Inject "bad data" into the suspect group by scaling
        sf = scale_factor_min + (scale_factor_max - scale_factor_min)*rand(1); 
        % sf = 1;
        [~, local_indices_to_corrupt_in_z_noisy] = ismember(suspect_group_for_corruption, kept_indices);
        local_indices_to_corrupt_in_z_noisy = local_indices_to_corrupt_in_z_noisy(local_indices_to_corrupt_in_z_noisy > 0); 
    
        if ~isempty(local_indices_to_corrupt_in_z_noisy)
            z_noisy_iter(local_indices_to_corrupt_in_z_noisy) = z_noisy_iter(local_indices_to_corrupt_in_z_noisy) * sf;
            fprintf('Iter %d: Injected bad data with scale factor %.3f to %d measurements in the group.\n', iter, sf, length(local_indices_to_corrupt_in_z_noisy));
        else
            fprintf('Iter %d: No measurements from suspect group are in the kept set. No bad data injected.\n', iter);
        end
            
        z_full_corrupted = ones(nz_actual, 1) * 999; 
        z_full_corrupted(kept_indices) = z_noisy_iter;
        
        % --- Options for State Estimator ---
        options_se.enable_group_correction = true;
        options_se.correction_group_full_indices = suspect_group_global_indices; 
        options_se.max_correction_iterations = 5; % Max number of *correction cycles* (e.g., 2 means up to 1 initial WLS + 2 WLS with corrections = 3 total WLS)
        options_se.correction_error_tolerance = 1e-3; % Norm of estimated error vector for the group
    
        fprintf('Running State Estimation (attempting iterative group correction)...\n');
        [lambdaN, success, r_norm_final, Omega_final, resid_raw_final, z_corrected_info] = ...
            LagrangianM_correct(z_full_corrupted, result_struct, 0, bus_base, options_se, R_variances_full);
        
        if ~success
            fprintf('Iteration %d: Overall WLS process did not succeed. Scale factor = %.3f\n', iter, sf);
            if isfield(z_corrected_info, 'skipped_reason') && ~isempty(z_corrected_info.skipped_reason)
                fprintf('Correction info: %s\n', z_corrected_info.skipped_reason);
            end
            continue;
        end
        
        fprintf('Overall WLS process successful.\n');
        if isfield(z_corrected_info, 'applied_any_correction') && z_corrected_info.applied_any_correction
            % fprintf('Iterative group bad data correction was applied (%d iteration(s)).\n', z_corrected_info.iterations_performed);
            if isfield(z_corrected_info, 'last_corrected_global_indices')
                % disp('Last Corrected Global Indices:'); disp(z_corrected_info.last_corrected_global_indices(:).');
                % disp('Original Bad Values (at last correction):'); disp(z_corrected_info.last_original_values(:).');
                disp('Estimated Errors (at last correction):'); disp(z_corrected_info.last_estimated_errors(:).');
                disp('Corrected Values (at last correction):'); disp(z_corrected_info.last_corrected_values(:).');
                fprintf('Norm of errors that triggered last correction: %.3e\n', z_corrected_info.last_applied_error_norm);
            end
        elseif isfield(z_corrected_info, 'skipped_reason') && ~isempty(z_corrected_info.skipped_reason)
            fprintf('Group bad data correction info: %s\n', z_corrected_info.skipped_reason);
        else
            fprintf('Group bad data correction was enabled but not applied (e.g., errors initially small or group empty).\n');
        end
    
        r_norm_fullsize = zeros(nz_actual, 1);
        r_norm_fullsize(kept_indices) = r_norm_final; 
        r_norm_fullsize(~keep_mask) = -1; 
    
        [max_r_val, max_r_idx_global] = max(r_norm_fullsize);
        fprintf('Largest normalized residual after final SE run: %.2f at index %d.\n', max_r_val, max_r_idx_global);
    
        if ismember(max_r_idx_global, suspect_group_for_corruption)
            fprintf('The largest residual is still within the original suspect group.\n');
        else
            fprintf('The largest residual is no longer in the original suspect group.\n');
        end
        
        if iter == max_iterations_main_loop && max_iterations_main_loop > 1
            disp('All iterations ended.');
        elseif max_iterations_main_loop == 1
            disp('Single run demonstration complete.');
        end
    end
