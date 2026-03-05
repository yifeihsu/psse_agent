%% ================== MAIN SCRIPT EXAMPLE (Revised for Paper Output, Single/Grouped Error Comparison, and Correction) ====================
clc;
clear;
close all;

%% --- 1) Load and Modify the Base Case ---
mpc_orig = loadcase('case9');
disp('Original system loaded (IEEE 9 Bus).');
mpc = mpc_orig;
nbus    = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);

%% --- 2) Run Base Case OPF ONCE ---
disp('Running base case OPF on modified system...');
mpopt_base = mpoption('verbose', 0, 'out.all', 0);
[results_opf, success_base] = runopf(mpc, mpopt_base);

if success_base ~= 1
    error('Base case OPF failed to converge! Cannot proceed.');
end
disp('Base case OPF successful.');
baseMVA = results_opf.baseMVA;
bus_base = results_opf.bus;
gen_base = results_opf.gen;
branch_base = results_opf.branch;

%% --- Build a "result" struct for the SE function ---
result_struct.bus    = bus_base;
result_struct.branch = branch_base;
result_struct.gen    = gen_base;
result_struct.baseMVA= baseMVA;
if isfield(results_opf, 'order')
     result_struct.order = results_opf.order;
end

%% --- 3) Generate a "complete" Measurement Vector (z_full) ---
nz_actual = nbus + nbus + nbus + 4*nbranch;
z_full = zeros(nz_actual, 1);
z_full(1:nbus) = bus_base(:, 8);  % VM
Sbus_base = makeSbus(baseMVA, bus_base, gen_base);
Pinj_base = real(Sbus_base); Qinj_base = imag(Sbus_base);
z_full(nbus+1 : 2*nbus)   = Pinj_base;
z_full(2*nbus+1 : 3*nbus) = Qinj_base;
z_full(3*nbus+1 : 3*nbus+nbranch)             = branch_base(:,14)/baseMVA;
z_full(3*nbus+nbranch+1 : 3*nbus+2*nbranch)   = branch_base(:,15)/baseMVA;
z_full(3*nbus+2*nbranch+1 : 3*nbus+3*nbranch) = branch_base(:,16)/baseMVA;
z_full(3*nbus+3*nbranch+1 : 3*nbus+4*nbranch) = branch_base(:,17)/baseMVA;

%% --- 4) Define Active Measurements ---
remove_indices = [];
keep_mask = true(1, nz_actual);
keep_mask(remove_indices) = false;
z_base_active = z_full(keep_mask);
kept_indices = find(keep_mask);

%% --- 5) Define Measurement Variances ---
R_variances_full = zeros(nz_actual, 1);
for i = 1:nz_actual
    if i <= nbus, R_variances_full(i) = (0.001)^2; else, R_variances_full(i) = (0.01)^2; end
end
R_variances_active = R_variances_full(kept_indices);
sigma_active = sqrt(R_variances_active);

%% --- 6) Simulate Grouped Bad Data Scenario & Initial Identification ---
bus_of_interest = 3;
sf = 1.3; % 30% error scaling factor
max_iterations_run = 1;
rng_seed_grouped = 42;
rng(rng_seed_grouped);

disp('========================================');
fprintf('Simulating GROUPED faulty PT/CT at Bus %d with %.2f%% error scaling.\n', bus_of_interest, (sf-1)*100);

corrupted_global_indices_grouped_error = [];
z_full_corrupted_grouped_for_correction = []; % To store the erroneous measurements for the correction step

for iter = 1:max_iterations_run
    z_noisy_active_grouped = z_base_active + sigma_active .* randn(size(sigma_active));
    original_noisy_values_for_corrupted_group = containers.Map('KeyType','double','ValueType','any');

    temp_corrupted_indices = [];
    volt_global_idx = bus_of_interest;
    if keep_mask(volt_global_idx)
        [~,loc] = ismember(volt_global_idx,kept_indices); 
        if loc>0
            original_noisy_values_for_corrupted_group(volt_global_idx) = z_noisy_active_grouped(loc);
            z_noisy_active_grouped(loc) = z_noisy_active_grouped(loc)*sf; 
            temp_corrupted_indices=[temp_corrupted_indices, volt_global_idx]; 
        end
    end
    branches_from = find(mpc.branch(:,1) == bus_of_interest);
    for ln = branches_from(:)'
        pf_idx=3*nbus+ln; qf_idx=3*nbus+nbranch+ln; 
        if keep_mask(pf_idx),[~,loc]=ismember(pf_idx,kept_indices); if loc>0, original_noisy_values_for_corrupted_group(pf_idx) = z_noisy_active_grouped(loc); z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, pf_idx];end;end
        if keep_mask(qf_idx),[~,loc]=ismember(qf_idx,kept_indices); if loc>0, original_noisy_values_for_corrupted_group(qf_idx) = z_noisy_active_grouped(loc); z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, qf_idx];end;end
    end
    branches_to = find(mpc.branch(:,2) == bus_of_interest);
    for ln = branches_to(:)'
        pt_idx=3*nbus+2*nbranch+ln; qt_idx=3*nbus+3*nbranch+ln; 
        if keep_mask(pt_idx),[~,loc]=ismember(pt_idx,kept_indices); if loc>0, original_noisy_values_for_corrupted_group(pt_idx) = z_noisy_active_grouped(loc); z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, pt_idx];end;end
        if keep_mask(qt_idx),[~,loc]=ismember(qt_idx,kept_indices); if loc>0, original_noisy_values_for_corrupted_group(qt_idx) = z_noisy_active_grouped(loc); z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, qt_idx];end;end
    end
    pinj_idx = nbus+bus_of_interest; qinj_idx = 2*nbus+bus_of_interest;
    if keep_mask(pinj_idx),[~,loc]=ismember(pinj_idx,kept_indices); if loc>0, original_noisy_values_for_corrupted_group(pinj_idx) = z_noisy_active_grouped(loc); z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, pinj_idx];end;end
    if keep_mask(qinj_idx),[~,loc]=ismember(qinj_idx,kept_indices); if loc>0, original_noisy_values_for_corrupted_group(qinj_idx) = z_noisy_active_grouped(loc); z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, qinj_idx];end;end
    corrupted_global_indices_grouped_error = unique(temp_corrupted_indices);

    z_full_corrupted_grouped = ones(nz_actual, 1) * NaN;
    z_full_corrupted_grouped(kept_indices) = z_noisy_active_grouped;
    z_full_corrupted_grouped_for_correction = z_full_corrupted_grouped; % Save for correction step

    % Initial SE run (before correction) using LagrangianM_partial
    [~, success_se_initial, r_norm_active_initial, Omega_active_initial, raw_resid_active_initial] = LagrangianM_partial(z_full_corrupted_grouped, result_struct, 0, bus_base);

    if ~success_se_initial, fprintf('Initial WLS for grouped error did not converge.\n'); continue; end

    r_norm_full_initial = zeros(nz_actual, 1);
    r_norm_full_initial(kept_indices) = r_norm_active_initial;
    [sorted_r_initial, sorted_idx_global_initial] = sort(abs(r_norm_full_initial), 'descend');

    fprintf('\n--- Top 5 Single Normalized Residuals (Before Correction - Grouped Error at Bus %d) ---\n', bus_of_interest);
    fprintf('%-6s | %-12s | %-7s | %-s\n', 'Rank', 'Global Index', 'NR Val', 'Measurement Description');
    fprintf('%s\n', repmat('-', 1, 70));
    for k_rank = 1:min(5, length(sorted_r_initial))
        global_idx = sorted_idx_global_initial(k_rank);
        val = sorted_r_initial(k_rank);
        desc = get_measurement_description(global_idx, nbus, nbranch, mpc.branch);
        fprintf('%-6d | %-12d | %-7.2f | %-s\n', k_rank, global_idx, val, desc);
    end
    max_r_val_initial = sorted_r_initial(1); % Store this for paper

    max_r_idx_global_initial = sorted_idx_global_initial(1);
    if ~ismember(max_r_idx_global_initial, corrupted_global_indices_grouped_error) && max_r_val_initial > 3.0
        fprintf('\nSingle NR test (before correction): Largest NR (%.2f at %s) is NOT among corrupted measurements.\n', ...
            max_r_val_initial, get_measurement_description(max_r_idx_global_initial, nbus, nbranch, mpc.branch));

        MD_initial = NaN(nbus, 1);
        for bID = 1:nbus
            current_bus_group_global_indices = get_bus_group_global_indices(bID, nbus, nbranch, mpc.branch);
            [active_in_group_mask, local_indices_in_kept] = ismember(current_bus_group_global_indices, kept_indices);
            group_local_indices = local_indices_in_kept(active_in_group_mask);
            if isempty(group_local_indices) || length(group_local_indices) < 2, MD_initial(bID) = NaN; continue; end
            r_sub = raw_resid_active_initial(group_local_indices);
            Omega_sub = Omega_active_initial(group_local_indices, group_local_indices);
            if rcond(full(Omega_sub)) < 1e-12, MD_initial(bID) = NaN; continue; end
            MD_initial(bID) = r_sub' * (Omega_sub \ r_sub) - length(r_sub);
        end

        [sorted_MD_initial, sorted_MD_bus_idx_initial] = sort(MD_initial, 'descend', 'MissingPlacement','last');
        fprintf('\n--- Top 5 Grouped Bus Indices (Before Correction - Grouped Error at Bus %d) ---\n', bus_of_interest);
        fprintf('%-6s | %-10s | %-15s | %-s\n', 'Rank', 'Bus Number', 'Grouped Index', 'Group Description');
        fprintf('%s\n', repmat('-', 1, 70));
        for k_rank = 1:min(5, nbus)
            bus_num = sorted_MD_bus_idx_initial(k_rank);
            md_val = sorted_MD_initial(k_rank);
            if isnan(md_val), continue; end
            fprintf('%-6d | %-10d | %-15.2f | Bus %d Group\n', k_rank, bus_num, md_val, bus_num);
        end
        if ismember(bus_of_interest, sorted_MD_bus_idx_initial(1:min(5,sum(~isnan(MD_initial)))))
             fprintf('\nGROUPED TEST SUCCESS (Before Correction): Bus %d correctly in top MDs.\n', bus_of_interest);
         else
             fprintf('\nGROUPED TEST INFO (Before Correction): Bus %d NOT in top MDs.\n', bus_of_interest);
         end
    else
         fprintf('\nSingle NR test (before correction): Largest NR (%.2f at %s) IS among corrupted or NR <= 3.0.\n', ...
             max_r_val_initial, get_measurement_description(max_r_idx_global_initial, nbus, nbranch, mpc.branch));
    end
end
disp('========================================');

%% --- 6.1) Apply Grouped Correction using LagrangianM_correct ---
if success_se_initial % Proceed only if initial SE converged
    fprintf('\n--- Applying Iterative Grouped Correction for Bus %d Fault ---\n', bus_of_interest);
    options_se.enable_group_correction = true;
    % The suspect group are all measurements related to bus_of_interest
    options_se.correction_group_full_indices = get_bus_group_global_indices(bus_of_interest, nbus, nbranch, mpc.branch);
    options_se.max_correction_iterations = 5; 
    options_se.correction_error_tolerance = 1e-3;

    [~, success_se_corrected, r_norm_final_corrected, ~, raw_resid_final_corrected, z_corrected_info] = ...
        LagrangianM_correct(z_full_corrupted_grouped_for_correction, result_struct, 0, bus_base, options_se, R_variances_full);

    if success_se_corrected
        fprintf('Iterative Grouped Correction Process Converged Successfully.\n');
        if z_corrected_info.applied_any_correction
            fprintf('  Correction was applied for %d iteration(s).\n', z_corrected_info.iterations_performed);
            fprintf('  Final norm of estimated errors that triggered last correction: %.3e\n', z_corrected_info.last_applied_error_norm);
            
            r_norm_full_after_correction = zeros(nz_actual, 1);
            r_norm_full_after_correction(kept_indices) = r_norm_final_corrected; % Assuming r_norm_final_corrected corresponds to kept_indices
            [max_r_val_after_correction, ~] = max(abs(r_norm_full_after_correction));
            fprintf('  Largest NR after correction: %.2f (was %.2f before correction).\n', max_r_val_after_correction, max_r_val_initial);

            % Prepare data for Table Y (Numerical Results of Grouped Measurement Correction)
            fprintf('\n--- Numerical Results of Grouped Measurement Correction for Bus %d Fault ---\n', bus_of_interest);
            fprintf('%-30s | %-15s | %-15s | %-15s | %-15s\n', 'Measurement', 'Original Err Val', 'True Noisy Val', 'Estimated Error', 'Corrected Val');
            fprintf('%s\n', repmat('-',1,100));
            
            % Display for a few key measurements from the corrected group
            display_limit = min(length(z_corrected_info.last_corrected_global_indices), 5); % Display up to 5
            for i_disp = 1:display_limit
                glob_idx = z_corrected_info.last_corrected_global_indices(i_disp);
                desc = get_measurement_description(glob_idx, nbus, nbranch, mpc.branch);
                original_err_val = z_full_corrupted_grouped_for_correction(glob_idx); % Value that went into LagrangianM_correct
                
                % Retrieve the original noisy value before gross error sf was applied
                % This requires that original_noisy_values_for_corrupted_group was populated correctly
                true_noisy_val = NaN; % Default if not found
                if isKey(original_noisy_values_for_corrupted_group, glob_idx)
                    true_noisy_val = original_noisy_values_for_corrupted_group(glob_idx);
                end

                est_err = z_corrected_info.last_estimated_errors(i_disp);
                corr_val = z_corrected_info.last_corrected_values(i_disp);
                fprintf('%-30s | %-15.4f | %-15.4f | %-15.4f | %-15.4f\n', desc, original_err_val, true_noisy_val, est_err, corr_val);
            end

        else
            fprintf('  Correction was enabled, but not applied (e.g., errors initially small or group empty).\n');
            if isfield(z_corrected_info, 'skipped_reason') && ~isempty(z_corrected_info.skipped_reason)
                fprintf('  Reason: %s\n', z_corrected_info.skipped_reason);
            end
        end
    else
        fprintf('Iterative Grouped Correction Process Did NOT Converge.\n');
         if isfield(z_corrected_info, 'skipped_reason') && ~isempty(z_corrected_info.skipped_reason)
            fprintf('  Reason: %s\n', z_corrected_info.skipped_reason);
        end
    end
else
    fprintf('\nInitial SE for grouped error failed, skipping correction step.\n');
end
disp('========================================');


%% --- 7) Test Single NR and Grouped Index Effectiveness for ISOLATED Errors from the Group ---
if ~isempty(corrupted_global_indices_grouped_error)
    fprintf('\n--- Testing NR & Grouped Index for ISOLATED Errors from Bus %d Group ---\n', bus_of_interest);
    num_isolated_tests = length(corrupted_global_indices_grouped_error);
    
    fprintf('\n--- Detailed Results for Isolated Error Tests (Original Fault Group from Bus %d) ---\n', bus_of_interest);
    fprintf('%-30s | %-12s | %-40s | %-12s | %-25s | %-10s\n', ...
        'Corrupted Measurement', 'NR Correct?', 'Largest NR (Value @ Location)', ...
        'Grouped Correct?', 'Largest Grouped (Value @ Bus)', 'Target Bus');
    fprintf('%s\n', repmat('-', 1, 135));

    for i_test = 1:num_isolated_tests
        cidx_global_isolated = corrupted_global_indices_grouped_error(i_test);
        desc_corrupted_isolated = get_measurement_description(cidx_global_isolated, nbus, nbranch, mpc.branch);
        
        rng(rng_seed_grouped + i_test); 
        z_noisy_active_isolated = z_base_active + sigma_active .* randn(size(sigma_active));
        
        [is_active_cidx_iso, local_cidx_in_active_iso] = ismember(cidx_global_isolated, kept_indices);
        if is_active_cidx_iso
            z_noisy_active_isolated(local_cidx_in_active_iso) = z_noisy_active_isolated(local_cidx_in_active_iso) * sf;
        else
            fprintf('%-30s | N/A (Not Active)\n', desc_corrupted_isolated);
            continue;
        end
        z_full_isolated_err = ones(nz_actual, 1) * NaN;
        z_full_isolated_err(kept_indices) = z_noisy_active_isolated;

        [~, success_se_iso, r_norm_active_iso, Omega_active_iso, raw_resid_active_iso] = ...
            LagrangianM_partial(z_full_isolated_err, result_struct, 0, bus_base);

        if ~success_se_iso
            fprintf('%-30s | %-12s | %-40s | %-12s | %-25s | %-10s\n', desc_corrupted_isolated, 'No Conv.', 'N/A', 'No Conv.', 'N/A', 'N/A');
            continue;
        end

        r_norm_full_iso = zeros(nz_actual, 1);
        r_norm_full_iso(kept_indices) = r_norm_active_iso;
        [max_r_iso_val, max_r_iso_global_idx] = max(abs(r_norm_full_iso));
        desc_identified_by_NR_iso = get_measurement_description(max_r_iso_global_idx, nbus, nbranch, mpc.branch);
        nr_correct_id = (max_r_iso_global_idx == cidx_global_isolated && max_r_iso_val > 3.0);
        
        MD_isolated = NaN(nbus, 1);
        chi2_threshold_md = chi2inv(0.99,1); % Placeholder, df should be group size
        for bID = 1:nbus
            current_bus_group_global_indices_iso = get_bus_group_global_indices(bID, nbus, nbranch, mpc.branch);
            [active_in_group_mask_iso, local_indices_in_kept_iso] = ismember(current_bus_group_global_indices_iso, kept_indices);
            group_local_indices_iso = local_indices_in_kept_iso(active_in_group_mask_iso);
            
            df_group = length(group_local_indices_iso);
            if isempty(group_local_indices_iso) || df_group < 2, MD_isolated(bID) = NaN; continue; end
            
            r_sub_iso = raw_resid_active_iso(group_local_indices_iso);
            Omega_sub_iso = Omega_active_iso(group_local_indices_iso, group_local_indices_iso);
            if rcond(full(Omega_sub_iso)) < 1e-12, MD_isolated(bID) = NaN; continue; end
            MD_isolated(bID) = r_sub_iso' * (Omega_sub_iso \ r_sub_iso); % MD value
            % For chi-squared adjusted: MD_isolated(bID) = r_sub_iso' * (Omega_sub_iso \ r_sub_iso) - df_group;
            if bID ==1 % Update threshold based on actual first group size for example, can be more specific
                 chi2_threshold_md = chi2inv(0.99,df_group) ;
            end

        end
        [max_MD_iso_val, identified_bus_by_MD_iso] = max(MD_isolated);
        true_bus_of_isolated_error = get_bus_from_measurement_idx(cidx_global_isolated, nbus, nbranch, mpc.branch, bus_of_interest);
        
        % Determine df for the identified group for thresholding
        df_identified_group = NaN;
        if ~isnan(identified_bus_by_MD_iso) && identified_bus_by_MD_iso > 0 && identified_bus_by_MD_iso <= nbus
            temp_group_indices_for_df = get_bus_group_global_indices(identified_bus_by_MD_iso, nbus, nbranch, mpc.branch);
            [active_mask_for_df, ~] = ismember(temp_group_indices_for_df, kept_indices);
            df_identified_group = sum(active_mask_for_df);
        end
        
        grouped_correct_id = false;
        if ~isnan(max_MD_iso_val) && ~isnan(df_identified_group) && df_identified_group >=1
             current_chi2_threshold = chi2inv(0.99, df_identified_group);
             grouped_correct_id = (identified_bus_by_MD_iso == true_bus_of_isolated_error && max_MD_iso_val > current_chi2_threshold );
        elseif isnan(max_MD_iso_val) % if MD calculation failed or group too small
            max_MD_iso_val = NaN; % Ensure it's NaN for display
            identified_bus_by_MD_iso = 0; % Indicate no bus identified
        end


        fprintf('%-30s | %-12s | %-40s | %-12s | %-25s | %-10d\n', ...
            desc_corrupted_isolated, ...
            bool_to_str(nr_correct_id), ...
            sprintf('%.2f at %s', max_r_iso_val, desc_identified_by_NR_iso), ...
            bool_to_str(grouped_correct_id), ...
            sprintf('%.2f at Bus %d', max_MD_iso_val, identified_bus_by_MD_iso), ...
            true_bus_of_isolated_error);
    end
else
    fprintf('\nNo corrupted indices from grouped error scenario to test individually.\n');
end
disp('========================================');


% Helper function to describe measurements
function desc = get_measurement_description(global_idx, nbus, nbranch, branch_data)
    if global_idx == 0, desc = 'N/A (No sig. NR)'; return; end
    if global_idx >= 1 && global_idx <= nbus, desc = sprintf('Vm Bus %d', global_idx);
    elseif global_idx > nbus && global_idx <= 2*nbus, desc = sprintf('Pinj Bus %d', global_idx - nbus);
    elseif global_idx > 2*nbus && global_idx <= 3*nbus, desc = sprintf('Qinj Bus %d', global_idx - 2*nbus);
    elseif global_idx > 3*nbus && global_idx <= 3*nbus + nbranch, branch_idx = global_idx - 3*nbus; desc = sprintf('Pf Br %d (%d-%d)', branch_idx, branch_data(branch_idx,1), branch_data(branch_idx,2));
    elseif global_idx > 3*nbus + nbranch && global_idx <= 3*nbus + 2*nbranch, branch_idx = global_idx - (3*nbus + nbranch); desc = sprintf('Qf Br %d (%d-%d)', branch_idx, branch_data(branch_idx,1), branch_data(branch_idx,2));
    elseif global_idx > 3*nbus + 2*nbranch && global_idx <= 3*nbus + 3*nbranch, branch_idx = global_idx - (3*nbus + 2*nbranch); desc = sprintf('Pt Br %d (%d-%d)', branch_idx, branch_data(branch_idx,1), branch_data(branch_idx,2));
    elseif global_idx > 3*nbus + 3*nbranch && global_idx <= 3*nbus + 4*nbranch, branch_idx = global_idx - (3*nbus + 3*nbranch); desc = sprintf('Qt Br %d (%d-%d)', branch_idx, branch_data(branch_idx,1), branch_data(branch_idx,2));
    else desc = sprintf('Unknown (Idx %d)', global_idx); end
end

% Helper function to get global indices for a bus group
function bus_group_global_indices = get_bus_group_global_indices(bID, nbus, nbranch, branch_data)
    bus_group_global_indices = [bID, nbus + bID, 2*nbus + bID]; % Vm, Pinj, Qinj
    lines_from_bID = find(branch_data(:,1) == bID);
    for ln_idx = lines_from_bID(:)', bus_group_global_indices = [bus_group_global_indices, 3*nbus + ln_idx, 3*nbus + nbranch + ln_idx]; end
    lines_to_bID = find(branch_data(:,2) == bID);
    for ln_idx = lines_to_bID(:)', bus_group_global_indices = [bus_group_global_indices, 3*nbus + 2*nbranch + ln_idx, 3*nbus + 3*nbranch + ln_idx]; end
    bus_group_global_indices = unique(bus_group_global_indices);
end

% Helper function to convert boolean to Yes/No string
function str = bool_to_str(bool_val)
    if bool_val, str = 'Yes'; else, str = 'No'; end
end

% Helper function to get associated bus for a measurement index
function bus_num = get_bus_from_measurement_idx(global_idx, nbus, ~, branch_data, bus_of_interest_orig)
    if global_idx >= 1 && global_idx <= nbus, bus_num = global_idx; % Vm
    elseif global_idx > nbus && global_idx <= 2*nbus, bus_num = global_idx - nbus; % Pinj
    elseif global_idx > 2*nbus && global_idx <= 3*nbus, bus_num = global_idx - 2*nbus; % Qinj
    else % Flows - primary association is the bus_of_interest_orig if it's a terminal
        branch_idx = 0; from_bus = 0; to_bus = 0;
        if global_idx > 3*nbus && global_idx <= 3*nbus + size(branch_data,1) % Pf
            branch_idx = global_idx - 3*nbus;
        elseif global_idx > 3*nbus + size(branch_data,1) && global_idx <= 3*nbus + 2*size(branch_data,1) % Qf
            branch_idx = global_idx - (3*nbus + size(branch_data,1));
        elseif global_idx > 3*nbus + 2*size(branch_data,1) && global_idx <= 3*nbus + 3*size(branch_data,1) % Pt
            branch_idx = global_idx - (3*nbus + 2*size(branch_data,1));
        elseif global_idx > 3*nbus + 3*size(branch_data,1) && global_idx <= 3*nbus + 4*size(branch_data,1) % Qt
            branch_idx = global_idx - (3*nbus + 3*size(branch_data,1));
        end
        if branch_idx > 0
            from_bus = branch_data(branch_idx,1);
            to_bus = branch_data(branch_idx,2);
            if from_bus == bus_of_interest_orig || to_bus == bus_of_interest_orig
                bus_num = bus_of_interest_orig; % If part of original structural fault, associate with that bus
            else % Otherwise, just pick one of its terminals (e.g. from_bus)
                bus_num = from_bus; 
            end
        else
            bus_num = 0; % Should not happen if global_idx is valid
        end
    end
end