%% ================== MAIN SCRIPT EXAMPLE (Revised for Paper Output & Single/Grouped Error Comparison) ====================
clc;
clear;
close all;
%% --- 1) Load and Modify the Base Case ---
mpc_orig = loadcase('case9');
disp('Original system loaded (IEEE 9 Bus).');

mpc = mpc_orig;  % Work with a copy

nbus    = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);

%% --- 2) Run Base Case OPF ONCE ---
disp('Running base case OPF on modified system...');
mpopt_base = mpoption('verbose', 0, 'out.all', 0);
[results_opf, success_base] = runopf(mpc, mpopt_base); % Capture full results_opf

if success_base ~= 1
    error('Base case OPF failed to converge! Cannot proceed.');
end
disp('Base case OPF successful.');
baseMVA = results_opf.baseMVA; % Get baseMVA from results_opf
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
% Using variances from the user-provided script
for i = 1:nz_actual
    if i <= nbus % Voltage
        R_variances_full(i) = (0.001)^2; % StdDev 0.1%
    else % Power
        R_variances_full(i) = (0.01)^2;  % StdDev 1%
    end
end
R_variances_active = R_variances_full(kept_indices);
sigma_active = sqrt(R_variances_active);

%% --- 6) Simulate Grouped Bad Data Scenario ---
bus_of_interest = 3; 
sf = 1.3;
max_iterations_run = 1;

disp('========================================');
fprintf('Simulating GROUPED faulty PT/CT at Bus %d with %.2f%% error scaling.\n', bus_of_interest, (sf-1)*100);

corrupted_global_indices_grouped_error = []; 

% Seed for reproducibility of the main grouped error scenario
rng_seed_grouped = 42; % Choose any integer for specific reproducible noise
rng(rng_seed_grouped); 

for iter = 1:max_iterations_run
    z_noisy_active_grouped = z_base_active + sigma_active .* randn(size(sigma_active));

    temp_corrupted_indices = [];
    % Apply errors and collect actually corrupted global indices
    volt_global_idx = bus_of_interest;
    if keep_mask(volt_global_idx), [~,loc] = ismember(volt_global_idx,kept_indices); if loc>0, z_noisy_active_grouped(loc) = z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, volt_global_idx]; end; end
    
    branches_from = find(mpc.branch(:,1) == bus_of_interest);
    for ln = branches_from(:)', pf_idx=3*nbus+ln; qf_idx=3*nbus+nbranch+ln; if keep_mask(pf_idx),[~,loc]=ismember(pf_idx,kept_indices); if loc>0,z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, pf_idx];end;end; if keep_mask(qf_idx),[~,loc]=ismember(qf_idx,kept_indices); if loc>0,z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, qf_idx];end;end; end
    branches_to = find(mpc.branch(:,2) == bus_of_interest);
    for ln = branches_to(:)', pt_idx=3*nbus+2*nbranch+ln; qt_idx=3*nbus+3*nbranch+ln; if keep_mask(pt_idx),[~,loc]=ismember(pt_idx,kept_indices); if loc>0,z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, pt_idx];end;end; if keep_mask(qt_idx),[~,loc]=ismember(qt_idx,kept_indices); if loc>0,z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, qt_idx];end;end; end
    
    pinj_idx = nbus+bus_of_interest; qinj_idx = 2*nbus+bus_of_interest;
    if keep_mask(pinj_idx),[~,loc]=ismember(pinj_idx,kept_indices); if loc>0,z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, pinj_idx];end;end
    if keep_mask(qinj_idx),[~,loc]=ismember(qinj_idx,kept_indices); if loc>0,z_noisy_active_grouped(loc)=z_noisy_active_grouped(loc)*sf; temp_corrupted_indices=[temp_corrupted_indices, qinj_idx];end;end
    corrupted_global_indices_grouped_error = unique(temp_corrupted_indices);

    z_full_corrupted_grouped = ones(nz_actual, 1) * NaN;
    z_full_corrupted_grouped(kept_indices) = z_noisy_active_grouped;
    
    [~, success_se_grouped, r_norm_active_grouped, Omega_active_grouped, raw_resid_active_grouped] = LagrangianM_partial(z_full_corrupted_grouped, result_struct, 0, bus_base);
    
    if ~success_se_grouped, fprintf('WLS for grouped error did not converge.\n'); continue; end

    r_norm_full_grouped = zeros(nz_actual, 1);
    r_norm_full_grouped(kept_indices) = r_norm_active_grouped;
    [sorted_r_grouped, sorted_idx_global_grouped] = sort(abs(r_norm_full_grouped), 'descend');
    
    fprintf('\n--- Top 5 Single Normalized Residuals (Grouped Error Scenario at Bus %d) ---\n', bus_of_interest);
    fprintf('%-6s | %-12s | %-7s | %-s\n', 'Rank', 'Global Index', 'NR Val', 'Measurement Description');
    fprintf('%s\n', repmat('-', 1, 70));
    for k_rank = 1:min(5, length(sorted_r_grouped))
        global_idx = sorted_idx_global_grouped(k_rank);
        val = sorted_r_grouped(k_rank);
        desc = get_measurement_description(global_idx, nbus, nbranch, mpc.branch);
        fprintf('%-6d | %-12d | %-7.2f | %-s\n', k_rank, global_idx, val, desc);
    end
    
    max_r_val_grouped = sorted_r_grouped(1);
    max_r_idx_global_grouped = sorted_idx_global_grouped(1);

    if ~ismember(max_r_idx_global_grouped, corrupted_global_indices_grouped_error) && max_r_val_grouped > 3.0
        fprintf('\nSingle NR test (grouped error): Largest NR (%.2f at %s) is NOT among corrupted measurements.\n', ...
            max_r_val_grouped, get_measurement_description(max_r_idx_global_grouped, nbus, nbranch, mpc.branch));
        
        MD_grouped = NaN(nbus, 1);
        for bID = 1:nbus
            current_bus_group_global_indices = get_bus_group_global_indices(bID, nbus, nbranch, mpc.branch);
            [active_in_group_mask, local_indices_in_kept] = ismember(current_bus_group_global_indices, kept_indices);
            group_local_indices = local_indices_in_kept(active_in_group_mask);
            if isempty(group_local_indices) || length(group_local_indices) < 2, MD_grouped(bID) = NaN; continue; end
            r_sub = raw_resid_active_grouped(group_local_indices);
            Omega_sub = Omega_active_grouped(group_local_indices, group_local_indices);
            if rcond(full(Omega_sub)) < 1e-12, MD_grouped(bID) = NaN; continue; end
            MD_grouped(bID) = r_sub' * (Omega_sub \ r_sub) - length(r_sub);
        end
        
        [sorted_MD_grouped, sorted_MD_bus_idx_grouped] = sort(MD_grouped, 'descend', 'MissingPlacement','last');
        fprintf('\n--- Top 5 Grouped Bus Indices (Grouped Error Scenario at Bus %d) ---\n', bus_of_interest);
        fprintf('%-6s | %-10s | %-15s | %-s\n', 'Rank', 'Bus Number', 'Grouped Index', 'Group Description');
        fprintf('%s\n', repmat('-', 1, 70));
        for k_rank = 1:min(5, nbus)
            bus_num = sorted_MD_bus_idx_grouped(k_rank);
            md_val = sorted_MD_grouped(k_rank);
            if isnan(md_val), continue; end
            fprintf('%-6d | %-10d | %-15.2f | Bus %d Group\n', k_rank, bus_num, md_val, bus_num);
        end
         if ismember(bus_of_interest, sorted_MD_bus_idx_grouped(1:min(5,sum(~isnan(MD_grouped)))))
             fprintf('\nGROUPED TEST SUCCESS (Grouped Error): Bus %d correctly in top MDs.\n', bus_of_interest);
         else 
             fprintf('\nGROUPED TEST INFO (Grouped Error): Bus %d NOT in top MDs.\n', bus_of_interest); 
         end
    else
        fprintf('\nSingle NR test (grouped error): Largest NR (%.2f at %s) IS among corrupted measurements or NR <= 3.0. Grouped test analysis for misidentification might not be as illustrative here.\n', ...
             max_r_val_grouped, get_measurement_description(max_r_idx_global_grouped, nbus, nbranch, mpc.branch));
    end
end
disp('========================================');

%% --- 7) Test Single NR and Grouped Index Effectiveness for ISOLATED Errors from the Group ---
if ~isempty(corrupted_global_indices_grouped_error)
    fprintf('\n--- Testing NR & Grouped Index for ISOLATED Errors from Bus %d Group ---\n', bus_of_interest);
    num_isolated_tests = length(corrupted_global_indices_grouped_error);
    
    % Prepare a table header for isolated error test results
    fprintf('\n--- Detailed Results for Isolated Error Tests (Original Fault Group from Bus %d) ---\n', bus_of_interest);
    fprintf('%-30s | %-10s | %-35s | %-10s | %-15s | %-10s\n', ...
        'Corrupted Measurement', 'NR Correct', 'Largest NR (Val @ Loc)', ...
        'Grouped Correct', 'Largest Grouped (Val @ Bus)', 'Target Bus');
    fprintf('%s\n', repmat('-', 1, 120));

    for i_test = 1:num_isolated_tests
        cidx_global_isolated = corrupted_global_indices_grouped_error(i_test);
        desc_corrupted_isolated = get_measurement_description(cidx_global_isolated, nbus, nbranch, mpc.branch);
        
        rng(rng_seed_grouped + i_test); % Different seed for each isolated test's noise, but reproducible
        z_noisy_active_isolated = z_base_active + sigma_active .* randn(size(sigma_active));
        
        [is_active_cidx_iso, local_cidx_in_active_iso] = ismember(cidx_global_isolated, kept_indices);
        if is_active_cidx_iso
            z_noisy_active_isolated(local_cidx_in_active_iso) = z_noisy_active_isolated(local_cidx_in_active_iso) * sf; % Apply same SF
        else
            % This case should ideally not be reached if corrupted_global_indices_grouped_error is correctly populated
            fprintf('%-30s | N/A (Not Active)\n', desc_corrupted_isolated);
            continue;
        end

        z_full_isolated_err = ones(nz_actual, 1) * NaN;
        z_full_isolated_err(kept_indices) = z_noisy_active_isolated;

        [~, success_se_iso, r_norm_active_iso, Omega_active_iso, raw_resid_active_iso] = ...
            LagrangianM_partial(z_full_isolated_err, result_struct, 0, bus_base);

        if ~success_se_iso
            fprintf('%-30s | No\n', desc_corrupted_isolated);
            continue;
        end

        % NR Test Analysis for Isolated Error
        r_norm_full_iso = zeros(nz_actual, 1);
        r_norm_full_iso(kept_indices) = r_norm_active_iso;
        [max_r_iso_val, max_r_iso_global_idx] = max(abs(r_norm_full_iso));
        desc_identified_by_NR_iso = get_measurement_description(max_r_iso_global_idx, nbus, nbranch, mpc.branch);
        nr_correct_id = (max_r_iso_global_idx == cidx_global_isolated && max_r_iso_val > 3.0);
        
        % Grouped Index (Mahalanobis) Test Analysis for Isolated Error
        MD_isolated = NaN(nbus, 1);
        for bID = 1:nbus
            current_bus_group_global_indices_iso = get_bus_group_global_indices(bID, nbus, nbranch, mpc.branch);
            [active_in_group_mask_iso, local_indices_in_kept_iso] = ismember(current_bus_group_global_indices_iso, kept_indices);
            group_local_indices_iso = local_indices_in_kept_iso(active_in_group_mask_iso);
            if isempty(group_local_indices_iso) || length(group_local_indices_iso) < 2, MD_isolated(bID) = NaN; continue; end
            r_sub_iso = raw_resid_active_iso(group_local_indices_iso);
            Omega_sub_iso = Omega_active_iso(group_local_indices_iso, group_local_indices_iso);
            if rcond(full(Omega_sub_iso)) < 1e-12, MD_isolated(bID) = NaN; continue; end
            MD_isolated(bID) = r_sub_iso' * (Omega_sub_iso \ r_sub_iso) - length(r_sub_iso);
        end
        [max_MD_iso_val, identified_bus_by_MD_iso] = max(MD_isolated);
        % Determine the "true" bus associated with the isolated error cidx_global_isolated
        true_bus_of_isolated_error = get_bus_from_measurement_idx(cidx_global_isolated, nbus, nbranch, mpc.branch, bus_of_interest);
        grouped_correct_id = (identified_bus_by_MD_iso == true_bus_of_isolated_error && max_MD_iso_val > chi2inv(0.99, length(get_bus_group_global_indices(identified_bus_by_MD_iso, nbus, nbranch, mpc.branch)))); % Example threshold
        if isnan(max_MD_iso_val) || length(get_bus_group_global_indices(identified_bus_by_MD_iso, nbus, nbranch, mpc.branch)) < 2 % Handle cases where MD couldn't be calculated or group too small
            grouped_correct_id = 0; % Or some other indicator for "not applicable"
             max_MD_iso_val = NaN; % Ensure it's NaN if not properly calculated
        end


        fprintf('%-30s | %-10s | %-35s | %-10s | %-20s | %-2d\n', ...
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


% Helper function to describe measurements (no change)
function desc = get_measurement_description(global_idx, nbus, nbranch, branch_data)
    if global_idx == 0, desc = 'N/A (No significant NR)'; return; end
    if global_idx >= 1 && global_idx <= nbus
        desc = sprintf('Vm Bus %d', global_idx);
    elseif global_idx > nbus && global_idx <= 2*nbus
        desc = sprintf('Pinj Bus %d', global_idx - nbus);
    elseif global_idx > 2*nbus && global_idx <= 3*nbus
        desc = sprintf('Qinj Bus %d', global_idx - 2*nbus);
    elseif global_idx > 3*nbus && global_idx <= 3*nbus + nbranch
        branch_idx = global_idx - 3*nbus;
        desc = sprintf('Pf Br %d (%d-%d)', branch_idx, branch_data(branch_idx,1), branch_data(branch_idx,2));
    elseif global_idx > 3*nbus + nbranch && global_idx <= 3*nbus + 2*nbranch
        branch_idx = global_idx - (3*nbus + nbranch);
        desc = sprintf('Qf Br %d (%d-%d)', branch_idx, branch_data(branch_idx,1), branch_data(branch_idx,2));
    elseif global_idx > 3*nbus + 2*nbranch && global_idx <= 3*nbus + 3*nbranch
        branch_idx = global_idx - (3*nbus + 2*nbranch);
        desc = sprintf('Pt Br %d (%d-%d)', branch_idx, branch_data(branch_idx,1), branch_data(branch_idx,2));
    elseif global_idx > 3*nbus + 3*nbranch && global_idx <= 3*nbus + 4*nbranch
        branch_idx = global_idx - (3*nbus + 3*nbranch);
        desc = sprintf('Qt Br %d (%d-%d)', branch_idx, branch_data(branch_idx,1), branch_data(branch_idx,2));
    else
        desc = sprintf('Unknown (Idx %d)', global_idx);
    end
end

% Helper function to get global indices for a bus group (no change)
function bus_group_global_indices = get_bus_group_global_indices(bID, nbus, nbranch, branch_data)
    bus_group_global_indices = [];
    bus_group_global_indices = [bus_group_global_indices, bID]; % Vm
    bus_group_global_indices = [bus_group_global_indices, nbus + bID, 2*nbus + bID]; % Pinj, Qinj
    lines_from_bID = find(branch_data(:,1) == bID);
    for ln_idx = lines_from_bID(:)', bus_group_global_indices = [bus_group_global_indices, 3*nbus + ln_idx, 3*nbus + nbranch + ln_idx]; end % Pf, Qf
    lines_to_bID = find(branch_data(:,2) == bID);
    for ln_idx = lines_to_bID(:)', bus_group_global_indices = [bus_group_global_indices, 3*nbus + 2*nbranch + ln_idx, 3*nbus + 3*nbranch + ln_idx]; end % Pt, Qt
    bus_group_global_indices = unique(bus_group_global_indices);
end

% Helper function to convert boolean to Yes/No string
function str = bool_to_str(bool_val)
    if bool_val, str = 'Yes'; else, str = 'No'; end
end

% Helper function to get associated bus for a measurement index
function bus_num = get_bus_from_measurement_idx(global_idx, nbus, nbranch, branch_data, default_bus_for_flows)
    if global_idx >= 1 && global_idx <= nbus % Vm
        bus_num = global_idx;
    elseif global_idx > nbus && global_idx <= 2*nbus % Pinj
        bus_num = global_idx - nbus;
    elseif global_idx > 2*nbus && global_idx <= 3*nbus % Qinj
        bus_num = global_idx - 2*nbus;
    elseif global_idx > 3*nbus && global_idx <= 3*nbus + 4*nbranch % Flows
        % For flows, determine if it's from or to default_bus_for_flows
        % This logic assumes flows from the `default_bus_for_flows` (bus_of_interest) are the primary association
        branch_idx_Pf = global_idx - 3*nbus;
        branch_idx_Qf = global_idx - (3*nbus + nbranch);
        branch_idx_Pt = global_idx - (3*nbus + 2*nbranch);
        branch_idx_Qt = global_idx - (3*nbus + 3*nbranch);
        if branch_idx_Pf >=1 && branch_idx_Pf <= nbranch && branch_data(branch_idx_Pf,1) == default_bus_for_flows
            bus_num = default_bus_for_flows; return;
        elseif branch_idx_Qf >=1 && branch_idx_Qf <= nbranch && branch_data(branch_idx_Qf,1) == default_bus_for_flows
            bus_num = default_bus_for_flows; return;
        elseif branch_idx_Pt >=1 && branch_idx_Pt <= nbranch && branch_data(branch_idx_Pt,2) == default_bus_for_flows
            bus_num = default_bus_for_flows; return;
        elseif branch_idx_Qt >=1 && branch_idx_Qt <= nbranch && branch_data(branch_idx_Qt,2) == default_bus_for_flows
            bus_num = default_bus_for_flows; return;
        else % If not directly related to default_bus_for_flows, assign to one of its terminals
            if branch_idx_Pf >=1 && branch_idx_Pf <= nbranch, bus_num = branch_data(branch_idx_Pf,1); return;
            elseif branch_idx_Qf >=1 && branch_idx_Qf <= nbranch, bus_num = branch_data(branch_idx_Qf,1); return;
            elseif branch_idx_Pt >=1 && branch_idx_Pt <= nbranch, bus_num = branch_data(branch_idx_Pt,2); return;
            elseif branch_idx_Qt >=1 && branch_idx_Qt <= nbranch, bus_num = branch_data(branch_idx_Qt,2); return;
            else bus_num = 0; % Should not happen
            end
        end
    else
        bus_num = 0; % Unknown
    end
end