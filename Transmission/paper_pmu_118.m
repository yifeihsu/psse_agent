%% ================== MAIN SCRIPT: Correctly Modeled Hybrid SE ====================

clc;
clear;
close all;

%% --- 0) Define MATPOWER constants ---
define_constants; 

%% --- 1) Load Base Case & Define PMU Scenario ---
mpc_orig = loadcase('case118');
disp('Original system loaded (IEEE 118 Bus).');
mpc = mpc_orig;
nbus = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);
baseMVA = mpc.baseMVA;
bus_of_interest = 10;

% Define the set of buses that have PMUs
one_hop_neighbors = find_neighbors_recursive(bus_of_interest, mpc.branch, 1);
two_hop_neighbors = find_neighbors_recursive(bus_of_interest, mpc.branch, 3);
pmu_buses_set = unique([bus_of_interest; one_hop_neighbors; two_hop_neighbors]);
disp(['PMU observable area (Bus 10 and its 1 & 2-hop neighbors): ', mat2str(pmu_buses_set')]);

% Map PMU bus numbers to their row indices in mpc.bus
pmu_bus_row_indices = arrayfun(@(b) find(mpc.bus(:, BUS_I) == b, 1), pmu_buses_set);

% Identify branches for PMU current measurements
pmu_branch_measurements = []; 
for i = 1:nbranch
    if ismember(mpc.branch(i, F_BUS), pmu_buses_set)
        pmu_branch_measurements = [pmu_branch_measurements; i, mpc.branch(i, F_BUS)];
    end
    if ismember(mpc.branch(i, T_BUS), pmu_buses_set)
        pmu_branch_measurements = [pmu_branch_measurements; i, mpc.branch(i, T_BUS)];
    end
end
pmu_branch_measurements = unique(pmu_branch_measurements, 'rows');
num_pmu_current_phasors = size(pmu_branch_measurements, 1);

pmu_config.pmu_buses = pmu_buses_set;
pmu_config.pmu_branch_currents = pmu_branch_measurements;

%% --- Get Ybus, Yf, Yt ---
[Ybus, Yf, Yt] = makeYbus(baseMVA, mpc.bus, mpc.branch);

%% --- 2) Run Base Case OPF ---
disp('Running base case OPF...');
mpopt_base = mpoption('verbose', 0, 'out.all', 0, 'model', 'AC');
[results_opf, success_base] = runopf(mpc, mpopt_base);
if ~success_base, error('Base case OPF failed!'); end
disp('Base case OPF successful.');

% It is better practice to not re-reference the angles here, 
% the SE should handle its own reference.
results_opf.bus(:, 9) = results_opf.bus(:, 9) - results_opf.bus(69, 9);
bus_sol = results_opf.bus; 
branch_sol = results_opf.branch;

%% --- 3) Generate True Hybrid Measurement Vector (z_true_hybrid) ---
disp('Corrected Model: Assuming SCADA measurements exist for ALL buses and branches.');
scada_buses_for_inj_mask = true(nbus, 1); 
scada_lines_indices = (1:nbranch)';       

num_scada_buses_for_inj = sum(scada_buses_for_inj_mask);
num_scada_lines_for_flow = length(scada_lines_indices);

%% --- MODIFICATION START: Measurement vector structure (idx) changed to rectangular ---
% This structure now correctly represents SCADA everywhere, plus PMUs in rectangular coords
idx.vm_all      = (1:nbus)';
idx.vr_pmu      = (idx.vm_all(end)+1 : idx.vm_all(end)+length(pmu_buses_set))';
idx.vi_pmu      = (idx.vr_pmu(end)+1 : idx.vr_pmu(end)+length(pmu_buses_set))';
idx.ir_pmu      = (idx.vi_pmu(end)+1 : idx.vi_pmu(end)+num_pmu_current_phasors)';
idx.ii_pmu      = (idx.ir_pmu(end)+1 : idx.ir_pmu(end)+num_pmu_current_phasors)';
idx.pinj_scada  = (idx.ii_pmu(end)+1 : idx.ii_pmu(end)+num_scada_buses_for_inj)';
idx.qinj_scada  = (idx.pinj_scada(end)+1 : idx.pinj_scada(end)+num_scada_buses_for_inj)';
idx.flows_scada = (idx.qinj_scada(end)+1 : idx.qinj_scada(end)+4*num_scada_lines_for_flow)';
%% --- MODIFICATION END ---

nz_actual_hybrid = idx.flows_scada(end);
z_true_hybrid = zeros(nz_actual_hybrid, 1);

% Generate true values
z_true_hybrid(idx.vm_all) = bus_sol(:, VM);

Vm_sol_complex = bus_sol(:,VM) .* exp(1j * deg2rad(bus_sol(:,VA)));

%% --- MODIFICATION START: Generate true PMU measurements in rectangular coordinates ---
z_true_hybrid(idx.vr_pmu) = real(Vm_sol_complex(pmu_bus_row_indices));
z_true_hybrid(idx.vi_pmu) = imag(Vm_sol_complex(pmu_bus_row_indices));

[If_true_complex, It_true_complex] = get_branch_currents(mpc, Vm_sol_complex, Yf, Yt);

for k = 1:size(pmu_branch_measurements,1)
    br_idx = pmu_branch_measurements(k,1);
    measuring_bus_num = pmu_branch_measurements(k,2);
    if measuring_bus_num == mpc.branch(br_idx, F_BUS)
        current_val_complex = If_true_complex(br_idx);
    else
        current_val_complex = It_true_complex(br_idx); 
    end
    
    % Get real and imaginary parts of the current phasor
    z_true_hybrid(idx.ir_pmu(k)) = real(current_val_complex);
    z_true_hybrid(idx.ii_pmu(k)) = imag(current_val_complex);
end
%% --- MODIFICATION END ---

Pgen_bus = accumarray(results_opf.gen(:, GEN_BUS), results_opf.gen(:, PG), [nbus 1]);
Qgen_bus = accumarray(results_opf.gen(:, GEN_BUS), results_opf.gen(:, QG), [nbus 1]);
Pinj_mw_all = Pgen_bus - bus_sol(:, PD); 
Qinj_mvar_all = Qgen_bus - bus_sol(:, QD);

scada_bus_true_indices = find(scada_buses_for_inj_mask);
z_true_hybrid(idx.pinj_scada) = Pinj_mw_all(scada_bus_true_indices) / baseMVA;
z_true_hybrid(idx.qinj_scada) = Qinj_mvar_all(scada_bus_true_indices) / baseMVA;

scada_flows = [branch_sol(scada_lines_indices, PF); branch_sol(scada_lines_indices, QF); branch_sol(scada_lines_indices, PT); branch_sol(scada_lines_indices, QT)]/baseMVA;
z_true_hybrid(idx.flows_scada) = scada_flows;

%% --- 4) Measurement Variances ---
%% --- MODIFICATION START: Define variances for rectangular coordinates ---
% For rectangular coordinates, it's common to assume the variance of the
% real and imaginary parts are equal. We can derive them from polar
% uncertainties, but for simplicity, we'll define them directly.
std_dev_pmu_v_rect = 0.0001;  % For Vr and Vi
std_dev_pmu_i_rect = 0.0001;  % For Ir and Ii

std_dev_scada_vm = 0.01; 
std_dev_scada_pq_inj = 0.02; 
std_dev_scada_pq_flow = 0.02;

R_variances_full_hybrid = zeros(nz_actual_hybrid, 1);
R_variances_full_hybrid(idx.vm_all) = std_dev_scada_vm^2;
R_variances_full_hybrid(idx.pinj_scada) = std_dev_scada_pq_inj^2;
R_variances_full_hybrid(idx.qinj_scada) = std_dev_scada_pq_inj^2;
R_variances_full_hybrid(idx.flows_scada) = std_dev_scada_pq_flow^2;

% Now, OVERWRITE the variances for measurements coming from PMUs
R_variances_full_hybrid(pmu_bus_row_indices) = std_dev_pmu_v_rect^2; % Vm at PMU buses get higher precision (overwrite SCADA Vm)
R_variances_full_hybrid(idx.vr_pmu) = std_dev_pmu_v_rect^2;
R_variances_full_hybrid(idx.vi_pmu) = std_dev_pmu_v_rect^2;
R_variances_full_hybrid(idx.ir_pmu) = std_dev_pmu_i_rect^2;
R_variances_full_hybrid(idx.ii_pmu) = std_dev_pmu_i_rect^2;
%% --- MODIFICATION END ---

sigma_hybrid = sqrt(R_variances_full_hybrid);

%% --- 5) Bad Data Injection & SE Loop ---
delta_theta_bad_deg = 1; % Increased bias to make it more visible
max_iterations = 1;
disp('=============================================================================');

V_opf_complex = bus_sol(:, VM) .* exp(1j * deg2rad(bus_sol(:, VA)));
initial_state.e = real(V_opf_complex);
initial_state.f = imag(V_opf_complex);

for iter = 1:max_iterations
    noise_hybrid = sigma_hybrid .* randn(nz_actual_hybrid, 1);
    z_hybrid_with_error = z_true_hybrid + noise_hybrid;
    
    delta_theta_bad_rad = deg2rad(delta_theta_bad_deg);
    fprintf('Iter %d: Applying PMU phase bias of %.2f deg at Bus %d.\n', iter, delta_theta_bad_deg, bus_of_interest);
    
    corrupted_indices = [];
    if abs(delta_theta_bad_deg) > 1e-4
        %% --- MODIFICATION START: Bad data injection for rectangular coordinates ---
        % A phase bias is a rotation in the complex plane.
        % Z_biased = Z_original * exp(j*delta_theta)
        rotation_phasor = exp(1j * delta_theta_bad_rad);

        % Bias the V phasor measurement from the faulty PMU
        pmu_list_idx_faulty = find(pmu_buses_set == bus_of_interest);
        idx_vr_faulty = idx.vr_pmu(pmu_list_idx_faulty);
        idx_vi_faulty = idx.vi_pmu(pmu_list_idx_faulty);
        
        % Reconstruct the complex voltage from measurements and rotate it
        V_meas_complex = z_hybrid_with_error(idx_vr_faulty) + 1j * z_hybrid_with_error(idx_vi_faulty);
        V_meas_biased = V_meas_complex * rotation_phasor;
        
        % Update the measurement vector with the biased values
        z_hybrid_with_error(idx_vr_faulty) = real(V_meas_biased);
        z_hybrid_with_error(idx_vi_faulty) = imag(V_meas_biased);
        corrupted_indices = [corrupted_indices; idx_vr_faulty; idx_vi_faulty];
        
        % Bias the I phasor measurements for currents measured by the faulty PMU
        for k_cur = 1:num_pmu_current_phasors
            if pmu_branch_measurements(k_cur, 2) == bus_of_interest
                idx_ir_faulty = idx.ir_pmu(k_cur);
                idx_ii_faulty = idx.ii_pmu(k_cur);
                
                % Reconstruct the complex current from measurements and rotate it
                I_meas_complex = z_hybrid_with_error(idx_ir_faulty) + 1j * z_hybrid_with_error(idx_ii_faulty);
                I_meas_biased = I_meas_complex * rotation_phasor;
                
                % Update the measurement vector
                z_hybrid_with_error(idx_ir_faulty) = real(I_meas_biased);
                z_hybrid_with_error(idx_ii_faulty) = imag(I_meas_biased);
                corrupted_indices = [corrupted_indices; idx_ir_faulty; idx_ii_faulty];
            end
        end
        fprintf('  Global indices of measurements directly biased by PMU error: %s\n', mat2str(unique(corrupted_indices)'));
        %% --- MODIFICATION END ---
    end
    
    [success_se, r_norm_active, ~, ~, ~] = LagrangianM_partial_hybrid(z_hybrid_with_error, mpc, pmu_config, R_variances_full_hybrid, initial_state);
    
    if ~success_se, fprintf('Iter %d: State Estimator did not converge.\n', iter); continue; end
    
    r_fullsize_normalized = nan(nz_actual_hybrid, 1);
    active_mask = ~isnan(z_hybrid_with_error);
    r_fullsize_normalized(active_mask) = r_norm_active;
    
    [max_residual_val, max_residual_global_idx] = max(r_fullsize_normalized);
    
    if max_residual_val > 3
        is_max_resid_in_corrupted_set = ismember(max_residual_global_idx, corrupted_indices);
        meas_type_info = get_measurement_type_by_global_idx(max_residual_global_idx, mpc, pmu_config, idx);
        fprintf('  Largest residual (%.2f) is at global_idx %d (%s).\n', max_residual_val, max_residual_global_idx, meas_type_info);
        if is_max_resid_in_corrupted_set
            fprintf('  Result: Largest residual IS in the set of directly biased PMU measurements.\n');
        else
            fprintf('  Result: SMEARING EFFECT / MISIDENTIFICATION OBSERVED!\n');
        end
    else
        fprintf('  Largest residual (%.2f) is below threshold. No bad data detected.\n', max_residual_val);
    end
end
disp('=============================================================================');


function neighbors_set = find_neighbors_recursive(start_bus, branches, max_hops)
    define_constants; 
    all_neighbors = {start_bus};
    current_hop_buses = [start_bus];
    visited_buses = [start_bus];
    for hop = 1:max_hops
        next_hop_buses = [];
        for b = current_hop_buses'
            connected_branches_f = branches(branches(:, F_BUS) == b, :);
            connected_branches_t = branches(branches(:, T_BUS) == b, :);
            direct_connections = [connected_branches_f(:, T_BUS); connected_branches_t(:, F_BUS)];
            unique_new_neighbors = setdiff(unique(direct_connections), visited_buses);
            next_hop_buses = [next_hop_buses; unique_new_neighbors];
            visited_buses = [visited_buses; unique_new_neighbors];
        end
        if isempty(next_hop_buses), break; end
        all_neighbors{hop+1} = unique(next_hop_buses);
        current_hop_buses = unique(next_hop_buses);
    end
    neighbors_set = vertcat(all_neighbors{:});
    neighbors_set = unique(neighbors_set);
end
function desc = get_measurement_type_by_global_idx(g_idx, mpc, pmu_config, idx)
    define_constants;
    bus_data = mpc.bus;
    branch_data = mpc.branch;
    pmu_buses_list = pmu_config.pmu_buses;
    pmu_branch_meas_info = pmu_config.pmu_branch_currents;
    desc = sprintf('Global Idx %d: ', g_idx);

    if ismember(g_idx, idx.vm_all)
        bus_idx = find(idx.vm_all == g_idx);
        bus_num = bus_data(bus_idx, BUS_I);
        % This Vm measurement has lower precision unless it's at a PMU bus
        if ismember(bus_num, pmu_buses_list)
             desc = [desc, sprintf('High-Precision SCADA Vm at PMU Bus %d', bus_num)];
        else
             desc = [desc, sprintf('SCADA Vm at Bus %d', bus_num)];
        end
    elseif ismember(g_idx, idx.vr_pmu)
        pmu_list_idx = find(idx.vr_pmu == g_idx);
        desc = [desc, sprintf('PMU Vr at Bus %d', pmu_buses_list(pmu_list_idx))];
    elseif ismember(g_idx, idx.vi_pmu)
        pmu_list_idx = find(idx.vi_pmu == g_idx);
        desc = [desc, sprintf('PMU Vi at Bus %d', pmu_buses_list(pmu_list_idx))];
    elseif ismember(g_idx, idx.ir_pmu)
        list_idx = find(idx.ir_pmu == g_idx);
        branch_idx = pmu_branch_meas_info(list_idx, 1);
        meas_bus = pmu_branch_meas_info(list_idx, 2);
        desc = [desc, sprintf('PMU Ir on Branch %d-%d, at Bus %d', branch_data(branch_idx,F_BUS),branch_data(branch_idx,T_BUS), meas_bus)];
    elseif ismember(g_idx, idx.ii_pmu)
        list_idx = find(idx.ii_pmu == g_idx);
        branch_idx = pmu_branch_meas_info(list_idx, 1);
        meas_bus = pmu_branch_meas_info(list_idx, 2);
        desc = [desc, sprintf('PMU Ii on Branch %d-%d, at Bus %d', branch_data(branch_idx,F_BUS),branch_data(branch_idx,T_BUS), meas_bus)];
    elseif ismember(g_idx, idx.pinj_scada) || ismember(g_idx, idx.qinj_scada) || ismember(g_idx, idx.flows_scada)
        desc = [desc, 'SCADA measurement'];
    else
        desc = [desc, 'Unknown Measurement Type'];
    end
end
function [If, It] = get_branch_currents(mpc, Vc, Yf, Yt)
    If = Yf * Vc; It = Yt * Vc;
end