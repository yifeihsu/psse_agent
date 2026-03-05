%% ================== MAIN SCRIPT EXAMPLE (Revised for Phase Angle Bad Data) ====================
clc;
clear;
close all;

%% --- 0) Define MATPOWER constants ---
define_constants; % Defines F_BUS, T_BUS, VM, VA, PD, QG, PG, QG, GEN_BUS etc.

%% --- 1) Load and Modify the Base Case ---
mpc_orig = loadcase('case118');
disp('Original system loaded (IEEE 118 Bus).');

mpc = mpc_orig;  % Work with a copy
% mpc.branch(:, 6:8) = 0;  % zero out constraints if needed

nbus    = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);

%% --- Get Ybus, Yf, Yt (needed for recalculating flows/injections) ---
% These are based on the network structure and parameters from mpc
% makeYbus returns admittances in p.u.
[Ybus, Yf, Yt] = makeYbus(mpc.baseMVA, mpc.bus, mpc.branch);
disp('Ybus, Yf, Yt matrices computed (in p.u.).');

%% --- 2) Run Base Case OPF ONCE ---
disp('Running base case OPF on modified system...');
mpopt_base = mpoption('verbose', 0, 'out.all', 1); 
% Explicitly request AC OPF
mpopt_base = mpoption(mpopt_base, 'model', 'AC');
[results_opf, success_base] = runopf(mpc, mpopt_base);

if success_base ~= 1
    error('Base case OPF failed to converge! Cannot proceed.');
end
disp('Base case OPF successful.');

% Extract results from the OPF output structure
bus_base    = results_opf.bus;    % Bus data (includes Vm in p.u., Va in degrees)
gen_base    = results_opf.gen;    % Gen data (includes PG, QG in MW/MVAr)
branch_base = results_opf.branch; % Branch data (includes PF, QF, PT, QT in MW/MVAr)
baseMVA     = results_opf.baseMVA;


%% --- Build a "result" struct for the SE function ---
% This struct should contain the true system state for the SE
result_struct.bus    = bus_base;
result_struct.branch = branch_base;
result_struct.gen    = gen_base;
result_struct.baseMVA= baseMVA;
result_struct.order = results_opf.order; % Include order if SE function needs it for internal bus ordering

%% --- 3) Generate a "complete" Measurement Vector (z_full) ---
% These are the TRUE values from the base case OPF, converted to p.u.
nz_actual = nbus + nbus + nbus + 4*nbranch; % VM, Pinj, Qinj, Pf, Qf, Pt, Qt
z_full = zeros(nz_actual, 1);

% 1) Voltage magnitudes (VM) - already in p.u. from bus_base
z_full(1:nbus) = bus_base(:, VM);

% 2) Nodal power injections (Pinj, Qinj)
%    Convert from MW/MVAr to p.u.
Pgen_bus = zeros(nbus, 1); % MW
Qgen_bus = zeros(nbus, 1); % MVAr
% Aggregate generation at each bus
if ~isempty(gen_base) % Check if there are any generators
    for i = 1:size(gen_base, 1)
        bus_idx = gen_base(i, GEN_BUS);
        % Ensure bus_idx is valid for Pgen_bus and Qgen_bus
        if bus_idx > 0 && bus_idx <= nbus
            Pgen_bus(bus_idx) = Pgen_bus(bus_idx) + gen_base(i, PG); % PG is in MW
            Qgen_bus(bus_idx) = Qgen_bus(bus_idx) + gen_base(i, QG); % QG is in MVAr
        else
            warning('Generator %d connected to invalid bus index %d. Skipping.', i, bus_idx);
        end
    end
end
% Net injection = Generation - Load (all in MW/MVAr initially)
Pinj_base_mw = Pgen_bus - bus_base(:, PD); % PD is in MW
Qinj_base_mvar = Qgen_bus - bus_base(:, QD); % QD is in MVAr

z_full(nbus+1 : 2*nbus)   = Pinj_base_mw / baseMVA;   % Convert to p.u.
z_full(2*nbus+1 : 3*nbus) = Qinj_base_mvar / baseMVA; % Convert to p.u.

% 3) Branch flows (PF, QF, PT, QT) - from branch_base (MW/MVAr), convert to p.u.
z_full(3*nbus+1 : 3*nbus+nbranch)             = branch_base(:,PF)/baseMVA;
z_full(3*nbus+nbranch+1 : 3*nbus+2*nbranch)   = branch_base(:,QF)/baseMVA;
z_full(3*nbus+2*nbranch+1 : 3*nbus+3*nbranch) = branch_base(:,PT)/baseMVA;
z_full(3*nbus+3*nbranch+1 : 3*nbus+4*nbranch) = branch_base(:,QT)/baseMVA;

%% --- 4) Remove (or De-Weight) Some Measurements ---
% Example: remove all P and Q injections (as in many SE setups)
buses_to_remove_inj = 1:nbus; 
remove_P_idx = nbus + buses_to_remove_inj;
remove_Q_idx = 2*nbus + buses_to_remove_inj;

all_indices = 1:nz_actual;
% remove_indices = [remove_P_idx, remove_Q_idx]; % Uncomment to remove injections
remove_indices = []; % Keep all measurements for this example
keep_mask = true(1, nz_actual);
keep_mask(remove_indices) = false;

z_base = z_full(keep_mask); % These are the TRUE values (in p.u.) of the KEPT measurements
kept_indices = find(keep_mask); % Global indices of the kept measurements

%% --- 5) Define Measurement Noise Variances ---
% R_variances will be for the 'kept' measurements (all in p.u.)
R_variances = zeros(length(z_base), 1);
for i = 1:length(z_base)
    global_idx = kept_indices(i);
    if global_idx <= nbus % Voltage magnitude measurements (p.u.)
        R_variances(i) = (0.005)^2; % e.g., std dev of 0.005 p.u. (0.5% of nominal V)
    elseif global_idx <= 3*nbus % Power injection measurements (p.u.)
        R_variances(i) = (0.015)^2; % e.g., std dev of 0.015 p.u.
    else % Power flow measurements (p.u.)
        R_variances(i) = (0.01)^2;  % e.g., std dev of 0.01 p.u.
    end
end
sigma = sqrt(R_variances); % Standard deviations for KEPT measurements (p.u.)

%% --- 6) Bad Data Injection: Perturb Voltage Phase Angle at 'bus_of_interest' ---
bus_of_interest = 10; % Example: Bus 10's phase angle measurement is faulty
disp(['Simulating bad data due to phase angle error at Bus: ', num2str(bus_of_interest)]);

% Phase angle error parameters (in degrees)
delta_theta_bad_deg_min = 2.0;
delta_theta_bad_deg_max = 10.0;

max_iterations   = 100;

disp('=============================================================================');
disp('Starting simulation loop for bad data injection (phase angle perturbation)...');
for iter = 1:max_iterations

    % 6a) Get true values for the kept measurements (this is z_base, already in p.u.)

    % 6b) Generate normal measurement noise for all kept measurements (noise is in p.u.)
    noise_on_kept = sigma .* randn(size(z_base));
    
    % This vector represents what would be measured if there were no bad data,
    % just true values (p.u.) + normal noise (p.u.). We will modify this.
    z_iter_final_kept = z_base + noise_on_kept;

    % 6c) Introduce the phase angle error and recalculate affected measurements
    % Generate a random phase angle error for this iteration
    delta_theta_bad_deg = delta_theta_bad_deg_min + (delta_theta_bad_deg_max - delta_theta_bad_deg_min) * rand(1);
    delta_theta_bad_rad = deg2rad(delta_theta_bad_deg);
    fprintf('Iter %d: Applying phase angle error of %.2f degrees at bus %d.\n', iter, delta_theta_bad_deg, bus_of_interest);

    % Get true Vm (p.u.) and Va (radians) from the base case OPF results
    Vm_true_all     = bus_base(:, VM);         % All bus voltage magnitudes (p.u.)
    Va_true_all_rad = deg2rad(bus_base(:, VA));% All bus voltage angles (radians)

    % Create the perturbed complex voltage vector for the entire system (p.u.)
    % Magnitudes (Vm_true_all) remain unchanged from the true state.
    % Only the phase angle at bus_of_interest is altered.
    Va_pert_all_rad = Va_true_all_rad;
    Va_pert_all_rad(bus_of_interest) = Va_true_all_rad(bus_of_interest) + delta_theta_bad_rad;
    V_complex_pert_all = Vm_true_all .* exp(1j * Va_pert_all_rad); % Complex voltages in p.u.

    % Recalculate ALL power flows and injections using the perturbed complex voltage vector.
    % Ybus, Yf, Yt are in p.u. V_complex_pert_all is in p.u.
    % So, the resulting complex powers will be in p.u.
    Sf_pert_all_pu = V_complex_pert_all(mpc.branch(:, F_BUS)) .* conj(Yf * V_complex_pert_all); % From-end complex power flow (p.u.)
    St_pert_all_pu = V_complex_pert_all(mpc.branch(:, T_BUS)) .* conj(Yt * V_complex_pert_all); % To-end complex power flow (p.u.)
    Sbus_pert_all_pu = V_complex_pert_all .* conj(Ybus * V_complex_pert_all); % Nodal complex power injection (p.u.)

    % Extract real and imaginary parts (already in per unit)
    Pf_pert_pu_all   = real(Sf_pert_all_pu);
    Qf_pert_pu_all   = imag(Sf_pert_all_pu);
    Pt_pert_pu_all   = real(St_pert_all_pu);
    Qt_pert_pu_all   = imag(St_pert_all_pu);
    Pinj_pert_pu_all = real(Sbus_pert_all_pu);
    Qinj_pert_pu_all = imag(Sbus_pert_all_pu);

    % This list will store the global indices of measurements whose underlying mean
    % has been changed due to the phase angle error.
    current_iter_corrupted_global_indices = [];

    % Iterate through the KEPT measurements and update those affected by the bad data
    for k_meas = 1:length(kept_indices)
        global_idx = kept_indices(k_meas); % Get the global index of this measurement
        original_noise_component = noise_on_kept(k_meas); % Noise for this specific measurement (p.u.)

        is_affected_by_phase_error = false;
        new_bad_mean_value_pu = 0; % This will store the new mean in p.u.

        % --- Check if this measurement is affected by the phase error at bus_of_interest ---
        
        % 1. Power Injection Measurements at bus_of_interest
        if global_idx == (nbus + bus_of_interest) % P_inj at bus_of_interest (p.u.)
            new_bad_mean_value_pu = Pinj_pert_pu_all(bus_of_interest);
            is_affected_by_phase_error = true;
        elseif global_idx == (2*nbus + bus_of_interest) % Q_inj at bus_of_interest (p.u.)
            new_bad_mean_value_pu = Qinj_pert_pu_all(bus_of_interest);
            is_affected_by_phase_error = true;
        
        % 2. Power Flow Measurements on branches connected to bus_of_interest (all p.u.)
        % Pf indices: (3*nbus + 1) to (3*nbus + nbranch)
        % Qf indices: (3*nbus + nbranch + 1) to (3*nbus + 2*nbranch)
        % Pt indices: (3*nbus + 2*nbranch + 1) to (3*nbus + 3*nbranch)
        % Qt indices: (3*nbus + 3*nbranch + 1) to (3*nbus + 4*nbranch)
        elseif global_idx > 3*nbus && global_idx <= (3*nbus + 4*nbranch)
            branch_idx = 0;
            flow_type = '';

            if global_idx <= (3*nbus + nbranch) % Pf
                branch_idx = global_idx - (3*nbus);
                flow_type = 'Pf';
            elseif global_idx <= (3*nbus + 2*nbranch) % Qf
                branch_idx = global_idx - (3*nbus + nbranch);
                flow_type = 'Qf';
            elseif global_idx <= (3*nbus + 3*nbranch) % Pt
                branch_idx = global_idx - (3*nbus + 2*nbranch);
                flow_type = 'Pt';
            else % Qt (global_idx <= (3*nbus + 4*nbranch))
                branch_idx = global_idx - (3*nbus + 3*nbranch);
                flow_type = 'Qt';
            end
            
            % Check if this branch is connected to bus_of_interest
            if mpc.branch(branch_idx, F_BUS) == bus_of_interest || mpc.branch(branch_idx, T_BUS) == bus_of_interest
                is_affected_by_phase_error = true;
                switch flow_type
                    case 'Pf'
                        new_bad_mean_value_pu = Pf_pert_pu_all(branch_idx);
                    case 'Qf'
                        new_bad_mean_value_pu = Qf_pert_pu_all(branch_idx);
                    case 'Pt'
                        new_bad_mean_value_pu = Pt_pert_pu_all(branch_idx);
                    case 'Qt'
                        new_bad_mean_value_pu = Qt_pert_pu_all(branch_idx);
                end
            end
        end
        
        % If the measurement is affected, replace its value in z_iter_final_kept
        % The new value is the (recalculated mean due to phase error, in p.u.) + (original noise, in p.u.)
        if is_affected_by_phase_error
            z_iter_final_kept(k_meas) = new_bad_mean_value_pu + original_noise_component;
            current_iter_corrupted_global_indices = [current_iter_corrupted_global_indices, global_idx];
        end
        
        % IMPORTANT: Voltage magnitude measurements (global_idx <= nbus) are NOT
        % modified here by the phase angle error. They retain their (true_value_pu + noise_pu).
    end
    actual_corrupted_global_indices_this_iter = unique(current_iter_corrupted_global_indices);

    % 6d) Re-insert the (potentially corrupted) kept measurements into a "full" measurement vector
    z_full_corrupted = zeros(nz_actual, 1); % Will hold all measurements in p.u.
    z_full_corrupted(keep_mask) = z_iter_final_kept; % z_iter_final_kept now has bad data (in p.u.)
    z_full_corrupted(~keep_mask) = 999; % Or NaN, for unmeasured/removed values

    % 6e) Run the state estimator (assuming LagrangianM_partial is your SE function)
    % The 'bus_base' here is used as the initial guess for the SE.
    % The 'result_struct' provides the true system parameters and OPF solution for comparison/reference if needed by SE.
    % The SE function should expect measurements in p.u.
    [lambdaN, success_se, r_normalized_residuals] = LagrangianM_partial(z_full_corrupted, result_struct, 0, bus_base);
    
    if ~success_se
        fprintf('Iteration %d: State Estimator did not converge. Phase error = %.3f deg.\n', iter, delta_theta_bad_deg);
        continue; % Skip to next iteration
    end

    % Extend the normalized residual vector back to full size for easy indexing
    r_fullsize_normalized = zeros(nz_actual, 1);
    r_fullsize_normalized(keep_mask) = r_normalized_residuals; % Assuming r is for kept measurements
    r_fullsize_normalized(~keep_mask) = -999; % Indicate not measured or ignored

    % 6f) Find the largest normalized measurement residual
    [max_residual_val, max_residual_global_idx] = max(r_fullsize_normalized);

    % 6g) Check if the largest residual corresponds to one of the truly corrupted measurements
    % actual_corrupted_global_indices_this_iter contains the global indices of measurements
    % whose means were altered by the phase angle error in this iteration.
    
    is_correctly_identified = ismember(max_residual_global_idx, actual_corrupted_global_indices_this_iter);

    if max_residual_val > 3 % Threshold for suspecting bad data
        if is_correctly_identified
            fprintf('Iter %d: Correctly identified. Largest residual (%.2f) at meas_idx %d (is corrupted).\n', ...
                    iter, max_residual_val, max_residual_global_idx);
        else
            % This is a detection failure or misidentification if the largest residual
            % is above threshold but does not point to an actually manipulated measurement.
            fprintf('Iter %d: MISSED/MISIDENTIFIED BAD DATA!\n', iter);
            fprintf('  Largest residual = %.2f at global_idx %d (NOT in this iter''s corrupted set).\n', ...
                    max_residual_val, max_residual_global_idx);
            fprintf('  Actually corrupted global indices this iter: %s\n', mat2str(actual_corrupted_global_indices_this_iter));
            
            % ---- Optional: Single-Error Tests ----
            % Consider adapting the single-error test logic if needed.
            % break; % Optionally break after a misidentification
        end
    else
        % Largest residual is below threshold.
        if ~isempty(actual_corrupted_global_indices_this_iter) && delta_theta_bad_deg ~= 0
             fprintf('Iter %d: Bad data INJECTED but largest residual (%.2f) at meas_idx %d is BELOW threshold.\n', ...
                     iter, max_residual_val, max_residual_global_idx);
        else
             fprintf('Iter %d: No significant bad data effect or no bad data injected. Max residual %.2f.\n', ...
                     iter, max_residual_val);
        end
    end

    if iter == max_iterations
        disp('All iterations completed.');
    end
end
disp('=============================================================================');

% Note: The function 'LagrangianM_partial' is assumed to be available and
% correctly implemented for state estimation, returning normalized residuals.
% It should expect all measurement inputs in per-unit.
% The 'result_struct' should contain all necessary true system data for the SE.
% Constants like VM, VA, PD, QG, PF, QF, F_BUS, T_BUS are from MATPOWER's define_constants.
