clc;
clear;
close all;

%% --- 1. Load and Modify the Base Case ---
mpc_orig = loadcase('case9');
disp('Original system loaded (IEEE 9 Bus).');
mpc = mpc_orig; % Work with a copy
% mpc.branch(:, 6:8) = 0; % Zero out line charging B, tap ratios, and phase shifters

nbus = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);
fprintf('System: IEEE 9 Bus. Number of buses: %d, Number of branches: %d\n', nbus, nbranch);
rng("default");
rng(4);
%% --- 2. Run Base Case OPF ONCE ---
disp('Running base case OPF on original system...');
mpopt_base = mpoption('verbose', 0, 'out.all', 1);
results_opf = runopf(mpc, mpopt_base);
if ~results_opf.success
    error('OPF did not converge');
end
baseMVA = results_opf.baseMVA;
bus_base = results_opf.bus; % bus_base will be used for SE initial state
gen_base = results_opf.gen;
branch_base = results_opf.branch;
disp('Base case OPF successful.');

%% --- 3. Generate ONE Base True Measurement Set (z_base) ---
disp('Generating base true measurement set...');
nz_actual = nbus + nbus + nbus + 4*nbranch;
z_base = zeros(nz_actual, 1);
z_base(1:nbus) = bus_base(:, 8);  % VM
Sbus_base = makeSbus(baseMVA, bus_base, gen_base);
Pinj_base = real(Sbus_base);
Qinj_base = imag(Sbus_base);
z_base(nbus+1 : 2*nbus)   = Pinj_base;
z_base(2*nbus+1 : 3*nbus) = Qinj_base;
z_base(3*nbus+1 : 3*nbus+nbranch)             = branch_base(:,14) / baseMVA;  % Pf
z_base(3*nbus+nbranch+1 : 3*nbus+2*nbranch)   = branch_base(:,15) / baseMVA;  % Qf
z_base(3*nbus+2*nbranch+1 : 3*nbus+3*nbranch) = branch_base(:,16) / baseMVA;  % Pt
z_base(3*nbus+3*nbranch+1 : 3*nbus+4*nbranch) = branch_base(:,17) / baseMVA;  % Qt

%% --- 4. Define Measurement Variances and Create Noisy Measurement Set ---
disp('Defining measurement variances and creating noisy measurement set...');
R_variances_vec = [
    (0.001)^2 * ones(nbus, 1);     % V
    (0.01)^2  * ones(nbus, 1);     % P_inj
    (0.01)^2  * ones(nbus, 1);     % Q_inj
    (0.01)^2  * ones(4*nbranch,1)  % P/Q flows
    ];
sigma_vec = sqrt(R_variances_vec);

% Generate one instance of noise to be used for all subsequent tests in this run
% This makes the comparison between simultaneous and isolated errors more direct
% as the underlying "true noisy state" before parameter error impact is similar.
base_noise = randn(nz_actual,1) .* sigma_vec;
z_base_with_noise = z_base + base_noise;
fprintf('Noisy measurement vector created and will be used for all tests.\n');

%% --- 5. Parameter Error Tests ---
target_line_idx = 2; % Line 2 (Bus 2 to Bus 7 in standard case9.m)
original_R = mpc.branch(target_line_idx, 3);
original_X = mpc.branch(target_line_idx, 4);

% Scaling factors as per paper description context
fixed_scale_R = 2.3380;
fixed_scale_X = 2.7072;

% --- Test 1: Simultaneous R and X Error ---
fprintf('\n\n--- Test 1: Simultaneous R & X Error ---\n');
fprintf('Target Error on: Line %d (Buses %d-%d)\n', target_line_idx, mpc.branch(target_line_idx,1), mpc.branch(target_line_idx,2));
fprintf('  Original R: %.4f, Original X: %.4f\n', original_R, original_X);
fprintf('  Scaling R by %.4f, X by %.4f\n', fixed_scale_R, fixed_scale_X);

mpc_test_simultaneous = mpc;
mpc_test_simultaneous.branch(target_line_idx, 3) = fixed_scale_R * original_R;
mpc_test_simultaneous.branch(target_line_idx, 4) = fixed_scale_X * original_X;
fprintf('  Erroneous R: %.4f, Erroneous X: %.4f\n', mpc_test_simultaneous.branch(target_line_idx,3), mpc_test_simultaneous.branch(target_line_idx,4));

run_parameter_error_identification_tests(z_base_with_noise, mpc_test_simultaneous, bus_base, target_line_idx, nbranch, mpc, 'Simultaneous R & X');


% --- Test 2: Isolated R-only Error ---
fprintf('\n\n--- Test 2: Isolated R-only Error ---\n');
fprintf('Target Error on: Line %d (Buses %d-%d)\n', target_line_idx, mpc.branch(target_line_idx,1), mpc.branch(target_line_idx,2));
fprintf('  Original R: %.4f, Original X: %.4f (X kept original)\n', original_R, original_X);
fprintf('  Scaling R by %.4f, X kept original\n', fixed_scale_R);

mpc_test_R_only = mpc;
mpc_test_R_only.branch(target_line_idx, 3) = fixed_scale_R * original_R;
mpc_test_R_only.branch(target_line_idx, 4) = original_X; % X is correct
fprintf('  Erroneous R: %.4f, Correct X: %.4f\n', mpc_test_R_only.branch(target_line_idx,3), mpc_test_R_only.branch(target_line_idx,4));

run_parameter_error_identification_tests(z_base_with_noise, mpc_test_R_only, bus_base, target_line_idx, nbranch, mpc, 'R-only');


% --- Test 3: Isolated X-only Error ---
fprintf('\n\n--- Test 3: Isolated X-only Error ---\n');
fprintf('Target Error on: Line %d (Buses %d-%d)\n', target_line_idx, mpc.branch(target_line_idx,1), mpc.branch(target_line_idx,2));
fprintf('  Original R: %.4f (R kept original), Original X: %.4f\n', original_R, original_X);
fprintf('  R kept original, Scaling X by %.4f\n', fixed_scale_X);

mpc_test_X_only = mpc;
mpc_test_X_only.branch(target_line_idx, 3) = original_R; % R is correct
mpc_test_X_only.branch(target_line_idx, 4) = fixed_scale_X * original_X;
fprintf('  Correct R: %.4f, Erroneous X: %.4f\n', mpc_test_X_only.branch(target_line_idx,3), mpc_test_X_only.branch(target_line_idx,4));

run_parameter_error_identification_tests(z_base_with_noise, mpc_test_X_only, bus_base, target_line_idx, nbranch, mpc, 'X-only');


disp('--- All Parameter Error Identification Tests Finished ---');


%% Helper function to run and display parameter error identification tests
function run_parameter_error_identification_tests(measurements_input, mpc_with_error, bus_data_for_se, true_erroneous_line_idx, num_branches, mpc_base_case, scenario_name)
    
    fprintf('--- Running SE for Scenario: %s ---\n', scenario_name);
    % Assuming LagrangianM_singlephase signature: [lambdaN, success, r, lambda_vec, cov_lambda_vec]
    [lambdaN_params, success_nlm, ~, lambda_vec_params, cov_lambda_params] = ...
        LagrangianM_singlephase(measurements_input', mpc_with_error, 0, bus_data_for_se);

    if ~success_nlm
        fprintf('  Lagrangian Multiplier estimation did not converge for %s scenario.\n', scenario_name);
        return;
    end

    % --- Single NLM Test ---
    [max_abs_lambdaN_val, max_elem_idx_param] = max(abs(lambdaN_params));
    identified_line_idx_nlm = ceil(max_elem_idx_param / 2);
    param_type_code = mod(max_elem_idx_param-1, 2) + 1; % 1 for R, 2 for X
    param_type_str = {'R', 'X'};

    fprintf('  Single NLM Test Results (%s):\n', scenario_name);
    fprintf('    Largest |NLM| value: %.4f\n', max_abs_lambdaN_val);
    fprintf('    This corresponds to parameter type %s of Line %d (Buses %d-%d).\n', ...
        param_type_str{param_type_code}, identified_line_idx_nlm, ...
        mpc_base_case.branch(identified_line_idx_nlm,1), mpc_base_case.branch(identified_line_idx_nlm,2));

    if identified_line_idx_nlm == true_erroneous_line_idx
        fprintf('    NLM Correctly identified the target Line %d.\n', true_erroneous_line_idx);
    else
        fprintf('    NLM INCORRECTLY identified Line %d. Target was Line %d.\n', identified_line_idx_nlm, true_erroneous_line_idx);
    end

    % --- Grouped Parameter Index (T_g*) Test ---
    T_star_g_values = NaN(1, num_branches);
    for k_line = 1:num_branches
        idx_R_k = 2*k_line - 1;
        idx_X_k = 2*k_line;
        
        if idx_X_k > length(lambda_vec_params) || idx_X_k > size(cov_lambda_params,1)
            T_star_g_values(k_line) = NaN; % Or some indicator of problem
            continue;
        end
        
        lambda_g_k = lambda_vec_params([idx_R_k, idx_X_k]);
        Sigma_gg_k = cov_lambda_params([idx_R_k, idx_X_k], [idx_R_k, idx_X_k]);
        
        if rcond(Sigma_gg_k) > 1e-12
            T_g_k = lambda_g_k' * inv(Sigma_gg_k) * lambda_g_k;
            M_g_k = 2; % Group size is 2 (R and X)
            T_star_g_values(k_line) = T_g_k - M_g_k;
        else
            T_star_g_values(k_line) = NaN;
        end
    end
    
    [max_T_star_g_val, identified_line_idx_grouped] = max(T_star_g_values);
    % Check if max_T_star_g_val is NaN (can happen if all are NaN)
    if isnan(max_T_star_g_val)
        identified_line_idx_grouped = 0; % Or some other indicator for "no valid group found"
        fprintf('  Grouped Index Test Results (%s): All T_g* values were NaN.\n', scenario_name);
    else
        fprintf('  Grouped Index Test Results (%s):\n', scenario_name);
        fprintf('    Largest Grouped Index T_g*: %.4f, for Line %d (Buses %d-%d).\n', ...
            max_T_star_g_val, identified_line_idx_grouped, ...
            mpc_base_case.branch(identified_line_idx_grouped,1), mpc_base_case.branch(identified_line_idx_grouped,2));

        if identified_line_idx_grouped == true_erroneous_line_idx
            fprintf('    Grouped Index Correctly identified the target Line %d.\n', true_erroneous_line_idx);
        else
            fprintf('    Grouped Index INCORRECTLY identified Line %d. Target was Line %d.\n', identified_line_idx_grouped, true_erroneous_line_idx);
        end
    end
    
    % --- Detailed Data for Paper Tables (for this scenario) ---
    if strcmp(scenario_name, 'Simultaneous R & X') % For Table for combined error
        fprintf('\n    --- Data for Table "%s" (Scenario: %s) ---\n', 'NLM vs Grouped Param (Simultaneous)', scenario_name);
        fprintf('    Line | Parameter | Single NLM Value | Grouped Index T_g* for Line {R,X} Group\n');
        fprintf('    --------------------------------------------------------------------------------\n');
        val_R_target_NLM = abs(lambdaN_params(2*true_erroneous_line_idx-1));
        val_X_target_NLM = abs(lambdaN_params(2*true_erroneous_line_idx));
        val_Grouped_target_Tval = T_star_g_values(true_erroneous_line_idx);
        fprintf('    %-4d | R         | %-16.2f | %.2f\n', true_erroneous_line_idx, val_R_target_NLM, val_Grouped_target_Tval);
        fprintf('    %-4d | X         | %-16.2f | \n', true_erroneous_line_idx, val_X_target_NLM);
        if identified_line_idx_nlm ~= true_erroneous_line_idx && identified_line_idx_nlm > 0
            val_R_mis_NLM = abs(lambdaN_params(2*identified_line_idx_nlm-1));
            val_X_mis_NLM = abs(lambdaN_params(2*identified_line_idx_nlm));
            val_Grouped_mis_Tval = T_star_g_values(identified_line_idx_nlm);
            fprintf('    %-4d | R (NLMTop)| %-16.2f | %.2f\n', identified_line_idx_nlm, val_R_mis_NLM, val_Grouped_mis_Tval);
            fprintf('    %-4d | X (NLMTop)| %-16.2f | \n', identified_line_idx_nlm, val_X_mis_NLM);
        end
    elseif strcmp(scenario_name, 'R-only') || strcmp(scenario_name, 'X-only') % For Table for isolated errors
         fprintf('\n    --- Data for Table "%s" (Scenario: %s, Target Line %d) ---\n', 'NLM vs Grouped Param (Isolated)', scenario_name, true_erroneous_line_idx);
         fprintf('    Parameter | Largest Single NLM (Val @ Param of Line) | Correct NLM ID? | Largest Grouped T_g* (Val @ Line) | Correct Grouped ID?\n');
         fprintf('    ---------------------------------------------------------------------------------------------------------------------------------\n');
         
         is_nlm_correct = (identified_line_idx_nlm == true_erroneous_line_idx);
         if strcmp(scenario_name, 'R-only')
             is_nlm_correct = is_nlm_correct && (param_type_code == 1); % Check if R was identified
             param_tested_str = sprintf('R of Line %d', true_erroneous_line_idx);
         else % X-only
             is_nlm_correct = is_nlm_correct && (param_type_code == 2); % Check if X was identified
             param_tested_str = sprintf('X of Line %d', true_erroneous_line_idx);
         end
         
         is_grouped_correct = (identified_line_idx_grouped == true_erroneous_line_idx);
         
         fprintf('    %-9s | %-38s | %-15s | %-31s | %-17s\n', ...
             param_tested_str, ...
             sprintf('%.2f @ %s of Line %d', max_abs_lambdaN_val, param_type_str{param_type_code}, identified_line_idx_nlm), ...
             bool_to_str(is_nlm_correct), ...
             sprintf('%.2f @ Line %d', max_T_star_g_val, identified_line_idx_grouped), ...
             bool_to_str(is_grouped_correct));
    end
    fprintf('    --------------------------------------------------------------------------------\n');
end

% Helper function to convert boolean to Yes/No string
function str = bool_to_str(bool_val)
    if bool_val, str = 'Yes'; else, str = 'No'; end
end