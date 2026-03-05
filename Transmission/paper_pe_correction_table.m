clc;
clear;
close all;
%% --- 1. Load and Modify the Base Case ---
mpc_orig = loadcase('case9'); % Changed to case9
disp('Original system loaded (IEEE 9 Bus).'); % Updated display message
mpc = mpc_orig;
mpc.branch(:, 6:8) = 0; 
nbus = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);
rng("default");
rng(4);
%% --- 2. Run Base Case OPF ONCE (on the TRUE system) ---
disp('Running base case OPF on original (true) system...');
mpopt_base = mpoption('verbose', 0, 'out.all', 1);
results_true_opf = runopf(mpc, mpopt_base);
if ~results_true_opf.success
    error('OPF on original system did not converge');
end
[baseMVA, bus_true, gen_true, ~, branch_true, ~, success_true_opf, ~] = runopf(mpc, mpopt_base);
if success_true_opf ~= 1
    error('Base case OPF on original system failed! Cannot proceed.');
end
disp('Base case OPF on original system successful.');
ref_bus_idx = find(bus_true(:,2)==3);
if isempty(ref_bus_idx)
    ref_bus_idx = 1; % Default if no slack bus found
    disp('Warning: No slack bus found, using bus 1 as reference for angle normalization.');
end
bus_true(:,9) = bus_true(:,9) - bus_true(ref_bus_idx,9);

%% --- 3. Generate Base TRUE Measurement Set (z_true_clean_base) ---
disp('Generating base TRUE measurement set...');
nz_actual = nbus + nbus + nbus + 4*nbranch;
z_true_clean_base = zeros(nz_actual, 1);
z_true_clean_base(1:nbus) = bus_true(:, 8);  % VM
Sbus_true = makeSbus(baseMVA, bus_true, gen_true);
z_true_clean_base(nbus+1 : 2*nbus)   = real(Sbus_true); % P_inj
z_true_clean_base(2*nbus+1 : 3*nbus) = imag(Sbus_true); % Q_inj
z_true_clean_base(3*nbus+1 : 3*nbus+nbranch)             = branch_true(:,14) / baseMVA;  % Pf
z_true_clean_base(3*nbus+nbranch+1 : 3*nbus+2*nbranch)   = branch_true(:,15) / baseMVA;  % Qf
z_true_clean_base(3*nbus+2*nbranch+1 : 3*nbus+3*nbranch) = branch_true(:,16) / baseMVA;  % Pt
z_true_clean_base(3*nbus+3*nbranch+1 : 3*nbus+4*nbranch) = branch_true(:,17) / baseMVA;  % Qt

R_variances = [
    (0.001)^2 * ones(nbus, 1);     % V
    (0.01)^2  * ones(nbus, 1);     % P_inj
    (0.01)^2  * ones(nbus, 1);     % Q_inj
    (0.01)^2  * ones(4*nbranch,1)  % P/Q flows
    ];
sigma_R = sqrt(R_variances);

%% --- IDENTIFICATION PHASE SETUP ---
z_ident_noisy = z_true_clean_base + randn(nz_actual,1) .* sigma_R;
disp('Single noisy measurement vector z_ident_noisy created for NLM identification.');

%% --- 5. NLM Test Loop (Identification) & CORRECTION ---
num_tests = 1;
results_nlm_identified_line = zeros(num_tests, 1);
corrected_R_values = NaN(num_tests, 1);
corrected_X_values = NaN(num_tests, 1);
actual_R_errors = NaN(num_tests, 1);
actual_X_errors = NaN(num_tests, 1);

target_tie_line_idx = 2; 
original_resistance = mpc.branch(target_tie_line_idx, 3);
original_reactance = mpc.branch(target_tie_line_idx, 4);

% --- DEFINE THE SPECIFIC ERROR SCALE FACTORS HERE ---
proven_wrong_scale_R = 2.3380; % Example scaling factor for R
proven_wrong_scale_X = 2.7072; % Example scaling factor for X

% --- Parameters for Correction Phase ---
num_scans_for_correction = 20;
fprintf('\n--- Starting %d NLM Identification & Correction Test(s) ---\n', num_tests);
fprintf('Target Line for error: %d (Buses %d-%d)\n', target_tie_line_idx, mpc.branch(target_tie_line_idx,1), mpc.branch(target_tie_line_idx,2));
fprintf('Using specific error scale factors for R: %.4f and X: %.4f on line %d\n', ...
    proven_wrong_scale_R, proven_wrong_scale_X, target_tie_line_idx);

num_success_nlm = 0;
num_successful_corrections = 0;

for i = 1 : num_tests
    fprintf('\nTest %d/%d: \n', i, num_tests);
    mpc_test = mpc;
    
    erroneous_R = proven_wrong_scale_R * original_resistance;
    erroneous_X = proven_wrong_scale_X * original_reactance;
    mpc_test.branch(target_tie_line_idx, 3) = erroneous_R;
    mpc_test.branch(target_tie_line_idx, 4) = erroneous_X;

    fprintf('  Target Line %d: Original R=%.4f, X=%.4f\n', target_tie_line_idx, original_resistance, original_reactance);
    fprintf('                  Erroneous R=%.4f, X=%.4f\n', erroneous_R, erroneous_X);
    actual_R_errors(i) = erroneous_R; % Storing the erroneous value
    actual_X_errors(i) = erroneous_X; % Storing the erroneous value

    % --- Call the single-phase Lagrangian Estimator for IDENTIFICATION ---
    [lambdaN, success_nlm_run, ~] = LagrangianM_singlephase(z_ident_noisy', mpc_test, 0, bus_true);

    if success_nlm_run
        fprintf('  Max |NLM| value for NLM identification: %.4f\n', max(abs(lambdaN)));
        num_success_nlm = num_success_nlm + 1;
        [~, max_elem_idx] = max(abs(lambdaN));
        identified_line_idx = ceil(max_elem_idx / 2); % Assuming NLM output is per parameter (R, X)
        results_nlm_identified_line(i) = identified_line_idx;
        fprintf('  NLM Identification Success. Identified line index by NLM: %d. (Target was %d)\n', identified_line_idx, target_tie_line_idx);
        
        fprintf('  Starting Parameter Correction for the true erroneous line %d...\n', target_tie_line_idx);
        
        multi_scan_measurements_z = zeros(nz_actual, num_scans_for_correction);
        for scan_idx = 1:num_scans_for_correction
            multi_scan_measurements_z(:, scan_idx) = z_true_clean_base + randn(nz_actual,1) .* sigma_R;
        end
        
        initial_states_multi_scan = zeros(2*nbus, num_scans_for_correction);
        for k_scan = 1:num_scans_for_correction
            initial_states_multi_scan(:, k_scan) = [bus_true(:,8); rad2deg(bus_true(:,9))];
        end

        [corrected_params_group, success_correction] = ...
            correct_parameter_group_multi_scan(mpc_test, target_tie_line_idx, ...
            multi_scan_measurements_z, ...
            initial_states_multi_scan, R_variances, baseMVA); % Pass R_variances (full vector)

        if success_correction
            num_successful_corrections = num_successful_corrections + 1;
            corrected_R_values(i) = corrected_params_group(1);
            corrected_X_values(i) = corrected_params_group(2);
            fprintf('  Parameter Correction SUCCEEDED for Line %d.\n', target_tie_line_idx);
            fprintf('    Original R: %.4f, Erroneous R: %.4f, Corrected R: %.4f\n', original_resistance, erroneous_R, corrected_R_values(i));
            fprintf('    Original X: %.4f, Erroneous X: %.4f, Corrected X: %.4f\n', original_reactance, erroneous_X, corrected_X_values(i));
            
            err_R_percent = (abs(corrected_R_values(i) - original_resistance) / abs(original_resistance)) * 100;
            err_X_percent = (abs(corrected_X_values(i) - original_reactance) / abs(original_reactance)) * 100;
            fprintf('    Percent Error after correction - R: %.2f%%, X: %.2f%%\n', err_R_percent, err_X_percent);

            % --- Output for Paper Table ---
            fprintf('\n--- Data for Paper Table: Parameter Correction Results ---\n');
            fprintf('Parameter | True Value (p.u.) | Erroneous Value (p.u.) | Corrected Value (p.u.) | %% Error After Corr.\n');
            fprintf('---------------------------------------------------------------------------------------------------\n');
            fprintf('R Line %-2d | %-17.4f | %-22.4f | %-22.4f | %.2f%%\n', target_tie_line_idx, original_resistance, erroneous_R, corrected_R_values(i), err_R_percent);
            fprintf('X Line %-2d | %-17.4f | %-22.4f | %-22.4f | %.2f%%\n', target_tie_line_idx, original_reactance, erroneous_X, corrected_X_values(i), err_X_percent);
            fprintf('---------------------------------------------------------------------------------------------------\n');

        else
            fprintf('  Parameter Correction FAILED for line %d.\n', target_tie_line_idx);
        end
    else
        results_nlm_identified_line(i) = -1; % NLM identification failed
        fprintf('  NLM Identification did not converge.\n');
    end
end
fprintf('\n--- NLM Identification & Correction Loop Finished ---\n');

%% --- 6. Analyze Results (brief summary, detailed analysis can be separate) ---
if num_tests == 1 % Analysis for a single test run
    if success_nlm_run
        fprintf('\nSummary for the test run:\n');
        fprintf('  NLM Identification identified Line: %d (Target was Line %d)\n', identified_line_idx, target_tie_line_idx);
        if identified_line_idx == target_tie_line_idx
            fprintf('  NLM Identification: CORRECT\n');
        else
            fprintf('  NLM Identification: INCORRECT\n');
        end
        if exist('success_correction', 'var') % Check if correction was attempted
            if success_correction
                fprintf('  Parameter Correction: SUCCESSFUL\n');
            else
                fprintf('  Parameter Correction: FAILED\n');
            end
        else
             fprintf('  Parameter Correction: NOT ATTEMPTED (NLM failed or other issue)\n');
        end
    else
        fprintf('\nSummary for the test run: NLM Identification FAILED to converge.\n');
    end
else % Analysis for multiple tests (original script's logic)
    valid_results_ident = results_nlm_identified_line(results_nlm_identified_line > 0 & ~isnan(results_nlm_identified_line));
    num_valid_ident = length(valid_results_ident);
    num_failures_ident = sum(results_nlm_identified_line == -1 | isnan(results_nlm_identified_line));

    fprintf('Total Tests: %d\n', num_tests);
    fprintf('Successful NLM Identification Runs (Converged): %d (%.1f%%)\n', num_success_nlm, ...
        100*num_success_nlm/num_tests);
    fprintf('NLM Identification Convergence Failures: %d\n', num_failures_ident);

    if num_valid_ident > 0
        correctly_identified_count = sum(valid_results_ident == target_tie_line_idx);
        identification_rate  = 100 * correctly_identified_count / num_valid_ident;
        fprintf('\nAnalysis of %d valid NLM identification runs:\n', num_valid_ident);
        fprintf('  Correctly Identified Target Line %d: %d times\n', ...
            target_tie_line_idx, correctly_identified_count);
        fprintf('  NLM Identification Accuracy (among converged runs): %.2f%%\n', identification_rate);
    else
        disp('No valid NLM identification results to analyze for accuracy.');
    end
    
    fprintf('\nParameter Correction Summary (over %d tests):\n', num_tests);
    fprintf('  Number of successful corrections (where NLM also converged): %d\n', num_successful_corrections);
    
    % More detailed correction analysis could be added here if num_tests > 1
    % e.g., average errors before/after for successfully corrected cases
    if num_successful_corrections > 0
        % Filter for successfully corrected cases where NLM identified the target correctly
        % This part of analysis is complex if identified_line_idx ~= target_tie_line_idx
        % The current script corrects target_tie_line_idx
        fprintf('  (Further correction analysis would require tracking if NLM was also correct for these successful corrections)\n');
    end
end

disp('--- Script Finished ---');