clc;
clear;
close all;
%% --- 1. Load and Modify the Base Case ---
mpc_orig = loadcase('case9');
disp('Original system loaded (IEEE 9 Bus).');
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
ref_bus = find(bus_true(:,2)==3);
bus_true(:,9) = bus_true(:,9) - bus_true(ref_bus,9);

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

% --- DEFINE THE SPECIFIC "PROVEN WRONG CASE" SCALE FACTORS HERE ---
% proven_wrong_scale_R = 1.6653;
% proven_wrong_scale_X = 4.6498;
proven_wrong_scale_R = 2.3380; % Example scaling factor for R
proven_wrong_scale_X = 2.7072; % Example scaling factor for X

% --- Parameters for Correction Phase ---
num_scans_for_correction = 10;
fprintf('\n--- Starting %d NLM Identification & Correction Tests ---\n', num_tests);
fprintf('Using specific error scale factors for R: %.4f and X: %.4f on line %d\n', ...
    proven_wrong_scale_R, proven_wrong_scale_X, target_tie_line_idx);

num_success_nlm = 0;
num_successful_corrections = 0;

for i = 1 : num_tests
    fprintf('\nTest %d/%d: \n', i, num_tests);
    mpc_test = mpc;
    mpc_test.branch(target_tie_line_idx, 3) = proven_wrong_scale_R * original_resistance;
    mpc_test.branch(target_tie_line_idx, 4) = proven_wrong_scale_X * original_reactance;

    fprintf('  Target Line %d: Erroneous R=%.4f (True=%.4f), Erroneous X=%.4f (True=%.4f).\n', ...
        target_tie_line_idx, ...
        mpc_test.branch(target_tie_line_idx, 3), original_resistance, ...
        mpc_test.branch(target_tie_line_idx, 4), original_reactance);
    actual_R_errors(i) = mpc_test.branch(target_tie_line_idx, 3);
    actual_X_errors(i) = mpc_test.branch(target_tie_line_idx, 4);

    % --- Call the single-phase Lagrangian Estimator for IDENTIFICATION ---
    [lambdaN, success_nlm_run, ~] = LagrangianM_singlephase(z_ident_noisy', mpc_test, 0, bus_true);

    % Display max NLM value for this specific error case
    if success_nlm_run
        fprintf('  Max |NLM| value for this case: %.4f\n', max(abs(lambdaN)));
    end

    if success_nlm_run
        num_success_nlm = num_success_nlm + 1;
        [~, max_elem_idx] = max(abs(lambdaN));
        identified_line_idx = ceil(max_elem_idx / 2);
        results_nlm_identified_line(i) = identified_line_idx;
        fprintf('  NLM Identification Success. Identified line: %d. ', identified_line_idx);
        % --- PARAMETER CORRECTION PHASE ---
        % Note that the NLM will pinpoint a wrong line, and ideally we
        % should use the grouped indices here.
        fprintf('  Starting Parameter Correction for line %d...\n', identified_line_idx);
        multi_scan_measurements_z = zeros(nz_actual, num_scans_for_correction);
        for scan_idx = 1:num_scans_for_correction
            multi_scan_measurements_z(:, scan_idx) = z_true_clean_base + randn(nz_actual,1) .* sigma_R;
        end
        initial_states_multi_scan = zeros(2*nbus, num_scans_for_correction);
        for k = 1:num_scans_for_correction
            initial_states_multi_scan(:, k) = [bus_true(:,8); bus_true(:,9)];
        end

        [corrected_params_group, success_correction] = ...
            correct_parameter_group_multi_scan(mpc_test, target_tie_line_idx, ...
            multi_scan_measurements_z, ...
            initial_states_multi_scan, R_variances, baseMVA);

        if success_correction
            num_successful_corrections = num_successful_corrections + 1;
            corrected_R_values(i) = corrected_params_group(1);
            corrected_X_values(i) = corrected_params_group(2);
            fprintf('  Parameter Correction Success. Line %d: Corrected R=%.4f, Corrected X=%.4f\n', ...
                identified_line_idx, corrected_params_group(1), corrected_params_group(2));
        else
            fprintf('  Parameter Correction Failed for line %d.\n', identified_line_idx);
        end
    else
        results_nlm_identified_line(i) = -1;
        fprintf('  NLM Identification did not converge.\n');
    end
end
fprintf('\n--- NLM Identification & Correction Loop Finished ---\n');

%% --- 6. Analyze Results ---
% (Your existing analysis code for identification accuracy)
valid_results_ident = results_nlm_identified_line(results_nlm_identified_line > 0);
num_valid_ident = length(valid_results_ident);
num_failures_ident = sum(results_nlm_identified_line == -1);

fprintf('Total Tests: %d\n', num_tests);
fprintf('Successful NLM Identification Runs: %d (%.1f%%)\n', num_success_nlm, ...
    100*num_success_nlm/num_tests);
fprintf('NLM Identification Failures: %d\n', num_failures_ident);

if num_valid_ident > 0
    correctly_identified_count = sum(valid_results_ident == target_tie_line_idx);
    identification_rate  = 100 * correctly_identified_count / num_valid_ident;
    fprintf('\nAnalysis of %d valid NLM identification runs:\n', num_valid_ident);
    fprintf('  Correctly Identified Target Line %d: %d times\n', ...
        target_tie_line_idx, correctly_identified_count);
    fprintf('  Identification Accuracy for this specific error case: %.2f%%\n', identification_rate);

    if num_tests > 1 % Only show histogram if multiple tests run for this specific case
        figure;
        histogram(valid_results_ident, nbranch);
        title(sprintf('Distribution of Identified Lines (Target %d, Specific Error Case)', ...
            target_tie_line_idx));
        xlabel('Line Index');
        ylabel('Frequency');
        grid on;
    end
else
    disp('No valid NLM identification results => no accuracy analysis.');
end

% Add analysis for correction
fprintf('\n--- Correction Analysis (for cases where target line was correctly identified) ---\n');
successful_correction_indices = find(results_nlm_identified_line == target_tie_line_idx & ~isnan(corrected_R_values) & corrected_R_values~=0); % Assuming 0 is not a valid corrected R
num_actually_corrected = length(successful_correction_indices);

fprintf('Total Tests where target line was correctly identified: %d\n', sum(results_nlm_identified_line == target_tie_line_idx));
fprintf('Number of these where correction algorithm reported success: %d\n', num_successful_corrections);
fprintf('Number of these with seemingly valid (non-NaN, non-zero) corrected values: %d\n', num_actually_corrected);


if num_actually_corrected > 0
    errors_R_initial = abs(actual_R_errors(successful_correction_indices) - original_resistance);
    errors_X_initial = abs(actual_X_errors(successful_correction_indices) - original_reactance);

    errors_R_final = abs(corrected_R_values(successful_correction_indices) - original_resistance);
    errors_X_final = abs(corrected_X_values(successful_correction_indices) - original_reactance);

    mean_initial_error_R = mean(errors_R_initial);
    mean_initial_error_X = mean(errors_X_initial);
    mean_final_error_R = mean(errors_R_final);
    mean_final_error_X = mean(errors_X_final);

    fprintf('  Average Initial Absolute Error R: %.4f (%.2f%% of true R)\n', mean_initial_error_R, 100*mean_initial_error_R/original_resistance);
    fprintf('  Average Corrected Absolute Error R: %.4f (%.2f%% of true R)\n', mean_final_error_R, 100*mean_final_error_R/original_resistance);
    fprintf('  Average Initial Absolute Error X: %.4f (%.2f%% of true X)\n', mean_initial_error_X, 100*mean_initial_error_X/original_reactance);
    fprintf('  Average Corrected Absolute Error X: %.4f (%.2f%% of true X)\n', mean_final_error_X, 100*mean_final_error_X/original_reactance);

    figure;
    subplot(2,1,1);
    plot(errors_R_initial, 'b-o', 'DisplayName', 'Initial Abs Error R'); hold on;
    plot(errors_R_final, 'r-x', 'DisplayName', 'Abs Error After Correction R');
    legend; title(sprintf('Correction Performance for R (Target Line %d, Specific Error Case)', target_tie_line_idx)); ylabel('Absolute Error');
    grid on;

    subplot(2,1,2);
    plot(errors_X_initial, 'b-o', 'DisplayName', 'Initial Abs Error X'); hold on;
    plot(errors_X_final, 'r-x', 'DisplayName', 'Abs Error After Correction X');
    legend; title(sprintf('Correction Performance for X (Target Line %d, Specific Error Case)', target_tie_line_idx)); ylabel('Absolute Error');
    xlabel('Test Case Index (successful identification & correction)');
    grid on;
else
    disp('No successful corrections with valid values to analyze.');
end

disp('--- Script Finished ---');