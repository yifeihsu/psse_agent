clc;
clear;
close all;
%% --- 1. Load and Modify the Base Case ---
mpc_orig = loadcase('case9');
disp('Original system loaded (IEEE 9 Bus).'); % Corrected case name
mpc = mpc_orig; % Work with a copy
mpc.branch(:, 6:8) = 0; % Zero out line charging B, tap ratios, and phase shifters
nbus = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);
fprintf('System modified. Number of buses: %d, Number of branches: %d\n', nbus, nbranch);

%% --- 2. Run Base Case OPF ONCE ---
disp('Running base case OPF on modified system...');
mpopt_base = mpoption('verbose', 0, 'out.all', 1); 
results = runopf(mpc, mpopt_base);
if ~results.success
    error('OPF did not converge');
end
[baseMVA, bus_base, gen_base, ~, branch_base, ~, success_base, ~] = runopf(mpc, mpopt_base);
if success_base ~= 1
    error('Base case OPF failed to converge! Cannot proceed.');
end
disp('Base case OPF successful.');

%% --- 3. Generate ONE Base Measurement Set (z_base) ---
disp('Generating base measurement set...');
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

%% --- 4. Add Noise to the Base Measurement Set ---
disp('Generating measurement set (currently no noise)...'); % Modified message
R_variances = [
    (0.001)^2 * ones(nbus, 1);     % V
    (0.01)^2  * ones(nbus, 1);     % P_inj
    (0.01)^2  * ones(nbus, 1);     % Q_inj
    (0.01)^2  * ones(4*nbranch,1)  % P/Q flows
    ];
if length(R_variances) ~= nz_actual
    error('Mismatch between measurement vector size (%d) and R_variances size (%d).', ...
        nz_actual, length(R_variances));
end
sigma = sqrt(R_variances);
% noise = randn(nz_actual,1) .* sigma; % Noise can be added here if needed
% z_base_noisy = z_base + noise;
z_base_noisy = z_base; % Using noise-free measurements for deterministic testing
disp('Measurement vector z_base_noisy created (currently noise-free).');

%% --- 5. NLM Test Loop with Grouped Index Calculation for ALL Lines ---
num_tests = 10; % Reduced for example, can be increased
results_nlm_identified_line = zeros(num_tests, 1);
% Store all T*g values for each test: (num_tests x nbranch) matrix
all_tests_Tstar_g_values = NaN(num_tests, nbranch); 

target_tie_line_idx = 2;
original_resistance = mpc.branch(target_tie_line_idx, 3);
original_reactance = mpc.branch(target_tie_line_idx, 4);

fixed_scale_R = 2.3380;
fixed_scale_X = 2.7072;

fprintf('\n--- Starting %d NLM Tests with Grouped Index Calculation for ALL Lines ---\n', num_tests);
fprintf('Target error on line %d. Fixed R_scale=%.4f, X_scale=%.4f\n', target_tie_line_idx, fixed_scale_R, fixed_scale_X);

num_success_nlm_runs = 0;

for i = 1 : num_tests
    fprintf(' Test %d/%d: \n', i, num_tests);
    mpc_test = mpc;
    mpc_test.branch(target_tie_line_idx, 3) = fixed_scale_R * original_resistance;
    mpc_test.branch(target_tie_line_idx, 4) = fixed_scale_X * original_reactance;

    % --- Call the single-phase Lagrangian Estimator ---
    [lambdaN, success_nlm, ~, lambda_vec, ea_cov_lambda] = LagrangianM_singlephase(z_base_noisy', mpc_test, 0, bus_base);
    
    if success_nlm
        num_success_nlm_runs = num_success_nlm_runs + 1;
        [max_abs_lambdaN, max_elem_idx] = max(abs(lambdaN));
        identified_line_idx_nlm = ceil(max_elem_idx / 2);
        results_nlm_identified_line(i) = identified_line_idx_nlm;
        
        fprintf('  NLM Success. Max|λN|=%.2f. NLM Identified line: %d. ', max_abs_lambdaN, identified_line_idx_nlm);
        if identified_line_idx_nlm == target_tie_line_idx
            fprintf('(Correct NLM ID). ');
        else
            fprintf('(Incorrect NLM ID). ');
        end

        % --- Calculate Grouped Index for ALL lines ---
        current_test_Tstar_g = NaN(1, nbranch);
        fprintf('\n    Calculating grouped indices for all lines: ');
        for k_line = 1:nbranch
            param_idx_R = 2*k_line - 1;
            param_idx_X = 2*k_line;
            
            if param_idx_X <= length(lambda_vec) % Ensure indices are within bounds
                lambda_g_k = lambda_vec([param_idx_R, param_idx_X]); % 2x1 vector for line k
                Sigma_gg_k = ea_cov_lambda([param_idx_R, param_idx_X], [param_idx_R, param_idx_X]); % 2x2 matrix for line k
                
                if rcond(Sigma_gg_k) > 1e-12
                    T_g_k = lambda_g_k' * inv(Sigma_gg_k) * lambda_g_k;
                    M_g_k = 2; 
                    T_star_g_k = T_g_k - M_g_k;
                    current_test_Tstar_g(k_line) = T_star_g_k;
                    % Optional: print T*g for each line if verbose
                    % fprintf('L%d: T*g=%.2f; ', k_line, T_star_g_k);
                else
                    %fprintf('Sigma_gg for line %d ill-conditioned. Grouped index not calculated.\n', k_line);
                end
            else
                 %fprintf('Parameter indices for line %d out of bounds. Grouped index not calculated.\n', k_line);
            end
        end
        all_tests_Tstar_g_values(i, :) = current_test_Tstar_g;
        
        % Identify line with max T*g for this test
        [max_Tstar_g_val, identified_line_idx_grouped] = max(current_test_Tstar_g);
        fprintf('\n    Max T*g = %.2f for line %d. ',max_Tstar_g_val, identified_line_idx_grouped);
        if identified_line_idx_grouped == target_tie_line_idx
            fprintf('(Grouped Index Correctly ID''d target line %d)\n', target_tie_line_idx);
        else
            fprintf('(Grouped Index Incorrectly ID''d line %d, target was %d)\n', identified_line_idx_grouped, target_tie_line_idx);
        end


    else
        results_nlm_identified_line(i) = -1; % indicate NLM result = fail
        fprintf('NLM did not converge.\n');
    end
end
fprintf('\n--- NLM Test Loop Finished ---\n');

%% --- 6. Analyze Results ---
valid_nlm_runs_mask = results_nlm_identified_line > 0;
valid_identified_lines_nlm = results_nlm_identified_line(valid_nlm_runs_mask);
num_valid_nlm_runs = length(valid_identified_lines_nlm);
num_nlm_failures = sum(results_nlm_identified_line == -1);

fprintf('Total Tests: %d\n', num_tests);
fprintf('Successful NLM Runs (convergence): %d (%.1f%%)\n', num_success_nlm_runs, ...
    100*num_success_nlm_runs/num_tests);
fprintf('NLM Convergence Failures: %d\n', num_nlm_failures);

if num_valid_nlm_runs > 0
    correctly_identified_nlm_count = sum(valid_identified_lines_nlm == target_tie_line_idx);
    identification_rate_nlm  = 100 * correctly_identified_nlm_count / num_valid_nlm_runs;
    fprintf('\nAnalysis of %d valid NLM identification runs:\n', num_valid_nlm_runs);
    fprintf('  Correctly Identified Target Line %d by NLM: %d times\n', ...
        target_tie_line_idx, correctly_identified_nlm_count);
    fprintf('  NLM Identification Accuracy: %.2f%%\n', identification_rate_nlm);
    
    figure;
    subplot(2,1,1); % Changed to subplot
    histogram(valid_identified_lines_nlm, 'BinMethod', 'integers'); % Use integers for line indices
    xlim([0.5, nbranch + 0.5]); % Set x-axis limits based on number of branches
    title(sprintf('NLM Identified Lines (Target %d, Fixed Error)', target_tie_line_idx));
    xlabel('Line Index Identified by NLM');
    ylabel('Frequency');
    grid on;

    % Analysis and Plotting for Grouped Indices
    valid_Tstar_g_tests = all_tests_Tstar_g_values(valid_nlm_runs_mask, :);
    
    % Identify line with max T*g for each valid test run
    identified_lines_grouped = zeros(num_valid_nlm_runs, 1);
    max_Tstar_g_per_valid_test = zeros(num_valid_nlm_runs, 1);
    Tstar_g_for_target_line_all_valid_tests = zeros(num_valid_nlm_runs, 1);

    for k_test = 1:num_valid_nlm_runs
        [max_val, idx] = max(valid_Tstar_g_tests(k_test, :));
        identified_lines_grouped(k_test) = idx;
        max_Tstar_g_per_valid_test(k_test) = max_val;
        Tstar_g_for_target_line_all_valid_tests(k_test) = valid_Tstar_g_tests(k_test, target_tie_line_idx);
    end

    correctly_identified_grouped_count = sum(identified_lines_grouped == target_tie_line_idx);
    identification_rate_grouped = 100 * correctly_identified_grouped_count / num_valid_nlm_runs;
    
    fprintf('\nAnalysis of Grouped Index (T*g) Identification (%d valid NLM runs):\n', num_valid_nlm_runs);
    fprintf('  Correctly Identified Target Line %d by Max T*g: %d times\n', ...
        target_tie_line_idx, correctly_identified_grouped_count);
    fprintf('  Grouped Index Identification Accuracy: %.2f%%\n', identification_rate_grouped);

    subplot(2,1,2); % Changed to subplot
    histogram(identified_lines_grouped, 'BinMethod', 'integers');
    xlim([0.5, nbranch + 0.5]); % Set x-axis limits
    title(sprintf('Line Identified by Max Grouped Index T*g (Target %d)', target_tie_line_idx));
    xlabel('Line Index Identified by Max T*g');
    ylabel('Frequency');
    grid on;
    
    % Plot distribution of T*g values for the target line vs other lines (from one test or averaged)
    if num_valid_nlm_runs > 0
        figure;
        % Example: Use T*g values from the first valid test run
        first_valid_test_idx = find(valid_nlm_runs_mask, 1, 'first');
        if ~isempty(first_valid_test_idx)
            Tstar_g_display = all_tests_Tstar_g_values(first_valid_test_idx, :);
            bar_colors = repmat('b', nbranch, 1);
            bar_colors(target_tie_line_idx) = 'r';
            
            bar_handles = bar(Tstar_g_display);
            % Manually set colors if possible or use different plot type
            % For newer MATLAB versions, you can set CData for bar objects
             hold on;
             bar(target_tie_line_idx, Tstar_g_display(target_tie_line_idx), 'r'); % Highlight target
             hold off;

            title(sprintf('Distribution of T*g for All Lines (Test %d, Target Line %d in Red)', first_valid_test_idx, target_tie_line_idx));
            xlabel('Line Index');
            ylabel('T*g Value');
            grid on;
            legend({'Other Lines', sprintf('Target Line %d (Erroneous)',target_tie_line_idx)});
        end
    end

else
    disp('No valid NLM runs => no accuracy analysis.');
end
disp('--- Script Finished ---');