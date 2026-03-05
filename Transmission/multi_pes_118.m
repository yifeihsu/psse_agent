clc;
clear;
close all;

%% --- 1. Load and Modify the Base Case ---
mpc_orig = loadcase('case118');

disp('Original system loaded (IEEE 118 Bus).');

% line_to_remove_orig_idx = 9;
% fprintf('Removing original line index %d (Bus %d <-> Bus %d)\n', ...
%     line_to_remove_orig_idx, ...
%     mpc_orig.branch(line_to_remove_orig_idx, 1), ...
%     mpc_orig.branch(line_to_remove_orig_idx, 2));

mpc = mpc_orig; % Work with a copy
mpc.branch(:, 6:8) = 0;
% mpc.branch(line_to_remove_orig_idx, :) = [];

% Update counts and identify new indices
nbus = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1); % Update after removal
% fprintf('System modified. Number of buses: %d, Number of branches: %d\n', nbus, nbranch);

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

% The total measurement vector:
%   [ V1..Vnbus ; Pinj1..PinjN ; Qinj1..QinjN ; Pf1..PfM ; Qf1..QfM ; Pt1..PtM ; Qt1..QtM ]
%   N = nbus, M = nbranch
nz_actual = nbus + nbus + nbus + 4*nbranch;
z_base = zeros(nz_actual, 1);

% 1) Voltage magnitudes from OPF result
z_base(1:nbus) = bus_base(:, 8);  % VM

% 2) Nodal power injections from the OPF result
Sbus_base = makeSbus(baseMVA, bus_base, gen_base);
Pinj_base = real(Sbus_base);
Qinj_base = imag(Sbus_base);
z_base(nbus+1 : 2*nbus)   = Pinj_base;         % P_inj
z_base(2*nbus+1 : 3*nbus) = Qinj_base;         % Q_inj

% 3) Branch flows from OPF result (PF, QF, PT, QT in columns 14..17 of branch)
z_base(3*nbus+1 : 3*nbus+nbranch)             = branch_base(:,14) / baseMVA;  % Pf
z_base(3*nbus+nbranch+1 : 3*nbus+2*nbranch)   = branch_base(:,15) / baseMVA;  % Qf
z_base(3*nbus+2*nbranch+1 : 3*nbus+3*nbranch) = branch_base(:,16) / baseMVA;  % Pt
z_base(3*nbus+3*nbranch+1 : 3*nbus+4*nbranch) = branch_base(:,17) / baseMVA;  % Qt

%% --- 4. Add Noise to the Base Measurement Set ---
disp('Adding noise to the measurement set...');
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
noise = randn(nz_actual,1) .* sigma;
z_base_noisy = z_base + noise;
% z_base_noisy = z_base;
disp('Noisy measurement vector z_base_noisy created.');

%% --- 5. NLM Test Loop ---
num_tests = 100;
results_nlm = zeros(num_tests, 1);  % store identified line index
% we introduce param error on this line
target_tie_line_idx = 2; 

% Original R&X Values
original_resistance = mpc.branch(target_tie_line_idx, 3);
original_reactance = mpc.branch(target_tie_line_idx, 4);

% Range for random scaling factor
min_scale_factor = 1.5;
max_scale_factor = 10;

fprintf('\n--- Starting %d NLM Tests ---\n', num_tests);

num_success_nlm = 0;

for i = 1 : num_tests
    fprintf(' Test %d/%d: ', i, num_tests);

    % create a model copy with parameter error
    mpc_test = mpc;
    scale_factor1 = min_scale_factor + rand*(max_scale_factor - min_scale_factor);
    scale_factor2 = min_scale_factor + rand*(max_scale_factor - min_scale_factor);
    % mpc_test.branch(target_tie_line_idx, 3) = scale_factor1 * original_resistance;
    % mpc_test.branch(target_tie_line_idx, 4) = scale_factor2 * original_reactance;
    mpc_test.branch(target_tie_line_idx, 3) = 1.6653 * original_resistance;
    mpc_test.branch(target_tie_line_idx, 4) = 4.6498 * original_reactance;
    % 1.5263 2.2435
    fprintf('X scale=%.2f. ', scale_factor2);

    % --- Call the single-phase Lagrangian Estimator ---
    [lambdaN, success_nlm, ~] = LagrangianM_singlephase(z_base_noisy', mpc_test, 0, bus_base);
    display(max(abs(lambdaN)))

    if success_nlm
        num_success_nlm = num_success_nlm + 1;
        % identify line with largest magnitude of lambda
        [~, max_elem_idx] = max(abs(lambdaN));
        % each line has 2 columns in the param Jacobian =>
        % line_idx = ceil(max_elem_idx / 2) if your code lumps them all
        line_idx = ceil(max_elem_idx / 2);

        results_nlm(i) = line_idx;
        fprintf('NLM Success. Identified line: %d. ', line_idx);
        if line_idx == target_tie_line_idx
            fprintf('(Correct)\n');
        else
            fprintf('(Incorrect)\n');
        end
    else
        results_nlm(i) = -1; % indicate NLM result = fail
        fprintf('NLM did not converge.\n');
    end
end

fprintf('\n--- NLM Test Loop Finished ---\n');

%% --- 6. Analyze Results ---
valid_results = results_nlm(results_nlm > 0);  % exclude -1, -99
num_valid     = length(valid_results);
num_failures  = sum(results_nlm == -1);
num_errors    = sum(results_nlm == -99);

fprintf('Total Tests: %d\n', num_tests);
fprintf('Successful NLM Runs: %d (%.1f%%)\n', num_success_nlm, ...
    100*num_success_nlm/num_tests);
fprintf('NLM Failures: %d\n', num_failures);
fprintf('Errors during NLM Execution: %d\n', num_errors);

if num_valid > 0
    correctly_identified = sum(valid_results == target_tie_line_idx);
    identification_rate  = 100 * correctly_identified / num_valid;
    fprintf('\nAnalysis of %d valid NLM runs:\n', num_valid);
    fprintf('  Correctly Identified Tie-Line %d: %d times\n', ...
        target_tie_line_idx, correctly_identified);
    fprintf('  Identification Accuracy: %.2f%%\n', identification_rate);

    % Optional distribution
    figure;
    histogram(valid_results, nbranch);
    title(sprintf('Distribution of Identified Lines (Target %d)', ...
        target_tie_line_idx));
    xlabel('Line Index');
    ylabel('Frequency');
    grid on;
else
    disp('No valid NLM results => no accuracy analysis.');
end

disp('--- Script Finished ---');
