
%% ================== MAIN SCRIPT EXAMPLE ====================
clc;
clear;
close all;

%% --- 1) Load and Modify the Base Case ---
mpc_orig = loadcase('case118');
disp('Original system loaded (IEEE 118 Bus).');

mpc = mpc_orig;  % Work with a copy
% mpc.branch(:, 6:8) = 0;  % zero out constraints if needed

nbus    = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);

%% --- 2) Run Base Case OPF ONCE ---disp('Running base case OPF on modified system...');
mpopt_base = mpoption('verbose', 0, 'out.all', 1); 
[baseMVA, bus_base, gen_base, ~, branch_base, ~, success_base, ~] = runopf(mpc, mpopt_base);

if success_base ~= 1
    error('Base case OPF failed to converge! Cannot proceed.');
end
disp('Base case OPF successful.');

%% --- Build a "result" struct for the SE function ---
result_struct.bus    = bus_base;
result_struct.branch = branch_base;
result_struct.gen    = gen_base;
result_struct.baseMVA= baseMVA;

%% --- 3) Generate a "complete" Measurement Vector (z_full) ---
nz_actual = nbus + nbus + nbus + 4*nbranch;
z_full = zeros(nz_actual, 1);

% 1) Voltage magnitudes
z_full(1:nbus) = bus_base(:, 8);  % VM

% 2) Nodal power injections
Sbus_base = makeSbus(baseMVA, bus_base, gen_base);
Pinj_base = real(Sbus_base);
Qinj_base = imag(Sbus_base);
z_full(nbus+1 : 2*nbus)   = Pinj_base;   % P_inj
z_full(2*nbus+1 : 3*nbus) = Qinj_base;   % Q_inj

% 3) Branch flows (PF, QF, PT, QT)
z_full(3*nbus+1 : 3*nbus+nbranch)             = branch_base(:,14)/baseMVA;  % Pf
z_full(3*nbus+nbranch+1 : 3*nbus+2*nbranch)   = branch_base(:,15)/baseMVA;  % Qf
z_full(3*nbus+2*nbranch+1 : 3*nbus+3*nbranch) = branch_base(:,16)/baseMVA;  % Pt
z_full(3*nbus+3*nbranch+1 : 3*nbus+4*nbranch) = branch_base(:,17)/baseMVA;  % Qt

%% --- 4) Remove (or De-Weight) Some Measurements ---
buses_to_remove_inj = 1:nbus; 
remove_P_idx = nbus + buses_to_remove_inj;
remove_Q_idx = 2*nbus + buses_to_remove_inj;

all_indices = 1:nz_actual;
remove_indices = [remove_P_idx, remove_Q_idx];
remove_indices = [];
keep_mask = true(1, nz_actual);
keep_mask(remove_indices) = false;

z_base = z_full(keep_mask);
kept_indices = find(keep_mask);

%% --- 5) Increase the Noise Variance for Voltage Measurements ---
R_variances = zeros(length(z_base), 1);
for i = 1:length(z_base)
    global_idx = kept_indices(i);
    if global_idx <= nbus
        R_variances(i) = (0.01)^2;  % bigger variance => less weight for volt
    else
        R_variances(i) = (0.002)^2; % flows (or injections for bus <5)
    end
end
sigma = sqrt(R_variances);

%% --- 6) "Bad PT" Injection at a Bus => bus_of_interest
% Suppose bus_of_interest = 3 has a faulty voltage PT. 
% Now, if bus_of_interest is the from_bus, we corrupt Pf/Qf; 
% if it's the to_bus, we corrupt Pt/Qt.

bus_of_interest = 10;

% (NEW/CHANGED) Find branches where "bus_of_interest" is the from-bus
bad_branches_from = find(mpc.branch(:,1) == bus_of_interest);
% (NEW/CHANGED) Find branches where "bus_of_interest" is the to-bus
bad_branches_to   = find(mpc.branch(:,2) == bus_of_interest);

scale_factor_min = 1.2;  
scale_factor_max = 1.5;  
max_iterations   = 100;

disp('========================================');
disp('Starting the loop of 100 runs to demonstrate a faulty PT at a single bus...');
for iter = 1:max_iterations

    % 6a) Start from the "clean" measurement subset
    z_noisy_iter = z_base;

    % 6b) Add normal measurement noise
    noise = sigma .* randn(size(z_noisy_iter));
    z_noisy_iter = z_noisy_iter + noise;

    % 6c) Inject the "faulty PT" effect
    % sf = scale_factor_min + (scale_factor_max - scale_factor_min)*rand(1); 
    sf = 1;
    % (1) Scale the bus-of-interest's voltage measurement if it is in the kept set
    volt_idx_full = bus_of_interest;  % index for V(bus_of_interest) in [1..nbus]
    if keep_mask(volt_idx_full)
        red_idx_volt = find(kept_indices == volt_idx_full);
        z_noisy_iter(red_idx_volt) = z_noisy_iter(red_idx_volt) * sf;
    end

    % (2a) Scale from-bus flows (Pf, Qf) for each branch in bad_branches_from
    for ln = bad_branches_from(:).'
        idxPf_full = 3*nbus + ln;             % Pf(ln)
        idxQf_full = 3*nbus + nbranch + ln;   % Qf(ln)

        if keep_mask(idxPf_full)
            red_idx_pf = find(kept_indices == idxPf_full);
            z_noisy_iter(red_idx_pf) = z_noisy_iter(red_idx_pf) * sf;
        end
        if keep_mask(idxQf_full)
            red_idx_qf = find(kept_indices == idxQf_full);
            z_noisy_iter(red_idx_qf) = z_noisy_iter(red_idx_qf) * sf;
        end
    end

    % (2b) (NEW/CHANGED) Scale to-bus flows (Pt, Qt) for each branch in bad_branches_to
    % Because from the bus_of_interest's perspective, that flow is "from" this bus side as well.
    for ln = bad_branches_to(:).'
        idxPt_full = 3*nbus + 2*nbranch + ln;  % Pt(ln)
        idxQt_full = 3*nbus + 3*nbranch + ln;  % Qt(ln)

        if keep_mask(idxPt_full)
            red_idx_pt = find(kept_indices == idxPt_full);
            z_noisy_iter(red_idx_pt) = z_noisy_iter(red_idx_pt) * sf;
        end
        if keep_mask(idxQt_full)
            red_idx_qt = find(kept_indices == idxQt_full);
            z_noisy_iter(red_idx_qt) = z_noisy_iter(red_idx_qt) * sf;
        end
    end

    % (3) Optionally, scale bus-of-interest P_inj and Q_inj if they exist in the kept set
    p_inj_idx_full = nbus + bus_of_interest;
    q_inj_idx_full = 2*nbus + bus_of_interest;
    if keep_mask(p_inj_idx_full)
        red_idx_pin = find(kept_indices == p_inj_idx_full);
        z_noisy_iter(red_idx_pin) = z_noisy_iter(red_idx_pin) * sf;
    end
    if keep_mask(q_inj_idx_full)
        red_idx_qin = find(kept_indices == q_inj_idx_full);
        z_noisy_iter(red_idx_qin) = z_noisy_iter(red_idx_qin) * sf;
    end

    % 6d) Re-insert into "full" vector for the SE function
    z_full_corrupted = zeros(nz_actual, 1);
    z_full_corrupted(keep_mask) = z_noisy_iter;
    z_full_corrupted(~keep_mask) = 999;  % or NaN

    % 6e) Run the state estimator (unchanged from your code)
    [lambdaN, success, r] = LagrangianM_partial(z_full_corrupted, result_struct, 0, bus_base);
    if ~success
        fprintf('Iteration %d: WLS did not converge. Scale factor = %.3f\n', iter, sf);
        continue;
    end

    % Extend the residual vector back to full size for easy indexing
    r_fullsize = zeros(nz_actual, 1);
    r_fullsize(keep_mask) = r;
    r_fullsize(~keep_mask) = -999;  % not measured or ignore

    % 6f) Largest normalized measurement residual
    [max_val, max_r_idx] = max(r_fullsize);
    if max_val >= 3
        print("error");
    end
    % 6g) Identify which full indices we actually corrupted
    corrupted_indices = [];
    % (a) The bus voltage at bus_of_interest
    corrupted_indices = [corrupted_indices, volt_idx_full];

    % (b) The from-bus flows from bus_of_interest
    for ln = bad_branches_from(:).'
        corrupted_indices = [corrupted_indices, (3*nbus + ln), (3*nbus + nbranch + ln)];
    end

    % (b') (NEW) The to-bus flows for lines in bad_branches_to
    for ln = bad_branches_to(:).'
        corrupted_indices = [corrupted_indices, (3*nbus + 2*nbranch + ln), (3*nbus + 3*nbranch + ln)];
    end

    % (c) Possibly bus_of_interest's P_inj, Q_inj
    corrupted_indices = [corrupted_indices, p_inj_idx_full, q_inj_idx_full];

    corrupted_indices = unique(corrupted_indices);

    % Check detection
    if ~ismember(max_r_idx, corrupted_indices) && max_val > 3
        fprintf('Iter %d => multi-error detection failure.\n', iter);
        fprintf('Largest residual = %d (val=%.2f), not in corrupted set.\n', max_r_idx, max_val);

        % ---- NEW PART: Single-Error Tests ----
        % We'll break from the main loop and do the individual corruption test.
        fprintf('Now testing single-error corruption for each measurement in the corrupted set...\n');

        for cidx = corrupted_indices
            if cidx > nz_actual, continue; end  % safety check

            % 1) Start from the same base measurement (z_base + noise)
            z_test = z_base + sigma .* randn(size(z_base));  % or you can keep the same noise

            % 2) Corrupt ONLY cidx by some factor
            single_sf = sf;  % or random, up to you
            if keep_mask(cidx)
                redidx_c = find(kept_indices == cidx);
                z_test(redidx_c) = z_test(redidx_c)*single_sf;
            else
                % If that measurement isn't even in the set, skip
                continue;
            end

            % 3) Rebuild full measurement vector
            z_full_test = zeros(nz_actual,1);
            z_full_test(keep_mask)  = z_test;
            z_full_test(~keep_mask) = 999;

            % 4) Run SE
            [~, ok_single, r_single] = LagrangianM_partial(z_full_test, result_struct, 0, bus_base);

            if ~ok_single
                fprintf('   Single-error test for meas %d => no converge.\n', cidx);
                continue;
            end

            % 5) Largest residual
            r_single_full = zeros(nz_actual,1);
            r_single_full(keep_mask)  = r_single;
            r_single_full(~keep_mask) = -999;
            [val_s, idx_s] = max(r_single_full);

            % 6) Check if it identifies cidx
            if idx_s == cidx
                fprintf('   Single-error test for meas %d => DETECTED (maxR=%.2f)!\n', cidx, val_s);
            else
                fprintf('   Single-error test for meas %d => Not detected; largestR at %d.\n', cidx, idx_s);
            end
        end

        % done testing single corruption
        break; 
    else
        fprintf('Iter %d => largest residual %d => detection is correct.\n', iter, max_r_idx);
    end

    if iter == max_iterations
        disp('All 100 iterations ended without detection failure under these conditions.');
    end
end
