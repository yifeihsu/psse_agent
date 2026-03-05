function gen_wls_nlm_series_ieee14()
% Generate a measurement series and run WLS + normalized Lagrange multiplier test
% using your LagrangianM_singlephase.m on IEEE 14-bus (switchable to IEEE 118-bus).

%% ---------------- User parameters ----------------
CASE_NAME        = 'case14';    % 'case14' -> switch to 'case118' to scale up
N_SAMPLES        = 1000;        % number of scenarios
YEAR             = 2025;        % BPA year to pull
IND_REMOVE_REF   = 0;           % pass as 'ind' to your function (0 recommended)
OUTPUT_MAT       = sprintf('wls_nlm_series_%s_%d.mat', CASE_NAME, N_SAMPLES);

% Noise levels (per-unit) consistent with W in your function:
SIGMA_VM         = 1e-3;  % voltage magnitude
SIGMA_INJ        = 1e-2;  % P/Q injections
SIGMA_FLOW       = 1e-2;  % Pf/Qf/Pt/Qt

% Clamp the scaling to keep OPF feasible across the year
SCALE_MIN        = 0.6;
SCALE_MAX        = 1.4;

rng(42); % reproducibility

%% ---------------- Setup & load case ----------------
mpopt = mpoption('verbose', 0, 'out.all', 0);
mpc0  = loadcase(CASE_NAME);  % IEEE test case provided by MATPOWER
PD = 3; QD = 4; VM = 8; VA = 9; F_BUS = 1; T_BUS = 2;

nb = size(mpc0.bus, 1);
nl = size(mpc0.branch, 1);

% Base loads (MW/MVAr) in the case that we will scale
Pd0 = mpc0.bus(:, PD);
Qd0 = mpc0.bus(:, QD);

%% ---------------- Load BPA total load & build a scale time series ----------------
[bpa_scale_raw, bpa_datetimes] = get_bpa_total_load_scale(YEAR);
% Normalize and clamp for feasibility
scale_series = bpa_scale_raw / nanmean(bpa_scale_raw);
scale_series = max(min(scale_series, SCALE_MAX), SCALE_MIN);

% Draw N_SAMPLES indices across the available time points
idx = randi(numel(scale_series), N_SAMPLES, 1);
scale_used = scale_series(idx);
time_used  = bpa_datetimes(idx);

%% ---------------- Pre-allocate outputs ----------------
r_cell         = cell(N_SAMPLES, 1);
lambdaN_cell   = cell(N_SAMPLES, 1);
lambdaN_rx     = cell(N_SAMPLES, 1);  % [lambda_r, lambda_x] per line
z_noisy_cell   = cell(N_SAMPLES, 1);
success_opf    = false(N_SAMPLES, 1);
success_wls    = false(N_SAMPLES, 1);
opf_solver     = strings(N_SAMPLES, 1);

%% ---------------- Main loop ----------------
for k = 1:N_SAMPLES
    s = scale_used(k);  % load scale factor

    % Scale loads in a copy of the base case
    mpc = mpc0;
    mpc.bus(:, PD) = s * Pd0;
    mpc.bus(:, QD) = s * Qd0;

    % --- Run AC-OPF; if it fails, try a power flow as fallback ---
    try
        results = runopf(mpc, mpopt);          % AC OPF by default
        success_opf(k) = results.success == 1;
        opf_solver(k)  = "runopf";
        if ~success_opf(k)
            error('OPF infeasible');
        end
    catch
        % Fallback to AC power flow to at least compute a feasible state
        results = runpf(mpc, mpopt);
        success_opf(k) = results.success == 1;
        opf_solver(k)  = "runpf";
        if ~success_opf(k)
            % mark as failed sample and continue
            continue
        end
    end

    % --- Build true measurement vector in *per-unit* (order matches your function) ---
    z_true = build_measurements_pu(results);

    % --- Add Gaussian noise consistent with your R (before zero-inj removal) ---
    nb_k = size(results.bus, 1);
    nl_k = size(results.branch, 1);
    sigma = [ ...
        SIGMA_VM   * ones(nb_k, 1);            % Vm
        SIGMA_INJ  * ones(2*nb_k, 1);          % Pinj, Qinj (your function will remove zero-inj rows)
        SIGMA_FLOW * ones(4*nl_k, 1)           % Pf, Qf, Pt, Qt
    ];
    z_noisy = z_true + sigma .* randn(size(z_true));

    % --- Call your WLS + normalized LM estimator ---
    bus_data = results.bus;  % used by your function for initial (Va, Vm)
    try
        [lambdaN, succ_wls, r_k, ~, ~] = LagrangianM_singlephase(z_noisy, results, IND_REMOVE_REF, bus_data);
        success_wls(k)  = succ_wls == 1;
        r_cell{k}       = r_k;
        lambdaN_cell{k} = lambdaN;
        % reshape to [nl, 2] for [r,x] parameters per line (your lambdaN is 2*nl x 1)
        if numel(lambdaN) == 2*nl_k
            lambdaN_rx{k} = reshape(lambdaN, 2, nl_k).';  % [nl x 2], columns: [lambda_r, lambda_x]
        else
            lambdaN_rx{k} = [];
        end
        z_noisy_cell{k} = z_noisy;
    catch ME
        % If WLS did not converge or any other error, record and continue
        success_wls(k)  = false;
        r_cell{k}       = [];
        lambdaN_cell{k} = [];
        lambdaN_rx{k}   = [];
        z_noisy_cell{k} = z_noisy;
        warning('Sample %d: WLS/NLM failed with message: %s', k, ME.message);
    end
end

%% ---------------- Save dataset ----------------
meta.case_name   = CASE_NAME;
meta.nb          = nb;
meta.nl          = nl;
meta.lines_f     = results.branch(:, F_BUS);
meta.lines_t     = results.branch(:, T_BUS);
meta.sigma       = struct('Vm', SIGMA_VM, 'inj', SIGMA_INJ, 'flow', SIGMA_FLOW);
meta.year        = YEAR;

dataset = struct();
dataset.r             = r_cell;
dataset.lambdaN       = lambdaN_cell;
dataset.lambdaN_rx    = lambdaN_rx;
dataset.z_noisy       = z_noisy_cell;
dataset.scale         = scale_used;
dataset.time          = time_used;
dataset.success_opf   = success_opf;
dataset.success_wls   = success_wls;
dataset.opf_solver    = opf_solver;
dataset.meta          = meta;

save(OUTPUT_MAT, 'dataset', '-v7.3');
fprintf('\nSaved dataset to %s\n', OUTPUT_MAT);

end

%% ---------------- Helper: measurement builder (per-unit) ----------------
function z_pu = build_measurements_pu(results)
% Construct z in the order:
% [ Vm (nb); Pinj (nb); Qinj (nb); Pf (nl); Qf (nl); Pt (nl); Qt (nl) ] in per-unit
% Using solved voltages and Y-bus/branch admittances.

VM = 8; VA = 9; F_BUS = 1; T_BUS = 2;
nb = size(results.bus, 1);
nl = size(results.branch, 1);

% Bus voltages
Vm  = results.bus(:, VM);
Va  = results.bus(:, VA) * pi/180;
V   = Vm .* exp(1j*Va);

% Network admittances
[Ybus, Yf, Yt] = makeYbus(results);

% Bus injections (per-unit complex power)
Ibus = Ybus * V;
Sbus = V .* conj(Ibus);      % S_inj = V * conj(I)
Pinj = real(Sbus);
Qinj = imag(Sbus);

% Branch flows (per-unit)
If = Yf * V; It = Yt * V;
Sf = V(results.branch(:, F_BUS)) .* conj(If);   % from side
St = V(results.branch(:, T_BUS)) .* conj(It);   % to side
Pf = real(Sf); Qf = imag(Sf);
Pt = real(St); Qt = imag(St);

% Stack to match the estimator's internal order
z_pu = [Vm; Pinj; Qinj; Pf; Qf; Pt; Qt];
z_pu = z_pu(:);
end

%% ---------------- Helper: BPA total-load scale series ----------------
function [scale, tstamp] = get_bpa_total_load_scale(year_num)
% Returns a vector 'scale' ~ (TotalLoad / mean(TotalLoad)) and a datetime vector.
% Pulls BPA "WindGenTotalLoadYTD_YEAR.xlsx" when possible.
% If download fails, throws with a message to place the file locally.

% BPA page with year-indexed links is documented here:
% https://transmission.bpa.gov/business/operations/wind/  (see Item #5 on the page)
% File pattern: OPITabularReports/WindGenTotalLoadYTD_YYYY.xlsx
base = 'https://transmission.bpa.gov/business/operations/wind/OPITabularReports/';
fname = sprintf('WindGenTotalLoadYTD_%d.xlsx', year_num);
url   = [base, fname];

local = fullfile(tempdir, fname);
try
    if ~isfile(local), websave(local, url); end
catch
    error(['Could not download BPA file %s.\n' ...
           'Download it manually from the BPA data page and set "local" accordingly.\nURL index: https://transmission.bpa.gov/business/operations/wind/'], fname);
end

T = readtable(local, 'FileType','spreadsheet');

% Find the column that looks like "Total Load" (MW)
varnames = string(T.Properties.VariableNames);
cand = contains(upper(varnames), "LOAD") & contains(upper(varnames), "TOTAL");
if ~any(cand)
    % fall back: any column with "Load" if "Total Load" not present
    cand = contains(upper(varnames), "LOAD");
end
load_col = find(cand, 1, 'first');

% Find a timestamp column
time_col = find(contains(upper(varnames), "TIME") | contains(upper(varnames), "DATE"), 1, 'first');

total_load_MW = T{:, load_col};
tstamp = T{:, time_col};
if ~isa(tstamp, 'datetime')
    try
        tstamp = datetime(tstamp, 'ConvertFrom','excel');
    catch
        tstamp = datetime(tstamp);
    end
end
total_load_MW = filloutliers(total_load_MW, 'previous', 'movmedian', 49, 'ThresholdFactor', 4);
scale = total_load_MW ./ nanmean(total_load_MW);
end
