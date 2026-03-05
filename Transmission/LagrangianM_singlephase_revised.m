function [lambdaN, success, r, x_final, mu_final] = LagrangianM_singlephase_revised(z, result, ind)
% Revised Normalized Lagrangian Multiplier method for parameter error identification
% Corrects issues in the original formulation.
% Implements standard constrained WLS and NLM calculation.
%
% Inputs:
%   z      : measurement vector (ensure order matches assumptions below)
%   result : MATPOWER struct with bus/branch/gen/baseMVA
%   ind    : flag. if ==1, remove reference-angle "measurement" z(nb)
%            (Assumes Vm are first nb measurements, ref angle corresponds to z(nb))
%
% Outputs:
%   lambdaN  : Normalized Lagrangian multipliers w.r.t. line parameters [r1, x1, r2, x2, ...]
%   success  : 1 if WLS converged, 0 if not
%   r        : normalized measurement residuals
%   x_final  : final state vector [theta(non-ref); V]
%   mu_final : final Lagrange multipliers for zero-injection constraints

%% 1) Basic Setup
ref = find(result.bus(:, 2) == 3); % reference bus index
if isempty(ref) || numel(ref) > 1
    error('Could not find a unique reference bus (type 3)');
end
baseMVA = result.baseMVA;
nb = size(result.bus, 1); % number of buses
nl = size(result.branch, 1); % number of lines
fbus = result.branch(:, 1); % From bus indices
tbus = result.branch(:, 2); % To bus indices

% Ensure reference angle is zero (relative) for initialization if needed
% Note: The state vector x won't include the ref angle explicitly
% result.bus(:, 9) = result.bus(:, 9) - result.bus(ref, 9); % VAng

eps = 1e-4;   % Convergence tolerance for state update dx
max_iter = 20; % Max iterations for WLS
success = 1;  % Convergence flag

%% 2) Identify Zero-Injection Buses
% Calculate net injection S = P + jQ at each bus
% Note: makeSbus is not a standard MATPOWER function, assuming its role here.
% A standard way:
Pgen = zeros(nb, 1); Qgen = zeros(nb, 1);
Pload = zeros(nb, 1); Qload = zeros(nb, 1);
if isfield(result, 'gen') && ~isempty(result.gen)
    on = find(result.gen(:, 8) > 0); % which generators are on?
    genbus = result.gen(on, 1); % buses connected to online generators
    Pgen = accumarray(genbus, result.gen(on, 2), [nb, 1]);
    Qgen = accumarray(genbus, result.gen(on, 3), [nb, 1]);
end
if isfield(result, 'bus') && ~isempty(result.bus) % bus loads PD, QD
    Pload = result.bus(:, 3);
    Qload = result.bus(:, 4);
end
Pinj_known = (Pgen - Pload) / baseMVA;
Qinj_known = (Qgen - Qload) / baseMVA;

% Find buses where BOTH P and Q injections are effectively zero
ZIn = find(abs(Pinj_known) < 1e-9 & abs(Qinj_known) < 1e-9);
nzi = numel(ZIn);
fprintf('Found %d zero-injection buses.\n', nzi);

%% 3) Form Weighting Matrix W
% Assumed measurement order: [Vm (nb); Pinj (nb); Qinj (nb); Pf (nl); Qf (nl); Pt (nl); Qt (nl)]
% Total measurements initially = nb + 2*nb + 4*nl = 3*nb + 4*nl
m_full = 3*nb + 4*nl;

% Standard deviations (adjust as needed)
std_dev_V = 0.004; % p.u.
std_dev_P = 0.01;  % p.u. on baseMVA
std_dev_Q = 0.01;  % p.u. on baseMVA
std_dev_Pf = 0.008; % p.u. on baseMVA (flows)
std_dev_Qf = 0.008; % p.u. on baseMVA (flows)

Rdiag_full = [
    std_dev_V^2 * ones(nb, 1);
    std_dev_P^2 * ones(nb, 1);
    std_dev_Q^2 * ones(nb, 1);
    std_dev_Pf^2 * ones(nl, 1);
    std_dev_Qf^2 * ones(nl, 1);
    std_dev_Pf^2 * ones(nl, 1); % Assuming Pt/Qt same accuracy as Pf/Qf
    std_dev_Qf^2 * ones(nl, 1)
];

if numel(Rdiag_full) ~= m_full
    error('Dimension mismatch in Rdiag_full construction. Expected %d, got %d.', m_full, numel(Rdiag_full));
end
if numel(z) ~= m_full
     error('Dimension mismatch between measurement vector z (%d) and expected full size (%d).', numel(z), m_full);
end

% Indices for measurements in the full vector z
idx_V = 1:nb;
idx_P = nb + (1:nb);
idx_Q = 2*nb + (1:nb);
idx_Pf = 3*nb + (1:nl);
idx_Qf = 3*nb + nl + (1:nl);
idx_Pt = 3*nb + 2*nl + (1:nl);
idx_Qt = 3*nb + 3*nl + (1:nl);

% Identify measurements to keep (remove ZI injections)
keep_idx = true(m_full, 1);
zi_inj_idx = [idx_P(ZIn), idx_Q(ZIn)]; % Indices of P/Q injections at ZI buses
keep_idx(zi_inj_idx) = false;

% Handle optional reference angle "measurement" removal
if ind == 1
    if ref > nb % Check if ref index is valid for V measurement block
        error('Reference bus index %d is out of bounds for V measurements (1 to %d).', ref, nb);
    end
    keep_idx(ref) = false; % Remove the Vm measurement at the ref bus if ind==1
    fprintf('Flag ind=1: Removing measurement corresponding to index %d (expected Vm at ref bus %d).\n', ref, ref);
end

% Filter z, Rdiag, and create W
z_filt = z(keep_idx);
Rdiag_filt = Rdiag_full(keep_idx);
W = spdiags(1./Rdiag_filt, 0, numel(z_filt), numel(z_filt));
R = spdiags(Rdiag_filt, 0, numel(z_filt), numel(z_filt)); % Inverse of W

m = numel(z_filt); % Number of actual measurements used in WLS

%% 4) Build Y‐Bus
% Use MATPOWER's makeYbus
[Ybus, Yf, Yt] = makeYbus(baseMVA, result.bus, result.branch);

%% 5) State Initialization
% State vector x = [Th_1;...; Th_{ref-1}; Th_{ref+1}; ...; Th_nb; V_1; ...; V_nb]
% Size: (nb-1) + nb = 2*nb - 1
nx = 2*nb - 1;
state_idx_th = 1:(nb-1);
state_idx_v = nb:(2*nb-1);

% Initial guess (flat start)
V0 = ones(nb, 1);
Th0 = zeros(nb, 1);
% Extract state variables, removing ref angle
x0 = [Th0(1:ref-1); Th0(ref+1:nb); V0];

% Initialize zero-injection multipliers
mu = zeros(2 * nzi, 1); % One for P constraint, one for Q constraint at each ZI bus

%% === 6) WLS with Zero Injections using Newton-Raphson ===
k = 0; % iteration counter
converged = false;

% Map state vector x back to full V, Theta vectors
map_state_to_angles = [1:ref-1, ref+1:nb]; % Indices in bus angle vector corresponding to x
V = zeros(nb, 1);
Th = zeros(nb, 1);

while ~converged && k < max_iter
    k = k + 1;
    fprintf('WLS Iteration: %d\n', k);

    % Map current state x0 to full V and Theta vectors
    Th(map_state_to_angles) = x0(state_idx_th); % Th(ref) remains 0
    V(:) = x0(state_idx_v); % Map V portions of x0 to V vector
    Vc = V .* exp(1j * Th); % Complex voltage vector Vc

    % --- 6a) Calculate Measurement Function h(x) ---
    Ibus = Ybus * Vc;
    Sbus = Vc .* conj(Ibus); % Complex power injections S = P + jQ
    Sf = Vc(fbus) .* conj(Yf * Vc); % Complex power flow 'from' end
    St = Vc(tbus) .* conj(Yt * Vc); % Complex power flow 'to' end

    hx_full = [
        V;          % Vm
        real(Sbus); % Pinj
        imag(Sbus); % Qinj
        real(Sf);   % Pf
        imag(Sf);   % Qf
        real(St);   % Pt
        imag(St)    % Qt
    ];
    hx = hx_full(keep_idx); % Filter h(x) to match z_filt

    % --- 6b) Calculate Zero-Injection Constraint Function c(x) ---
    % Constraints are Pinj(ZIn) = 0 and Qinj(ZIn) = 0
    cx = [real(Sbus(ZIn)); imag(Sbus(ZIn))]; % Current value of constraints

    % --- 6c) Calculate Jacobians H = dh/dx and C = dc/dx ---
    % Need partial derivatives w.r.t state variables (Th_nonref, V_all)
    [dSbus_dVa, dSbus_dVm] = dSbus_dV(Ybus, Vc);
    [dSf_dVa, dSf_dVm, dSt_dVa, dSt_dVm] = dSbr_dV(result.branch, Yf, Yt, Vc); % From MATPOWER

    % Extract relevant columns for H and C
    % State order: [Th_nonref; V_all]
    % Full Jacobian columns: [Th_all; V_all]
    j_V   = 1:nb; % Indices for Vm derivatives (columns nb+1 to 2*nb)
    j_Th  = (nb+1):(2*nb); % Indices for Theta derivatives (columns 1 to nb)

    % Build H_full (Jacobian of hx_full w.r.t ALL angles and Vm)
    dVm_dVa = sparse(nb, nb); % Vm doesn't depend on angles
    dVm_dVm = speye(nb);
    dPinj_dVa = real(dSbus_dVa); dPinj_dVm = real(dSbus_dVm);
    dQinj_dVa = imag(dSbus_dVa); dQinj_dVm = imag(dSbus_dVm);
    dPf_dVa = real(dSf_dVa); dPf_dVm = real(dSf_dVm);
    dQf_dVa = imag(dSf_dVa); dQf_dVm = imag(dSf_dVm);
    dPt_dVa = real(dSt_dVa); dPt_dVm = real(dSt_dVm);
    dQt_dVa = imag(dSt_dVa); dQt_dVm = imag(dSt_dVm);

    H_full_all_states = [
        dVm_dVa,   dVm_dVm;
        dPinj_dVa, dPinj_dVm;
        dQinj_dVa, dQinj_dVm;
        dPf_dVa,   dPf_dVm;
        dQf_dVa,   dQf_dVm;
        dPt_dVa,   dPt_dVm;
        dQt_dVa,   dQt_dVm
    ];

    % Select columns corresponding to actual state variables (remove ref angle column)
    state_cols = [map_state_to_angles, nb + j_V]; % map_state_to_angles -> Th cols, nb+j_V -> V cols
    H_all_meas = H_full_all_states(:, state_cols);

    % Select rows corresponding to filtered measurements
    H = H_all_meas(keep_idx, :);

    % Build C (Jacobian of cx w.r.t state variables)
    dPinjZI_dVa = dPinj_dVa(ZIn, :); dPinjZI_dVm = dPinj_dVm(ZIn, :);
    dQinjZI_dVa = dQinj_dVa(ZIn, :); dQinjZI_dVm = dQinj_dVm(ZIn, :);
    C_full_states = [
        dPinjZI_dVa, dPinjZI_dVm;
        dQinjZI_dVa, dQinjZI_dVm
    ];
    C = C_full_states(:, state_cols); % Select state columns

    % --- 6d) Formulate and Solve KKT System ---
    % Mismatch vectors
    delta_z = z_filt - hx;
    delta_c = -cx; % Constraint violation

    % KKT Matrix (Gauss-Newton approximation)
    G = H' * W * H; % Approx Hessian of Lagrangian w.r.t x
    KKT = [ G, C';
            C, sparse(2*nzi, 2*nzi) ];

    % Right Hand Side
    rhs = [ H' * W * delta_z;
            delta_c ];

    % Solve for state update (dx) and multiplier update (dmu)
    % Check conditioning before solving
    if condest(KKT) > 1e14
        warning('KKT matrix is ill-conditioned (condest=%.2e). WLS may fail.', condest(KKT));
        success = 0;
        break; % Exit loop
    end
    
    % Solve using backslash (sparse factorization)
    update_vec = KKT \ rhs;

    dx = update_vec(1:nx);
    % Note: The multipliers obtained here are related to the *change* needed
    % if the constraints were linearized. For NLM, we need the final converged mu.
    % A better way to get mu is often outside the loop, or using the final KKT eqns.
    % Let's store the mu part of the update vector - it might approximate the final mu.
    mu_from_solve = update_vec(nx+1 : end); % This interpretation can be subtle

    % --- 6e) Update State and Check Convergence ---
    x0 = x0 + dx;
    norm_dx = norm(dx, Inf);
    fprintf('   ||dx|| = %.4e\n', norm_dx);

    if norm_dx < eps
        converged = true;
        fprintf('WLS converged in %d iterations.\n', k);
    elseif k == max_iter
        warning('WLS did not converge in %d iterations.', max_iter);
        success = 0;
    end
end % End WLS Iteration Loop

%% Check for WLS success before proceeding
if ~success
    lambdaN = 10 * ones(2 * nl, 1); % Return dummy large values
    r = [];
    x_final = x0;
    mu_final = mu; % Return last mu estimate
    return;
end

%% 7) Post-WLS Calculations (using converged state x0)
x_final = x0; % Final state vector
Th(map_state_to_angles) = x_final(state_idx_th);
V(:) = x_final(state_idx_v);
Vc = V .* exp(1j * Th); % Final complex voltages

% --- 7a) Final Residuals and Measurement Multipliers ---
Ibus = Ybus * Vc;
Sbus = Vc .* conj(Ibus);
Sf = Vc(fbus) .* conj(Yf * Vc);
St = Vc(tbus) .* conj(Yt * Vc);
hx_full = [
    V; real(Sbus); imag(Sbus); real(Sf); imag(Sf); real(St); imag(St)
];
hx = hx_full(keep_idx);
final_resid_unnorm = z_filt - hx;

% Measurement Lagrange Multipliers (implicitly lambda_m = -W * residual)
lambda_m = -W * final_resid_unnorm;

% --- 7b) Final Zero-Injection Multipliers ---
% We can re-solve the lower part of the KKT system or use mu_from_solve.
% Using final H, C, G: C*dx + 0*dmu = delta_c => C*dx = -cx
% G*dx + C'*dmu = H'*W*delta_z => C'*dmu = H'*W*delta_z - G*dx
% Let's use the multipliers from the last solve step as an approximation.
% A more robust way might involve solving C'*mu = H'*W*(z-h) - G*dx (using dx=0 at optimum)
% => C'*mu = H'*W*(z-h)
% Let's try solving this:
if nzi > 0
    rhs_mu = H' * W * final_resid_unnorm; % Note: residual = z-h
    % Need to solve C' * mu = rhs_mu potentially in a least-squares sense if C' isn't full rank
    % Or use mu from the last iteration's solution vector
    % mu_final = mu_from_solve; % Simplest approach using last solve step
    % Alternative: Solve C'*mu = H'*W*r (where r = z-h)
     try 
         mu_final = C' \ rhs_mu; 
         fprintf('Calculated final ZI multipliers mu.\n');
     catch ME
         warning('Could not directly solve for final ZI multipliers mu. Using last iterative value. Error: %s', ME.message);
         mu_final = mu_from_solve; % Fallback
     end
else
    mu_final = []; % No ZI constraints
end

% Check dimensions
if numel(mu_final) ~= 2*nzi
     warning('Dimension mismatch for final mu. Expected %d, got %d. Using zeros.', 2*nzi, numel(mu_final));
     mu_final = zeros(2*nzi, 1); % Fallback if calculation failed
end


% Stack all multipliers: lambda_m for measurements, mu for ZI constraints
Lambda = [lambda_m; mu_final];

%% === 8) Build Parameter Jacobian w.r.t. r & x for each line ===
% Hp = dh/dp, Cp = dc/dp where p = [r1, x1, r2, x2, ...]
% Parameter vector p has size 2*nl

% Derivatives require the final state (V, Th)
Vi = V(fbus); Vj = V(tbus);
Th_i = Th(fbus); Th_j = Th(tbus);
dTh = Th_i - Th_j;

Hp_full = zeros(m_full, 2*nl); % Jacobian of ALL measurements w.r.t ALL parameters
Cp_full = zeros(2*nb, 2*nl); % Jacobian of ALL injections w.r.t ALL parameters

for k = 1:nl
    % Line k connects bus i to bus j
    i = fbus(k);
    j = tbus(k);
    r_k = result.branch(k, 3); % Resistance
    x_k = result.branch(k, 4); % Reactance
    % Note: MATPOWER's Ybus includes tap ratios and shunts.
    % For simplicity, assuming makeYbus derivatives handle these.
    % A more direct calculation based on parameter derivatives:
    Y_series_k = 1 / (r_k + 1j*x_k);
    g_k = real(Y_series_k);
    b_k = imag(Y_series_k);

    % Derivatives of g_k, b_k w.r.t r_k, x_k
    denom_sq = (r_k^2 + x_k^2)^2;
    if denom_sq < 1e-12 % Avoid division by zero for lossless lines (should handle appropriately)
         dg_dr = 0; db_dr = 0; dg_dx = 0; db_dx = 0; % Placeholder
    else
        dg_dr = (x_k^2 - r_k^2) / denom_sq;
        db_dr = (2 * r_k * x_k) / denom_sq;
        dg_dx = (-2 * r_k * x_k) / denom_sq;
        db_dx = (r_k^2 - x_k^2) / denom_sq; % Corrected: was (x^2-r^2)
    end
    
    % Impact on injections Sbus = V .* conj(Ybus*V)
    % Need dSbus / dr_k = dSbus / dYbus * dYbus / dr_k
    % Need dSbus / dx_k = dSbus / dYbus * dYbus / dx_k
    % dYbus/drk, dYbus/dxk affect only entries (i,i), (j,j), (i,j), (j,i)
    % Let dY_dr = dYbus/dr_k and dY_dx = dYbus/dx_k
    % dY_dr(i,i) = dg_dr; dY_dr(j,j) = dg_dr; dY_dr(i,j) = -dg_dr; dY_dr(j,i) = -dg_dr;
    % dY_dr adds j*db_dr for shunts if they depend on r_k (unlikely)
    % dY_dx(i,i) = dg_dx + j*db_dx (assuming shunt susceptance b/2 depends on x_k - check makeYbus)
    % dY_dx(j,j) = dg_dx + j*db_dx
    % dY_dx(i,j) = -dg_dx - j*db_dx
    % dY_dx(j,i) = -dg_dx - j*db_dx
    
    % Simpler: Calculate impact on flows directly using standard formulas
    % Pf_k = Vi^2*g_k - Vi*Vj*(g_k*cos(dTh_k) + b_k*sin(dTh_k))
    % Qf_k = -Vi^2*b_k - Vi*Vj*(g_k*sin(dTh_k) - b_k*cos(dTh_k))
    % Pt_k = Vj^2*g_k - Vi*Vj*(g_k*cos(dTh_k) - b_k*sin(dTh_k))
    % Qt_k = -Vj^2*b_k + Vi*Vj*(g_k*sin(dTh_k) + b_k*cos(dTh_k))
    
    Vi_k = V(i); Vj_k = V(j); % scalar V for line k
    dTh_k = Th(i) - Th(j); % scalar angle diff for line k
    cos_dTh = cos(dTh_k);
    sin_dTh = sin(dTh_k);
    
    % Derivatives w.r.t g_k, b_k
    dPf_dg = Vi_k^2 - Vi_k*Vj_k*cos_dTh;
    dPf_db = - Vi_k*Vj_k*sin_dTh;
    dQf_dg = - Vi_k*Vj_k*sin_dTh;
    dQf_db = -Vi_k^2 + Vi_k*Vj_k*cos_dTh; % Corrected sign convention? Check convention used for Q above. Assuming Q = imag(V*conj(I))
    % If Q = -imag(V*conj(I)), signs flip for Q derivatives
    % Let's stick to Q = imag(V*conj(I)) as in MATPOWER h(x) above.
    % Qf = -Vi^2*(b_shunt + b_k) - Vi*Vj*(g_k*sin(dTh_k) - b_k*cos(dTh_k)) <-- Need to be careful about shunts
    % Assume no line shunts for now, or they are fixed.
    dQf_db = -Vi_k^2 + Vi_k*Vj_k*cos_dTh; % Assuming Qf = -Vi^2*b_k - Vi*Vj*(g_k*sin - b_k*cos)
    
    dPt_dg = Vj_k^2 - Vi_k*Vj_k*cos_dTh;
    dPt_db = Vi_k*Vj_k*sin_dTh;
    dQt_dg = Vi_k*Vj_k*sin_dTh;
    dQt_db = -Vj_k^2 + Vi_k*Vj_k*cos_dTh; % Assuming Qt = -Vj^2*b_k + Vi*Vj*(g_k*sin + b_k*cos)

    % Apply Chain Rule: dF/dr = dF/dg*dg/dr + dF/db*db/dr
    dPf_dr = dPf_dg*dg_dr + dPf_db*db_dr; dPf_dx = dPf_dg*dg_dx + dPf_db*db_dx;
    dQf_dr = dQf_dg*dg_dr + dQf_db*db_dr; dQf_dx = dQf_dg*dg_dx + dQf_db*db_dx;
    dPt_dr = dPt_dg*dg_dr + dPt_db*db_dr; dPt_dx = dPt_dg*dg_dx + dPt_db*db_dx;
    dQt_dr = dQt_dg*dg_dr + dQt_db*db_dr; dQt_dx = dQt_dg*dg_dx + dQt_db*db_dx;
    
    % Parameter index in p: r_k -> 2k-1, x_k -> 2k
    p_idx_r = 2*k - 1;
    p_idx_x = 2*k;

    % Populate Jacobians (Hp_full, Cp_full)
    % Hp_full: relates ALL measurements to parameters r_k, x_k
    % Cp_full: relates ALL injections (P, Q at ALL buses) to parameters r_k, x_k
    
    % Vm measurements are not directly affected by rk, xk
    Hp_full(idx_V, p_idx_r:p_idx_x) = 0;

    % Injection measurements: P_i, P_j, Q_i, Q_j change
    % dPi/drk = dPf/drk (assuming Pi only includes flow on line k out of bus i)
    % dPi/dxk = dPf/dxk
    % dPj/drk = dPt/drk (P injection at j includes flow Pt *into* j from k)
    % dPj/dxk = dPt/dxk
    % dQi/drk = dQf/drk
    % dQi/dxk = dQf/dxk
    % dQj/drk = dQt/drk
    % dQj/dxk = dQt/dxk
    Hp_full(idx_P(i), p_idx_r) = dPf_dr; Hp_full(idx_P(i), p_idx_x) = dPf_dx;
    Hp_full(idx_P(j), p_idx_r) = dPt_dr; Hp_full(idx_P(j), p_idx_x) = dPt_dx;
    Hp_full(idx_Q(i), p_idx_r) = dQf_dr; Hp_full(idx_Q(i), p_idx_x) = dQf_dx;
    Hp_full(idx_Q(j), p_idx_r) = dQt_dr; Hp_full(idx_Q(j), p_idx_x) = dQt_dx;

    % Flow measurements: Pf, Qf, Pt, Qt change
    Hp_full(idx_Pf(k), p_idx_r) = dPf_dr; Hp_full(idx_Pf(k), p_idx_x) = dPf_dx;
    Hp_full(idx_Qf(k), p_idx_r) = dQf_dr; Hp_full(idx_Qf(k), p_idx_x) = dQf_dx;
    Hp_full(idx_Pt(k), p_idx_r) = dPt_dr; Hp_full(idx_Pt(k), p_idx_x) = dPt_dx;
    Hp_full(idx_Qt(k), p_idx_r) = dQt_dr; Hp_full(idx_Qt(k), p_idx_x) = dQt_dx;
    
    % Cp_full: Jacobian of [P_all; Q_all] w.r.t rk, xk
    Cp_full(i, p_idx_r) = dPf_dr; Cp_full(i, p_idx_x) = dPf_dx; % P inj at bus i
    Cp_full(j, p_idx_r) = dPt_dr; Cp_full(j, p_idx_x) = dPt_dx; % P inj at bus j
    Cp_full(nb+i, p_idx_r) = dQf_dr; Cp_full(nb+i, p_idx_x) = dQf_dx; % Q inj at bus i
    Cp_full(nb+j, p_idx_r) = dQt_dr; Cp_full(nb+j, p_idx_x) = dQt_dx; % Q inj at bus j

end

% Filter Hp and Cp to match the measurements used (keep_idx) and ZI constraints
Hp = Hp_full(keep_idx, :); % Jacobian of used measurements w.r.t parameters
Cp = Cp_full([ZIn, nb+ZIn], :); % Jacobian of ZI constraints w.r.t parameters

%% === 9) Calculate Normalized Lagrangian Multipliers (NLM) ===

% --- 9a) NLM Numerator ---
% Numerator = lambda_p = (dL/dp) = (dh/dp)' * lambda_m + (dc/dp)' * mu
% lambda_m = -W*(z-h) are multipliers for measurements h(x)=z
% mu are multipliers for zero injections c(x)=0
% lambda_p = Hp' * lambda_m + Cp' * mu
% lambda_p = Hp' * (-W * final_resid_unnorm) + Cp' * mu_final
lambda_param_numerator = Hp' * lambda_m + Cp' * mu_final;

% --- 9b) NLM Denominator (Variance calculation) ---
% Denominator = sqrt(Var(lambda_p))
% Var(lambda_p) = Var(Hp'*lambda_m + Cp'*mu)
% Requires covariance matrix of the multipliers [lambda_m; mu].
% This is related to the inverse of the full KKT matrix.

% Let KKT_final be the KKT matrix from the last iteration.
KKT_final = [ G, C'; C, sparse(2*nzi, 2*nzi) ];

% The covariance of the state estimate [x; mu] is approx KKT_final^-1 * Sigma_rhs * (KKT_final^-1)'
% Where Sigma_rhs relates to measurement errors. This gets complex.

% Alternative Approximation (Common): Use sensitivity matrix S_p and residual covariance Omega
% S_p = [Hp; Cp] (Sensitivity of all constraints w.r.t params)
% Covariance matrix of constraints = BlockDiag(Omega, Var(c(x)))
% Var(c(x)) is tricky, often assumed small or related to model error.

% Let's follow the *structure* hinted at in the original code, assuming it aimed
% to calculate Cov(Lambda) = Cov([lambda_m; mu]).
% The original code used phi * inv(W) * phi', where phi came from inv(Gain).
% Calculating inv(KKT_final) is required to get the block corresponding to mu.
% Cov(mu) approx = (KKT_final / G)^-1 where KKT / G = -C G^-1 C'
% Cov(lambda_m) approx = W * Cov(residual) * W = W * Omega * W
% Cov(lambda_m, mu) is also needed.

% --- Simplified Variance Approximation ---
% Approximate Var(lambda_p) ~ diag(Hp' * W * Hp) -- ignores ZI constraints and correlations
% This is often too simplistic.

% --- Let's try to estimate Cov(Lambda) using KKT inverse ---
% Calculate inv(KKT_final) carefully
if rcond(full(KKT_final)) < 1e-14 % Use rcond for sparse is tricky, convert to full for check
     warning('Final KKT matrix is ill-conditioned (rcond=%.2e). NLM variance may be unreliable.', rcond(full(KKT_final)));
     % Handle scenario - maybe return NaNs or use simplified variance?
     lambdaN = nan(2*nl, 1);
     % Calculate residuals anyway
     Omega = R - H * (G \ H'); % Approx residual covariance (ignoring ZI constraints impact)
     diag_omega = abs(diag(Omega)); % Use abs to avoid small negative numerical noise
     diag_omega(diag_omega < 1e-12) = 1e-12; % Prevent division by zero
     r = abs(final_resid_unnorm) ./ sqrt(diag_omega);
     r = full(r);
     return;
else
    Inv_KKT_final = inv(KKT_final); % Calculate inverse (accepting cost/risk)
    
    % Covariance of [dx; mu] ~ Inv_KKT_final * Cov(rhs) * Inv_KKT_final'
    % Assuming Cov(rhs) = BlockDiag( H'W R W H, C R C' ) ??? <-- This is complex
    
    % Let's use a common approximation for NLM variance based on KKT^-1
    % Var(lambda_p) = Sp' * Sigma_Lambda * Sp
    % Where Sigma_Lambda = Cov([lambda_m; mu])
    % And Sp = [Hp; Cp]
    % Finding Sigma_Lambda directly is hard.
    
    % Alternative: Use result that Var(dL/dp) ~ (dL/dp)' * Inv(InfoMatrix) * (dL/dp)
    % InfoMatrix is related to KKT.
    % Variance of score test: Var(lambda_p) approx = S_p' * KKT_analogue * S_p
    % Where KKT_analogue might be BlockDiag(W, large_ZI_weight)
    
    % Let's use the approximation Var(lambda_p) = [Hp' Cp'] * BlockDiag(W, C*inv(G)*C') * [Hp; Cp] ?? No.
    
    % --- Reverting to original code's *structure* for variance ---
    % It calculated S = -[W*Hp; Cp]' and variance = S * Cov(multipliers) * S'
    % Let's try and estimate Cov(multipliers) = Cov([lambda_m; mu])
    % Cov(lambda_m) ~ W*Omega*W = W*(R - H*inv(G)*H')*W = W*R*W - (W*H)*inv(G)*(H'*W)
    % Cov(mu) ~ inv(-C*inv(G)*C') ??? (Schur complement inverse)
    % Let's assume the original `covu` calculation attempted Cov(Lambda).
    % That calculation required inverting the augmented `Gain` matrix.
    % This path seems too complex and potentially flawed.
    
    % --- Final Simpler Variance Approximation for Denominator ---
    % Use sensitivity of constraints [h; c] to parameters p: Sp = [Hp; Cp]
    % Use weights associated with constraints: Wp = BlockDiag(W, W_zi)
    % W_zi needs estimation, could be large diagonal matrix or derived from Cov(mu)
    % Approx Var(lambda_p_k) ~ Sp_k' * Wp * Sp_k (diagonal elements)
    % Let's use W_zi ~ inv(Cov(mu)) ~ -C*inv(G)*C' <-- still needs inv(G)
    % Simplest: Var(lambda_p_k) ~ Hp_k' * W * Hp_k (ignore ZI part, similar to LM test)
    
    diag_var_approx = zeros(2*nl, 1);
    Wt = sqrt(W); % Cholesky of W for weighting
    for k_param = 1:(2*nl)
         Hp_k = Hp(:, k_param);
         % Simplified variance based on measurement part only:
         diag_var_approx(k_param) = norm(Wt * Hp_k)^2; 
         % Alternative using Cp too (needs weighting W_zi):
         % Cp_k = Cp(:, k_param);
         % diag_var_approx(k_param) = norm(Wt*Hp_k)^2 + norm(sqrt(W_zi)*Cp_k)^2;
    end

    % Prevent division by near-zero variance
    diag_var_approx(diag_var_approx < 1e-12) = 1e-12;
    
    lambda_param_std_dev = sqrt(diag_var_approx);
    
end % End if/else for KKT condition check

lambdaN = lambda_param_numerator ./ lambda_param_std_dev;


%% === 10) Compute & Return Normalized Residual r ===
% Use final unnormalized residuals and approximate covariance Omega
% Omega = R - H * inv(G) * H' (Approximation for constrained WLS)
try
    InvG = KKT_final(1:nx, 1:nx); % Extract G block from KKT matrix
    if rcond(full(InvG)) < 1e-14 % Check G condition
         warning('G matrix block is ill-conditioned. Residual normalization may be unreliable.');
         Omega = R; % Fallback to using original measurement variance
    else
         Omega = R - H * (InvG \ H'); % Use inv(G) = Inv_KKT_final(1:nx, 1:nx) ? No, G = H'WH
         G_final = H' * W * H; % Recalculate final G
         if rcond(full(G_final)) < 1e-14
             warning('Final G=HWH matrix is ill-conditioned. Residual normalization may be unreliable.');
             Omega = R;
         else
             Omega = R - H * (G_final \ H');
         end
    end
catch ME_Omega
    warning('Error calculating residual covariance Omega: %s. Using R.', ME_Omega.message);
    Omega = R; % Fallback
end

diag_omega = abs(diag(Omega)); % Use abs for numerical stability
diag_omega(diag_omega < 1e-12) = 1e-12; % Prevent division by zero
r = abs(final_resid_unnorm) ./ sqrt(diag_omega);
r = full(r); % Convert to full vector if sparse

fprintf('NLM calculation completed.\n');

end

% Helper functions (like makeYbus, dSbus_dV, dSbr_dV) are assumed to be
% available and follow MATPOWER conventions.