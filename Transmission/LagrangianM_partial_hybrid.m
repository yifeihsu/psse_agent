function [success, r_norm_active, x_rect, Omega_act, final_resid_act] = ...
    LagrangianM_partial_hybrid(z_full, mpc, pmu_cfg, Rvar_full, init_state)
%--------------------------------------------------------------------------
% REVISED Hybrid state estimator (PMU-Rect + SCADA) in rectangular coordinates.
% Assumes SCADA is present everywhere, with PMUs providing additional and/or
% higher-precision measurements.
%--------------------------------------------------------------------------
%% 0. CONSTANTS & BASIC DATA
define_constants;
tol           = 1e-5;      % Stopping tolerance on |dx|
max_iter      = 25;        % WLS iterations
ridge_eps     = 1e-6;      % Tiny ridge if Gain matrix is ill-conditioned
nbus          = size(mpc.bus,   1);
nbranch       = size(mpc.branch, 1);
[Ybus,Yf,Yt]  = makeYbus(mpc.baseMVA, mpc.bus, mpc.branch);
ref_idx       = find(mpc.bus(:,BUS_TYPE)==REF, 1);
if isempty(ref_idx), ref_idx = 1; end
nz_full       = numel(z_full);

%% 2. BUILD MEASUREMENT INDEX MAP (Rectangular PMU)
pmu_bus          = pmu_cfg.pmu_buses(:);
pmu_Iinfo        = pmu_cfg.pmu_branch_currents;
n_pmu_v          = numel(pmu_bus);
n_pmu_I          = size(pmu_Iinfo,1);
n_scada_inj      = nbus;
n_scada_lines    = nbranch;

idx.vm_all    = (1:nbus).';
idx.vr_pmu    = (idx.vm_all(end)+1             : idx.vm_all(end)+n_pmu_v).';
idx.vi_pmu    = (idx.vr_pmu(end)+1             : idx.vr_pmu(end)+n_pmu_v).';
idx.ir_pmu    = (idx.vi_pmu(end)+1             : idx.vi_pmu(end)+n_pmu_I).';
idx.ii_pmu    = (idx.ir_pmu(end)+1             : idx.ir_pmu(end)+n_pmu_I).';
idx.pinj      = (idx.ii_pmu(end)+1             : idx.ii_pmu(end)+n_scada_inj).';
idx.qinj      = (idx.pinj(end)+1               : idx.pinj(end)+n_scada_inj).';
idx.flows     = (idx.qinj(end)+1               : idx.qinj(end)+4*n_scada_lines).';

assert(idx.flows(end)==nz_full,'Index map size in SE function does not match z_full size from main script.');

%% 3. ACTIVE MEASUREMENTS & WEIGHTS
act_mask        = ~isnan(z_full);
z_act           = z_full(act_mask);
R_act           = Rvar_full(act_mask);
assert(all(R_act>0),'Found zero or negative variance in active measurements.');
W_act           = spdiags(1./R_act,0,length(R_act),length(R_act));

%% 4. INITIAL STATE
if nargin<5 || isempty(init_state)
    Vr = ones(nbus,1);  Vi = zeros(nbus,1);
else
    Vr = init_state.e(:); Vi = init_state.f(:);
end
x = [Vr; Vi];

%% 5. STATE-VARIABLE MAP (remove Vi at reference bus)
state_map = [ 1:nbus, nbus+1:nbus+ref_idx-1, nbus+ref_idx+1:2*nbus ]';

%% 6. WLS ITERATION LOOP
k = 0; success = 0;
while k < max_iter
    k = k+1;
    % 6a. Unpack state vector
    Vr = x(1:nbus);      Vi = x(nbus+1:2*nbus);
    Vi(ref_idx) = 0;     % Enforce angle reference
    V  = Vr + 1j*Vi;
    
    % 6b. Calculate estimated measurements h(x)
    hx = zeros(nz_full,1);
    pmu_rows = arrayfun(@(b) find(mpc.bus(:,BUS_I)==b,1), pmu_bus);

    hx(idx.vm_all) = abs(V);
    hx(idx.vr_pmu) = Vr(pmu_rows);
    hx(idx.vi_pmu) = Vi(pmu_rows);
    
    [If,It] = get_branch_currents(mpc,V,Yf,Yt);
    for ii = 1:n_pmu_I
        br     = pmu_Iinfo(ii,1); meas_b = pmu_Iinfo(ii,2);
        if meas_b==mpc.branch(br,F_BUS), I = If(br); else, I = It(br); end
        hx(idx.ir_pmu(ii)) = real(I);
        hx(idx.ii_pmu(ii)) = imag(I);
    end
    
    Sbus = V .* conj(Ybus*V);
    Sf   = V(mpc.branch(:,F_BUS)).*conj(Yf*V);
    St   = V(mpc.branch(:,T_BUS)).*conj(Yt*V);
    hx(idx.pinj)     = real(Sbus);
    hx(idx.qinj)     = imag(Sbus);
    hx(idx.flows)    = [ real(Sf); imag(Sf); real(St); imag(St) ];
    hx_act = hx(act_mask);
    
    % 6c. Calculate Jacobian H
    H = sparse(nz_full, 2*nbus);
    
    Vm = abs(V); Vm(Vm<1e-12) = 1e-12;
    dVm_dVr = spdiags(Vr./Vm,0,nbus,nbus); dVm_dVi = spdiags(Vi./Vm,0,nbus,nbus);
    H(idx.vm_all,:) = [dVm_dVr, dVm_dVi];
    
    pmu_v_selector = sparse(1:n_pmu_v, pmu_rows, 1, n_pmu_v, nbus);
    H(idx.vr_pmu,:) = [pmu_v_selector, sparse(n_pmu_v, nbus)];
    H(idx.vi_pmu,:) = [sparse(n_pmu_v, nbus), pmu_v_selector];
    
    for ii = 1:n_pmu_I
        br = pmu_Iinfo(ii,1); meas_b = pmu_Iinfo(ii,2);
        if meas_b==mpc.branch(br,F_BUS), Yrow = Yf(br,:); else, Yrow = Yt(br,:); end
        H(idx.ir_pmu(ii), :) = [real(Yrow), -imag(Yrow)];
        H(idx.ii_pmu(ii), :) = [imag(Yrow),  real(Yrow)];
    end
    
    [dSbus_dVr,dSbus_dVi] = dSbus_dV(Ybus, V, 'cart');
    [dSf_dVr,dSf_dVi,dSt_dVr,dSt_dVi] = dSbr_dV(mpc.branch,Yf,Yt,V, 'cart');
    H(idx.pinj,:) = [ real(dSbus_dVr), real(dSbus_dVi) ];
    H(idx.qinj,:) = [ imag(dSbus_dVr), imag(dSbus_dVi) ];
    H(idx.flows,:) = [ real(dSf_dVr), real(dSf_dVi); ...
                       imag(dSf_dVr), imag(dSf_dVi); ...
                       real(dSt_dVr), real(dSt_dVi); ...
                       imag(dSt_dVr), imag(dSt_dVi) ];
    
    % 6d. Solve for state update Δx
    H_act = H(act_mask, :);
    H_act = H_act(:, state_map);
    mis = z_act - hx_act;
    
    G = H_act' * W_act * H_act;
    if rcond(full(G)) < 1e-12, G = G + ridge_eps * speye(size(G)); end
    dx = G \ (H_act' * W_act * mis);
    
    x(state_map) = x(state_map) + dx;
    
    if max(abs(dx)) < tol, success = 1; break; end
end

%% 7. FORMULATE OUTPUTS
if ~success
    fprintf('State estimator failed to converge after %d iterations.\n', k);
    r_norm_active = nan; x_rect = []; Omega_act = []; final_resid_act = [];
    return
end
fprintf('Hybrid WLS converged in %d iterations.\n',k);
Vr = x(1:nbus); Vi = x(nbus+1:2*nbus); Vc = Vr + 1j*Vi;

% =============================== FIX IS HERE ===============================
% Recalculate final residual based on the h(x) from the last converged iteration.
% No need to call another function.
final_resid_act  = z_act - hx(act_mask);
% ===========================================================================

% Calculate normalized residuals
% Re-use the Jacobian H from the last iteration for Omega calculation
H_act = H(act_mask, state_map);
G = H_act' * W_act * H_act;
if rcond(full(G)) < 1e-12, G = G + ridge_eps * speye(size(G)); end

Omega_act  = inv(W_act) - H_act * (G \ H_act');
diagOm     = diag(Omega_act);
diagOm(diagOm < 1e-12) = 1e-12; 
r_norm_active = abs(final_resid_act) ./ sqrt(diagOm);
x_rect = struct('e',Vr,'f',Vi,'Vc',Vc);
end

function [If, It] = get_branch_currents(mpc, Vc, Yf, Yt)
    If = Yf * Vc; It = Yt * Vc;
end