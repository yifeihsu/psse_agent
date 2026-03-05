%% ================== MAIN SCRIPT: HYBRID SE WITH BAD DATA ANALYSIS ====================
clc;
clear;
close all;

%% --- 0) Define MATPOWER constants & Simulation Parameters ---
define_constants; 
bus_of_interest = 10;
delta_theta_bad_deg = 1.0; % Bias in degrees to match the paper's text

%% --- 1) Load Base Case & Define PMU Scenario ---
mpc_orig = loadcase('case118');
disp('Original system loaded (IEEE 118 Bus).');
mpc = mpc_orig;
nbus = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);
baseMVA = mpc.baseMVA;

% Define the set of buses that have PMUs based on hops from bus_of_interest
one_hop_neighbors = find_neighbors_recursive(bus_of_interest, mpc.branch, 1);
two_hop_neighbors = find_neighbors_recursive(bus_of_interest, mpc.branch, 2);
pmu_buses_set = unique([bus_of_interest; one_hop_neighbors; two_hop_neighbors]);
disp(['PMU observable area (Bus 10 and its 1 & 2-hop neighbors): ', mat2str(pmu_buses_set')]);

pmu_bus_row_indices = arrayfun(@(b) find(mpc.bus(:, BUS_I) == b, 1), pmu_buses_set);
pmu_branch_measurements = []; 
for i = 1:nbranch
    if ismember(mpc.branch(i, F_BUS), pmu_buses_set), pmu_branch_measurements = [pmu_branch_measurements; i, mpc.branch(i, F_BUS)]; end
    if ismember(mpc.branch(i, T_BUS), pmu_buses_set), pmu_branch_measurements = [pmu_branch_measurements; i, mpc.branch(i, T_BUS)]; end
end
pmu_branch_measurements = unique(pmu_branch_measurements, 'rows');
num_pmu_current_phasors = size(pmu_branch_measurements, 1);
pmu_config.pmu_buses = pmu_buses_set;
pmu_config.pmu_branch_currents = pmu_branch_measurements;

%% --- Get Ybus, Yf, Yt ---
[Ybus, Yf, Yt] = makeYbus(baseMVA, mpc.bus, mpc.branch);

%% --- 2) Run Base Case OPF ---
disp('Running base case OPF...');
mpopt_base = mpoption('verbose', 0, 'out.all', 0, 'model', 'AC');
[results_opf, success_base] = runopf(mpc, mpopt_base);
if ~success_base, error('Base case OPF failed!'); end
disp('Base case OPF successful.');
results_opf.bus(:, 9) = results_opf.bus(:, 9) - results_opf.bus(69, 9);
bus_sol = results_opf.bus; 
branch_sol = results_opf.branch;

%% --- 3) Generate True Hybrid Measurement Vector (z_true_hybrid) ---
scada_buses_for_inj_mask = true(nbus, 1); scada_lines_indices = (1:nbranch)';       
idx.vm_all=(1:nbus)'; idx.vr_pmu=(idx.vm_all(end)+1:idx.vm_all(end)+length(pmu_buses_set))';
idx.vi_pmu=(idx.vr_pmu(end)+1:idx.vr_pmu(end)+length(pmu_buses_set))';
idx.ir_pmu=(idx.vi_pmu(end)+1:idx.vi_pmu(end)+num_pmu_current_phasors)';
idx.ii_pmu=(idx.ir_pmu(end)+1:idx.ir_pmu(end)+num_pmu_current_phasors)';
idx.pinj_scada=(idx.ii_pmu(end)+1:idx.ii_pmu(end)+nbus)';
idx.qinj_scada=(idx.pinj_scada(end)+1:idx.pinj_scada(end)+nbus)';
idx.flows_scada=(idx.qinj_scada(end)+1:idx.qinj_scada(end)+4*nbranch)';
nz_actual_hybrid = idx.flows_scada(end);
z_true_hybrid = zeros(nz_actual_hybrid, 1);
Vm_sol_complex = bus_sol(:,VM) .* exp(1j * deg2rad(bus_sol(:,VA)));
[If_true_complex, It_true_complex] = get_branch_currents(mpc, Vm_sol_complex, Yf, Yt);
z_true_hybrid(idx.vm_all) = bus_sol(:, VM);
z_true_hybrid(idx.vr_pmu) = real(Vm_sol_complex(pmu_bus_row_indices));
z_true_hybrid(idx.vi_pmu) = imag(Vm_sol_complex(pmu_bus_row_indices));
for k = 1:size(pmu_branch_measurements,1)
    br_idx=pmu_branch_measurements(k,1); meas_bus=pmu_branch_measurements(k,2);
    if meas_bus == mpc.branch(br_idx,F_BUS), current_val_complex=If_true_complex(br_idx); else, current_val_complex=It_true_complex(br_idx); end
    z_true_hybrid(idx.ir_pmu(k))=real(current_val_complex); z_true_hybrid(idx.ii_pmu(k))=imag(current_val_complex);
end
Pgen_bus=accumarray(results_opf.gen(:,GEN_BUS),results_opf.gen(:,PG),[nbus 1]); Qgen_bus=accumarray(results_opf.gen(:,GEN_BUS),results_opf.gen(:,QG),[nbus 1]);
z_true_hybrid(idx.pinj_scada)=(Pgen_bus - bus_sol(:,PD))/baseMVA; z_true_hybrid(idx.qinj_scada)=(Qgen_bus - bus_sol(:,QD))/baseMVA;
z_true_hybrid(idx.flows_scada)=[branch_sol(:,PF); branch_sol(:,QF); branch_sol(:,PT); branch_sol(:,QT)]/baseMVA;

%% --- 4) Measurement Variances ---
std_dev_pmu_v_rect = 0.0001; std_dev_pmu_i_rect = 0.0001;
std_dev_scada_vm = 0.001; std_dev_scada_pq_inj = 0.01; std_dev_scada_pq_flow = 0.01;
R_variances_full_hybrid = zeros(nz_actual_hybrid, 1);
R_variances_full_hybrid(idx.vm_all)=std_dev_scada_vm^2; R_variances_full_hybrid(idx.pinj_scada)=std_dev_scada_pq_inj^2;
R_variances_full_hybrid(idx.qinj_scada)=std_dev_scada_pq_inj^2; R_variances_full_hybrid(idx.flows_scada)=std_dev_scada_pq_flow^2;
R_variances_full_hybrid(pmu_bus_row_indices)=std_dev_pmu_v_rect^2;
R_variances_full_hybrid(idx.vr_pmu)=std_dev_pmu_v_rect^2; R_variances_full_hybrid(idx.vi_pmu)=std_dev_pmu_v_rect^2;
R_variances_full_hybrid(idx.ir_pmu)=std_dev_pmu_i_rect^2; R_variances_full_hybrid(idx.ii_pmu)=std_dev_pmu_i_rect^2;
sigma_hybrid = sqrt(R_variances_full_hybrid);

%% --- 5) Bad Data Injection & SE Run ---
disp('=============================================================================');
V_opf_complex=bus_sol(:,VM).*exp(1j*deg2rad(bus_sol(:,VA)));
initial_state.e=real(V_opf_complex); initial_state.f=imag(V_opf_complex);
z_hybrid_with_error=z_true_hybrid+sigma_hybrid.*randn(size(sigma_hybrid));
delta_theta_bad_rad=deg2rad(delta_theta_bad_deg);
fprintf('Applying PMU phase bias of %.2f deg at Bus %d.\n', delta_theta_bad_deg, bus_of_interest);
rotation_phasor=exp(1j*delta_theta_bad_rad);
corrupted_indices=[];
pmu_list_idx_faulty=find(pmu_buses_set==bus_of_interest);
idx_vr_faulty=idx.vr_pmu(pmu_list_idx_faulty); idx_vi_faulty=idx.vi_pmu(pmu_list_idx_faulty);
V_meas_complex=z_hybrid_with_error(idx_vr_faulty)+1j*z_hybrid_with_error(idx_vi_faulty);
V_meas_biased=V_meas_complex*rotation_phasor;
z_hybrid_with_error(idx_vr_faulty)=real(V_meas_biased); z_hybrid_with_error(idx_vi_faulty)=imag(V_meas_biased);
corrupted_indices=[corrupted_indices;idx_vr_faulty;idx_vi_faulty];
for k_cur=1:num_pmu_current_phasors
    if pmu_branch_measurements(k_cur,2)==bus_of_interest
        idx_ir_faulty=idx.ir_pmu(k_cur); idx_ii_faulty=idx.ii_pmu(k_cur);
        I_meas_complex=z_hybrid_with_error(idx_ir_faulty)+1j*z_hybrid_with_error(idx_ii_faulty);
        I_meas_biased=I_meas_complex*rotation_phasor;
        z_hybrid_with_error(idx_ir_faulty)=real(I_meas_biased); z_hybrid_with_error(idx_ii_faulty)=imag(I_meas_biased);
        corrupted_indices=[corrupted_indices;idx_ir_faulty;idx_ii_faulty];
    end
end
fprintf('  Global indices of measurements directly biased by PMU error: %s\n', mat2str(unique(corrupted_indices)'));

% Run State Estimator with bad data
[success_se, r_norm_active, est_state, Omega_active, final_resid_act] = ...
    LagrangianM_partial_hybrid(z_hybrid_with_error, mpc, pmu_config, R_variances_full_hybrid, initial_state, idx);
if ~success_se, error('State Estimator did not converge.'); end

%% --- 6) Analyze Results: Single vs. Grouped Indices ---

% --- A. Calculate Single Normalized Residuals ---
r_fullsize_normalized = nan(nz_actual_hybrid, 1);
active_mask = ~isnan(z_hybrid_with_error);
r_fullsize_normalized(active_mask) = r_norm_active;
[sorted_r_norm, sorted_r_norm_idx] = sort(r_fullsize_normalized, 'descend');

% --- B. Calculate Grouped Bus Indices using Mahalanobis Distance ---
fprintf('\nCalculating Grouped Indices (Mahalanobis Distance) for all buses...\n');
grouped_bus_indices_MD = NaN(nbus, 1);
kept_indices = find(active_mask);

for b_idx = 1:nbus
    bus_num = mpc.bus(b_idx, BUS_I);
    
    % Get all measurement indices associated with this bus
    current_bus_group_global_indices = get_bus_group_global_indices_hybrid(bus_num, mpc, pmu_config, idx);
    
    % Find which of these are active measurements and their local indices
    [active_in_group_mask, local_indices_in_kept] = ismember(current_bus_group_global_indices, kept_indices);
    group_local_indices = local_indices_in_kept(active_in_group_mask);
    
    % Need at least 2 measurements in a group to calculate MD
    if isempty(group_local_indices) || length(group_local_indices) < 2
        continue;
    end
    
    % Extract the sub-vector of residuals and sub-matrix of Omega
    r_sub = final_resid_act(group_local_indices);
    Omega_sub = Omega_active(group_local_indices, group_local_indices);

    % Check for conditioning before inverting
    if rcond(full(Omega_sub)) < 1e-12
        continue;
    end
    
    % Calculate Mahalanobis Distance
    grouped_bus_indices_MD(b_idx) = r_sub' * (Omega_sub \ r_sub) - length(r_sub);
end
[sorted_grouped_val, sorted_grouped_bus_idx] = sort(grouped_bus_indices_MD, 'descend', 'MissingPlacement','last');


%% --- 7) Generate Console and Figure Output ---

% --- A. Generate Console Table 1: Single vs. Grouped Index Comparison ---
disp(' ');
disp('==================== TABLE 1: SINGLE VS. GROUPED INDEX COMPARISON ====================');
fprintf('%-5s %-25s %-10s | %-5s %-10s %-15s\n', 'Rank', 'Top-5 Single NRs', '|r_N|', 'Rank', 'Bus #', 'Grouped Index');
disp(repmat('-', 1, 80));
for i = 1:5
    rank1=i; val1=sorted_r_norm(i);
    desc1_full=get_measurement_type_by_global_idx(sorted_r_norm_idx(i), mpc, pmu_config, idx);
    desc1=strrep(strrep(strrep(desc1_full,'PMU ',''),' on Branch',''),', at Bus','@');
    rank2=i; val2=sorted_grouped_val(i); bus_num2=mpc.bus(sorted_grouped_bus_idx(i),BUS_I);
    fprintf('%-5d %-25s %-10.2f | %-5d %-10d %-15.2f\n', rank1, desc1, val1, rank2, bus_num2, val2);
end
disp(repmat('-', 1, 80));

% --- B. Simulate Correction and Generate Console Table 2 ---
disp(' ');
disp('==================== TABLE 2: GROUPED MEASUREMENT CORRECTION RESULTS ===================');
bus_to_correct_row = sorted_grouped_bus_idx(1); bus_to_correct_num = mpc.bus(bus_to_correct_row, BUS_I);
z_hybrid_corrected = z_hybrid_with_error;
indices_to_remove = get_bus_group_global_indices_hybrid(bus_to_correct_num, mpc, pmu_config, idx);
z_hybrid_corrected(indices_to_remove) = NaN;
[success_se_corr, r_norm_corr, ~, ~, ~] = ...
    LagrangianM_partial_hybrid(z_hybrid_corrected, mpc, pmu_config, R_variances_full_hybrid, initial_state, idx);
fprintf('%-40s %-20s %-20s %-20s\n', 'Measurement (from PMU @ Bus 10)', 'Initial (p.u.)', 'True (p.u.)', 'Estimated (p.u.)');
disp(repmat('-', 1, 105));
hx_final = get_hx_from_state(est_state, mpc, pmu_config, Ybus, Yf, Yt, idx);
for i = 1:length(corrupted_indices)
    g_idx=corrupted_indices(i); desc=get_measurement_type_by_global_idx(g_idx, mpc, pmu_config, idx);
    desc=strrep(desc,'PMU ',''); initial_val=z_hybrid_with_error(g_idx); true_val=z_true_hybrid(g_idx); est_val=hx_final(g_idx);
    fprintf('%-40s %-20.4f %-20.4f %-20.4f\n', desc, initial_val, true_val, est_val);
end
disp(repmat('-', 1, 105));
fprintf('%-82s %.2f\n', 'Largest system NR (before correction):', sorted_r_norm(1));
if success_se_corr, fprintf('%-82s %.2f\n', 'Largest system NR (after correction):', max(r_norm_corr));
else, fprintf('%-82s %s\n', 'Largest system NR (after correction):', 'N/A (SE failed)'); end
disp('======================================================================================');

function [success, r_norm_active, x_rect, Omega_active, final_resid_act] = ...
    LagrangianM_partial_hybrid(z_full, mpc, pmu_config, Rvar_full, init_state, idx)
%% 0. CONSTANTS & BASIC DATA
define_constants;
tol           = 1e-5;      % Stopping tolerance on |dx|
max_iter      = 25;        % WLS iterations
ridge_eps     = 1e-6;      % Tiny ridge if Gain matrix is ill-conditioned
nbus          = size(mpc.bus,   1);
[Ybus,Yf,Yt]  = makeYbus(mpc.baseMVA, mpc.bus, mpc.branch);
ref_idx       = find(mpc.bus(:,BUS_TYPE)==REF, 1);
if isempty(ref_idx), ref_idx = 1; end

% =================== DELETED REDUNDANT idx CREATION =====================
% The idx structure is now passed in as an argument.
% ========================================================================

%% 3. ACTIVE MEASUREMENTS & WEIGHTS
act_mask        = ~isnan(z_full);
z_act           = z_full(act_mask);
R_act           = Rvar_full(act_mask);
W_act           = spdiags(1./R_act,0,length(R_act),length(R_act));

%% 4. INITIAL STATE & STATE MAP
if nargin<6 || isempty(init_state) % <-- Check for nargin < 6 now
    Vr = ones(nbus,1);  Vi = zeros(nbus,1);
else
    Vr = init_state.e(:); Vi = init_state.f(:);
end
x = [Vr; Vi];
state_map = [ 1:nbus, nbus+1:nbus+ref_idx-1, nbus+ref_idx+1:2*nbus ]';

%% 5. WLS ITERATION LOOP
k = 0; success = 0;
while k < max_iter
    k = k+1;
    % =================== CORRECTED HELPER FUNCTION CALLS ====================
    hx = get_hx_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);
    H = get_H_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);
    % ========================================================================
    
    H_act = H(act_mask, state_map);
    mis = z_act - hx(act_mask);
    
    G = H_act' * W_act * H_act;
    if rcond(full(G)) < 1e-12, G = G + ridge_eps * speye(size(G)); end
    dx = G \ (H_act' * W_act * mis);
    
    x(state_map) = x(state_map) + dx;
    if max(abs(dx)) < tol, success = 1; break; end
end

%% 6. FORMULATE OUTPUTS
if ~success, fprintf('SE failed to converge.\n'); r_norm_active=nan; x_rect=[]; Omega_active=[]; final_resid_act=[]; return; end
fprintf('Hybrid WLS converged in %d iterations.\n',k);
Vr = x(1:nbus); Vi = x(nbus+1:2*nbus); Vc = Vr + 1j*Vi;
final_resid_act = z_act - hx(act_mask);

% --- MODIFIED PART: Calculate and return the full Omega matrix ---
% Need to re-calculate H at converged point for accurate Omega
H = get_H_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);
H_act = H(act_mask, state_map);
G = H_act' * W_act * H_act;
if rcond(full(G)) < 1e-12, G = G + ridge_eps * speye(size(G)); end

Omega_active = inv(W_act) - H_act * (G \ H_act');
diagOm = diag(Omega_active); diagOm(diagOm < 1e-12) = 1e-12;
r_norm_active = abs(final_resid_act) ./ sqrt(diagOm);
x_rect = struct('e',Vr, 'f',Vi, 'Vc',Vc);
end

function bus_group_global_indices = get_bus_group_global_indices_hybrid(bID, mpc, pmu_config, idx)
    define_constants;
    bus_group_global_indices = [];
    bus_row = find(mpc.bus(:, BUS_I) == bID, 1);
    if isempty(bus_row), return; end

    % Add SCADA measurements for this bus
    bus_group_global_indices = [bus_group_global_indices; idx.vm_all(bus_row); idx.pinj_scada(bus_row); idx.qinj_scada(bus_row)];
    
    % Add SCADA flow measurements
    lines_from = find(mpc.branch(:, F_BUS) == bID);
    lines_to = find(mpc.branch(:, T_BUS) == bID);
    bus_group_global_indices = [bus_group_global_indices; idx.flows_scada(lines_from); idx.flows_scada(size(mpc.branch,1)+lines_from)]; % Pf, Qf
    bus_group_global_indices = [bus_group_global_indices; idx.flows_scada(2*size(mpc.branch,1)+lines_to); idx.flows_scada(3*size(mpc.branch,1)+lines_to)]; % Pt, Qt
    
    % Add PMU measurements if the bus has a PMU
    if ismember(bID, pmu_config.pmu_buses)
        pmu_list_idx = find(pmu_config.pmu_buses == bID);
        bus_group_global_indices = [bus_group_global_indices; idx.vr_pmu(pmu_list_idx); idx.vi_pmu(pmu_list_idx)];
    end
    
    % Add PMU current measurements originating from this bus
    for k = 1:size(pmu_config.pmu_branch_currents, 1)
        if pmu_config.pmu_branch_currents(k, 2) == bID
            bus_group_global_indices = [bus_group_global_indices; idx.ir_pmu(k); idx.ii_pmu(k)];
        end
    end
    bus_group_global_indices = unique(bus_group_global_indices);
end

% --- Other helper functions (get_hx_from_state, get_H_from_state, find_neighbors_recursive, etc.) remain unchanged ---
% --- from the previous provided solution. They are included here for completeness. ---
function hx = get_hx_from_state(x, mpc, pmu_cfg, Ybus, Yf, Yt, idx)
    define_constants; nbus=size(mpc.bus,1);
    Vr=x(1:nbus); Vi=x(nbus+1:2*nbus); Vc=Vr+1j*Vi;
    hx=zeros(idx.flows_scada(end),1);
    pmu_bus=pmu_cfg.pmu_buses(:); pmu_Iinfo=pmu_cfg.pmu_branch_currents;
    pmu_rows=arrayfun(@(b)find(mpc.bus(:,BUS_I)==b,1),pmu_bus);
    [If,It]=get_branch_currents(mpc,Vc,Yf,Yt);
    hx(idx.vm_all)=abs(Vc); hx(idx.vr_pmu)=Vr(pmu_rows); hx(idx.vi_pmu)=Vi(pmu_rows);
    for k=1:size(pmu_Iinfo,1)
        br=pmu_Iinfo(k,1);meas_b=pmu_Iinfo(k,2);
        if meas_b==mpc.branch(br,F_BUS),I=If(br);else,I=It(br);end
        hx(idx.ir_pmu(k))=real(I);hx(idx.ii_pmu(k))=imag(I);
    end
    Sbus=Vc.*conj(Ybus*Vc); Sf=Vc(mpc.branch(:,F_BUS)).*conj(Yf*Vc); St=Vc(mpc.branch(:,T_BUS)).*conj(Yt*Vc);
    hx(idx.pinj_scada)=real(Sbus);hx(idx.qinj_scada)=imag(Sbus);
    hx(idx.flows_scada)=[real(Sf);imag(Sf);real(St);imag(St)];
end
function H = get_H_from_state(x, mpc, pmu_cfg, Ybus, Yf, Yt, idx)
    define_constants; nbus=size(mpc.bus,1);
    Vr=x(1:nbus); Vi=x(nbus+1:2*nbus); Vc=Vr+1j*Vi;
    H=sparse(idx.flows_scada(end),2*nbus);
    pmu_bus=pmu_cfg.pmu_buses(:); pmu_Iinfo=pmu_cfg.pmu_branch_currents; n_pmu_v=numel(pmu_bus);
    pmu_rows=arrayfun(@(b)find(mpc.bus(:,BUS_I)==b,1),pmu_bus);
    Vm=abs(Vc);Vm(Vm<1e-12)=1e-12;
    H(idx.vm_all,:)=[spdiags(Vr./Vm,0,nbus,nbus),spdiags(Vi./Vm,0,nbus,nbus)];
    pmu_v_selector=sparse(1:n_pmu_v,pmu_rows,1,n_pmu_v,nbus);
    H(idx.vr_pmu,:)=[pmu_v_selector,sparse(n_pmu_v,nbus)]; H(idx.vi_pmu,:)=[sparse(n_pmu_v,nbus),pmu_v_selector];
    for ii=1:size(pmu_Iinfo,1)
        br=pmu_Iinfo(ii,1);meas_b=pmu_Iinfo(ii,2);
        if meas_b==mpc.branch(br,F_BUS),Yrow=Yf(br,:);else,Yrow=Yt(br,:);end
        H(idx.ir_pmu(ii),:)=[real(Yrow),-imag(Yrow)];H(idx.ii_pmu(ii),:)=[imag(Yrow),real(Yrow)];
    end
    [dSbus_dVr,dSbus_dVi]=dSbus_dV(Ybus,Vc,'cart');[dSf_dVr,dSf_dVi,dSt_dVr,dSt_dVi]=dSbr_dV(mpc.branch,Yf,Yt,Vc,'cart');
    H(idx.pinj_scada,:)=[real(dSbus_dVr),real(dSbus_dVi)];H(idx.qinj_scada,:)=[imag(dSbus_dVr),imag(dSbus_dVi)];
    H(idx.flows_scada,:)=[real(dSf_dVr),real(dSf_dVi);imag(dSf_dVr),imag(dSf_dVi);real(dSt_dVr),real(dSt_dVi);imag(dSt_dVr),imag(dSt_dVi)];
end
function neighbors_set=find_neighbors_recursive(start_bus,branches,max_hops)
    define_constants;all_neighbors={start_bus};current_hop_buses=[start_bus];visited_buses=[start_bus];
    for hop=1:max_hops
        next_hop_buses=[];
        for b=current_hop_buses'
            direct_connections=[branches(branches(:,F_BUS)==b,T_BUS);branches(branches(:,T_BUS)==b,F_BUS)];
            unique_new_neighbors=setdiff(unique(direct_connections),visited_buses);
            next_hop_buses=[next_hop_buses;unique_new_neighbors];visited_buses=[visited_buses;unique_new_neighbors];
        end
        if isempty(next_hop_buses),break;end
        all_neighbors{hop+1}=unique(next_hop_buses);current_hop_buses=unique(next_hop_buses);
    end
    neighbors_set=unique(vertcat(all_neighbors{:}));
end
function desc=get_measurement_type_by_global_idx(g_idx,mpc,pmu_config,idx)
    define_constants;bus_data=mpc.bus;branch_data=mpc.branch;pmu_buses=pmu_config.pmu_buses;pmu_I=pmu_config.pmu_branch_currents;
    if ismember(g_idx,idx.vm_all),bus_idx=find(idx.vm_all==g_idx);desc=sprintf('Vm@Bus %d',bus_data(bus_idx,BUS_I));
    elseif ismember(g_idx,idx.vr_pmu),list_idx=find(idx.vr_pmu==g_idx);desc=sprintf('Vr@Bus %d',pmu_buses(list_idx));
    elseif ismember(g_idx,idx.vi_pmu),list_idx=find(idx.vi_pmu==g_idx);desc=sprintf('Vi@Bus %d',pmu_buses(list_idx));
    elseif ismember(g_idx,idx.ir_pmu),list_idx=find(idx.ir_pmu==g_idx);br=pmu_I(list_idx,1);mb=pmu_I(list_idx,2);desc=sprintf('Ir Line %d-%d@%d',branch_data(br,F_BUS),branch_data(br,T_BUS),mb);
    elseif ismember(g_idx,idx.ii_pmu),list_idx=find(idx.ii_pmu==g_idx);br=pmu_I(list_idx,1);mb=pmu_I(list_idx,2);desc=sprintf('Ii Line %d-%d@%d',branch_data(br,F_BUS),branch_data(br,T_BUS),mb);
    elseif ismember(g_idx,idx.pinj_scada),bus_idx=find(idx.pinj_scada==g_idx);desc=sprintf('Pinj@Bus %d',bus_data(bus_idx,BUS_I));
    elseif ismember(g_idx,idx.qinj_scada),bus_idx=find(idx.qinj_scada==g_idx);desc=sprintf('Qinj@Bus %d',bus_data(bus_idx,BUS_I));
    else, desc='SCADA Flow'; end
end
function [If,It]=get_branch_currents(mpc,Vc,Yf,Yt)
    If=Yf*Vc;It=Yt*Vc;
end