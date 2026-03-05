%% ================== MAIN SCRIPT: HYBRID SE WITH AUTOMATED GROUP CORRECTION ====================
clc;
clear;
close all;

%% --- 0) Define MATPOWER constants & Simulation Parameters ---
define_constants; 
bus_of_interest = 10;
delta_theta_bad_deg = 1; % A significant bias to demonstrate correction

%% --- 1) Load Base Case & Define PMU Scenario ---
mpc_orig = loadcase('case118');
disp('Original system loaded (IEEE 118 Bus).');
mpc = mpc_orig;
nbus = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);
baseMVA = mpc.baseMVA;

% Define PMU observable area
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
results_opf.bus(:, 9) = results_opf.bus(:, 9) - results_opf.bus(69, 9); % Re-reference angles
bus_sol = results_opf.bus; 
branch_sol = results_opf.branch;

%% --- 3) Generate True Hybrid Measurement Vector ---
% (This section remains unchanged)
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
std_dev_pmu_v_rect = 1e-4; std_dev_pmu_i_rect = 1e-4;
std_dev_scada_vm = 1e-3; std_dev_scada_pq_inj = 1e-2; std_dev_scada_pq_flow = 1e-2;
R_variances_full_hybrid = zeros(nz_actual_hybrid, 1);
R_variances_full_hybrid(idx.vm_all)=std_dev_scada_vm^2; R_variances_full_hybrid(idx.pinj_scada)=std_dev_scada_pq_inj^2;
R_variances_full_hybrid(idx.qinj_scada)=std_dev_scada_pq_inj^2; R_variances_full_hybrid(idx.flows_scada)=std_dev_scada_pq_flow^2;
R_variances_full_hybrid(idx.vr_pmu)=std_dev_pmu_v_rect^2; R_variances_full_hybrid(idx.vi_pmu)=std_dev_pmu_v_rect^2;
R_variances_full_hybrid(idx.ir_pmu)=std_dev_pmu_i_rect^2; R_variances_full_hybrid(idx.ii_pmu)=std_dev_pmu_i_rect^2;

%% --- 5) Bad Data Injection & State Estimation with Correction ---
disp('=============================================================================');
V_opf_complex=bus_sol(:,VM).*exp(1j*deg2rad(bus_sol(:,VA)));
initial_state.e=real(V_opf_complex); initial_state.f=imag(V_opf_complex);
z_hybrid_with_error=z_true_hybrid + sqrt(R_variances_full_hybrid).*randn(size(z_true_hybrid)); % Add Gaussian noise

% --- Inject Gross Error ---
delta_theta_bad_rad=deg2rad(delta_theta_bad_deg);
fprintf('Applying PMU phase bias of %.2f deg at Bus %d.\n', delta_theta_bad_deg, bus_of_interest);
rotation_phasor=exp(1j*delta_theta_bad_rad);
pmu_list_idx_faulty=find(pmu_buses_set==bus_of_interest);
idx_vr_faulty=idx.vr_pmu(pmu_list_idx_faulty); idx_vi_faulty=idx.vi_pmu(pmu_list_idx_faulty);
V_meas_complex=z_hybrid_with_error(idx_vr_faulty)+1j*z_hybrid_with_error(idx_vi_faulty);
V_meas_biased=V_meas_complex*rotation_phasor;
z_hybrid_with_error(idx_vr_faulty)=real(V_meas_biased); z_hybrid_with_error(idx_vi_faulty)=imag(V_meas_biased);

for k_cur=1:num_pmu_current_phasors
    if pmu_branch_measurements(k_cur,2)==bus_of_interest
        idx_ir_faulty=idx.ir_pmu(k_cur); idx_ii_faulty=idx.ii_pmu(k_cur);
        I_meas_complex=z_hybrid_with_error(idx_ir_faulty)+1j*z_hybrid_with_error(idx_ii_faulty);
        I_meas_biased=I_meas_complex*rotation_phasor;
        z_hybrid_with_error(idx_ir_faulty)=real(I_meas_biased); z_hybrid_with_error(idx_ii_faulty)=imag(I_meas_biased);
    end
end

% --- Define Options for Iterative Group Correction ---
options_se.enable_group_correction = true;
options_se.correction_group_full_indices = get_bus_group_global_indices_hybrid(bus_of_interest, mpc, pmu_config, idx);
options_se.max_correction_iterations = 5;
options_se.correction_error_tolerance = 1e-4;

% --- Run State Estimator with Correction Enabled ---
[success_se, r_norm_final, ~, ~, ~, correction_info] = ...
    LagrangianM_partial_hybrid_correct(z_hybrid_with_error, mpc, pmu_config, R_variances_full_hybrid, initial_state, idx, options_se);

if ~success_se, error('State Estimator process failed.'); end

%% --- 6) Analyze and Display Correction Results ---
disp(' ');
disp('==================== GROUPED CORRECTION OUTPUT ====================');
if correction_info.applied_any_correction
    fprintf('✅ Iterative group bad data correction was successful (%d iteration(s)).\n\n', correction_info.iterations_performed);
    disp('--- Final Correction Details ---');
    
    % --- Create and Display Results Table ---
    col_headers = {'Measurement', 'Original Bad Value', 'Estimated Error', 'Corrected Value', 'True Value'};
    results_data = cell(length(correction_info.last_corrected_global_indices), length(col_headers));
    
    for i = 1:length(correction_info.last_corrected_global_indices)
        g_idx = correction_info.last_corrected_global_indices(i);
        desc = get_measurement_type_by_global_idx(g_idx, mpc, pmu_config, idx);
        
        results_data{i,1} = strrep(desc, 'PMU ', '');
        results_data{i,2} = correction_info.last_original_values(i);
        results_data{i,3} = correction_info.last_estimated_errors(i);
        results_data{i,4} = correction_info.last_corrected_values(i);
        results_data{i,5} = z_true_hybrid(g_idx);
    end
    
    results_table = cell2table(results_data, 'VariableNames', col_headers);
    disp(results_table);
    
    fprintf('\nNorm of errors that triggered the last correction: %.3e\n', correction_info.last_applied_error_norm);

elseif ~isempty(correction_info.skipped_reason)
    fprintf('ℹ️ Group bad data correction was not applied. Reason: %s\n', correction_info.skipped_reason);
end

fprintf('Largest normalized residual after final SE run: %.2f\n', max(r_norm_final));
disp('===================================================================');

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
