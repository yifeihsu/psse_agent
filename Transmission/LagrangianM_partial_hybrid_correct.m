function [success, r_norm, x_rect, Omega, final_resid_raw, correction_info] = ...
    LagrangianM_partial_hybrid_correct(z_full, mpc, pmu_config, Rvar_full, init_state, idx, options)
%LAGRANGIANM_PARTIAL_HYBRID Hybrid State Estimator with Iterative Group Correction
% (REVISED to correctly report original measurement values)
%
%  Inputs:
%    (As before, with the addition of...)
%    options : Struct for SE options:
%              .enable_group_correction (true/false)
%              .correction_group_full_indices (global indices of suspect group)
%              .max_correction_iterations (max correction cycles)
%              .correction_error_tolerance (norm of error vector to trigger correction)
%
%  Outputs:
%    (As before, with the addition of...)
%    correction_info : Struct with details on the correction process.

%% 1. CONSTANTS & BASIC DATA
define_constants;
tol       = 1e-5;      % Stopping tolerance on |dx| for inner WLS loop
max_iter  = 25;        % Max iterations for each individual WLS run
ridge_eps = 1e-6;      % Tiny ridge if Gain matrix is ill-conditioned
nbus      = size(mpc.bus, 1);
[Ybus,Yf,Yt] = makeYbus(mpc.baseMVA, mpc.bus, mpc.branch);
ref_idx   = find(mpc.bus(:, BUS_TYPE) == REF, 1);
if isempty(ref_idx), ref_idx = 1; end

% Initialize correction info structure
correction_info.applied_any_correction = false;
correction_info.iterations_performed = 0;
correction_info.skipped_reason = 'Correction not enabled or no group specified.';

%% 2. ACTIVE MEASUREMENTS & WEIGHTS
act_mask = ~isnan(z_full);
z_act    = z_full(act_mask); % This vector will be modified if corrections are applied
z_act_initial = z_act;       % --- REVISION: Preserve the original active measurements ---
R_act    = Rvar_full(act_mask);
W_act    = spdiags(1./R_act, 0, length(R_act), length(R_act));
global_indices_of_active_meas = find(act_mask);

%% 3. ITERATIVE CORRECTION SETUP
max_total_wls_runs = 1; % Default: 1 WLS run (no correction)
if nargin < 7, options.enable_group_correction = false; end % Default if options not passed

if options.enable_group_correction
    max_total_wls_runs = 1 + options.max_correction_iterations;
    error_tol_for_correction_norm = options.correction_error_tolerance;
end

current_wls_run_number = 0;
perform_next_wls_run = true;
overall_wls_success = false;

% Initialize loop variables
G = []; H_act = []; x = [];

%% 4. ITERATIVE WLS AND CORRECTION LOOP
while perform_next_wls_run && current_wls_run_number < max_total_wls_runs
    current_wls_run_number = current_wls_run_number + 1;
    fprintf('--- Starting WLS Run: %d of max %d ---\n', current_wls_run_number, max_total_wls_runs);

    % --- State Initialization for this WLS run ---
    if isempty(init_state)
        Vr = ones(nbus, 1); Vi = zeros(nbus, 1);
    else
        Vr = init_state.e(:); Vi = init_state.f(:);
    end
    x = [Vr; Vi];
    state_map = [1:nbus, nbus+1:nbus+ref_idx-1, nbus+ref_idx+1:2*nbus]';

    k = 0;
    wls_converged_this_run = false;
    
    % --- Inner WLS Iteration Loop ---
    while k < max_iter
        k = k + 1;
        hx = get_hx_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);
        H  = get_H_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);

        H_act = H(act_mask, state_map);
        mis = z_act - hx(act_mask);

        G = H_act' * W_act * H_act;
        if rcond(full(G)) < ridge_eps, G = G + ridge_eps * speye(size(G)); end
        
        dx = G \ (H_act' * W_act * mis);
        x(state_map) = x(state_map) + dx;

        if max(abs(dx)) < tol
            wls_converged_this_run = true;
            break;
        end
    end % End of Inner WLS Loop

    % --- Post WLS Run Analysis & Correction Decision ---
    if ~wls_converged_this_run
        fprintf('SE failed to converge on WLS Run %d.\n', current_wls_run_number);
        overall_wls_success = false;
        perform_next_wls_run = false; % Stop all further attempts
        correction_info.skipped_reason = 'WLS failed during the correction process.';
    else
        fprintf('Hybrid WLS Run %d converged in %d iterations.\n', current_wls_run_number, k);
        overall_wls_success = true; % Mark that at least one run converged
        
        can_attempt_correction = options.enable_group_correction && ...
                                 ~isempty(options.correction_group_full_indices) && ...
                                 current_wls_run_number < max_total_wls_runs;

        if can_attempt_correction
            fprintf('--- Performing Group Correction Check ---\n');
            % Recalculate H and h(x) at the converged point for accuracy
            hx = get_hx_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);
            H  = get_H_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);
            H_act = H(act_mask, state_map);
            G = H_act' * W_act * H_act;
            
            Omega_active = inv(W_act) - H_act * (G \ H_act');
            S_active = Omega_active * W_act;
            
            [is_member_of_group, ~] = ismember(global_indices_of_active_meas, options.correction_group_full_indices);
            group_local_indices = find(is_member_of_group);

            if ~isempty(group_local_indices)
                r_s_raw_group = (z_act - hx(act_mask));
                r_s_raw_group = r_s_raw_group(group_local_indices);
                S_ss = S_active(group_local_indices, group_local_indices);

                if rcond(full(S_ss)) > ridge_eps
                    estimated_errors_this_step = S_ss \ r_s_raw_group;
                    
                    if norm(estimated_errors_this_step) > error_tol_for_correction_norm
                        fprintf('Estimated error norm (%.3e) > tol (%.3e). Applying correction.\n', norm(estimated_errors_this_step), error_tol_for_correction_norm);
                        
                        % Apply correction to the working measurement vector
                        z_act(group_local_indices) = z_act(group_local_indices) - estimated_errors_this_step;
                        
                        % --- REVISION: Store results for reporting ---
                        correction_info.applied_any_correction = true;
                        correction_info.iterations_performed = correction_info.iterations_performed + 1;
                        correction_info.last_applied_error_norm = norm(estimated_errors_this_step);
                        correction_info.last_corrected_global_indices = global_indices_of_active_meas(group_local_indices);
                        
                        % Store the actual initial bad values
                        correction_info.last_original_values = z_act_initial(group_local_indices);
                        
                        % Store the most up-to-date corrected values
                        correction_info.last_corrected_values = z_act(group_local_indices);
                        
                        % Store the total cumulative error found so far
                        correction_info.last_estimated_errors = z_act_initial(group_local_indices) - z_act(group_local_indices);
                        
                        correction_info.skipped_reason = ''; % Clear any previous skip reason
                        perform_next_wls_run = true; % Signal to do another WLS run
                    else
                        fprintf('Estimated error norm <= tolerance. Correction converged.\n');
                        perform_next_wls_run = false; % Stop iterating
                        correction_info.skipped_reason = sprintf('Errors below tolerance after %d correction(s).', correction_info.iterations_performed);
                    end
                else
                    fprintf('S_ss matrix ill-conditioned. Correction skipped.\n');
                    perform_next_wls_run = false;
                    correction_info.skipped_reason = 'S_ss matrix ill-conditioned.';
                end
            else
                fprintf('Correction group empty in active set. Stopping.\n');
                perform_next_wls_run = false;
                correction_info.skipped_reason = 'Correction group is empty.';
            end
        else
            perform_next_wls_run = false; % Stop if correction not enabled or max runs reached
        end
    end
end % End of Outer Correction Loop

%% 5. FORMULATE FINAL OUTPUTS
success = overall_wls_success;
if ~success
    r_norm = []; x_rect = []; Omega = []; final_resid_raw = [];
    return;
end

% Use results from the LAST successful converged run
Vr = x(1:nbus); Vi = x(nbus+1:2*nbus); Vc = Vr + 1j*Vi;
x_rect = struct('e', Vr, 'f', Vi, 'Vc', Vc);

hx = get_hx_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);
H  = get_H_from_state(x, mpc, pmu_config, Ybus, Yf, Yt, idx);
H_act = H(act_mask, state_map);
G = H_act' * W_act * H_act;
if rcond(full(G)) < ridge_eps, G = G + ridge_eps*speye(size(G)); end

final_resid_raw = z_act - hx(act_mask); % Use the potentially corrected z_act
Omega = inv(W_act) - H_act * (G \ H_act');
diagOm = diag(Omega);
diagOm(diagOm < 1e-12) = 1e-12; % Prevent division by zero or negative variances
r_norm = abs(final_resid_raw) ./ sqrt(diagOm);

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