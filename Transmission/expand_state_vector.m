function [theta_full, V_full] = expand_state_vector(x_scan_k, nb, ref_bus)
%EXPAND_STATE_VECTOR Re-inserts the reference angle = 0 for a "reduced" state
%
% x_scan_k is (2*nb -1) long = [ angles_(nb-1); V(nb) ].
% The first (nb-1) entries are the angles for all buses except the ref_bus.
% The last nb entries are the voltage magnitudes for all buses.
%
% Outputs:
%  theta_full(nb,1), V_full(nb,1)

    a_sub = x_scan_k(1 : nb-1);             % angles for non-ref buses
    v_sub = x_scan_k(nb : (2*nb -1));       % voltages for all nb buses

    theta_full = zeros(nb,1);
    idx_a = 1;
    for b = 1:nb
        if b == ref_bus
            % reference bus angle = 0
            theta_full(b) = 0;
        else
            theta_full(b) = a_sub(idx_a);
            idx_a = idx_a + 1;
        end
    end

    V_full = v_sub;  % all nb entries for V
end
