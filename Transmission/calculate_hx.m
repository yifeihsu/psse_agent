function hx_k = calculate_hx(mpc_case, theta_full, V_full)
%CALCULATE_HX Returns the predicted measurements for
% z = [V1..Vnb, Pinj1..Pinj_nb, Qinj1..Qinj_nb, Pf1..Pf_nl, Qf1..Qf_nl, Pt1..Pt_nl, Qt1..Qt_nl]

    %% 1) Build Y-bus, Yf, Yt
    [Ybus, Yf, Yt] = makeYbus(mpc_case);
    nb = size(mpc_case.bus, 1);
    nl = size(mpc_case.branch,1);

    %% 2) Form Vc = V * e^{j theta}
    Vc = V_full .* exp(1j*theta_full);

    %% 3) Bus injections
    Sbus = Vc .* conj(Ybus * Vc);  % complex bus injections
    Pinj = real(Sbus);
    Qinj = imag(Sbus);

    %% 4) Line flows
    fbus = mpc_case.branch(:,1);
    tbus = mpc_case.branch(:,2);

    Sf = diag(Vc(fbus)) * conj(Yf*Vc); % complex power "from bus" flows
    St = diag(Vc(tbus)) * conj(Yt*Vc); % complex power "to bus" flows

    Pf = real(Sf);
    Qf = imag(Sf);
    Pt = real(St);
    Qt = imag(St);

    %% 5) Assemble the measurement predictions
    hx_k = [
        V_full;      % nb x 1
        Pinj;        % nb x 1
        Qinj;        % nb x 1
        Pf; Qf;      % nl x 1 each
        Pt; Qt       % nl x 1 each
    ];
end
