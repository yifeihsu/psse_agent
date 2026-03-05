function [lambdaN, success, r, lambda_vec, ea] = LagrangianM_singlephase(z, result, ind, bus_data)
%% Standard State Estimation Program
%  Zero injections are considered
%  Residual Analysis and Lagrangian Multiplier method
%  Single-phase model
%  Here the reference angle should be 0 in radian.
%  Inputs:
%       z      : measurement vector
%       result : MATPOWER struct with bus/branch/gen/baseMVA
%       ind    : flag. if ==1, we remove a reference-angle "measurement"
%
%  Outputs:
%       lambdaN : Normalized Lagrangian multipliers w.r.t. line parameters
%       success : 1 if WLS converged, 0 if not
%       r       : normalized measurement residuals

%% 1) Basic Setup
ref = find(result.bus(:, 2)==3);
% reference bus % ensure reference angle is zero result.bus( :, 9) =
    result.bus( :, 9) - result.bus(ref, 9);

eps = 1e-3;
% tolerance k = 0;
% iteration counter flag = 1;
success = 1;

nb = size(result.bus, 1);
nl = size(result.branch, 1);
fbus = result.branch( :, 1);
tbus = result.branch( :, 2);

%% 2) Identify Zero‐Injection Buses
ZI   = makeSbus(result.baseMVA, result.bus, result.gen);
% ZIn = find(abs(ZI) < 1e-6);
% or exactly == 0 if you prefer ZIn = [];
nzi = numel(ZIn);

%% 3) Form Weighting Matrix W
% measurement order: [Vm (nb), Pinj (~ nb), Qinj (~ nb), Pf/Qf/Pt/Qt (4nl)]
% but zero‐inj removes 2 per zero‐inj bus from P/Q set
Rdiag = [ ...
     0.001^2 * ones(nb, 1);                     % volt mag 
     0.01^2  * ones(2*nb - 2*nzi, 1);           % P/Q inj minus zero inj 
     0.01^2 * ones(4*nl, 1) ];
% flows R = spdiags(Rdiag, 0, numel(Rdiag), numel(Rdiag));
W = inv(R);
% Convert z to row(just to match code that uses z( :).'): z =
                       z( :).';

                       % % 4) Build Y‐Bus[Ybus, Yf, Yt] = makeYbus(result);

%% 5) State Initialization
% x = [theta_1..theta_nb, V_1..V_nb]
% remove ref from the "solved" angles
nstate      = 1 : 2*nb;
nstate(ref) = [];

% Start from(theta = 0, V = 1) x = sparse(1 : 2 * nb, 1,
                                          [zeros(nb, 1); ones(nb, 1)]);
x0 = sparse(1 : 2 * nb - 1, 1, [zeros(nb - 1, 1); ones(nb, 1)]);
% excludes ref angle % Augmented x xl = zeros(5 * nb + 4 * nl - 1, 1);
xl(nb : 2 * nb - 1) = 1;

%% 6) Remove Zero‐Injection Buses from z
% zero‐inj rows: P_inj(ZIn), Q_inj(ZIn)
z([nb+ZIn; 2*nb+ZIn]) = [];
x(1 : nb) = bus_data( :, 9) * (pi / 180);
x(nb + 1 : 2 * nb) = bus_data( :, 8);
x0 = x(2 : end);

%% === 7) Standard WLS with Zero Injections ===
while flag
    %% 7a) Build H and measurement prediction hx
    Vc = x(nb+1:2*nb) .* exp(1j*x(1:nb));
% V_i[H_full, Sf, St] = makeJaco(x, Ybus, Yf, Yt, nb, nl, fbus, tbus, Vc);
% remove ref angle column H_full( :, ref) = [];

% remove zero‐injection rows from H_full C =
    H_full([nb + ZIn; 2 * nb + ZIn], :);
H = H_full;
H([nb + ZIn; 2 * nb + ZIn], :) = [];

% compute measurement predictions Ibus = Ybus * Vc;
Sinj = diag(Vc) * conj(Ibus);
Pinj = real(Sinj);
Qinj = imag(Sinj);

Pf = real(Sf);
Qf = imag(Sf);
Pt = real(St);
Qt = imag(St);

hx_full = [x(nb + 1 : 2 * nb); Pinj; Qinj; Pf; Qf; Pt; Qt];

% remove zero‐injection entries cx = hx_full([nb + ZIn; 2 * nb + ZIn]);
hx = hx_full;
hx([nb + ZIn; 2 * nb + ZIn]) = [];
    %% 7b) Form the big Lagrangian "Gain" system
    Gain = [
        sparse(2*nb-1, 2*nb-1),    H'*W,               C';
        H,                         speye(size(H,1)),   sparse(size(H,1),size(C,1));
        C,                         sparse(size(C,1),size(H,1)), sparse(size(C,1),size(C,1))
    ];

    %% 7c) Right‐Hand Side
    mismatch = z(:) - hx(:);
    tk = [zeros(2 * nb - 1, 1); mismatch; -cx];

    %% 7d) Solve
    o        = symamd(Gain);
    [ Lfac, Ufac, Pmat ] = lu(Gain(o, o));
    dy = Lfac \ (Pmat * tk(o));
    dxl = Ufac \ dy;
    dxl(o) = dxl;
    % revert permutation

            dx = dxl(1 : 2 * nb - 1);

    % Check for convergence
    if max(abs(dx)) < eps
        flag = 0;
    else x0 = x0 + dx;
    % update states xl = xl + dxl;
    % update multipliers x(nstate) = x0;
    % fill back k = k + 1;
    if k
      >= 20 % did not converge lambdaN = 10 * ones(nl, 1);
    success = 0;
    r = [];
    % no residual lambda_vec = [];
    ea = [];
    return;
        end
    end
end

%% === 8) Build Parameter Jacobian w.r.t. r & x for each line ===
    ntheta = 1:nb;
        ntheta(ref) = [];
        theta_est = zeros(nb, 1);
        theta_est(ntheta) = x0(1 : nb - 1);

        V_est = x0(nb : 2 * nb - 1);
        % -- -assume V_est and theta_est are defined as before-- - Hp_full =
            zeros(3 * nb + 4 * nl, 2 * nl);

    for
      kk = 1 : nl i = fbus(kk);
    j = tbus(kk);
    r_k = result.branch(kk, 3);
    x_k = result.branch(kk, 4);

    denom = r_k ^ 2 + x_k ^ 2;
    g_ij = r_k / denom;
    b_ij = -x_k / denom;

    dg_dr = (x_k ^ 2 - r_k ^ 2) / (denom ^ 2);
    db_dr = (2 * r_k * x_k) / (denom ^ 2);
    dg_dx = (-2 * r_k * x_k) / (denom ^ 2);
    db_dx = (x_k ^ 2 - r_k ^ 2) / (denom ^ 2);

    Vi = V_est(i);
    Vj = V_est(j);
    dth = theta_est(i) - theta_est(j);
    cosd = cos(dth);
    sind = sin(dth);

        % Partial derivatives of injections w.r.t g_ij, b_ij for bus i
        dPi_dg = Vi^2 - Vi*Vj*cosd;
        % -- -CORRECTED LINE-- - dPi_db = Vi * Vj * sind;
        % Was - Vi *Vj *sind

                    % Q_i dQi_dg = Vi * Vj * sind;
        % -- -CORRECTED LINE-- - dQi_db = -Vi ^ 2 + Vi * Vj * cosd; % Was -Vi^2 - Vi*Vj*cosd

        % Partial derivatives of injections w.r.t g_ij, b_ij for bus j
        dPj_dg = Vj^2 - Vi*Vj*cosd;
        % -- -CORRECTED LINE-- - dPj_db = -Vi * Vj * sind;
        % Was + Vi *Vj *sind

                    % Q_j dQj_dg = -Vi * Vj * sind;
        % -- -CORRECTED LINE-- - dQj_db = -Vj ^ 2 + Vi * Vj * cosd;
        % Was - Vj ^ 2 - Vi *Vj *cosd

                             dPi_dr = dPi_dg * dg_dr + dPi_db * db_dr;
        dPi_dx = dPi_dg * dg_dx + dPi_db * db_dx;
        dPj_dr = dPj_dg * dg_dr + dPj_db * db_dr;
        dPj_dx = dPj_dg * dg_dx + dPj_db * db_dx;
        dQi_dr = dQi_dg * dg_dr + dQi_db * db_dr;
        dQi_dx = dQi_dg * dg_dx + dQi_db * db_dx;
        dQj_dr = dQj_dg * dg_dr + dQj_db * db_dr;
        dQj_dx = dQj_dg * dg_dx + dQj_db * db_dx;

        % place into Hp_full Hp_full(nb + i, 2 * kk - 1) = dPi_dr;
        Hp_full(nb + i, 2 * kk) = dPi_dx;
        Hp_full(nb + j, 2 * kk - 1) = dPj_dr;
        Hp_full(nb + j, 2 * kk) = dPj_dx;
        Hp_full(2 * nb + i, 2 * kk - 1) = dQi_dr;
        Hp_full(2 * nb + i, 2 * kk) = dQi_dx;
        Hp_full(2 * nb + j, 2 * kk - 1) = dQj_dr;
        Hp_full(2 * nb + j, 2 * kk) = dQj_dx;

        % Pf(i->j) = > same as P_i Hp_full(3 * nb + kk, 2 * kk - 1) = dPi_dr;
        Hp_full(3 * nb + kk, 2 * kk) = dPi_dx;
        % Qf(i->j) = > same as Q_i
                     Hp_full(3 * nb + nl + kk, 2 * kk - 1) = dQi_dr;
        Hp_full(3 * nb + nl + kk, 2 * kk) = dQi_dx;
        % Pt(j->i) = > same as P_j
                     Hp_full(3 * nb + 2 * nl + kk, 2 * kk - 1) = dPj_dr;
        Hp_full(3 * nb + 2 * nl + kk, 2 * kk) = dPj_dx;
        % Qt(j->i) = > same as Q_j
                     Hp_full(3 * nb + 3 * nl + kk, 2 * kk - 1) = dQj_dr;
        Hp_full(3 * nb + 3 * nl + kk, 2 * kk) = dQj_dx;
        end % % remove zero - injection rows from Hp_full Cp =
            Hp_full([nb + ZIn; 2 * nb + ZIn], :);
        Hp = Hp_full;
        Hp([nb + ZIn; 2 * nb + ZIn], :) = [];

% If ind==1, also remove row nb from Hp
%% 9) Form S = -[W*Hp; Cp]' for lambda test
S = -[W*Hp; Cp]';

% Invert the big Gain from the last iteration
% (We have it in 'Gain' from above in the final iteration)
temp_inv = inv(Gain);

% The block used in the formula for the param covariance
mH = size(H,1);
E5 = temp_inv(2 * nb : 2 * nb - 1 + mH, 2 * nb : 2 * nb - 1 + mH);
E8 = temp_inv(2 * nb + mH : end, 2 * nb : 2 * nb - 1 + mH);
phi = [E5; E8];
covu = phi * inv(W) *phi'; ea = S * covu * S';

                                % The multipliers in dxl(2 * nb : end)
                                      lambda_vec = S * dxl(2 * nb : end);

% normalized tt = sqrt(diag(ea));
lambdaN = lambda_vec./ tt;

%% 10) Compute & Return Residual r
% We'll compute the final measurement residual (normalized).
% The final mismatch is z - hx from the last iteration:
final_Vc = x(nb+1 : 2*nb) .* exp(1j*x(1:nb));
Ibus_f = Ybus * final_Vc;
Sinj_f = diag(final_Vc) * conj(Ibus_f);
Pinj_f = real(Sinj_f);
Qinj_f = imag(Sinj_f);
[ ~, Sf, St ] = makeJaco(x, Ybus, Yf, Yt, nb, nl, fbus, tbus, Vc);

Pf_f = real(Sf);
Qf_f = imag(Sf);
Pt_f = real(St);
Qt_f = imag(St);

hx_full = [x(nb + 1 : 2 * nb); Pinj_f; Qinj_f; Pf_f; Qf_f; Pt_f; Qt_f];
hx_full([nb + ZIn; 2 * nb + ZIn]) = [];
if ind
  == 1 hx_full(nb) = [];
end final_resid = z( :) - hx_full( :);
% unnormalized

    % Normalized residual : Rfull_inv = W;  % Weighted by W
% Cov(resid) = R - H*(H'*W*H)\H'
% but we have H (the final version used in the iteration)
omega = inv(Rfull_inv) - H*inv(H'*W*H)*H';
rvals = abs(final_resid) ./ sqrt(diag(omega));
r = full(rvals);

end

function [J, SF, ST] = makeJaco(~, Ybus, Yf, Yt, nb, nl, fbus, tbus, V)
%% build Jacobian
[dSbus_dVa, dSbus_dVm] = dSbus_dV(Ybus, V);
[DSF_DVA, DSF_DVM, DST_DVA, DST_DVM, SF, ST] = dSbr_dV1(Yf, Yt, V, nb, nl, fbus, tbus);
j11 = real(dSbus_dVa);
j12 = real(dSbus_dVm);
j21 = imag(dSbus_dVa);
j22 = imag(dSbus_dVm);
j31 = real(DSF_DVA);
j32 = real(DSF_DVM);
j41 = real(DST_DVA);
j42 = real(DST_DVM);
j51 = imag(DSF_DVA);
j52 = imag(DSF_DVM);
j61 = imag(DST_DVA);
j62 = imag(DST_DVM);
j71 = spdiags([], [], nb, nb);
j72 = spdiags(ones(nb, 1), 0, nb, nb);
J = [   j71 j72;
        j11 j12;
        j21 j22;
        j31 j32;
        j51 j52;
        j41 j42;
        j61 j62;
];
end

function [dSf_dV1, dSf_dV2, dSt_dV1, dSt_dV2, Sf, St] = dSbr_dV1(Yf, Yt, V, nb, nl, f, t)
%DSBR_DV   Computes partial derivatives of branch power flows w.r.t. voltage.
%   All in polar coordinates edition
%   If = Yf * V;
%   Sf = diag(Vf) * conj(If) = diag(conj(If)) * Vf
%
%   Polar coordinates:
%     Partials of V, Vf & If w.r.t. voltage angles
%       dV/dVa  = j * diag(V)
%       dVf/dVa = sparse(1:nl, f, j * V(f)) = j * sparse(1:nl, f, V(f))
%       dIf/dVa = Yf * dV/dVa = Yf * j * diag(V)
%
%     Partials of V, Vf & If w.r.t. voltage magnitudes
%       dV/dVm  = diag(V./abs(V))
%       dVf/dVm = sparse(1:nl, f, V(f)./abs(V(f))
%       dIf/dVm = Yf * dV/dVm = Yf * diag(V./abs(V))
%
%     Partials of Sf w.r.t. voltage angles
%       dSf/dVa = diag(Vf) * conj(dIf/dVa)
%                       + diag(conj(If)) * dVf/dVa
%               = diag(Vf) * conj(Yf * j * diag(V))
%                       + conj(diag(If)) * j * sparse(1:nl, f, V(f))
%               = -j * diag(Vf) * conj(Yf * diag(V))
%                       + j * conj(diag(If)) * sparse(1:nl, f, V(f))
%               = j * (conj(diag(If)) * sparse(1:nl, f, V(f))
%                       - diag(Vf) * conj(Yf * diag(V)))
%
%     Partials of Sf w.r.t. voltage magnitudes
%       dSf/dVm = diag(Vf) * conj(dIf/dVm)
%                       + diag(conj(If)) * dVf/dVm
%               = diag(Vf) * conj(Yf * diag(V./abs(V)))
%                       + conj(diag(If)) * sparse(1:nl, f, V(f)./abs(V(f)))

%% compute intermediate values
Yfc = conj(Yf);
Ytc = conj(Yt);
Vc = conj(V);
Ifc = Yfc * Vc;     %% conjugate of "from" current
Itc = Ytc * Vc;     %% conjugate of "to" current

diagVf  = sparse(1:nl, 1:nl, V(f), nl, nl);
diagVt  = sparse(1:nl, 1:nl, V(t), nl, nl);
diagIfc = sparse(1:nl, 1:nl, Ifc, nl, nl);
diagItc = sparse(1:nl, 1:nl, Itc, nl, nl);
Vnorm       = V ./ abs(V);
diagVc      = sparse(1:nb, 1:nb, Vc, nb, nb);
diagVnorm   = sparse(1:nb, 1:nb, Vnorm, nb, nb);
CVf  = sparse(1:nl, f, V(f), nl, nb);
CVnf = sparse(1:nl, f, Vnorm(f), nl, nb);
CVt  = sparse(1:nl, t, V(t), nl, nb);
CVnt = sparse(1:nl, t, Vnorm(t), nl, nb);
dSf_dV1 = 1j * (diagIfc * CVf - diagVf * Yfc * diagVc);     %% dSf_dVa
dSf_dV2 = diagVf * conj(Yf * diagVnorm) + diagIfc * CVnf;   %% dSf_dVm
dSt_dV1 = 1j * (diagItc * CVt - diagVt * Ytc * diagVc);     %% dSt_dVa
dSt_dV2 = diagVt * conj(Yt * diagVnorm) + diagItc * CVnt;   %% dSt_dVm
Sf = V(f) .* Ifc;
St = V(t) .* Itc;
end
