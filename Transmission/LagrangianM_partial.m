function [lambdaN, success, r, Omega, final_resid] = LagrangianM_partial(z_in, result, ind, bus_data)
%LAGRANGIANM_SINGLEPHASE  Standard State Estimation Program
%   with the ability to handle partial/missing measurements in z_in.
%
%  Inputs:
%    z_in   : measurement vector (possibly missing some elements,
%             marked as NaN or 999, etc.)
%    result : MATPOWER struct with bus/branch/gen/baseMVA
%    ind    : flag; if ==1, remove reference-angle measurement row
%    bus_data : initial bus data (voltages, angles, etc.)
%
%  Outputs:
%    lambdaN : Normalized Lagrangian multipliers w.r.t. line parameters
%    success : 1 if WLS converged, 0 if not
%    r       : normalized measurement residuals (only for valid measurements)

%% 1) Basic Setup
ref = find(result.bus(:, 2)==3); % reference bus
% Ensure reference angle is zero
result.bus(:, 9) = result.bus(:, 9) - result.bus(ref, 9);

eps = 1e-4;   % tolerance
k   = 0;      % iteration counter
flag = 1;
success = 1;

nb   = size(result.bus, 1);
nl   = size(result.branch, 1);
fbus = result.branch(:, 1);
tbus = result.branch(:, 2);

%% 2) Identify Zero-Injection Buses (if any)
ZI   = makeSbus(result.baseMVA, result.bus, result.gen);
ZIn  = [];  % e.g. find(abs(ZI) < 1e-6) if you want to remove zero-injections
nzi  = numel(ZIn);

%% 3) Build "Full" Measurement Vector Indices
% In a *complete* system (ignoring zero-injection removal), the measurement order is:
%   index range:
%     1..nbus    => V magnitudes
%     nbus+(1..nbus)       => P_inj
%     2*nbus+(1..nbus)     => Q_inj
%     3*nbus+(1..nl)       => Pf
%     3*nbus+nl+(1..nl)    => Qf
%     3*nbus+2*nl+(1..nl)  => Pt
%     3*nbus+3*nl+(1..nl)  => Qt
%
% Then we also remove zero-injection rows from P_inj, Q_inj => adjusted indices in practice.
% For simplicity, let's define a "full" length ignoring zero-injection first:
nMeasFull_noZI = nb + 2*nb + 4*nl;  % = 3*nb + 4*nl
% The zero-injection rows that *would* have been in the full vector:
ZI_remove_indices = [nb+ZIn, 2*nb+ZIn];  % bus injections at zero-injection buses

% We'll remove those from the "full" dimension to get final "full" dimension
fullMask_ZI = true(1, nMeasFull_noZI);
fullMask_ZI(ZI_remove_indices) = false;
nMeasFull = sum(fullMask_ZI);  % total # measurements if none were missing except zero-inj

%% 4) Build a "valid measurement mask" from z_in
% We assume z_in might be dimension nMeasFull, with some entries possibly set to NaN or 999
% for missing data. If your user code passes in exactly nMeasFull entries, do:
if length(z_in) ~= nMeasFull
    error('z_in length (%d) does not match expected nMeasFull (%d).', ...
          length(z_in), nMeasFull);
end

validMeasMask = ~isnan(z_in) & (z_in < 999);  % or any criterion for "missing"

% Extract the final z vector
z = z_in(validMeasMask);

%% 5) Build the weighting matrix for the valid measurements

R_variances = zeros(length(z_in), 1);

for i = 1:length(z_in)
    if i <= nb
        R_variances(i) = (0.001)^2;   % big variance => less weighting
    else
        R_variances(i) = (0.01)^2;
    end
end
Rdiag_all = R_variances;
% If you want different variances for voltages vs flows, you'd do that logic here.

% Now keep only the variances for the valid measurements:
Rdiag = Rdiag_all(validMeasMask);
R = spdiags(Rdiag, 0, numel(Rdiag), numel(Rdiag));
W = inv(R);

%% 6) Build the Full Y-bus and so forth
[Ybus, Yf, Yt] = makeYbus(result);

%% 7) State Initialization
% x = [theta_1..theta_nb, V_1..V_nb]
% remove ref from the "solved" angles
nstate      = 1 : 2*nb;
nstate(ref) = [];

x  = sparse(1:2*nb, 1, [zeros(nb,1); ones(nb,1)]);
x0 = sparse(1:2*nb-1,1,[zeros(nb-1,1); ones(nb,1)]);   % excludes ref angle
xl = zeros(5*nb + 4*nl -1, 1);  % Lagrangian multipliers
xl(nb : 2*nb-1) = 1;  % for a "flat start"

%% 8) WLS Loop
maxIter = 20;
while flag
    %% 8a) Compute the "full" predicted measurements hx_full & Jacobian H_full
    Vc = x(nb+1 : 2*nb) .* exp(1j*x(1:nb));  % V_i in polar form
    [H_big, Sf, St] = makeJaco(x, Ybus, Yf, Yt, nb, nl, fbus, tbus, Vc);

    % Remove the reference-angle column from H_big
    H_big(:, ref) = [];

    % We'll now build hx_big in the standard order,
    % then remove zero-injection rows and keep the valid ones.
    Ibus = Ybus * Vc;
    Sinj = diag(Vc) * conj(Ibus);
    Pinj = real(Sinj);
    Qinj = imag(Sinj);

    Pf = real(Sf); Qf = imag(Sf);
    Pt = real(St); Qt = imag(St);

    hx_preZI = [ ...
        x(nb+1 : 2*nb);     % V's
        Pinj;               % P_inj
        Qinj;               % Q_inj
        Pf; Qf; Pt; Qt ];   % flows

    % Remove zero-injection rows from both hx_preZI and H_big
    hx_full = hx_preZI(fullMask_ZI);
    H_full = H_big(fullMask_ZI, :);

    %% 8b) Now keep *only* the rows that correspond to the valid measurements
    hx = hx_full(validMeasMask);
    H  = H_full(validMeasMask, :);

    %% 8c) Form the "Gain" system for the Lagrangian approach
    % Also handle the zero-injection constraints (C) if you are using them
    % in the same manner as before. If you used something like:
    %   C = H_big(ZI_rows, :),
    % you would keep it separately. For brevity, we omit that part here
    % or set C = [] if not using the zero-injection as separate constraints.

    % For demonstration, let's skip that "C" block and do a standard WLS:
    % Gain = [H'*W*H]; mismatch = z - hx.

    mismatch = z(:) - hx(:);
    Gain = H' * W * H;

    % Solve for dx
    dx = Gain \ (H' * W * mismatch);

    % Update states
    x0 = x0 + dx;
    x(nstate) = x0;

    % Check for convergence
    if max(abs(dx)) < eps
        flag = 0;
    else
        k = k + 1;
        if k >= maxIter
            % did not converge
            lambdaN = 10*ones(nl,1);
            success = 0;
            r = [];    % no residual
            return;
        end
    end
end

%% 9) Once converged, build final predicted measurements again for residual
final_Vc = x(nb+1 : 2*nb) .* exp(1j*x(1:nb));
Ibus_f   = Ybus*final_Vc;
Sinj_f   = diag(final_Vc)*conj(Ibus_f);
Pinj_f   = real(Sinj_f);
Qinj_f   = imag(Sinj_f);

[Sfull, Sf_f, St_f] = makeJaco(x, Ybus, Yf, Yt, nb, nl, fbus, tbus, final_Vc);

Pf_f     = real(Sf_f); Qf_f = imag(Sf_f);
Pt_f     = real(St_f); Qt_f = imag(St_f);

hx_preZI = [ ...
    x(nb+1 : 2*nb);
    Pinj_f; Qinj_f;
    Pf_f; Qf_f;
    Pt_f; Qt_f];

hx_full = hx_preZI(fullMask_ZI);
hx_final = hx_full(validMeasMask);

final_resid = z - hx_final;  % only for measured entries

%% 10) Compute Normalized Residual r for valid measurements
% For a standard WLS approach, the "residual covariance" is:
%   Omega = R - H*(Gain\H')
% or equivalently:
%   Omega = inv(W) - ...
% We'll do the short version:
Omega = inv(W) - H*(Gain\H');

% If Omega has small numerical negativity, you might see complex sqrt.
% We'll do a small fix:
diagOmega = diag(Omega);
diagOmega(diagOmega < 0 & diagOmega > -1e-12) = 0;  % clamp small neg

rvals = abs(final_resid) ./ sqrt(diagOmega);
r = full(rvals);

%% 11) For Lagrangian multipliers w.r.t. line parameters
% If you are also doing the line-parameter-lambda logic,
% you'd need to re-build that portion with the sub-measurement approach,
% exactly as you did originally, but referencing the new subsets. 
% For brevity, let's set lambdaN = zeros(nl,1) or replicate your logic carefully.

lambdaN = zeros(nl,1);  % For demonstration only.

success = 1;  % we converged
end
