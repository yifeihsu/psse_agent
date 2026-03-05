function [r, success, hx] = SE(z, result, ind)
%% Measurement data and default settings
% z is row vector
ref = find(result.bus(:, 2)==3);
result.bus(:, 9) = result.bus(:, 9) - result.bus(ref, 9); % Ensure that the ref phase is 0.

eps = 0.001; % tol
k = 0;
flag = 1;
success = 1;

% Basic Settings
nb = size(result.bus, 1);
nl = size(result.branch, 1);
fbus = result.branch(:, 1);
tbus = result.branch(:, 2);
%% Find the Zero Injection Nodes
R = spdiags([0.004^2 * ones(nb, 1); 0.01^2 * ones(2*nb, 1); 0.008^2 * ones(4*nl, 1)], 0, 3*nb+4*nl, 3*nb+4*nl);
W = inv(R);
if ind == 1
    W(nb, :) = [];
    W(:, nb) = [];
end

[Ybus, Yf, Yt] = makeYbus(result);

nstate = 1: 2*nb;
nstate(ref) = [];

x = sparse(1:2*nb, 1, [zeros(nb, 1); ones(nb, 1)]);
x0 = sparse(1:2*nb-1, 1, [zeros(nb-1, 1); ones(nb, 1)]);

z = z';
z = sparse(z);

%% Standard WLS Estimator with Zero Injections
while flag
    V = x(nb+1: 2*nb) .* exp(1j * x(1: nb));
    [H, Sf, St] = makeJaco(x, Ybus, Yf, Yt, nb, nl, fbus, tbus, V);
    H(:, ref) = [];

    Ibus = Ybus * V;
    Sinj = diag(V) * conj(Ibus);
    Pinj = real(Sinj);
    Qinj = imag(Sinj);
    Pf = real(Sf); Pt = real(St);
    Qf = imag(Sf); Qt = imag(St);

    hx = [x(nb+1: 2*nb); Pinj; Qinj; Pf; Qf; Pt; Qt];
    if ind == 1
        hx(nb) = [];
        H(nb, :) = [];
    end

    Gain = H' * W * H;
    tk = H' * W * (z - hx);

    o = symamd(Gain);
    L = chol(Gain(o, o), 'lower');
    dxl = L'\(L\tk(o));
    dxl(o) = dxl; 
    dx = dxl(1: 2*nb-1);

    if max( abs(dx) ) < eps
        flag = 0;
    else
        x0 = x0 + dx;
        x(nstate) = x0;     % State Variable Output
        k = k + 1;          % Iteration Index
        if k >= 10          % If tol exceeds, break and set r = 0, success = 0.
            r = 0;
            success = 0;
            return
        end
    end
end

r = z - hx;
R = inv(W);
omega = R - H * inv(Gain) * H'; % Residual Covariance Matrix
r = abs(r) ./ sqrt(diag(omega)); % Normalize the residual
r = full(r);

% %% State Estimation Result Check
% result.bus(:, 7) = x(nb+1: 2*nb);
% result.bus(:, 10) = x(1:nb)*180/pi;
% %% Post examination of Bad Data
% % g, b, yc form the whole matrix first
% ntheta = 1: nb;
% ntheta(ref) = [];
% theta(ntheta) = x0(1:nb-1); % In radian
% V = x0(nb: 2*nb-1);
% % Form the parameter Jacobian using the state estimates
% Hp = zeros(3*nb+4*nl, nl);
% for k = 1: nl
%     i = fbus(k);
%     j = tbus(k);
%     % Pinj
%     Hp(i ,k) = - V(i) * V(j) * sin(theta(i)-theta(j));
%     Hp(j, k) = - V(i) * V(j) * sin(theta(j)-theta(i));
%     % Qinj
%     Hp(i+nb ,k) = - V(i)^2 + V(i) * V(j) * cos(theta(i)-theta(j));
%     Hp(j+nb ,k) = - V(j)^2 + V(i) * V(j) * cos(theta(j)-theta(i));
%     % Pf, Pt
%     Hp(2*nb+k, k) = - V(i) * V(j) * sin(theta(i)-theta(j));
%     Hp(2*nb+nl+k, k) = - V(i) * V(j) * sin(theta(j)-theta(i));
%     % Qf, Qt
%     Hp(2*nb+2*nl+k, k) = - V(i)^2 + V(i) * V(j) * cos(theta(i)-theta(j));
%     Hp(2*nb+3*nl+k, k) = - V(j)^2 + V(i) * V(j) * cos(theta(j)-theta(i));
% end
% cov1 = H * inv(Gain) * H';
% cov2 = full(R + cov1);
% dcov2 = diag(cov2);
% dcovp = dcov2(29: 28+nl);
% cov1 = full(cov1);
% a = diag(cov1).^(-1/2);
% b = diag(a);
% corr = b * cov1 * b;
%% Residual Analysis
end