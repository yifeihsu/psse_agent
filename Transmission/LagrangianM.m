function [lambdaN, success, r] = LagrangianM(z, result, ind)
%% Standard State Estimation Program
% Zero injections are considered
% Residual Analysis and Lagrangian Multiplier method
% This program involves RTU measurements only
% Here the reference angle should be 0 in radian.
% z: measurement vector, result: standard database case OPF
%% Measurement data and default settings
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
ZI = makeSbus(result.baseMVA, result.bus, result.gen); % 标准各节点注入
ZIn = find(ZI == 0); % 找到零注入节点的编号
nzi = size(ZIn, 1);

R = spdiags([0.004^2 * ones(nb, 1); 0.01^2 * ones(2*nb-2*nzi, 1); 0.008^2 * ones(4*nl, 1)], 0, 3*nb+4*nl-2*nzi, 3*nb+4*nl-2*nzi);
W = inv(R);
if ind == 1
    W(nb, :) = [];
    W(:, nb) = [];
end

[Ybus, Yf, Yt] = makeYbus(result); % Form the system model

nstate = 1: 2*nb;
nstate(ref) = [];

x = sparse(1:2*nb, 1, [zeros(nb, 1); ones(nb, 1)]);
x0 = sparse(1:2*nb-1, 1, [zeros(nb-1, 1); ones(nb, 1)]);
% Lagrangian State Variables
if ind == 1
    xl = zeros(5*nb + 4*nl -2, 1);
else
    xl = zeros(5*nb + 4*nl -1, 1);
end
xl(nb: 2*nb-1) = 1; % Flat start

z = z';
z = sparse(z);
z([nb+ZIn; 2*nb+ZIn]) = [];

%% Standard WLS Estimator with Zero Injections
while flag
    V = x(nb+1: 2*nb) .* exp(1j * x(1: nb));
    [H, Sf, St] = makeJaco(x, Ybus, Yf, Yt, nb, nl, fbus, tbus, V);
    H(:, ref) = [];
    C = H([nb+ZIn; 2*nb+ZIn] , :);
    H([nb+ZIn; 2*nb+ZIn] , :) = []; % Remove the Pinj and Qinj with Zero injection

    Ibus = Ybus * V;
    Sinj = diag(V) * conj(Ibus);
    Pinj = real(Sinj);
    Qinj = imag(Sinj);
    Pf = real(Sf); Pt = real(St);
    Qf = imag(Sf); Qt = imag(St);

    hx = [x(nb+1: 2*nb); Pinj; Qinj; Pf; Qf; Pt; Qt];

    cx = hx([nb+ZIn; 2*nb+ZIn]);
    hx([nb+ZIn; 2*nb+ZIn]) = [];

    if ind == 1
        hx(nb) = [];
        H(nb, :) = [];
    end

    % Form the coefficient matrix
    Gain = [zeros(2*nb-1), H'*W, C';
        H, eye(size(H, 1)), zeros(size(H, 1), size(C, 1));
        C, zeros(size(C, 1), size(H, 1)), zeros(size(C, 1))];
    % Form the right hand side
    tk = z - hx;
    tk = [zeros(2*nb-1, 1); tk; -cx];

    o = symamd(Gain);
    [L,U,P] = lu(Gain(o, o));
    dy = L\(P*tk(o));
    dxl = U\dy;
    dxl(o) = dxl;
    dx = dxl(1: 2*nb-1);

    if max( abs(dx) ) < eps % dxl: dx, r, mu, here we only consider the convergence of dx
        flag = 0;
    else
        x0 = x0 + dx;
        xl = xl + dxl;      % Augmentation
        x(nstate) = x0;     % State Variable Output
        k = k + 1;          % Iteration Index
        if k >= 20
            lambdaN = 10 * ones(nl, 1);
            success = 0;
            return
        end
    end
end
%% State Estimation Result Check
% result.bus(:, 7) = x(nb+1: 2*nb);
% result.bus(:, 10) = x(1:nb)*180/pi;
%% Post examination of Bad Data
% g, b, yc form the whole matrix first
ntheta = 1: nb;
ntheta(ref) = [];
theta(ntheta) = x0(1:nb-1); % In radian
V = x0(nb: 2*nb-1);
% Form the parameter Jacobian using the state estimates
Hp = zeros(3*nb + 4*nl, nl);
for k = 1: nl
    i = fbus(k);
    j = tbus(k);
    % Pinj
    Hp(nb+i, k) = - V(i) * V(j) * sin(theta(i)-theta(j));
    Hp(nb+j, k) = - V(i) * V(j) * sin(theta(j)-theta(i));
    % Qinj
    Hp(i+2*nb, k) = - V(i)^2 + V(i) * V(j) * cos(theta(i)-theta(j));
    Hp(j+2*nb, k) = - V(j)^2 + V(i) * V(j) * cos(theta(j)-theta(i));
    % Pf, Pt
    Hp(3*nb+k, k) = - V(i) * V(j) * sin(theta(i)-theta(j));
    Hp(3*nb+2*nl+k, k) = - V(i) * V(j) * sin(theta(j)-theta(i));
    % Qf, Qt
    Hp(3*nb+nl+k, k) = - V(i)^2 + V(i) * V(j) * cos(theta(i)-theta(j));
    Hp(3*nb+3*nl+k, k) = - V(j)^2 + V(i) * V(j) * cos(theta(j)-theta(i));
end
Cp = Hp([nb+ZIn; 2*nb+ZIn] , :);
Hp([nb+ZIn; 2*nb+ZIn] , :) = [];
if ind == 1
    Hp(nb, :) = [];
end

S = -[W*Hp; Cp]';

temp = inv(Gain);
E5 = temp(2*nb: 2*nb-1+size(H, 1), 2*nb: 2*nb-1+size(H, 1));
E8 = temp(2*nb+size(H, 1):size(Gain, 1), 2*nb: 2*nb-1+size(H, 1));
phi = [E5 ;  E8];
covu = phi * inv(W) * phi';
ea = S * covu * S';

lambda = S * dxl(2*nb: size(dxl, 1)); % lambda = S * [r; mu]
tt = sqrt(diag(ea));
lambdaN = zeros(size(lambda, 1), 1);
for i = 1 : size(lambda)
    lambdaN(i) = lambda(i) / tt(i);
end
%% Calculate the normalized residuals
r = z - hx;
R = inv(W);
omega = R - H * inv(H' * W * H) * H'; % Residual Covariance Matrix
r = abs(r) ./ sqrt(diag(omega)); % Normalize the residual
r = full(r);
end