function [lambdaN, flag_fail] = LagrangianMtopo(z, result)
%% Measurement data and default settings
ref = find(result.bus(:, 2)==3);
result.bus(:, 9) = result.bus(:, 9) - result.bus(ref, 9); % Ensure that the ref phase is 0.

eps = 0.001; % tol
k = 0;
flag = 1;
flag_fail = 0;

% Basic Settings
nsec = 5;
ncb = 6;
nb = size(result.bus, 1);
nl = size(result.branch, 1);
fbus = result.branch(:, 1);
tbus = result.branch(:, 2);
%% Find the Zero Injection Nodes
nzi = 0;
R0 = spdiags([0.004^2 * ones(nb, 1); 0.01^2 * ones(2*nb-2*nzi, 1); 0.008^2 * ones(4*nl, 1); 0.01^2 * ones(4, 1)], 0, 3*nb+4*nl-2*nzi, 3*nb+4*nl-2*nzi);
% W = inv(R);

[Ybus, Yf, Yt] = makeYbus(result); % Form the system model

nstate = 1: 2*(nb+nsec+ncb);
nstate(ref) = [];

x = [zeros(nb, 1); ones(nb, 1)]; % V, theta for bus/branch model
xa = [zeros(nb+nsec, 1); ones(nb+nsec, 1); zeros(2*ncb, 1)]; % theta, V, Pf, Qf
x0 = [zeros(nb+nsec-1, 1); ones(nb+nsec, 1); zeros(2*ncb, 1)]; % theta-ref, V, Pf, Qf


% Analog measurements input
z = z';
z = sparse(z);

%% Standard WLS Estimator with Zero Injections
while flag
    V = x(nb+1: 2*nb) .* exp(1j * x(1: nb)); % bus-branch模型中电压相量
    Va = xa(nb+nsec+1: 2*(nb+nsec)) .* exp(1j * xa(1: nb+nsec)); % 所有节点的电压相量
    topog;
    H = Hfin;
    H(:, ref) = [];

    a = diag(R0);
    a([nb+subn, 2*nb+subn]) = [];
    Rm = diag(a);
    R = zeros(size(H, 1), size(H, 1));
    R(1:size(Rm, 1), 1:size(Rm, 1)) = Rm;

    Gain = [R, H; H', zeros(size(H, 2), size(H, 2))];
    r = [rall; zeros(size(H, 2), 1)];

    dxl = inv(Gain) * r;
    dx = dxl(size(H, 1)+1: size(Gain, 1));

    if max( abs(dx) ) < eps % dxl: dx, r, mu, here we only consider the convergence of dx
        flag = 0;
    else
        x0 = x0 + dx;
        xa(nstate) = x0;     % State Variable Output
        x(1:nb) = xa(1:nb);
        x(nb+1:2*nb) = xa(nb+nsec+1:2*nb+nsec);
        k = k + 1;          % Iteration Index
    end
    if k >= 20
        flag_fail = 1;
        break
    end
end
if flag_fail == 1
    lambdaN = 0;
else
    ea = inv(Gain);
    ea = ea(1:size(R, 1), 1:size(R, 1));
    tt = sqrt(diag(ea));
    lambda = dxl(1:size(H, 1));
    lambdaN = zeros(size(lambda, 1), 1);
    for i = 1 : size(lambda)
        lambdaN(i) = lambda(i) / tt(i);
    end
end
end