cd(fullfile(matlabroot,"extern","engines","python")); system("python -m pip install .")function WLAV(z, result)
%% Standard State Estimation Program
% Weighted least absolute value estimator(LP with solver)
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
R = spdiags([0.004^2 * ones(nb, 1); 0.01^2 * ones(2*nb, 1); 0.008^2 * ones(4*nl, 1)], 0, 3*nb+4*nl, 3*nb+4*nl);
W = inv(R);

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

    r = z - hx;

    xu = sdpvar(2*nb-1, 1);
    xv = sdpvar(2*nb-1, 1);
    u = sdpvar(nm, 1);
    v = sdpvar(nm, 1);
    obj = Wi * (u + v);
    constraints = [];
    constraints = [constraints, H * xu - H * xv + u - v == r];
    constraints = [constraints, xu>=0, xv>=0, u>=0, v>=0];
    options = sdpsettings('solver', 'gurobi', 'verbose', 0, 'showprogress', 0);
    sol = optimize(constraints, obj, options);
    if sol.problem ~= 0
        display('Something went wrong!');
        sol.info
        yalmiperror(sol.problem)
    end
    dx = value(xu - xv);

    if max( abs(dx) ) < eps
        flag = 0;
        resi = value(u - v);
    else
        x(nstate) = x(nstate) + dx;
        k = k + 1;
    end
end
end