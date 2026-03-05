%% 针对IEEE 14-bus 生成nsample个支路参数错误量测，并运行Lagrangian Multiplier Test测试精度
clc;
clear;

mpc = loadcase(case14);
% mpc.branch(15, :) = [];
% mpc.branch(8, :) = [];
% mpc.gen(5, :) = [];
% mpc.bus(7, 3:4) = mpc.bus(12, 3:4);
% mpc.bus(8, 3:4) = mpc.bus(11, 3:4);
% mpc.gencost(5, :) = [];


nbus = size(mpc.bus, 1);
nbranch = size(mpc.branch, 1);
nz = nbus + nbus + 4 * nbranch + nbus;

ratio = xlsread('loadwotime.xlsx', 'Sheet1');
ratio = ratio / max(ratio);

[r, c] = size(ratio);

measurements = [];
nsample = 200;
mpopt = mpoption('verbose', 0, 'out.all', 0);

for i = 1:nbranch
    k = 0;
    while(k < nsample)
        rl = ratio(randi(r)); % 负荷水平

        mpc1 = mpc;
        mpc1.branch(i, 4) = (1.2 + randi(91)/10) * mpc1.branch(i, 4);
        mpc1.bus(:, 3) = rl * mpc1.bus(:, 3);
        mpc1.bus(:, 4) = rl * mpc1.bus(:, 4);

        [baseMVA, bus, gen, gencost, branch, f, success, et] = runopf(mpc1, mpopt);
        if success ~= 1
            disp("Something wrong with the OPF!");
            continue
        end
        msr = zeros(nz+1, 1);
        msr(1:nbus) = bus(:, 8);
        Sbus = makeSbus(baseMVA, bus, gen);
        Pinj = real(Sbus);
        Qinj = imag(Sbus);
        msr(nbus+1: 3*nbus) = [Pinj; Qinj];
        branch(:, 14:17) = branch(:, 14:17)/100;
        msr(3*nbus+1:3*nbus+4*nbranch) = [branch(:, 14); branch(:, 15); branch(:, 16); branch(:, 17)];
        msr(nz+1) = i;
        measurements = horzcat(measurements, msr);
        k = k+1;
    end
    disp('Error branch = ');
    disp(i);
end
measurements = measurements';

R = [ones(nbus, 1)*0.000016; 0.0001*ones(2*nbus, 1); 0.000064*ones(4*nbranch, 1)];
R = R';
[r,c] = size(measurements);
c = c-1;
sigma = sqrt(R);
tt = [];
for col = 1:c
    noise = randn(r,1)*sigma(col);
    tmp = measurements(:, col) + noise;
    tt = horzcat(tt, tmp);
end
tt(:, size(tt, 2)+1) = measurements(:, size(measurements, 2));
mea = tt;
disp("Finish, please save the 'mea' matrix.");
mpc2 = mpc;
[errorb, accu] = LagAccuTest(mpc2, mea, nsample);