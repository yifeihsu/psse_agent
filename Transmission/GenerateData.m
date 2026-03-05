function [measurements, tt, kall] = GenerateData(~)
%% 此函数生成标准模式下的量测与潮流真值
ratio = xlsread('loadwotime.xlsx', 'Sheet1');
ratio = ratio / max(ratio);
% ratio = ratio + 0.5; % 增大负荷水平，避免轻载情况
mpopt = mpoption('verbose', 0, 'out.all', 0, 'opf.ac.solver', 'IPOPT');
measurements = [];
tol = size(ratio, 1);
row = 1; 

secn = [3, 6]; % 变电站内部 Bus Bar
kall = [];
fail_ind = [];

% 运行模式的时间相关性
start = sort(randperm(tol, 50))';
intv = 1000 + randi(3000, size(start, 1), 1); % 每个错误持续4-7个时段
endi = start + intv;
kall = ones(tol, 1);
for i = 1: length(start)
    kall(start(i): endi(i)) = 1 + randi(3);
end
kall = kall(1:tol);


while row <= tol
    disp(row)
    disp('****************************************');
    ratld = ratio(row);
    k = kall(row);
    mpc1 = modelselection(k);
    mpc1.bus(:, 3) = ratld .* mpc1.bus(:, 3);
    mpc1.bus(:, 4) = ratld .* mpc1.bus(:, 4);

    nbus = size(mpc1.bus, 1);
    nbranch = size(mpc1.branch, 1);
    nz = 3 * nbus + 4 * nbranch;

    msr = zeros(nz+4, 1); % 39 + 4 or 42 + 4

    [baseMVA, bus, gen, ~, branch, ~, success, ~] = runopf(mpc1, mpopt);

    if success ~= 1
        fail = fail + 1;
        fail_ind = [fail_ind, row];
        row = row + 1;
        continue
    end

    msr(1:nbus) = bus(:, 8);
    Sbus = makeSbus(baseMVA, bus, gen);

    Pinj = real(Sbus);
    Qinj = imag(Sbus);
    msr(nbus+1: 3*nbus) = [Pinj; Qinj];
    branch(:, 14:17) = branch(:, 14:17) / 100;
    msr(3*nbus+1: 3*nbus+4*nbranch) = [branch(:, 14); branch(:, 15); branch(:, 16); branch(:, 17)];
    if k <= 2 || k == 5 % 若系统全运行，则在量测后面加入发电机和负荷的注入量测
        Sg = gen(3, 2:3)/100; % 发电功率注入
        Sl = -bus(3, 3:4)/100; % 负荷负注入
        msr(nz+1:nz+4) = [Sg(1); Sl(1); Sg(2); Sl(2)];
    else % 若为Split/Merge模式，把注入量测放到最后同时删掉6号节点的节点量
        mst = msr([nbus+secn, 2*nbus+secn]); % 3,6 的有功和无功注入放到最后
        msr(nz+1:nz+4) = mst;
        msr([nbus, 2*nbus, 3*nbus]) = []; % 移除6号节点的注入测量
    end
    measurements = vertcat(measurements, msr');
%     kall = vertcat(kall, k);
    row = row + 1;

end

R = [ones(nbus, 1)*0.000016; 0.0001*ones(2*nbus, 1); 0.000064*ones(4*nbranch, 1); 0.0001*ones(4, 1)];
[r,c] = size(measurements);
sigma = sqrt(R);
tt = [];
for col = 1:c
    noise = randn(r,1) * sigma(col);
    tmp = measurements(:, col) + noise;
    tt = horzcat(tt, tmp);
end
end