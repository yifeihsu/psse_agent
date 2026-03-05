mpc = loadcase(case5);
mpc.branch(:, 6:8) = 0;

nb = size(mpc.bus, 1);
nl = size(mpc.branch, 1);
nz = 3*nb + 4*nl;

[r, c] = size(tt);

resi = [];

for i = 1:r
    mpc1 = mpc;
    mea = tt(i, :);
    k = kall(i);
    if k == 1
        mpc1 = mpc;
        mpc1.bus(6, :) = mpc1.bus(3, :);
        mpc1.bus(6, 1) = 6;
        mpc1.bus(3, 3:4) = 0;
        mpc1.branch(4, 2) = 6;
        mea(nb+3) = mea(nz+1);
        mea(2*nb+3) = mea(nz+3);
        mea = [mea(1:2*nb), mea(nz+2), mea(2*nb+1:3*nb), mea(nz+4), mea(3*nb+1:nz)];
        res = SE(mea, mpc1, 1);
        res([nb+6, nb+6+6]) = [];
    elseif k == 2
        mpc1 = mpc;
        mpc1.bus(6, :) = mpc1.bus(3, :);
        mpc1.bus(6, 1) = 6;
        mpc1.bus(3, 3:4) = 0;
        mpc1.branch(5, 1) = 6;
        mea(nb+3) = mea(nz+1);
        mea(2*nb+3) = mea(nz+3);
        mea = [mea(1:2*nb), mea(nz+2), mea(2*nb+1:3*nb), mea(nz+4), mea(3*nb+1:nz)];
        res = SE(mea, mpc1, 1);
        res([nb+6, nb+6+6]) = [];
    elseif k == 3
        mpc1 = mpc;
        mea(nb+3) = mea(nz+1) + mea(nz+2);
        mea(2*nb+3) = mea(nz+3) + mea(nz+4);
        mea = mea(1:nz);
        res = SE(mea, mpc1, 0);
    elseif k == 4
        mpc1 = mpc;
        mea(nb+3) = mea(nz+1) + mea(nz+2);
        mea(2*nb+3) = mea(nz+3) + mea(nz+4);
        mea = mea(1:nz);
        res = SE(mea, mpc1, 0);
    end
    resi = horzcat(resi, res);
    disp(i);
end
resi = resi';
labelall = [];
for i = 1: r
    if kall(i) == 1
        label = [0, 1, 0, 0, 1, 0];
    elseif kall(i) == 2
        label = [0, 0, 0, 1, 0, 1];
    elseif kall(i) == 3
        label = [0, 0, 1, 0, 0, 1];
    elseif kall(i) == 4
        label = [0, 1, 0, 0, 0, 0];
    end
    labelall = vertcat(labelall, label);
end
resi = horzcat(resi, labelall);