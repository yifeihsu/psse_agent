%% 请先根据./GCN-BDD中的时序数据生成xb, elabel, tt, kall等
% 模型和相关参数初始化
clear;
clc;

load './Data and index/op_49_5.mat'
% load 'Data and index'\kall.mat
% load traditional_process\op_standard.mat
% cd './traditional_process'
% load('kall.mat')
% load('label.mat')
% load('para_label.mat')
% load('tt.mat')
% load('xb.mat')
%
% cd '../'
% 读取系统基本模型和信息
kall = op.kall;
tt = op.tt;
tt1 = op.tt1;
label = op.label;
para_label1 = op.para_label;
xb = op.xb;

mpc = loadcase(case5);
mpc.branch(:, 6:8) = 0;
mpc.zeroimpe = [3, 7; 7, 8; 6, 8; 3, 9; 9, 10; 6, 10]; % 关联矩阵
mpc.nsec = 5;
mpc.ncb = 6;

nb = size(mpc.bus, 1);
nl = size(mpc.branch, 1);
nz = 3*nb + 4*nl;

tol = size(tt, 1);

label(:, 4) = kall;

num_suc = 0;
num_para_suc = 0;
num_topo_suc = 0;

topo_iden_all = zeros(tol, 1);
para_iden_all = zeros(tol, 1);

for i = 1:tol % 对全部断面进行BDD / BDI
    disp(i);
    % 首先进行TOPO BDI
    mpc1 = mpc;
    mpc1.branch(:, 4) = xb(:, i);
    mea = tt(i, :);
    mea1 = tt1(i, :);
    k = kall(i);
    topo_label = label(i, 3);
    flag_topo = 1; % 检测是否存在拓扑错误
    fail_topo = 0;
    count_topo = 0;
    topo_iden = 0;

    NLM_topo;
    % while flag_topo == 1
    %     [lambda, to_flag, flag_fail] = maxM_topo(mea, mpc1);
    %     if flag_fail == 1
    %         fail_topo = 1;
    %         break
    %     end
    %     if max(abs(lambda(42:61))) > 10 && to_flag == 1
    %         % Identification
    %         max_index = maxindex(abs(lambda(42:61)));
    %         icbt = ceil(max_index/2);
    %         icb = zeros(1, 6);
    %         icb(icbt) = 1;
    %         %             % Correction：把可疑集的全部reverse
    %         %             mpc1.cb(icbt) = 1 - mpc1.cb(icbt);
    %         %             if all(mpc1.cb) == 1
    %         %                 mpc1.cb(2) = 0;
    %         %             end
    %         %             topo_iden = 1;
    %         %             topo_iden_all(i) = 1;
    %         %             count_topo = count_topo + 1;
    %         if k == 1 || k == 4
    %             true_cb = [2, 5];
    %         elseif k == 2
    %             true_cb = [4, 6];
    %         elseif k == 3
    %             true_cb = [3, 6];
    %         end
    %         if all(ismember(icbt, true_cb)) == 1
    %             topo_label = 1;
    %             topo_iden = 1;
    %             flag_topo = 0;
    %         else
    %             fail_topo = 1;
    %             break
    %         end
    % 
    %     else
    %         flag_topo = 0; % 拓扑错误清空！
    %     end
    % 
    %     %         if count_topo >= 5
    %     %             fail_topo = 1;
    %     %             break
    %     %         end
    % end

    if fail_topo == 1 % 拓扑检测程序报错：1.拓扑错误检测不到 2.可能是参数错误
        topo_iden_flag = 0;
        disp('The topology error cannot be identified!');
    elseif fail_topo == 0 && topo_iden == 1 && topo_label == 1 % 拓扑检测程序通过： 确实有拓扑错误，且检测成功
        topo_iden_flag = 1;
        disp('The topology is correct now, please continue.')
    elseif fail_topo == 0 && topo_iden == 0 && topo_label == 0 % 拓扑检测程序通过： 没有拓扑错误
        topo_iden_flag = 1;
        disp('There is no topology error and correct!')
    end

    if topo_iden_flag == 1
        num_topo_suc = num_topo_suc + 1;
    end

    %% 如果通过了拓扑错误检测，那么更换成参数错误检测模型
    mpc1 = mpc;
    mpc1.branch(:, 4) = xb(:, i); % 加载标准mpc casefile，将参数做替换
    % 分几种k拓扑情况，进行拓扑分析，bing
    if k == 1
        mea(nb+3) = mea(nz+1) + mea(nz+2); % 3号节点的有功注入
        mea(2*nb+3) = mea(nz+3) + mea(nz+4); % 3号节点的无功注入
        mea = mea(1:nz);

        mea1(nb+3) = mea1(nz+1) + mea1(nz+2); % 3号节点的有功注入
        mea1(2*nb+3) = mea1(nz+3) + mea1(nz+4); % 3号节点的无功注入
        mea1 = mea1(1:nz);

        para_ind = 0;
    elseif k == 2
        mea(nb+3) = mea(nz+1) + mea(nz+2);
        mea(2*nb+3) = mea(nz+3) + mea(nz+4);
        mea = mea(1:nz);

        mea1(nb+3) = mea1(nz+1) + mea1(nz+2);
        mea1(2*nb+3) = mea1(nz+3) + mea1(nz+4);
        mea1 = mea1(1:nz);
        para_ind = 0;
    elseif k == 3
        mpc1.bus(6, :) = mpc1.bus(3, :);
        mpc1.bus(6, 1) = 6;
        mpc1.bus(3, 3:4) = 0;
        mpc1.branch(5, 1) = 6;
        mea(nb+3) = mea(nz+1); % P3
        mea(2*nb+3) = mea(nz+3); % Q3
        mea = [mea(1:2*nb), mea(nz+2), mea(2*nb+1:3*nb), mea(nz+4), mea(3*nb+1:nz)];

        mea1(nb+3) = mea1(nz+1); % P3
        mea1(2*nb+3) = mea1(nz+3); % Q3
        mea1 = [mea1(1:2*nb), mea1(nz+2), mea1(2*nb+1:3*nb), mea1(nz+4), mea1(3*nb+1:nz)];
        para_ind = 1;
    elseif k == 4
        mpc1.bus(6, :) = mpc1.bus(3, :);
        mpc1.bus(6, 1) = 6;
        mpc1.bus(3, 3:4) = 0;
        mpc1.branch(4, 2) = 6;
        mea(nb+3) = mea(nz+1);
        mea(2*nb+3) = mea(nz+3);
        mea = [mea(1:2*nb), mea(nz+2), mea(2*nb+1:3*nb), mea(nz+4), mea(3*nb+1:nz)];

        mea1(nb+3) = mea1(nz+1);
        mea1(2*nb+3) = mea1(nz+3);
        mea1 = [mea1(1:2*nb), mea1(nz+2), mea1(2*nb+1:3*nb), mea1(nz+4), mea1(3*nb+1:nz)];
        para_ind = 1;
    end

    disp('The exact model is loaded.');

    %% NLM for PE Identification
    count_para = 0;
    flag_para = 1; % 控制循环是否结束

    flag_pe_error = 0; % 判断循环结束后检测结果是否正确

    para_iden = []; % 检测到的参数错误集合

    while flag_para == 1
        [para_lambda, NLM_success, res] = LagrangianM(mea, mpc1, para_ind);
        % If diverge
        if NLM_success == 0
            break
        end
        para_lambda = abs(para_lambda);
        res = abs(res);

        [lambda_max, lambda_ind] = max(para_lambda);
        [res_max, res_ind] = max(res);

        if max(res_max, lambda_max) >= 3
            if lambda_max >= res_max % 如果是参数错误，那么对应去修正参数(mpc1)
                if mpc1.branch(lambda_ind, 4) == mpc.branch(lambda_ind, 4)
                    flag_pe_error = 1; % 辨识出来的错误参数实际是正确的，那么直接跳出！
                    disp('PE error.')
                    break
                else
                    mpc1.branch(lambda_ind, 4) = mpc.branch(lambda_ind, 4);
                    para_iden = [para_iden, lambda_ind];
                    count_para = count_para + 1; % 辨识了1个参数
                end
            elseif lambda_max < res_max % 如果是遥测错误，检查错误位置，修正mea向量
                % 如果最大正则化残差大于阈值，那么1) 错误检测 2） 正确检测
                if mea(res_ind) == mea1(res_ind)
                    flag_pe_error = 1; % 辨识出来的错误参数实际是正确的，那么直接跳出！
                    disp('PE error.')
                    break
                else
                    mea(res_ind) = mea1(res_ind);
                end
            end
        else % 当全部小于阈值时，没有进一步检测；
            flag_para = 0; % 循环结束
        end

        if count_para >= 5 % count的次数和参数错误的数目有关
            break
        end
    end

    if count_para > 0
        para_iden_all(i) = 1;
    end

    % 检验结果是否正确，label是否一致（如果不一致则存在没有检测到的参数错误）

    para_label = para_label1(i, :); % Para label表示每个支路参数的正确与否
    para_iden1 = zeros(1, nl);
    para_iden1(para_iden) = 1;

    if all(para_iden1 == para_label)
        disp('All the parameter errors are corrected!');
        para_iden_flag = 1; % 成功辨识所有参数错误
    else
        disp('There are parameter errors!');
        para_iden_flag = 0;
    end

    if flag_para == 1
        para_iden_flag = 0;
    end

    %% 最终准确率指标计算

    if para_iden_flag == 1 % 参数错误准确率指标
        num_para_suc = num_para_suc + 1;
    end

    if para_iden_flag == 1 && topo_iden_flag == 1 % 如果TE与PE都没有问题，则成功！
        num_suc = num_suc + 1;
    end

    %% 06/29/22 新增部分，可能能提高拓扑检测准确率
    if topo_iden_flag == 0 && para_iden_flag == 1 % 检测成功参数错误，而拓扑错误没检测成功，再检测一遍拓扑！
        mpc1 = mpc;
        mea = tt(i, :);
        NLM_topo;
        if fail_topo == 1
            topo_iden_flag = 0;
            disp('The topology error cannot be identified!');
        elseif fail_topo == 0 && topo_iden == 1 && topo_label == 1
            topo_iden_flag = 1;
            disp('The topology is correct now, please continue.')
        elseif fail_topo == 0 && topo_iden == 0 && topo_label == 0
            topo_iden_flag = 1;
            disp('There is no topology error and correct!')
        end
        if topo_iden_flag == 1
            num_topo_suc = num_topo_suc + 1;
        end
    end

    %     if fail_para == 1 && flag_para == 1
    %         disp('The parameter error cannot be identified!')
    %         num_pe = num_pe + 1;
    %         num_error = num_error + 1;
    %     elseif flag_para == 0 && fail_topo == 0
    %         disp('All parameters are correct now.')
    %     elseif flag_para == 0 && fail_topo == 1
    %         num_error = num_error + 1;
    %     end
end
disp((num_suc / tol) * 100);
disp((num_para_suc / tol) * 100);
disp((num_topo_suc / tol) * 100);