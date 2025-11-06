function wls_cli_main(in_json_path, out_json_path)
% wls_cli_main Read input JSON, run WLS via LagrangianM_singlephase, write output JSON.

try
    % Ensure this folder (server dir) is on MATLAB path for helper + LagrangianM_singlephase
    here = fileparts(mfilename('fullpath'));
    addpath(genpath(here));

    % Optionally add MATPOWER from env
    mp = getenv('MATPOWER_PATH');
    if ~isempty(mp)
        addpath(genpath(mp));
    end

    % Verify MATPOWER available
    if isempty(which('runpf'))
        error('MATPOWER not found on the MATLAB path. Set MATPOWER_PATH or add MATPOWER to path.');
    end

    % Read inputs
    raw = fileread(in_json_path);
    D = jsondecode(raw);
    case_path = D.case_path;
    Z = D.z(:)';

    % Build case and run WLS
    define_constants;
    mpc = loadcase(case_path);
    bus_data = mpc.bus;

    % Basic dimension check: [Vm(nb), Pinj(nb), Qinj(nb), Pf/Qf/Pt/Qt (4*nl)]
    nb = size(mpc.bus,1); nl = size(mpc.branch,1);
    expN = 3*nb + 4*nl;
    if numel(Z) ~= expN
        error('WLS input error: |z|=%d, expected %d (=3*nb + 4*nl).', numel(Z), expN);
    end

    if isempty(which('LagrangianM_singlephase'))
        error('LagrangianM_singlephase.m not found on path');
    end

    Ind = 0;
    [lambdaN, success, r, lambda_vec, ea] = LagrangianM_singlephase(Z, mpc, Ind, bus_data);

    S.success    = logical(success);
    S.lambdaN    = full(lambdaN);
    S.r          = full(r);
    S.lambda_vec = full(lambda_vec);
    S.ea         = full(ea);

    fid = fopen(out_json_path, 'w');
    if fid == -1, error('Failed to open output: %s', out_json_path); end
    fwrite(fid, jsonencode(S), 'char');
    fclose(fid);
catch e
    try
        S = struct('success', false, 'error', e.message);
        fid = fopen(out_json_path, 'w');
        if fid ~= -1
            fwrite(fid, jsonencode(S), 'char');
            fclose(fid);
        end
    catch
        % ignore secondary errors
    end
    rethrow(e);
end

