function report = build_ieee57_matpower_3p_reference(output_dir, load_scale)
%BUILD_IEEE57_MATPOWER_3P_REFERENCE Verify the normalized balanced expansion.
% report = build_ieee57_matpower_3p_reference(output_dir, load_scale)
% Requires MATLAB with MATPOWER 8.1 and its prototype three-phase API.
% Loads this repository's pinned case57.m by absolute path. BASE_KV=1 is an
% explicit normalized line-to-line realization, not recovered equipment data.
% The canonical case asset is never modified. Both solves use PV regulation
% with reactive-limit enforcement disabled; this is a balanced PF reference,
% not an OpenDSS exporter or validation of unbalanced/zero-sequence physics.
%
% Primary API/source references checked against MATPOWER 8.1:
% https://matpower.org/doc/howto/three-phase.html
% https://github.com/MATPOWER/matpower/blob/8.1/lib/+mp/case_utils.m
% https://github.com/MATPOWER/matpower/blob/8.1/lib/t/t_convert_1p_to_3p.m

repo = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(output_dir)
    output_dir = fullfile(repo, 'output', ...
        ['ieee57_matpower3p_reference_' datestr(now, 'yyyymmddTHHMMSS')]);
end
if nargin < 2, load_scale = 1.0; end
assert(isscalar(load_scale) && isfinite(load_scale) && load_scale > 0, ...
    'load_scale must be finite and positive.');
assert(exist('run_pf', 'file') == 2 && exist('mpver', 'file') == 2, ...
    'MATPOWER 8.1 must be available on the MATLAB path.');
assert(strcmp(mpver, '8.1'), ...
    'This reference is validated against MATPOWER 8.1; review other versions explicitly.');
assert(~exist(output_dir, 'dir') && ~exist(output_dir, 'file'), ...
    'Choose a new output directory to preserve earlier evidence.');
output_dir = char(java.io.File(output_dir).getCanonicalPath());

define_constants;
source_path = fullfile(repo, 'mcp_server', 'case57.m');
mpc_source = loadcase(source_path);
assert(size(mpc_source.bus, 1) == 57 && size(mpc_source.branch, 1) == 80 ...
    && size(mpc_source.gen, 1) == 7 && mpc_source.baseMVA == 100, ...
    'The canonical IEEE57 dimensions/base changed.');
assert(isequal(mpc_source.bus(:, BUS_I), (1:57)'), ...
    'Review row mapping before using a reordered or renumbered case.');
assert(all(mpc_source.bus(:, BASE_KV) == 0), ...
    'Review existing voltage bases before assigning a normalized realization.');
assert(all(mpc_source.branch(:, SHIFT) == 0), ...
    'MATPOWER 8.1 three-phase conversion does not support phase shifts.');
assert(all(mpc_source.branch(:, BR_STATUS) == 1) ...
    && all(mpc_source.gen(:, GEN_STATUS) == 1), ...
    'This reference requires the canonical all-in-service case.');
assert(numel(unique(mpc_source.gen(:, GEN_BUS))) == 7, ...
    'Co-located generators need a separate reactive-dispatch convention.');
is_xfmr = mpc_source.branch(:, TAP) ~= 0; % includes both nominal-tap circuits
line_rows1 = find(~is_xfmr);
xfmr_rows1 = find(is_xfmr);
assert(numel(line_rows1) == 63 && numel(xfmr_rows1) == 17, ...
    'Canonical line/transformer identity changed.');
assert(isequal(find(is_xfmr & mpc_source.branch(:, TAP) == 1), [35; 36]), ...
    'Nominal-tap parallel transformers must retain distinct row identities.');
assert(all(mpc_source.branch(is_xfmr, BR_B) == 0), ...
    'Transformer charging requires terminal-power accounting after shunt relocation.');

mpc = mpc_source;
mpc.bus(:, BASE_KV) = 1.0;
mpc.bus(:, [PD QD]) = mpc.bus(:, [PD QD]) * load_scale;
assert(isequal(mpc.branch, mpc_source.branch), 'Source impedances/taps changed.');
mpopt = mpoption('verbose', 0, 'out.all', 0, 'pf.tol', 1e-10, ...
    'pf.nr.max_it', 100, 'pf.enforce_q_lims', 0);
task1 = run_pf(mpc, mpopt);
assert(task1.mm.soln.eflag == 1, 'Positive-sequence PF did not converge.');
% Assign BASE_KV before conversion. Passing a new base over source zeros is
% a base conversion, not a valid way to recover missing source voltages.
mpc3p = mp.case_utils.convert_1p_to_3p(mpc);
task3 = run_pf(mpc3p, mpopt, 'mpx', mp.xt_3p);
assert(task3.mm.soln.eflag == 1, 'Balanced three-phase PF did not converge.');

b1 = task1.dm.elements.bus.tab;
b3 = task3.dm.elements.bus3p.tab;
g1 = task1.dm.elements.gen.tab;
g3 = task3.dm.elements.gen3p.tab;
br1 = task1.dm.elements.branch.tab;
ln3 = task3.dm.elements.line3p.tab;
xf3 = task3.dm.elements.xfmr3p.tab;
sh1 = task1.dm.elements.shunt.tab;
sh3 = task3.dm.elements.shunt3p.tab;
ld3 = task3.dm.elements.load3p.tab;
assert(isequal(b1.uid, mpc.bus(:, BUS_I)) && isequal(b3.uid, b1.uid), ...
    'Bus row alignment changed.');
assert(isequal(g1.bus, mpc.gen(:, GEN_BUS)) && isequal(g3.bus, g1.bus), ...
    'Generator row alignment changed.');
assert(isequal([br1.bus_fr br1.bus_to], mpc.branch(:, [F_BUS T_BUS])), ...
    'Single-phase branch row alignment changed.');
assert(isequal([ln3.bus_fr ln3.bus_to], mpc.branch(line_rows1, [F_BUS T_BUS])) ...
    && isequal([xf3.bus_fr xf3.bus_to], mpc.branch(xfmr_rows1, [F_BUS T_BUS])), ...
    'Three-phase branch row alignment changed.');
assert(isequal(sh1.bus, sh3.bus), 'Shunt row alignment changed.');

base_mva = mpc.baseMVA;
vm3 = [b3.vm1 b3.vm2 b3.vm3];
va3 = [b3.va1 b3.va2 b3.va3];
v3 = vm3 .* exp(1j * pi / 180 * va3);
angle_error = angle(exp(1j * pi / 180 * ...
    (va3 - (b1.va + [0 -120 120])))) * 180 / pi;
sf3 = complex(zeros(80, 3));
st3 = complex(zeros(80, 3));
[sf3(line_rows1, :), st3(line_rows1, :)] = phase_flows(ln3);
[sf3(xfmr_rows1, :), st3(xfmr_rows1, :)] = phase_flows(xf3);
sf1 = br1.pl_fr + 1j * br1.ql_fr;
st1 = br1.pl_to + 1j * br1.ql_to;
sg3 = [g3.pg1 g3.pg2 g3.pg3] + 1j * [g3.qg1 g3.qg2 g3.qg3];
sg1 = g1.pg + 1j * g1.qg;
ss3 = [sh3.p1 sh3.p2 sh3.p3] + 1j * [sh3.q1 sh3.q2 sh3.q3];
ss1 = sh1.p + 1j * sh1.q; % demand sign: capacitive Q is negative
pd3 = [ld3.pd1 ld3.pd2 ld3.pd3];
pf3 = [ld3.pf1 ld3.pf2 ld3.pf3];
sl3 = pd3 + 1j * pd3 .* tan(acos(pf3));
load_bus3 = complex(zeros(57, 3));
gen_bus3 = complex(zeros(57, 3));
shunt_bus3 = complex(zeros(57, 3));
terminal_bus3 = complex(zeros(57, 3));
for phase = 1:3
    load_bus3(:, phase) = accumarray(ld3.bus, sl3(:, phase), [57 1]);
    gen_bus3(:, phase) = accumarray(g3.bus, sg3(:, phase), [57 1]);
    shunt_bus3(:, phase) = accumarray(sh3.bus, ss3(:, phase), [57 1]);
    terminal_bus3(:, phase) = accumarray(mpc.branch(:, F_BUS), sf3(:, phase), [57 1]) ...
        + accumarray(mpc.branch(:, T_BUS), st3(:, phase), [57 1]);
end
net1 = accumarray(g1.bus, sg1, [57 1]) - (mpc.bus(:, PD) + 1j * mpc.bus(:, QD));
net3 = sum(gen_bus3 - load_bus3, 2) / 1000;
if3 = conj((sf3 / (base_mva * 1000 / 3)) ./ v3(mpc.branch(:, F_BUS), :));
it3 = conj((st3 / (base_mva * 1000 / 3)) ./ v3(mpc.branch(:, T_BUS), :));
a = exp(1j * 2 * pi / 3);
seq = [1 1 1; 1 a a^2; 1 a^2 a] / 3;
vseq = v3 * seq.';
ifseq = if3 * seq.';
itseq = it3 * seq.';

metrics = struct;
metrics.voltage_magnitude_max_error_pu = maxabs(vm3 - b1.vm);
metrics.voltage_angle_max_error_deg = maxabs(angle_error);
metrics.branch_terminal_power_max_error_pu = ...
    maxabs([sum(sf3, 2)/1000 - sf1; sum(st3, 2)/1000 - st1]) / base_mva;
metrics.generator_power_max_error_pu = maxabs(sum(sg3, 2)/1000 - sg1) / base_mva;
metrics.shunt_power_max_error_pu = maxabs(sum(ss3, 2)/1000 - ss1) / base_mva;
metrics.load_power_max_error_pu = maxabs(sum(load_bus3, 2)/1000 ...
    - (mpc.bus(:, PD) + 1j*mpc.bus(:, QD))) / base_mva;
metrics.net_bus_injection_max_error_pu = maxabs(net3 - net1) / base_mva;
metrics.loss_power_error_pu = abs(sum(sum(sf3 + st3))/1000 - sum(sf1 + st1)) / base_mva;
metrics.phase_power_imbalance_max_pu = maxabs([sf3 - mean(sf3, 2); ...
    st3 - mean(st3, 2); sg3 - mean(sg3, 2); ss3 - mean(ss3, 2)]) / (1000 * base_mva / 3);
metrics.phase_nodal_power_balance_max_pu = maxabs(gen_bus3 - load_bus3 ...
    - shunt_bus3 - terminal_bus3) / (1000 * base_mva / 3);
metrics.voltage_zero_negative_sequence_max_pu = maxabs(vseq(:, [1 3]));
metrics.current_zero_negative_sequence_max_pu = maxabs([ifseq(:, [1 3]); itseq(:, [1 3])]);
metrics.shunt_voltage_dependence_max_error_pu = maxabs(ss3 - ...
    ((mpc.bus(sh3.bus, GS) - 1j * mpc.bus(sh3.bus, BS)) * 1000 / 3) ...
    .* vm3(sh3.bus, :).^2) / (1000 * base_mva / 3);

limits = struct('voltage_magnitude_max_error_pu', 1e-6, ...
    'voltage_angle_max_error_deg', 1e-4, ...
    'branch_terminal_power_max_error_pu', 1e-5, ...
    'generator_power_max_error_pu', 1e-5, 'shunt_power_max_error_pu', 1e-5, ...
    'load_power_max_error_pu', 1e-10, 'net_bus_injection_max_error_pu', 1e-5, ...
    'loss_power_error_pu', 1e-5, 'phase_power_imbalance_max_pu', 1e-8, ...
    'phase_nodal_power_balance_max_pu', 1e-8, ...
    'voltage_zero_negative_sequence_max_pu', 1e-8, ...
    'current_zero_negative_sequence_max_pu', 1e-8, ...
    'shunt_voltage_dependence_max_error_pu', 1e-10);
names = fieldnames(limits);
checks = struct;
for k = 1:numel(names)
    name = names{k};
    checks.(name) = isfinite(metrics.(name)) && metrics.(name) < limits.(name);
end
component = struct;
component.line_r_ohm_max_error = maxabs(mpc3p.lc(:, [2 5 7]) ...
    - mpc.branch(line_rows1, BR_R) * 0.01);
component.line_x_ohm_max_error = maxabs(mpc3p.lc(:, [8 11 13]) ...
    - mpc.branch(line_rows1, BR_X) * 0.01);
component.line_c_nf_max_error = maxabs(mpc3p.lc(:, [14 17 19]) ...
    - mpc.branch(line_rows1, BR_B) / (2*pi*60*0.01) * 1e9);
component.line_mutual_max = maxabs(mpc3p.lc(:, [3 4 6 9 10 12 15 16 18]));
component.transformer_rx_tap_max_error = maxabs(mpc3p.xfmr3p(:, [5 6 9]) ...
    - mpc.branch(xfmr_rows1, [BR_R BR_X TAP]));
component.transformer_phase_base_kva_max_error = maxabs(mpc3p.xfmr3p(:, 7) - 100000/3);
component.transformer_phase_base_kv_max_error = maxabs(mpc3p.xfmr3p(:, 8) - 1/sqrt(3));
checks.component_parameters = component.line_r_ohm_max_error < 1e-12 ...
    && component.line_x_ohm_max_error < 1e-12 && component.line_c_nf_max_error < 1e-5 ...
    && component.line_mutual_max == 0 && component.transformer_rx_tap_max_error < 1e-12 ...
    && component.transformer_phase_base_kva_max_error < 1e-8 ...
    && component.transformer_phase_base_kv_max_error < 1e-12;

report = struct('contract', 'ieee57_matpower81_balanced_3p_reference_v1', ...
    'executed', true, 'passed', all(cell2mat(struct2cell(checks))), ...
    'matlab_version', version, 'matpower_version', mpver, ...
    'source_case_path', source_path, 'source_case_sha256', file_sha256(source_path), ...
    'converter_path', which('mp.case_utils'), ...
    'converter_sha256', file_sha256(which('mp.case_utils')), ...
    'script_sha256', file_sha256(mfilename('fullpath') + ".m"), ...
    'output_dir', output_dir, 'load_scale', load_scale, 'base_mva_3phase', base_mva, ...
    'normalized_base_kv_ll', 1, 'z_base_ohm', 0.01, 'frequency_hz', 60, ...
    'bus_count', 57, 'branch_count', 80, 'line_count', 63, 'transformer_count', 17, ...
    'generator_count', 7, 'shunt_count', height(sh3), ...
    'line_branch_rows1', line_rows1, 'transformer_branch_rows1', xfmr_rows1, ...
    'nominal_tap_transformer_branch_rows1', [35;36], ...
    'positive_sequence_converged', true, 'three_phase_converged', true, ...
    'reactive_limit_enforcement', false, 'operating_limits_checked', false, ...
    'solved_voltage_min_pu', min(b1.vm), 'solved_voltage_max_pu', max(b1.vm), ...
    'metrics', metrics, 'limits', limits, ...
    'checks', checks, 'component_checks', component);
report.scope = ['Balanced PF and diagonal-phase conversion equivalence only; ' ...
    '1kV is normalized, winding/grounding are a chosen Y-Y completion, ' ...
    'no unbalanced/zero-sequence equipment or harmonic model is validated.'];
report.sources = {'https://matpower.org/doc/howto/three-phase.html', ...
    'https://github.com/MATPOWER/matpower/blob/8.1/lib/+mp/case_utils.m', ...
    'https://github.com/MATPOWER/matpower/blob/8.1/lib/t/t_convert_1p_to_3p.m'};
results = struct('bus_ids', b1.uid, 'vm1_pu', b1.vm, 'va1_deg', b1.va, ...
    'vm3_pu', vm3, 'va3_deg', va3, ...
    'branch_rows1', (1:80)', 'from_bus', br1.bus_fr, 'to_bus', br1.bus_to, ...
    'branch_pf_mw_1p', real(sf1), 'branch_qf_mvar_1p', imag(sf1), ...
    'branch_pt_mw_1p', real(st1), 'branch_qt_mvar_1p', imag(st1), ...
    'branch_pf_kw_3p', real(sf3), 'branch_qf_kvar_3p', imag(sf3), ...
    'branch_pt_kw_3p', real(st3), 'branch_qt_kvar_3p', imag(st3), ...
    'generator_bus', g1.bus, 'generator_pg_mw_1p', real(sg1), ...
    'generator_qg_mvar_1p', imag(sg1), 'generator_pg_kw_3p', real(sg3), ...
    'generator_qg_kvar_3p', imag(sg3));
mkdir(output_dir);
savecase(fullfile(output_dir, 'case57_normalized_1p.m'), mpc);
savecase(fullfile(output_dir, 'case57_balanced_3p.m'), mpc3p);
save(fullfile(output_dir, 'reference.mat'), 'mpc', 'mpc3p', 'report', 'results');
write_json(fullfile(output_dir, 'reference_report.json'), report);
write_json(fullfile(output_dir, 'reference_results.json'), results);
fprintf('IEEE57 MATPOWER 8.1 balanced conversion passed=%d\n', report.passed);
disp(metrics);
assert(report.passed, 'One or more equivalence checks failed; inspect reference_report.json.');
end

function [sf, st] = phase_flows(tab)
sf = [tab.pl1_fr tab.pl2_fr tab.pl3_fr] + 1j * [tab.ql1_fr tab.ql2_fr tab.ql3_fr];
st = [tab.pl1_to tab.pl2_to tab.pl3_to] + 1j * [tab.ql1_to tab.ql2_to tab.ql3_to];
end

function value = maxabs(array)
if isempty(array), value = 0; else, value = max(abs(array(:))); end
end

function write_json(path, payload)
fid = fopen(path, 'w');
assert(fid >= 0, 'Cannot write output JSON.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
fprintf(fid, '%s\n', jsonencode(payload, 'PrettyPrint', true));
end

function digest = file_sha256(path)
fid = fopen(path, 'rb');
assert(fid >= 0, 'Cannot read file for provenance hash.');
cleanup = onCleanup(@() fclose(fid)); %#ok<NASGU>
bytes = fread(fid, Inf, '*uint8');
md = java.security.MessageDigest.getInstance('SHA-256');
md.update(bytes);
digest = lower(reshape(dec2hex(typecast(md.digest(), 'uint8'), 2).', 1, []));
end
