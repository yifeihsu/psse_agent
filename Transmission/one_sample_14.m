% =========================================================================
% Script: generate_measurements_ieee14.m
% Description: Runs power flow for IEEE 14 Bus system and generates a 
%              measurement vector in per-unit values.
%
% Sequence:
%   1. Vm   (Voltage Magnitude)          - Indices: 0 to nb-1
%   2. Pinj (Active Power Injection)     - Indices: nb to 2*nb-1
%   3. Qinj (Reactive Power Injection)   - Indices: 2*nb to 3*nb-1
%   4. Pf   (Active Power Flow - From)   - Indices: 3*nb to 3*nb+nl-1
%   5. Qf   (Reactive Power Flow - From) - Indices: 3*nb+nl to 3*nb+2*nl-1
%   6. Pt   (Active Power Flow - To)     - Indices: 3*nb+2*nl to 3*nb+3*nl-1
%   7. Qt   (Reactive Power Flow - To)   - Indices: 3*nb+3*nl to 3*nb+4*nl-1
% =========================================================================

% 1. Load MATPOWER constants (for easy indexing like VM, PF, etc.)
define_constants;

% 2. Load the IEEE 14 Case
mpc = loadcase('case14');

% 3. Run Power Flow (suppressing verbose output with mpoption if desired)
opt = mpoption('verbose', 0, 'out.all', 0);
results = runpf(mpc, opt);

% Check if power flow converged
if results.success == 1
    fprintf('Power Flow Converged.\n');
else
    error('Power Flow failed to converge.');
end

% -------------------------------------------------------------------------
% 4. Extract Data and Convert to Per-Unit
% -------------------------------------------------------------------------
baseMVA = results.baseMVA;
nb = size(results.bus, 1); % Number of buses (14)
nl = size(results.branch, 1); % Number of branches (20)

% --- A. Voltage Magnitude (Vm) ---
% Already in p.u. in the results struct
Vm = results.bus(:, VM); 

% --- B. Injections (Pinj, Qinj) ---
% Use makeSbus to calculate complex power injection vector in p.u.
% Sbus = (Pg - Pl) + j(Qg - Ql) in per-unit
Sbus = makeSbus(baseMVA, results.bus, results.gen);
Pinj = real(Sbus);
Qinj = imag(Sbus);

% --- C. Branch Flows (Pf, Qf, Pt, Qt) ---
% MATPOWER branch results (PF, QF, PT, QT) are in MW/MVAr.
% Divide by baseMVA to convert to p.u.
Pf = results.branch(:, PF) / baseMVA;
Qf = results.branch(:, QF) / baseMVA;
Pt = results.branch(:, PT) / baseMVA;
Qt = results.branch(:, QT) / baseMVA;

% -------------------------------------------------------------------------
% 5. Construct the Measurement Vector (z)
% -------------------------------------------------------------------------
z = [Vm; Pinj; Qinj; Pf; Qf; Pt; Qt];

% -------------------------------------------------------------------------
% 6. Display and Validation
% -------------------------------------------------------------------------
fprintf('\nMeasurement Vector Generation Complete.\n');
fprintf('Total Measurements: %d\n', length(z));
fprintf('Expected Length: %d (nb*3 + nl*4) -> (14*3 + 20*4) = 122\n', nb*3 + nl*4);

% Display indices mapping (0-based for User Reference)
fprintf('\n--- Index Mapping (0-based) ---\n');
offset = 0;
fprintf('Vm:   %d to %d\n', offset, offset + nb - 1); offset = offset + nb;
fprintf('Pinj: %d to %d\n', offset, offset + nb - 1); offset = offset + nb;
fprintf('Qinj: %d to %d\n', offset, offset + nb - 1); offset = offset + nb;
fprintf('Pf:   %d to %d\n', offset, offset + nl - 1); offset = offset + nl;
fprintf('Qf:   %d to %d\n', offset, offset + nl - 1); offset = offset + nl;
fprintf('Pt:   %d to %d\n', offset, offset + nl - 1); offset = offset + nl;
fprintf('Qt:   %d to %d\n', offset, offset + nl - 1);

% Optional: Save to file
% csvwrite('measurements_ieee14.csv', z);
% fprintf('\nData saved to measurements_ieee14.csv\n');

% Display first few values
disp(' ');
disp('First 10 values of the measurement vector:');
disp(z(1:10));

bus_data = results.bus;
[lambdaN, success, r, lambda_vec, ea] = LagrangianM_singlephase(z, results, 0, bus_data);