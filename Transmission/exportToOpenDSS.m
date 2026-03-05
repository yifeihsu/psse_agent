function exportToOpenDSS(mpc, filename)
%EXPORTTOOPENDSS  Exports a MATPOWER 'mpc' to a basic OpenDSS .dss file.
%
%  filename = 'IEEE14_Opendss.dss', for example.

    if nargin < 2
        filename = 'IEEE14_Opendss.dss';
    end

    baseMVA = mpc.baseMVA;
    nb = size(mpc.bus, 1);
    nl = size(mpc.branch, 1);
    
    fid = fopen(filename, 'w');
    if fid < 0
       error('Could not open file %s for writing.', filename);
    end

    fprintf(fid, 'Clear\n');
    fprintf(fid, 'New Circuit.IEEE14 basekV=69 pu=1.0 phases=3 bus1=BUS1\n');
    fprintf(fid, 'set frequency=60\n\n');

    % ---------------------------------------------------------------------
    % 1) Create (optional) Vsource at the Slack Bus (bus type = 3 in MATPOWER)
    %    Typically, bus #1 is the slack. Or find the bus with type=3
    slackBusIdx = find(mpc.bus(:,2) == 3, 1);
    if isempty(slackBusIdx)
        % fallback if no type=3 found
        slackBusIdx = 1;
    end
    slackBusNum = mpc.bus(slackBusIdx,1);
    slackKV     = mpc.bus(slackBusIdx,10);  % baseKV
    if slackKV<=0, slackKV=69; end  % fallback if not set
    slackName   = sprintf('Bus%d', slackBusNum);

    % "Vsource" in OpenDSS
    % We'll assume it's 3-phase, line-to-line voltage = slackKV
    fprintf(fid, 'New Vsource.Slack bus1=%s.1.2.3  basekV=%.3f  phases=3  angle=0  pu=1.0\n\n', ...
        slackName, slackKV);
    
    % ---------------------------------------------------------------------
    % 2) Define Buses as needed in OpenDSS
    %    (Strictly speaking, OpenDSS auto-creates them if lines reference them)
    %    But we can set up a nominal "kV=" property for each bus.
    for i=1:nb
        busNum  = mpc.bus(i,1);
        busName = sprintf('Bus%d', busNum);
        busKV   = mpc.bus(i,10);  % from the 'baseKV' column
        if busKV<=0, busKV=69; end   % fallback if missing

        % Create a Bus object, 3-phase
        % There's no direct "New Bus..." in OpenDSS, typically we define them 
        % in lines/loads. But we can do "New Transformer"/"New Line" referencing it.
        % We'll just put a comment or set up an element we can refer to:
        fprintf(fid, '! Define %s at base kV=%.3f\n', busName, busKV);
    end
    fprintf(fid, '\n');

    % ---------------------------------------------------------------------
    % 3) Define Lines OR Transformers
    %    If bus i and bus j have the same baseKV, define a line
    %    Otherwise, define a transformer
    lineCount = 0;
    txCount   = 0;
    for k=1:nl
        fromBus = mpc.branch(k,1);
        toBus   = mpc.branch(k,2);
        rpu     = mpc.branch(k,3);  % p.u. resistance
        xpu     = mpc.branch(k,4);  % p.u. reactance
        bpu     = mpc.branch(k,5);  % p.u. line charging (often ignored in distribution)

        busNameF = sprintf('Bus%d', fromBus);
        busNameT = sprintf('Bus%d', toBus);

        kvF = mpc.bus(fromBus,10);
        if kvF<=0, kvF=69; end
        kvT = mpc.bus(toBus,10);
        if kvT<=0, kvT=69; end

        % We'll assume system frequency 60Hz
        % Convert from p.u. to ohms, roughly using busF's base kV
        % But if they differ, handle carefully...
        if abs(kvF - kvT) < 1e-3
            % same base kv, treat as a line
            lineCount = lineCount + 1;
            LName = sprintf('Line%d', lineCount);

            % base for ohm calc
            vbase_lin = kvF;  
            r_ohm = rpu * (vbase_lin^2 / baseMVA);
            x_ohm = xpu * (vbase_lin^2 / baseMVA);

            fprintf(fid, 'New Line.%s Bus1=%s.1.2.3 Bus2=%s.1.2.3 ', ...
                LName, busNameF, busNameT);
            fprintf(fid, 'phases=3 R1=%.6f X1=%.6f ', r_ohm, x_ohm);
            fprintf(fid, 'basefreq=60 length=1 units=None\n');

        else
            % different kv => treat as a transformer
            txCount = txCount + 1;
            TName = sprintf('Xfmr%d', txCount);

            % typical approach: use from-bus side as "winding1", to-bus side as "winding2"
            % OpenDSS wants %R, Xhl in % ...
            % Also we might consider a typical reference MVA for the Xfmr rating
            % This is highly approximate!
            % We'll do a rough approach:
            vbase1 = kvF;
            vbase2 = kvT;
            % per-unit on the system base => approximate X % for transformer
            %  X% = xpu * 100 => if xpu=0.1, X%=10. 
            %  R% = rpu * 100
            xpct = xpu * 100;  
            rpct = rpu * 100;

            fprintf(fid, 'New Transformer.%s Phases=3 Xhl=%.2f %%r=%.2f bank=IEEE14bank\n', ...
                TName, xpct, rpct);
            fprintf(fid, '~ wdg=1 bus=%s.1.2.3 kv=%.3f kva=1000\n', busNameF, vbase1);
            fprintf(fid, '~ wdg=2 bus=%s.1.2.3 kv=%.3f kva=1000\n', busNameT, vbase2);
        end
    end
    fprintf(fid, '\n! Done lines/transformers\n\n');

    % ---------------------------------------------------------------------
    % 4) Define Loads
    %    In MATPOWER: Pd (MW), Qd (Mvar).  We convert to kW, kvar in OpenDSS.
    for i=1:nb
        Pd = mpc.bus(i,3);  % MW
        Qd = mpc.bus(i,4);  % Mvar
        if Pd>0 || Qd>0
            busName = sprintf('Bus%d', mpc.bus(i,1));
            loadName = sprintf('Load%d', mpc.bus(i,1));
            kvLoad   = mpc.bus(i,10); if kvLoad<=0, kvLoad=69; end
            % convert MW => kW, Mvar => kvar
            kWval   = Pd * 1000;
            kvarVal = Qd * 1000;
            % Simple PF-based or model=1 (constant Z) or model=2 (constant I) or model=5 (ZIP)
            % We'll pick model=1 or 2 arbitrarily. 
            fprintf(fid, 'New Load.%s bus1=%s.1.2.3  Phases=3', loadName, busName);
            fprintf(fid, ' Conn=Delta kV=%.3f kW=%.2f kvar=%.2f Model=1\n', ...
                         kvLoad, kWval, kvarVal);
        end
    end
    fprintf(fid, '\n! Done loads\n\n');

    % ---------------------------------------------------------------------
    % 5) Define Generators or PV (for buses with mpc.gen)
    ng = size(mpc.gen,1);
    for g=1:ng
        genBus = mpc.gen(g,1);
        busName = sprintf('Bus%d', genBus);
        kvGen   = mpc.bus(genBus,10); if kvGen<=0, kvGen=69; end
        PG = mpc.gen(g,2);   % MW
        QG = mpc.gen(g,3);   % Mvar
        if PG < 0, PG=0; end
        % convert to kW
        fprintf(fid, 'New Generator.gen%d bus1=%s.1.2.3 phases=3 ', g, busName);
        fprintf(fid, 'kv=%.3f kW=%.3f pf=1.0\n', kvGen, PG*1000);
    end

    fprintf(fid, '\n! Done generators\n');

    % ---------------------------------------------------------------------
    % Finish
    fprintf(fid, 'Solve\n\n');
    fclose(fid);

    fprintf('Exported OpenDSS model to file: %s\n', filename);
end
