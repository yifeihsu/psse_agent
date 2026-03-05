function [errorb, accu] = LagAccuTest(mpc, mea, nsample)
% Following the data generation part.
% mpc: standard case file; nsample: sample for each location
nsam = nsample;
mpopt = mpoption('verbose', 0, 'out.all', 0);
result = runopf(mpc, mpopt);
nbr = size(mpc.branch, 1);
errorb = zeros(nbr, 1);
c = size(mea, 2); % c is the column shape of the measurement vector(including error indice)
for i= 1:size(mea, 1)
    z = mea(i, 1:c-1);
    lambdaN = LagrangianM(z, result);
    [b , a] = max(abs(lambdaN));
    if a ~= mea(i, c)
         errorb(ceil(i/nsam)) = errorb(ceil(i/nsam)) + 1; % Every 50 change the error location
    end
    i
end
accu = 1 - sum(errorb) / i;
end