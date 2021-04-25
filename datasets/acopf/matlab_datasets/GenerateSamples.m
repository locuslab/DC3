%% This code was created by Ahmed Zamzam on August 10, 2019 at St. Paul, MN,
%%   and modified by Priya Donti and David Rolnick.
% The code generates feasible ACOPF training samples
% The training samples represent load profiles and their corresponding
% optimal generators set-points

%% Include libraries
addpath(genpath('/Users/priyadonti/MATLAB/matpower6.0'));
addpath(genpath('/Users/priyadonti/MATLAB/tspopf5.1_maci64'));
%%
clearvars;
clc;
warning off;

my_model = case57;
Ybus = makeYbus(my_model);

%% define named indices into data matrices
[PQ, PV, REF, NONE, BUS_I, BUS_TYPE, PD, QD, GS, BS, BUS_AREA, VM, VA, ...
    BASE_KV, ZONE, VMAX, VMIN, LAM_P, LAM_Q, MU_VMAX, MU_VMIN] = idx_bus;
[GEN_BUS, PG, QG, QMAX, QMIN, VG, MBASE, GEN_STATUS, PMAX, PMIN, ...
    MU_PMAX, MU_PMIN, MU_QMAX, MU_QMIN, PC1, PC2, QC1MIN, QC1MAX, ...
    QC2MIN, QC2MAX, RAMP_AGC, RAMP_10, RAMP_30, RAMP_Q, APF] = idx_gen;
[F_BUS, T_BUS, BR_R, BR_X, BR_B, RATE_A, RATE_B, RATE_C, ...
    TAP, SHIFT, BR_STATUS, PF, QF, PT, QT, MU_SF, MU_ST, ...
    ANGMIN, ANGMAX, MU_ANGMIN, MU_ANGMAX] = idx_brch;
[PW_LINEAR, POLYNOMIAL, MODEL, STARTUP, SHUTDOWN, NCOST, COST] = idx_cost;

%% data dimensions
nb   = size(my_model.bus, 1);    %% number of buses
nl   = size(my_model.branch, 1); %% number of branches
ng   = size(my_model.gen, 1);    %% number of dispatchable injections

pv   = find(my_model.bus(:, BUS_TYPE) == 2);
pv_  = find(ismember(my_model.gen(:, GEN_BUS), pv));
gens = my_model.gen(:, GEN_BUS); 
pq   = find(my_model.bus(:, BUS_TYPE) == 1);
%% Data Copy
bus = my_model.bus;
br  = my_model.branch; 
gen = my_model.gen;
MVA_base = my_model.baseMVA;

%% Load base load profile
BaseLoadP  = bus(:, PD);
BaseLoadQ  = bus(:, QD);
BaseLoadPF = cos(atan(bus(:,QD)./bus(:,PD)));
BaseLoadPF(isnan(BaseLoadPF)) = 0;

NL = nnz(BaseLoadP);
LoadBuses = find(BaseLoadP>0);

%% Generate load samples
disp('Generating load samples');
NSamples   = 1e5;
CorrCoeff  = 0.5;
MAX_PF     = 1;
MIN_PF     = 0.8;
MaxChangeLoad = 0.7;
ScaleSigma = MaxChangeLoad/1.645;  %1.645 is the 95th percentile of a normal Gaussian with 1 std

mu    = ones(NL, 1);
sigma = ScaleSigma^2 * (CorrCoeff*ones(NL) + (1-CorrCoeff)*eye(NL));

LoadFactor = mvnrnd_trn((1-MaxChangeLoad)*ones(NL,1)', ...
            (1+MaxChangeLoad)*ones(NL,1)', mu', sigma, NSamples)';
PFFactor  = (rand(NL, NSamples)*(MAX_PF-MIN_PF) + MIN_PF);

%% Solving ACOPF problem for the samples and storing the input and output 
% pairs 
% 
Dem = NaN(nb, NSamples);
Gen = NaN(ng, NSamples);
Vol = NaN(nb, NSamples);

% EPS_INTERIOR = 0.01;
EPS_INTERIOR = 0;

my_model.bus(:, VMIN) = bus(:, VMIN) + EPS_INTERIOR;
my_model.bus(:, VMAX) = bus(:, VMAX) - EPS_INTERIOR;

my_model.branch(:,6:8) = 0; % turn off line flow limits

mpopt = mpoption('model','ac','opf.ac.solver','MIPS','verbose',0);

for t = 1:NSamples
    disp(t);
    my_model.bus(LoadBuses, PD) = BaseLoadP(LoadBuses) .* LoadFactor(:, t);
    my_model.bus(LoadBuses, QD) = (my_model.bus(LoadBuses, PD) .* ...
        tan(acos(PFFactor(:, t)))).*sign(BaseLoadQ(LoadBuses));

    
    % solve acopf
    my_result  = opf(my_model, mpopt);
    if(my_result.success == 1)
        Dem(:, t) = my_model.bus(:, PD) + 1j * my_model.bus(:, QD);
        Gen(:, t)= my_result.gen(:, PG) + 1j * my_result.gen(:, QG);
        vm = my_result.bus(:, VM);
        va = deg2rad(my_result.bus(:, VA));
        Vol(:, t) = vm .* exp(va * 1j);
    end
    if (rem(t, 1000) == 0)
        % Save Data
        save('FeasiblePairs_Case39.mat', 'Dem', 'Gen', 'Vol', 'Ybus', 'EPS_INTERIOR', ...
            'CorrCoeff','MaxChangeLoad');
        fprintf('Done with %d iterations \n', t);
    end
end

%% Save Data
% save('FeasiblePairs_Case118.mat', 'X', 'Y', 'EPS_INTERIOR');