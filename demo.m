clear all;
close all;
clc

%% set options of Gr(n,p)
N = 21; % sample size
n = 3; p = 1; % matrix n x p
G = grassmannfactory(n,p); % geometric tools of the Grassmannian Gr(n,p)
time = linspace(0,1,N); % time instances between 0 and 1


%% generate N random points on Gr(n,p) following the geodesic path
norm_vel=pi/2; % injectivity radius

while norm_vel >= pi/2
p0_true=G.rand(); % initial position
sim=normrnd(0, 1, [n,p]); 
v0_true=(eye(n) - p0_true * p0_true') * sim; % initial velocity
norm_vel=norm(v0_true,'fro');
end

p1_true=G.exp(p0_true,v0_true); % end position

var_noise = 0.1; % noise variance

for i=1:N
    Q_true{i}=G.exp(p0_true,time(i)*v0_true); % clean data
end

Q=addNoiseToData(Q_true, var_noise); % add noise to clean data


%% plot simulated data on Gr(n=3,p=1)
plotSimulationsGr31


                                        
%% run GLHMC 
obs.shapes = Q;
obs.ts = time;

% Initialization
value_init = random_initialization(Q);

nIters=1e3; % desired number of samples
LF=10; % number of iterations for leap-frog
nBurnIn=50; % burn-in parameter
epsilon=1e-4; % step size

[chain, acceptance_rate, P] = GLHMC(nIters, nBurnIn, obs, value_init,epsilon,LF);