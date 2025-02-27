
%
% INPUTS:
% nIters = the number of desired samples
% NBurnIn = the burn-in parameter to avoid bad first values
% obs = the set of time instances with corresponding objects on the
% Grassmannian
% value_init = initialisation for position and velocity
% step = the step size
% L = the number of iterations for leap-frog
%
%
% OUTPUTS:
% chain = a list of sampling for position and velocity
% acceptance = acceptance rate of GLHMC
% P = the log-posterior applied to each sample



function [chain, acceptance, post] = GLHMC(nIters, NBurnIn, obs, value_init, step, L)
 
chain = cell(nIters-NBurnIn,1); post=[];
theta = value_init;
accp=0;
acceptance=0;
 
% Start the timer
tic;

for iI = 1:nIters
    
    if mod(iI, 100) == 0 
    fprintf(['\r Acceptance rate between %d and %d: %f'], iI-99, iI, accp/100);
    accp = 0;
    drawnow; % Flush the output buffer
    end

    [theta,ind,HL] = leap_frog(theta,obs,step,L);
    accp = accp + ind;
    
    if iI > NBurnIn
    chain{iI-NBurnIn}=theta;  
    acceptance=acceptance+ind;
    post(iI-NBurnIn) = HL; 
    end 
end
 
% Stop the timer
elapsed_time = toc;
 
% Display the elapsed time
disp(['Executed time: ' num2str(elapsed_time) ' seconds']);

acceptance=acceptance/(nIters-NBurnIn);

end
 


function [params,accp,HL] = leap_frog(theta_curr,obs,step,L)

theta=theta_curr;
  
[n,p]=size(theta.X1);

G = grassmannfactory(n,p);
 
V_r=normrnd(theta.X2, 0.01, [n, p]);
 
V=(eye(n) - theta.X1 * theta.X1') * V_r;
 
[H1,~]=PosteriorAndGradient(theta, obs);
   
L=ceil(rand()*L);

i=0;
 
while i<L
    
  [~,du]=LikelihoodAndGradient(theta,obs);   
      
  du.X1=(eye(n) - theta.X1 * theta.X1') * du.X1;
  
  V=V+step*du.X1/2; 
  
  A0=theta.X1;
  
  Pos = G.exp(A0,V,step);
 
  V=G.grad_exp(A0,V,step); 

  theta.X1=Pos;
  
  [~,du]=LikelihoodAndGradient(theta,obs);
    
  du.X1=(eye(n) - theta.X1 * theta.X1') * du.X1;
 
  V=V+step*du.X1/2; 
  
  theta.X2=V;
    
  i=i+1;
  
end
 
[HL,~]=PosteriorAndGradient(theta,obs);  
 
prob=exp(HL-H1); 
 
if(unifrnd(0, 1) < prob)
    accp = 1;
    params.X1=theta.X1;
    params.X2=V;
else 
    accp=0;
    params=theta_curr;
end

end



function [post,grad] = PosteriorAndGradient( params, obs )

[like,gradlike]=LikelihoodAndGradient(params, obs);
[prior,gradprior]=PriorAndGradient(params) ; 
post=like+prior;

grad.X1=gradlike.X1+gradprior.X1;
grad.X2=gradlike.X2+gradprior.X2;

end


function [prior,grad] = PriorAndGradient(params)

pos = params.X1;
vel = params.X2;


posPrior = -10*norm(pos, 'fro')^2/2;
velPrior = -10*norm(vel, 'fro')^2/2;

prior = posPrior + velPrior;


gradientX10=+10*pos;
gradientX20=+10*vel;

grad.X1 = -gradientX10;
grad.X2 = -gradientX20;

end


function [likelihood,grad] = LikelihoodAndGradient(params, obs)

grParams = setRegressParams();
grParams.Ys = obs.shapes;
grParams.ts = obs.ts;
grParams.wi = ones(length(grParams.ts), 1);
[grParams.ts, idSort] = sort(grParams.ts);
grParams.Ys = grParams.Ys(idSort);
grParams.wi = grParams.wi(idSort);
grParams.ts_pre = grParams.ts;   
grParams.ts = grParams.ts - grParams.ts(1);
grParams.ts = grParams.ts ./ grParams.ts(end);
grParams.ts = min( max( round(grParams.ts / grParams.h) + 1, 1 ), grParams.nt+1 );

[~, X1s, X2s] = integrateForwardWithODE45(params.X1, params.X2, (0:grParams.nt)*grParams.h);
sigmaSq = estimateSigmaOfNoiseModel(X1s(grParams.ts), grParams.Ys);

energyV = grParams.alpha * trace( (X2s{1})' * X2s{1} );
energyS = 0;
for iI = 1:length(grParams.ts)
    [~, ~, ~, vTmp] = grgeo(X1s{grParams.ts(iI)}, grParams.Ys{iI}, 1, 'v3', 'v2');
    tmpDistSq = computeGeodesciDistanceWithSigma(sigmaSq{iI}, vTmp);
    energyS = energyS + tmpDistSq * grParams.wi(iI);
end
energyS = energyS / 2.0;  % here we compute energy / 2sigma^2
likelihood = - energyV - energyS;


Y=[params.X1 ; params.X2];
% forward in time
[X1, X2, Y_RK] = integrateForwardMP( Y(1:length(params.X1), :), Y(length(params.X1)+1:end, :), grParams.nt, grParams.h );

% backward in time
lam1_end = zeros( size(X1{end}) );
lam2_end = zeros( size(lam1_end) );
[lam1, lam2] = integrateBackwardMP( lam1_end, lam2_end, Y_RK, X1s, grParams );

% compute the gradient to update
% update on 05/12/2016, from negative gradient to positive 
tmp1 = eye(size(X1{1}, 1)) - X1{1} * (X1{1})';
gradientX10 = -tmp1 * lam1{end} + X2{1} * (lam2{end})' * X1{1};
gradientX20 = 2*X2{1}*grParams.alpha - tmp1 * lam2{end};

grad.X1 = -gradientX10;
grad.X2 = -gradientX20;

    
end