function grParams = setRegressParams()

grParams.sigmaSqr = 1;            % sigma square for balancing the mismatching term
grParams.alpha = 0;               % alpha for balancing the prior knowledge of the slope of the geodesic
grParams.nt = 100;                % the number of the discretized grids 
grParams.h = 1./grParams.nt;
grParams.deltaT = 0.01;           % the step size for updating the initial conditions
grParams.maxReductionSteps = 10;  % the number of times for tracing back
grParams.rho = 0.5;               % the step size for tracing back
grParams.nIterMax = 250;          % the number of iterations for updating the initial conditions
grParams.nIterBoth = 10;
grParams.stopThreshold = 1e-7;    % the threshold for stopping the iterations
grParams.useODE45 = true;         % chose ODE45 for solving the PDEs, more stable
grParams.minFunc = 'linesearch';  % use linesearch for updating the initial conditions, other methods could be implemented later
grParams.useRansac = false;
grParams.fullGGRInitZero = true;

grParams.fidSplineEnergy = 500;
grParams.fidSplineFitting = 100;