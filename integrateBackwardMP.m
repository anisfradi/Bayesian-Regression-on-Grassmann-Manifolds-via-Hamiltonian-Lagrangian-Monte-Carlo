%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%  integrate backward in time
%  
%  Author: Yi Hong
%  Date: 08-30-2013
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [lam1, lam2, lam_RK] = integrateBackwardMP(lam1_end, lam2_end, Y_RK, X1s, params)

nt = params.nt;
%nps: fix
h = -params.h;

lam1_k = lam1_end;
lam2_k = lam2_end;
lam_k = [lam1_k; lam2_k];
lam1 = {};
lam2 = {};
nSize = size(lam1_end, 1);
lam_RK = cell( nt, 4 );


% 4th Runge-Kutta
for it = 1:nt
    id = find(params.ts == nt-it+2);
    for iI = 1:length(id)
        [~, ~, ~, tanTmp] = grgeo(X1s{id(iI)}, params.Ys{id(iI)}, 1, 'v3', 'v2');
        if isfield(params, 'diagSigmaSqs')
%             matSigma = diag(1.0 ./ params.diagSigmaSqs);
            lam_k(1:nSize, :) = lam_k(1:nSize, :) - computeJumpWithSigma(params.diagSigmaSqs{id(iI)}, -2*tanTmp);
%                 + reshape(matSigma * tanTmp(:), size(tanTmp)) * params.wi(id(iI)); % jump, here energy/(2sigma^2)
        else
            lam_k(1:nSize, :) = lam_k(1:nSize, :) + tanTmp * 2.0 / params.sigmaSqr * params.wi(id(iI));   % jump
        end
    end
    lam1{it} = lam_k(1:nSize, :);
    lam2{it} = lam_k(nSize+1:end, :);
    Y4_k = lam_k;
    f4 = f(Y_RK{nt-it+1, 4}, Y4_k);
    Y3_k = lam_k + h/2.0 * f4; %f(Y_RK{nt-it+1, 4}, Y4_k);
    f3 = f(Y_RK{nt-it+1, 3}, Y3_k);
    Y2_k = lam_k + h/2.0 * f3; %f(Y_RK{nt-it+1, 3}, Y3_k);
    f2 = f(Y_RK{nt-it+1, 2}, Y2_k);
    Y1_k = lam_k + h * f2; %f(Y_RK{nt-it+1, 2}, Y2_k);
    lam_k = lam_k + h * ( f(Y_RK{nt-it+1, 1}, Y1_k)/6.0 + f2/3.0 + f3/3.0 + f4/6.0 );
    %lam_k = lam_k + h * ( f(Y_RK{nt-it+1, 1}, Y1_k)/6.0 + f(Y_RK{nt-it+1, 2}, Y2_k)/3.0 + ...
    %                      f(Y_RK{nt-it+1, 3}, Y3_k)/3.0 + f(Y_RK{nt-it+1, 4}, Y4_k)/6.0 );
    lam_RK{it, 4} = Y4_k;
    lam_RK{it, 3} = Y3_k;
    lam_RK{it, 2} = Y2_k;
    lam_RK{it, 1} = Y1_k;
end
id = find(params.ts == 1);
for iI = 1:length(id)
    [~, ~, ~, tanTmp] = grgeo(X1s{id(iI)}, params.Ys{id(iI)}, 1, 'v3', 'v2');
    if isfield(params, 'diagSigmaSqs')
%         matSigma = diag(1.0 ./ params.diagSigmaSqs);
        lam_k(1:nSize, :) = lam_k(1:nSize, :) - computeJumpWithSigma(params.diagSigmaSqs{id(iI)}, -2*tanTmp);
%             + reshape(matSigma * tanTmp(:), size(tanTmp)) * params.wi(id(iI)); % jump, here energy/(2sigma^2)
    else
        lam_k(1:nSize, :) = lam_k(1:nSize, :) + tanTmp * 2.0 / params.sigmaSqr * params.wi(id(iI));   % jump
    end
end
lam1{nt+1} = lam_k(1:nSize, :);
lam2{nt+1} = lam_k(nSize+1:end, :);
end


% function: a(X, lam) = ( lam2*X2'*X2; -lam1 + X2(lam2'*X1 + X1'*lam2)
function Y = f(X, lam)

nSizeX = floor(size(X, 1)/2);
X1 = X(1:nSizeX, :);
X2 = X(nSizeX+1:end, :);

nSizeL = floor(size(lam, 1)/2);
lam1 = lam(1:nSizeL, :);
lam2 = lam(nSizeL+1:end, :);

Y1 = lam2 * ( X2' * X2 );
Y2 = -lam1 + X2 * ( lam2'*X1 + X1'*lam2 );
Y = [Y1; Y2];
end
