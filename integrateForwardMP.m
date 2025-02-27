%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
%  integrate forward in time
%
%  Author: Yi Hong
%  Date: 08-30-2013
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [X1, X2, Y_RK] = integrateForwardMP( X10, X20, nt, h )

X1_k = X10;
X2_k = X20;
X_k = [X1_k; X2_k];
X1 = {};
X2 = {};
nSize = size(X10, 1);
Y_RK = cell( nt, 4 );

% 4-th Runge-Kutta
for it = 1:nt
	X1{it} = X_k(1:nSize, :);
    X2{it} = X_k(nSize+1:end, :);
	Y1_k = X_k;
    f1 = f(Y1_k);
    Y2_k = X_k + h/2.0 * f1; %f(Y1_k);
    f2 = f(Y2_k);
    Y3_k = X_k + h/2.0 * f2; %f(Y2_k);
    f3 = f(Y3_k);
    Y4_k = X_k + h * f3; %f(Y3_k);
    X_k = X_k + h * ( f1/6.0 + f2/3.0 + f3/3.0 + f(Y4_k)/6.0 );
    %X_k = X_k + h * ( f(Y1_k)/6.0 + f(Y2_k)/3.0 + f(Y3_k)/3.0 + f(Y4_k)/6.0 );
    Y_RK{it, 1} = Y1_k;
    Y_RK{it, 2} = Y2_k;
    Y_RK{it, 3} = Y3_k;
    Y_RK{it, 4} = Y4_k;
end
X1{nt+1} = X_k(1:nSize, :);
X2{nt+1} = X_k(nSize+1:end, :);
end

% function: f(X) = (X2; -X1(X2'X2))
function Y = f(X)
nSize = floor(size(X, 1)/2);
X1 = X(1:nSize, :);
X2 = X(nSize+1:end, :);
Y1 = X2;
Y2 = -X1*(X2'*X2);
Y = [Y1;Y2];
end
