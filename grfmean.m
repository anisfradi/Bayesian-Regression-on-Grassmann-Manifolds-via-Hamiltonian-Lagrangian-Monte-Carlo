function Q = grfmean(P, tol)
% GRFMEAN computes the Frechet mean for a set of subspaces.
%
% Example:
%
% Let P be a Nx1 cell array with P{i} being the basis of subspace i. Then
% Q = grfmean(P,0.1) returns the Frechet mean Q.
%
% The algorithm is implemented based ons
%
% W. Dreisigmeyer,
% "Direct Search Algorithms over Riemannian Manifolds"
% Los Alamos National Laboratory Los Alamos, NM
%
% Author: Roland Kwitt, Yi Hong, 2014

    m = length(P);
    Q = P{1};
    n = size(Q,1);
    p = size(Q,2);
    G = grassmannfactory(n,p);
    
    while 1
        S = 0;
        for i=1:m
            S = S + G.log(Q,P{i}); % log-map
        end
        S = 1/m*S;
        c = norm(S,'fro');
        fprintf('||.||_{fro}=%.5f\n', c);
        if c < tol
            break;
        end
        Q_new = G.exp(Q,S);
        Q = Q_new;
    end  
end