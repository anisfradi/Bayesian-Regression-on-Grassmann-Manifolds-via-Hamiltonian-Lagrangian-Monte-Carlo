function That = ptEdelman(Y,H,T,t)
% PTEDELMAN Parallel transport according to Edelman et al. (Eq. 2.68)
%
% PTEDELMAN(Y,H,T,t) transports the tangent vector T at Y along the
% geodesic starting at Y in the direction H.
%
% Author: Roland Kwitt, 2013

    % economy size SVD
    [U,S,V] = svd(H,0);
    
    Sigmat = diag(S)*t;
    M = [diag(-sin(Sigmat)); diag(cos(Sigmat))];
  
    part0 =[Y*V U]*(M*(U'*T));    
    
    part1 = T - U*(U'*T);
    That = part0 + part1;
end