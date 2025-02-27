function [Ac, Bc, Phi, Tan] = grgeo(varargin)
% GRGEO Grassmann geodesics.
%
%   [Ac, Bc, Phi] = GRGEO(A,B,t) computes an element along the 
%   geodesic that connects two points G0 and G1 on the Grassmanian 
%   manifold (represented as orthogonal (nxk) matrices A and B 
%   that span the subspaces G0 and G1).
%
%   For convenience, we actually compute a point along the geodesic
%   connecting those points in the bases S0bar = S0*U1 and S1*V1 
%   with U*S*V' = S0'*S1, s.t. Phi(0) = A and Phi(1) = B. The algorithm
%   that is used can be found (Task 3) in:
%
%   [1] K. Gallivan, A. Srivastrava, X. Liu and P. Van Dooren: "Efficient
%       Algorithms for Inferences on Grassmann Manifolds". In: IEEE 
%       Workshop on Statistical Signal Processing, 2003.
%
%   [Ac, Bc, Phi] = GRGEO(A,B,t,variant) with variant='v2' computes the 
%   same geodesic, but using the direct (less efficient) approach that is 
%   outlined in Section 2 of [1], i.e. Phi(t) = Q*exp(t*B)J (see paper 
%   for details). Default for variant = 'v1'.
%
%   [Ac, Bc, Phi] = GRGEO(A,B,t,variant,basisChoice) also allows to 
%   specify the choice of basis. In case of 'v1', we use the canonical 
%   basis of [1], i.e., Ac=A*U1, Bc=B*V1 for [U1,~,V1] = svd(A'*B). In
%   case of 'v2', we use Ac=A, Bc=B*V1*U1^T which is usefull in cases 
%   where we need to keep the first argument constant.
%
%   Example:
%
%       S0 = orth(rand(10,3));
%       S1 = orth(rand(10,3));
%       [S0bar, S1bar, Phi] = grgeo(S0, S1, 0);
%
%       assert(norm(S0bar - Phi,'fro') < eps);   
%
%       [S0bar, S1bar, Phi] = grgeo(S0, S1, 1);
%
%       assert(norm(S1bar - Phi,'fro') < eps); 
%
% Author: Roland Kwitt, 2013
% E-Mail: roland.kwitt@kitware.com

    if nargin < 3
        return
    end
         
    A = varargin{1};
    B = varargin{2};
    t = varargin{3};

    [~,k] = size(A);
       
    variant = 'v1';
    if nargin == 4
        variant = varargin{4};
    end
    
    [U1,Gamma,V1] = svd(A'*B,0);
    Delta = -1;

    basisChoice = 'v1';
    if nargin == 5
        variant = varargin{4};
        basisChoice = varargin{5};
    end
    
    switch basisChoice
        case 'v1'
            %fprintf('Basis choice is Ac=A*U1,Bc=B*V1\n');
            Ac = A*U1;
            Bc = B*V1;
        case 'v2'
            %fprintf('Basis choice is Ac=A, Bc=B*V1*U1^T\n');
            Ac = A;
            Bc = B*V1*U1';
    end
       
    Phi = -1; % default
    switch variant
        case 'v1'
            theta = acos(diag(Gamma));
            Sigma = diag(sin(theta));

            Dbar = Bc - Ac*Gamma;

            tG = scaleG(Gamma, t);
            tS = scaleS(Sigma, t);
            tO = Sigma\tS; % inv(Sigma)*tS
            
            Phi = Ac*tG + Dbar*tO;
        case 'v2'
            % orthogonal complement of Ac in SO(n)
            Qt = ocomp(Ac);
            
            % compute direction matrix, see Sect. 2, Task 2 of [1]
            QtB = Qt*Bc;
            X = +QtB(1:k, 1:k);
            Y = -QtB(k+1:end,:);
    
            [U1,U2,~,~,Sig] = gsvd(X,Y,0);
            theta = diag(asin(diag(Sig)));
            C = U2*theta*U1';
            
            Q = Qt';
            %Delta = [zeros(k,k) C'; -C zeros(n-k,n-k)];
            %J = [eye(k,k); zeros(n-k,k)];
            
            % This is the standard way to do it ...
            %Phi = Q*expm(t*Delta)*J;
            %Geo.Delta = Delta;
            %Geo.Q = Q;
            %Geo.J = J;
            %Geo.C = C;
            Tan = Q(:, k+1:end) * (-C);
            %Tan = null(A*A')*(-C);
        case 'v3'
            % This is an adaption of the algorithm in:
            %
            % E. Begelfor and W. Werman
            % "Affince Invariance Revisited"
            % CVPR 2006
            
            X = Ac;
            Y = Bc;
            
            [U,S,V] = svd(Y/(X'*Y)-X,0);
            Tan = U*diag(atan(diag(S)))*V';
            
            %x = Ac;
            %y = Bc;
            
            %ytx = y.'*x;
            %At = y.'-ytx*x.';
            %Bt = ytx\At;
            %[U, S, V] = svd(Bt.', 'econ');

            %U = U(:, 1:k);
            %S = diag(S);
            %S = S(1:k);
            %V = V(:, 1:k);
            %Tan = U*diag(atan(S))*V.';
        otherwise
            disp('variant not supported!');
            return
    end
      
    
    function S = scaleS(M, t)
        alpha = asin(diag(M));
        S = diag(sin(t*alpha));
    end


    function G = scaleG(X, t)
        alpha = acos(diag(X));
        G = diag(cos(t*alpha));
    end


    function Q = ocomp(A)
    % Orthogonal completion of A, see Sect. 2.1. of [1]
        [u,v] = size(A);
        A_1 = A(1:v,1:v);
        A_2 = A(v+1:end,:);
        Q = eye(u)-[A_1-eye(v);A_2]*pinv(eye(v)-A_1)*[A_1'-eye(v),A_2'];
        %Q = sparse(1:u, 1:u, 1) - sparse([A_1-eye(v);A_2])*sparse(pinv(eye(v)-A_1))*sparse([A_1'-eye(v),A_2']);
    end
end

