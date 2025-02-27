% 
% compute distance with sigma and log-map
%

function distSq = computeGeodesciDistanceWithSigma(sigmaSqs, logMap)

[n, p] = size(logMap);
assert(size(sigmaSqs, 1) == n && size(sigmaSqs, 2) == p^2);

vecDist = zeros(size(logMap));
for iI = 1:n
    vecDist(iI, :) = sqrt(inv(reshape(sigmaSqs(iI, :), p, p))) * (logMap(iI, :))';
end
distSq = norm(vecDist(:)).^2;

end