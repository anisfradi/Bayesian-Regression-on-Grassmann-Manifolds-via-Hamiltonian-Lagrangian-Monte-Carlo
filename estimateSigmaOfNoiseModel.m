% 
%  estimate the sigmas of the noise model 
%  currently, only consider the diagnonal entries
%
%  Author: Yi Hong, yihong@cs.unc.edu
%  Date: 05/18/2016
%

function [sigmaSq] = estimateSigmaOfNoiseModel(X1s, Ys)

tanVec = cell(length(Ys), 1);
for iI = 1:length(tanVec)
    [~, ~, ~, vTmp] = grgeo(X1s{iI}, Ys{iI}, 1, 'v3', 'v2');
    tanVec{iI} = vTmp;
end

sigmaSq = cell(length(Ys), 1);
[n, p] = size(X1s{1});


% for iI = 1:length(sigmaSq)
%     sigmaSq{iI}=100*ones([n,p]);
% end


for iI = 1:length(sigmaSq)
    curTanVec = cell(length(Ys), 1);
    for iJ = 1:length(Ys)
        if iJ == iI
            curTanVec{iJ} = tanVec{iJ};
        else
            % do parallel transport
            [~, ~, ~, vTmp] = grgeo(X1s{iJ}, X1s{iI}, 1, 'v3', 'v2');
        	curTanVec{iJ} = ptEdelman(X1s{iJ}, vTmp, tanVec{iJ}, 1);
        end
    end
    
    tmpSigma = zeros(n, p^2);
    for iJ = 1:length(curTanVec)
        for iK = 1:n
            tmpC = curTanVec{iJ}(iK, :)' * curTanVec{iJ}(iK, :);
            tmpSigma(iK, :) = tmpSigma(iK, :) + (tmpC(:))';
        end
    end
    sigmaSq{iI} = tmpSigma ./ length(curTanVec);
end

end


