% 
% add some noises to the data
% 
% Author: Yi Hong, yihong@cs.unc.edu
% Date: 09/26/2014
%

function [YsWithNoise] = addNoiseToData(Ys, delta)

if delta < 1e-7
    YsWithNoise = Ys;
else
    rng('shuffle');
    YsWithNoise = cell(length(Ys), 1);
    for iI = 1:length(Ys)
        vel = rand(size(Ys{iI})) - 0.5;
        vel = vel - Ys{iI} * (Ys{iI}' * vel);
        [~, YsTmp] = integrateForwardWithODE45(Ys{iI}, vel, [0 delta/2 delta]);
        YsWithNoise{iI} = YsTmp{end};
    end
end