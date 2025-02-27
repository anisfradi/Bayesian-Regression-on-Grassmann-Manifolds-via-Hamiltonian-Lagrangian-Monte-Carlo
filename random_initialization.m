
function startValue = random_initialization(shapes)

shapeMean = grfmean(shapes, 1e-5);
nSize = size(shapeMean);
zrnd = normrnd(shapeMean, 0.005, nSize);
startValue.X1 = zrnd * (zrnd' * zrnd)^(-0.5);
zrnd = normrnd(0, 0.01, nSize);
startValue.X2 = (eye(nSize(1)) - startValue.X1 * startValue.X1') * zrnd;

end
