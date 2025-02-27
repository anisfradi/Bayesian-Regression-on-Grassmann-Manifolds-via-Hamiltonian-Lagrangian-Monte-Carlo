function [Tend, Yend, Ydot] = integrateForwardWithODE45( Y0, Y0dot, tSpan )

[s1, s2] = size(Y0);
len = s1*s2;
YInit = zeros( 2*len, 1 );
YInit(1:len) = reshape(Y0, [], 1);
YInit(len+1:end) = reshape(Y0dot, [], 1);

options = odeset('RelTol', 1e-3, 'AbsTol', 1e-4);
[Tend, YTmp] = ode45( @(t, Y)vdp(t, Y, s1, s2), tSpan, YInit, options );

Yend = cell(length(Tend), 1);
Ydot = cell(length(Tend), 1);
for iI = 1:length(Tend)
    Yend{iI} = reshape(YTmp(iI, 1:len), s1, s2);
    Ydot{iI} = reshape(YTmp(iI, len+1:end), s1, s2);
end



    function dY = vdp( t, Y, s1, s2 )
        len = s1*s2;
        X1 = reshape( Y(1:len), s1, s2 );
        X2 = reshape( Y(len+1:end), s1, s2 );
        dY = zeros( 2*len, 1 );
        dY(1:len) = Y(len+1:end);
        dY(len+1:end) = reshape( -X1*(X2'*X2), [], 1 );
    end
end

