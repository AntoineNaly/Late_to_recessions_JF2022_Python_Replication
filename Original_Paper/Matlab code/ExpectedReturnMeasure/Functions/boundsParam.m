%=========================================================================
%            LOWER AND UPPER BOUND FOR PARAMETER VALUES
%=========================================================================

bmu    = [-1 1];
brho   = [-0.9999 0.9999];
bsigma = [0.00003 1];
bcorr   = [-0.9999 0.9999];
rsquared = [0 0.16];
hparam = [0, 5];


lubound  = [bmu; brho;brho;rsquared;rsquared;hparam;  bsigma];
    
 
clearvars  bmu brho bsigma bcorr
