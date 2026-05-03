%==========================================================================
%                     IMPORT PRIOR SPECIFICATION
%==========================================================================

%   1: BETA(mean,stdd)
%   2: GAMMA(mean,stdd)
%   3: NORMAL(mean,stdd)
%   4: INVGAMMA(s^2,nu)
%   5: UNIFORM(a,b)
%   0: no prior

prior = [3,   mean(YY(:,1)),          0.0001,        0,    mean(YY(:,1));... 
                        3,              0.97,         0.001,        0,           0.97;... 
                        3,              -0.985,        0.05,         0,           0.99;... 
                        5,               0,            0.2,        0,           0.99;... 
                        5,               0,            0.2,        0,           0.99;... 
                        5,               0,              4,        0,           0.99;... 
                        5,               0,              4,        1,          sigma2_1_fix];

                   

pshape   = prior(:,1);  
pmean    = prior(:,2); 
pstdd    = prior(:,3);
pmask    = prior(:,4); 
pfix     = prior(:,5); 

% Fix non-volatility parameters
pmaskinv = 1 - pmask;
pshape   = pshape.*pmaskinv;


sigscale = [    0.0006,         0,         0,         0,         0,         0,         0;...
                     0,    0.0010,         0,         0,         0,         0,         0;...
                     0,         0,    0.0010,         0,         0,         0,         0;...
                     0,         0,         0,    0.0001,         0,         0,         0;...
                     0,         0,         0,         0,    0.0001,         0,         0;...
                     0,         0,         0,         0,         0,    0.1384,         0;...
                     0,         0,         0,         0,         0,         0,    0.0002];




