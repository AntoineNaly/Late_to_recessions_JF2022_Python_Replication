%--------------------------------------------------------------------------
% Specify Prior Parameters
%--------------------------------------------------------------------------

% Specify Prior Parameters for the Macro Variables

% Prior for the transition probabilties        
     U1_01_=1;  U1_00_=1;   
     U1_10_=1;  U1_11_=1;    
     
%    @FOR PSI1 OF INDIVIDUAL COMPONENT@
     R0_V=eye(1)/4;
     T0_V=[0];
     
%   @FOR SIG2_V OF INDIVIDUAL COMPONENT@
     D0_ =  0;
     V0_ =  0; 
     
%  @FOR phi_i OF INDIVIDUAL COMPONENT@
     R00_=1/4;
     T00_=0;  
     
    
 % @FOR LAMDA'S OF EMPLOYMENT EQN@
     R00_4=eye(4)/1;
     T00_4=[0;0;0;0];  
     
%---------------------------------------------------------
%---------------------------------------------------------
  % @FOR PHI  OF COMMON COMPONENT@
     R0_=eye(1)/4;
     T0_=0;     
     
  %  @For MU0, MU1 OF COMMON COMPONENT@
     R0_M=eye(2)/2;
     T0_M=[0;0]; 
%---------------------------------------------------------
%---------------------------------------------------------
     
     
     priorsMacroGibbs.R0_ = R0_;
     priorsMacroGibbs.T0_ = T0_;
     priorsMacroGibbs.R0_M = R0_M;
     priorsMacroGibbs.T0_M = T0_M;
     priorsMacroGibbs.R0_V = R0_V;
     priorsMacroGibbs.T0_V = T0_V;
     priorsMacroGibbs.D0_ = D0_;
     priorsMacroGibbs.V0_ = V0_;
     priorsMacroGibbs.R00_ = R00_;
     priorsMacroGibbs.T00_ = T00_;
     priorsMacroGibbs.R00_4 = R00_4;
     priorsMacroGibbs.T00_4 = T00_4;
