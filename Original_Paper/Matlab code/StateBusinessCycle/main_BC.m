
%--------------------------------------------------------------------------
%  Source:
%  Kim, C. J., & Nelson, C. R. (1999). State-space models with regime switching: 
%  classical and Gibbs-sampling approaches with applications. MIT Press Books, 1.
%--------------------------------------------------------------------------


ls=path;
path('Data',path);
path('Functions',path); 


%--------------------------------------------------------------------------
%   Load DATA
%--------------------------------------------------------------------------

load('dataMacroFinance_1950_2019_updated.mat')

yy_monthly = (y_monthly - repmat(nanmean(y_monthly),[size(y_monthly,1),1]))./repmat(nanstd(y_monthly),[size(y_monthly,1),1]);
yy_quarterly = (y_quarterly - repmat(nanmean(y_quarterly),[size(y_quarterly,1),1]))./repmat(nanstd(y_quarterly),[size(y_monthly,1),1]);


[T, N_m] = size(yy_monthly);
[~, N_q] = size(yy_quarterly);

indexQuarter = zeros(T,1);
indexQuarter(3:3:T) =1;


%--------------------------------------------------------------------------
% Specify Prior Parameters
%--------------------------------------------------------------------------


specifyPriorsGibbsMacro;


%--------------------------------------------------------------------------
% INITIAL VALUES FOR THE GIBBS SAMPLER
%-------------------------------------------------------------------------- 


initialValuesMacro;

%--------------------------------------------------------------------------
% THE GIBBS SAMPLER
%-------------8-------------------------------------------------------------

N0 = 15000;       % NUMBER OF DRAWS TO LEAVE OUT
MM = 25000;      % NUMBER OF DRAWS


CAPN = N0 + MM;  % TOTAL NUMBER OF DRAWS@   
 for indexSimul =1 :CAPN 
     %--------------------------------------------------------------------------
% DRAW COMMON GROWTH COMPONENT
%-------------------------------------------------------------------------- 
 
 [loglh, z_t] = generate_xt_sv(yy_monthly, yy_quarterly, s_t, param_macro_MH, param_macro_gibbs,indexQuarter);
  x_t   =  z_t(:,1);

%--------------------------------------------------------------------------
% DRAW STATES St
%-------------------------------------------------------------------------- 

[S_T, FLT_PR]=hamiltonfilter_xt_sv(x_t,param_macro_MH);
STT = [0;0;0; S_T];
s_t = STT;

%--------------------------------------------------------------------------
% DRAW  MACRO PARAM
%-------------------------------------------------------------------------- 
indexMonthly =1;
param_macro_gibbs_aux.gamma_macro = param_macro_gibbs.gamma_macro_m;
param_macro_gibbs_aux.psi_macro = param_macro_gibbs.psi_macro_m;
param_macro_gibbs_aux.SIG2_i_macro = param_macro_gibbs.SIG2_i_macro_m;
[gamma_macro, psi_macro, SIG2_i_macro] = gibbSamplingMacro(yy_monthly(3:end,:), x_t, param_macro_gibbs_aux, priorsMacroGibbs, indexMonthly);
param_macro_gibbs.gamma_macro_m=gamma_macro;
param_macro_gibbs.psi_macro_m=psi_macro;
param_macro_gibbs.SIG2_i_macro_m=SIG2_i_macro;

gamma_macro_m = gamma_macro;
psi_macro_m = psi_macro;
SIG2_i_macro_m = SIG2_i_macro;

clear  param_macro_gibbs_aux

indexMonthly = 0;
 param_macro_gibbs_aux.gamma_macro = param_macro_gibbs.gamma_macro_q;
 param_macro_gibbs_aux.psi_macro = param_macro_gibbs.psi_macro_q;
 param_macro_gibbs_aux.SIG2_i_macro = param_macro_gibbs.SIG2_i_macro_q;

[gamma_macro, psi_macro, SIG2_i_macro] = gibbSamplingMacro(yy_quarterly(3:end,:), x_t, param_macro_gibbs_aux, priorsMacroGibbs, indexMonthly);
param_macro_gibbs.gamma_macro_q=gamma_macro;
param_macro_gibbs.psi_macro_q=psi_macro;
param_macro_gibbs.SIG2_i_macro_q=SIG2_i_macro;

gamma_macro_qrt = gamma_macro;
psi_macro_qrt = psi_macro;
SIG2_i_macro_qrt = SIG2_i_macro;

gamma_macro = [gamma_macro_m;gamma_macro_qrt];
psi_macro = [psi_macro_m;psi_macro_qrt];  
SIG2_i_macro = [SIG2_i_macro_m;SIG2_i_macro_qrt];   

clear  param_macro_gibbs_aux

%--------------------------------------------------------------------------
% DRAW COMMON MACRO PARAM
%-------------------------------------------------------------------------- 

D0_MU = 0;
V0_MU = 0;

[phi_cc, paramMU, Sigma2_0_cc,h_cc] =generate_MU_PHI_sv(x_t, STT(3:end),param_macro_MH, R0_,T0_,R0_M, T0_M, D0_, V0_);

% update Parameters
param_macro_MH.paramMU=paramMU;
param_macro_MH.Sigma2_0_cc=Sigma2_0_cc;
param_macro_MH.h_cc=h_cc;
param_macro_MH.phi_cc  =phi_cc;


%--------------------------------------------------------------------------
% DRAW p and q
%-------------------------------------------------------------------------- 

states = [1,2];
tranmat = generate_ChangeState(STT(5:T-2,1)+1,states);


A1TT = betarnd(tranmat(1,2)+U1_01_,tranmat(1,1)+U1_00_);
B1TT = betarnd(tranmat(2,1)+U1_10_, tranmat(2,2)+U1_10_);

q = 1 - B1TT;
p = 1 - A1TT;
paramProb =[A1TT; B1TT]; 

param_macro_MH.paramProb = paramProb;


%--------------------------------------------------------------------------
% Store parameter draws
%-------------------------------------------------------------------------- 


  if indexSimul > N0

   State_specific(:,:,indexSimul-N0) = e_t;
   State_common(:,:,indexSimul-N0) = z_t;
   probState(:,indexSimul-N0) = [p, q];
   Gamma_tot(:,indexSimul-N0) =  gamma_macro;
   PSI_tot(:,indexSimul-N0) =  psi_macro;
   SIG_tot(:,indexSimul-N0) =  SIG2_i_macro;
   MU_G_tot(:,indexSimul-N0) = paramMU;
   PHI_tot(:,indexSimul-N0)   = phi_cc;
   Prob_tot(:,:,indexSimul-N0) = FLT_PR;
   Sigma_X(:,indexSimul-N0) = [Sigma2_0_cc, h_cc];

  end
     
 end
 
 

%% Plots 

burnIn = floor(indexSimul*.0)+1;
recessionProbability = [nan(3,1); median(Prob_tot(:,1,burnIn:end),3)];
commonGrowth = [nan(2,1); median(State_common(:,1,burnIn:end),3)];


% plot recession probability
plot(recessionProbability(:,1))

% plot common growth components
plot(commonGrowth)

p1=5;
p2=50;
p3= 95;

% compute CI  
 % this variables can be estimated via gibbs sampling
gamma_macro = prctile(Gamma_tot,[p1, p2, p3],2);
psi_macro = prctile(PSI_tot,[p1, p2, p3], 2);
SIG2_i_macro = prctile(SIG_tot,[p1, p2, p3],2);

    
  
 % metopolis-hastings to estimate this variables in the full model
 paramMU_macro = prctile(MU_G_tot,[p1, p2, p3],2);
 Sigma_Macro =  prctile(Sigma_X,[p1, p2, p3],2);
 phi_macro = prctile(PHI_tot,[p1, p2, p3],2);
 paramProb_macro =  prctile(probState,[p1, p2, p3],2);


pq = mean(probState,2);

p = pq(1);
q = pq(2);      
pr_tr =[p  (1-q);...
     (1-p)  q];     

pr_tr_vec = pr_tr(:);

% steady-state probability matrix   

A = [(eye(2)-pr_tr); ones(1,2)];
EN=[0;0;1];
pr_ss=  (A'*A)\(A'*EN);  
if isnan(pr_ss)==1
pr_ss=[0.5;0.5];
end   


