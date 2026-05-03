
%-------------------------------------------------------------------------  
% Process Macro Variables
% y_t = gamma x_t + e_t
% e_t = psi_1 e_{t-1}  + epsilon_t 
% var(epsilon_t)=SIG_i;  
%
% Process Common component
% x_t = mu_{s_t} + phi ( x_{t-1} - mu_{s_{t-1}}) + phi ( x_{t-2} - mu_{s_{t-2}})  + omega_{t+1}
%
% Var( omega_{t+1}) = Sigma^2_0 (1+h_1 s_t)
%  normalize Sigma^2_0  = 1;
%-------------------------------------------------------------------------- 

s_t = ~logical(NBER_rec_index);


yy =  yy_monthly;
x_t = nanmean(yy_monthly,2)./std(nanmean(yy_monthly,2));



gamma_macro_m = zeros(N_m+3,1); 
e_t = zeros(T,N_m);

 for i=1:N_m-1
     aux = fitlm(x_t,yy_monthly(:,i),'intercept',false);
     gamma_macro_m(i) = table2array(aux.Coefficients(1,1));
     e_t(:,i)= aux.Residuals.Raw;
 end

 clear aux

 
 Xaux = [x_t(4:end-1), x_t(3:end-2), x_t(2:end-3), x_t(1:end-4)];
 Yaux = yy(5:end,end);
 aux = fitlm(Xaux,Yaux,'intercept',false);
 gamma_macro_last = table2array(aux.Coefficients(:,1));
 e_t(:,end)= [nan(4,1);aux.Residuals.Raw];
 gamma_macro_m(end-3:end) = gamma_macro_last;

         
psi_macro_m = zeros(1, N_m);
SIG2_i_macro_m= zeros( N_m,1);             

  for i=1:N_m

     e_select = e_t(:,i);

     indexData = ~isnan(e_select);
     e_select = e_select(indexData);
     aux = fitlm([e_select(1:end-1)],e_select(2:end ),'intercept',false);
     psi_macro_m(:,i) = table2array(aux.Coefficients(1,1));
     resid = aux.Residuals.Raw;
     SIG2_i_macro_m(i)= nanmean(resid.^2);    
  end    

% storage monthly estimates
param_macro_gibbs.gamma_macro_m = gamma_macro_m;
param_macro_gibbs.psi_macro_m = psi_macro_m;
param_macro_gibbs.SIG2_i_macro_m = SIG2_i_macro_m;


% For first N-1 Variables
gamma_macro_q = zeros(N_q,1); % three extra coef associated with the last variable
e_t = zeros(T,N_q);

 for i=1:N_q
     aux = fitlm(x_t,yy_quarterly(:,i),'intercept',false);
     gamma_macro_q(i) = table2array(aux.Coefficients(1,1));
     e_t(:,i)= aux.Residuals.Raw;
 end

 clear aux

% e_t = psi_1 e_{t-1}  + epsilon_t  

 psi_macro_q = zeros(1, N_q);
 SIG2_i_macro_q= zeros( N_q,1);             

  for i=1:N_q

     e_select = e_t(:,i);

     indexData = ~isnan(e_select);
     e_select = e_select(indexData);
     aux = fitlm([e_select(1:end-1)],e_select(2:end ),'intercept',false);
     psi_macro_q(:,i) = table2array(aux.Coefficients(1,1));
     resid = aux.Residuals.Raw;
     SIG2_i_macro_q(i)= nanmean(resid.^2);    
  end    


%--------------------------------------------------------------------------  
% Process Common component
%-------------------------------------------------------------------------- 


aux = fitlm([x_t(1:end-1) ],x_t(2:end),'intercept', false);
phi_macro = table2array(aux.Coefficients(1,1));
phi_cc =phi_macro;
paramMU   = [ -2; 2.5];
paramPHI  = phi_macro;
Sigma2_0_cc = 1;

      
%--------------------------------------------------------------------------  
% Process for transition probabilities 
% p, q
% note that state 0 is recession and state 1 is expansion
%-------------------------------------------------------------------------- 

states = [1,2];
tranmat = generate_ChangeState(s_t(5:end,1)+1,states);

A1TT = betarnd(tranmat(1,2)+U1_01_,tranmat(1,1)+U1_00_); % 0 ->1
B1TT = betarnd(tranmat(2,1)+U1_10_, tranmat(2,2)+U1_10_);% 1 ->0


paramProb =[A1TT;B1TT];  


h_cc = -0.3;
param_macro_MH.paramMU = paramMU;
param_macro_MH.Sigma2_0_cc = Sigma2_0_cc;
param_macro_MH.h_cc = h_cc;
param_macro_MH.phi_cc = phi_cc;
param_macro_MH.paramProb = paramProb;



indexNaN  = isnan(yy_monthly);
perNaN = sum(indexNaN,1)/T;
perNaNExlcude = perNaN>0.99;
indexVars =1:N_m;
indexVars = indexVars(~perNaNExlcude);

yy_monthly = yy_monthly(:,~perNaNExlcude);

SIG2_i_macro_m = SIG2_i_macro_m(~perNaNExlcude);
psi_macro_m = psi_macro_m(:,~perNaNExlcude');

indexGamma_macro = logical([~perNaNExlcude';1;1;1]);
gamma_macro_m = gamma_macro_m(indexGamma_macro);

[T, N_m] = size(yy_monthly);

% storage monthly estimates
param_macro_gibbs.gamma_macro_m = gamma_macro_m;
param_macro_gibbs.psi_macro_m = psi_macro_m';
param_macro_gibbs.SIG2_i_macro_m = SIG2_i_macro_m;


indexNaN  = isnan(yy_quarterly);
perNaN = sum(indexNaN,1)/T;
perNaNExlcude = perNaN>0.8;
indexVars =1:N_q;
indexVars = indexVars(~perNaNExlcude);

yy_quarterly = yy_quarterly(:,~perNaNExlcude);

SIG2_i_macro_q = SIG2_i_macro_q(~perNaNExlcude);
psi_macro_q = psi_macro_q(:,~perNaNExlcude');

indexGamma_macro = logical([~perNaNExlcude']);
gamma_macro_q = gamma_macro_q(indexGamma_macro);

[~, N_q] = size(yy_quarterly);

param_macro_gibbs.gamma_macro_q = gamma_macro_q;
param_macro_gibbs.psi_macro_q = psi_macro_q';
param_macro_gibbs.SIG2_i_macro_q = SIG2_i_macro_q;


