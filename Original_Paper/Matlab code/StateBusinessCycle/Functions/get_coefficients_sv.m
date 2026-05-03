

function [Ystar,H0,H1,RR,F0_t,F1,Q_t, A_select, Ystar_m, Ystar_q] = get_coefficients_sv(yy_monthly, yy_quarterly, s_t, param_macro_MH, param_macro_gibbs)
 
% Compute State-space Matrices
% state-space form
% YY(t) = H0 + H1*S(t) + e(t)
% S(t)  = F0 + F1*S(t-1) + v(t)
% Var(e(t)) = RR;
% Var(v(t)) = Q;

[T, N_m]= size(yy_monthly);

[~, N_q]= size(yy_quarterly);

 
 tau = 5; % sum of monthly macro growth to generate qrt macro growth
%-------------------------------------------------------------------------------------------------------------
%-------------------------------------------------------------------------------------------------------------
% Specify parameters

% Monthly data
gamma_macro_m = param_macro_gibbs.gamma_macro_m;
SIG2_i_macro_m = param_macro_gibbs.SIG2_i_macro_m;
psi_macro_m= param_macro_gibbs.psi_macro_m;

% Quarterly data
gamma_macro_q = param_macro_gibbs.gamma_macro_q;
SIG2_i_macro_q = param_macro_gibbs.SIG2_i_macro_q;
psi_macro_q= param_macro_gibbs.psi_macro_q;

% Estimated using Metropolis Hastings
% Common Component
paramMU = param_macro_MH.paramMU;
Sigma2_0_cc = param_macro_MH.Sigma2_0_cc;
h_cc = param_macro_MH.h_cc;
phi_cc = param_macro_MH.phi_cc;


%-------------------------------------------------------------------------------------------------------------
%-------------------------------------------------------------------------------------------------------------
% Coefficients Associated with the measurement equation
% YY(t) =A_{t}(H0 + H1*S(t) + e(t))

% if t is the last month of the quarter
tau_aux = [1, 2, 3, 4, 5]/3;

mDim  = N_m + N_q*tau+6;
nStates_Y = N_m + N_q*tau;
A_last = [eye(N_m), zeros(N_m, N_q*tau);...
          zeros(N_q,N_m), kron(eye(N_q),tau_aux )];


A_NotLast= [eye(N_m), zeros(N_m, N_q*tau)];

A_select.A_last = A_last;
A_select.A_NotLast = A_NotLast;

% ~~~~~~~~~~ H0 ~~~~~~~~~~

H0 = zeros(nStates_Y,1);
        
  
% ~~~~~~~~~~ H1 ~~~~~~~~~~ 

% H1 associated with Macro variables monthly
gamma_f = gamma_macro_m(1:end-4);  %  gamma associated with the first N-1 Observations
gamma_l = gamma_macro_m(end-3:end); %  gamma associated with the last observation

psi_1_f = psi_macro_m(1:end-1);    %  Psi associated with the first N-1 Observations
psi_1_l = psi_macro_m(end)';  %  Psi associated with the last observation


% build auxiliar variables for the N observation for the macro variables
gamma_0_l =   gamma_l(1);
gamma_1_l =   gamma_l(2) - psi_1_l*gamma_l(1);
gamma_2_l =   gamma_l(3) - psi_1_l*gamma_l(2);
gamma_3_l =   gamma_l(4) - psi_1_l*gamma_l(3);
gamma_4_l =              - psi_1_l*gamma_l(4);


H1_f = [gamma_f, -gamma_f.*psi_1_f];
H1_l = [gamma_0_l, gamma_1_l, gamma_2_l, gamma_3_l, gamma_4_l];


H1_macro_m_aux1 = [H1_f, zeros(N_m-1, 4);...
             H1_l,0];        
H1_macro_m_aux2 = eye(N_m);

H1_macro_m = [H1_macro_m_aux1,H1_macro_m_aux2,zeros(N_m,N_q*length(tau_aux))];

% H1 associated with Macro variables quarterly   
gamma_q = gamma_macro_q(1:end);  
psi_1_q = psi_macro_q(1:end);   

gamma_psi = -gamma_q.*psi_1_q;

auxVar = [gamma_q, gamma_psi];

H1_macro_q_1 =[];

for index_q =1:N_q
    aux1 = auxVar(index_q,:);

     mat_aux1= zeros(length(tau_aux),6);
     for ii=1:length(tau_aux)
         mat_aux1(ii,ii)= aux1(1);
     end
     for jj=1:5
         mat_aux1(jj,jj+1)= aux1(2);
     end   
      H1_macro_q_1 =[H1_macro_q_1;mat_aux1];
end

      H1_macro_q_2 = zeros(length(tau_aux)*N_q,N_m);    

      H1_macro_q_3 = kron(eye(N_q),eye(length(tau_aux)));

H1_macro_q = [   H1_macro_q_1,  H1_macro_q_2,  H1_macro_q_3]; 

H1 = [  H1_macro_m;   H1_macro_q]; 

    
% ~~~~~~~~~~ RR ~~~~~~~~~~ 
RR = zeros(nStates_Y); 



%-------------------------------------------------------------------------------------------------------------
% Coefficients Associated with the measurement equation
% S(t)  = F0 + F1*S(t-1) + v(t)
%-------------------------------------------------------------------------------------------------------------

nDim = size(H1,2);

% ~~~~~~~~~~ F0 ~~~~~~~~~~

mu_0 = paramMU(1);
mu_1 = paramMU(2);

mu_aux = mu_0 + mu_1.*s_t;

mu_t = mu_aux(2:end) - phi_cc(1)*mu_aux(1:end-1);

F0_t = [mu_t';...
   zeros(mDim-1, length(mu_t))];

   
% ~~~~~~~~~~ F1 ~~~~~~~~~~    

F1 = zeros(nDim);
F1(1,1) = phi_cc;

F1(2:tau+1,1:tau) = eye(tau);
    
% ~~~~~~~~~~ Q ~~~~~~~~~~   

Q_t = zeros(nDim,nDim, T); 
Sigma2_t = Sigma2_0_cc*(1+h_cc*s_t);

 Q_t(1,1,:) = Sigma2_t;

 jjj = 1;
 for kk=7:6+N_m

      Q_t(kk,kk,:) = repmat(SIG2_i_macro_m(jjj),[T,1]);
      jjj=jjj+1;
 end

   jjj = 1;
 for kk=6+N_m+1:5:6+(N_m+N_q*5)
      Q_t(kk,kk,:) = repmat(SIG2_i_macro_q(jjj),[T,1]);
      jjj=jjj+1;
 end

    
%-------------------------------------------------------------------------------------------------------------
%-------------------------------------------------------------------------------------------------------------
% Adjust Data and matrices
psi_1_m =  psi_macro_m;
psi_1_q =  psi_macro_q;


yy_star_m = yy_monthly(2:end,:) - repmat(psi_1_m',[T-1,1]).*yy_monthly(1:end-1,:);
yy_star_q = yy_quarterly(4:end,:)- repmat(psi_1_q',[T-3,1]).*yy_quarterly(1:end-3,:);

Ystar_m =yy_star_m(3:end,:);
Ystar_q =yy_star_q;   

Ystar = [Ystar_m, Ystar_q];

Tstar = length(Ystar);
Q_t = Q_t(:,:,end-Tstar+1:end);
F0_t= F0_t(:,end-Tstar+1:end);

end
