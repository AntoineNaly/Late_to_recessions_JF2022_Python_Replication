function [S_T, fprob]=hamiltonfilter_xt_sv(x_t,param_macro_MH)

% Specify Parameters
paramMU = param_macro_MH.paramMU;
mu_0 = paramMU(1);
mu_1 = paramMU(2);

Sigma2_0_cc = param_macro_MH.Sigma2_0_cc;

h_cc = param_macro_MH.h_cc;
phi_cc = param_macro_MH.phi_cc;
paramProb = param_macro_MH.paramProb;

p   = 1 - paramProb(1); % @Pr[St=0/St-1=0]@ % Prob of staying in a recession
q   = 1 - paramProb(2); % @Pr[St=1/St-1=1]@ % Prob of staying in an expansion
        

% @ NUMBER OF STATES TO BE CONSIDERED@
LAG_AR = 1;
NO_ST=LAG_AR+1;
nDim =2^NO_ST;
   
% different possible states
st_mat = zeros(nDim,NO_ST);
    
%  @ S_{t-1}    S_t @
%  @    0        0  @
%  @    0        1  @
%  @    1        0  @
%  @    1        1  @ 

 j=1;
for     st0 =0:1
         for     st1 =0:1

            st_mat(j,:) = [st0, st1];
            j=j+1;

         end
end


% Compute Different Posible Values for mu and sigma

Ystar = x_t(2:end) - phi_cc*x_t(1:end-1); 
Tstar = length(Ystar);

mu_mat=mu_0*ones(size(st_mat)) + st_mat*mu_1;

Sigma2_mat  = Sigma2_0_cc*ones(size(st_mat,1),1) + st_mat(:,end)*Sigma2_0_cc*h_cc; 

% transition probability matrix        
        
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


  % Start from steady state
prob_0_aux = [pr_ss,pr_ss]';
prob_0 =prob_0_aux(:);     

   
prob_1_vec = pr_tr_vec.*prob_0;



 % start Hamilton Filter   
auxPhi = [-phi_cc; 1];
fprob = zeros(2,Tstar);


 for t =1:Tstar   

     y_t = Ystar(t);

     y_error = y_t*ones(nDim,1) - mu_mat*auxPhi;


     prob_dd = pr_tr_vec.* prob_1_vec;

     liki =(1./sqrt(2.*pi.*Sigma2_mat)).*exp(-0.5*(y_error.^2)./Sigma2_mat).*prob_dd;

     sum_liki = sum(liki);
     liki_adj=liki/sum_liki;

     prob_1=liki_adj(1:nDim/2,1)+liki_adj(nDim/2+1:nDim,1);

     aux = [prob_1, prob_1]';
     prob_1_vec = aux(:);


      fprob(:,t)=prob_1;
     
 end     
 
 fprob = fprob';
 
 %  @<<<<<<<<<<<<<<<<<<<GENERATE S_T>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>@
  

S_T = zeros(Tstar,1);

p0 = fprob(1,end);
p1 = fprob(2,end);

S_T(end,1) = bingen(p0,p1, 1);

 for it=Tstar-1:-1:1
     
       if  S_T(it+1,1) == 0
           p0  = p*fprob(it,1); 
           p1  = (1-q)*fprob(it,2); 
           
       elseif S_T(it+1,1) == 1
           
          p0  = (1-p)*fprob(it,1); 
          p1  = q*fprob(it,2);

       end
           
    
        S_T(it,1) = bingen(p0,p1, 1);
     
 end
 
end
