
function [phi_cc, paramMU, Sigma2_0_cc, h_cc] =generate_MU_PHI_sv(x_t, STT,param_macro_MH, R0_,T0_,R0_M, T0_M, D0_, V0_)



    % generate PHI
        paramMU = param_macro_MH.paramMU;
            mu_0 = paramMU(1);
            mu_1 = paramMU(2);
        Sigma2_0_cc = param_macro_MH.Sigma2_0_cc;
        h_cc = param_macro_MH.h_cc;
        
    
        T = length(x_t);
    
        mu_t = mu_0 + mu_1*STT;
        sigma2_t = Sigma2_0_cc*(1+h_cc*STT);
       
  % generate PHI      
        Ystar  = x_t - mu_t;
        Xstar = [Ystar(4:end-1,1)];
        Ystar = Ystar(5:end,1);
        sigma_t = sigma2_t(5:end,1).^0.5;
        
        Xstar = Xstar./repmat(sigma_t,[1,size(Xstar,2)]);
        Ystar = Ystar./sigma_t;
        
        V = (R0_ + (Xstar'*Xstar))\(eye(1));
        PHI =  V*(R0_*T0_ + Xstar'*Ystar);
        C = chol(V);
    
        PHI_G = PHI + C'*randn(1,1);
        phi_cc = PHI_G;
   % generate MU_0 and MU_1
   
      Ystar = x_t(5:end,1) - PHI_G(1)*x_t(4:end-1,1);
      Xstar = [ones(T-4,1), STT(5:end,1) - PHI_G(1)*STT(4:end-1,1)];
      
      Ystar = Ystar./sigma_t;
      Xstar = Xstar./repmat(sigma_t,[1,size(Xstar,2)]);
  
      
        V = (R0_M + (Xstar'*Xstar))\(eye(2));
        MU =  V*(R0_M*T0_M + Xstar'*Ystar);
        C = chol(V);
        
         accept = 0;
         while accept == 0

             MU_G = MU + C'*randn(2,1);

             % only accept if mu_1>0
             if MU_G(2)>0
                 accept =1;
             end
         end
       
         MU_G(1) = MU_G(1)/(1 -PHI_G(1));
         
         paramMU = MU_G;
                mu_0 = paramMU(1);
                mu_1 = paramMU(2);
         
  % Generate Sigma2_0
  
  
      mu_t = mu_0 + mu_1*STT;
      Ystar  = x_t - mu_t;
      
      e_mat = Ystar(5:end,1) - PHI_G(1)*Ystar(4:end-1,1);
      tempDenom = (1+ h_cc*STT(5:end)).^(0.5);
      
       e_mat = e_mat./tempDenom;
            
      Tstar = length(e_mat);

      nn = Tstar + V0_;
     
      d = D0_ + e_mat'*e_mat;
      

      temp = chol(d\eye(1),'lower')*randn(1,nn); % compute draw from inv(PHI).
      Sigma2_0_cc = (temp*temp')\eye(1);


  
  % generate h

      mu_t = mu_0 + mu_1*STT;
      Ystar  = x_t - mu_t;
      
      e_mat = Ystar(5:end,1) - PHI_G(1)*Ystar(4:end-1,1);
  
      tempDenom = Sigma2_0_cc.^(0.5);
      
      e_mat = e_mat./tempDenom;
      e_mat = e_mat(logical(STT(5:end)));
      
      
      Tstar = length(e_mat);


      nn = Tstar + V0_;
     
      d = D0_ + e_mat'*e_mat;
      
      accept=0; 
        
        
    while accept==0
            
       temp = chol(d\eye(1),'lower')*randn(1,nn); % compute draw from inv(PHI).
       h_hat = (temp*temp')\eye(1);
       
     
              if   h_hat >  2/3
                     accept = 1;
              end 

    end
               
      h_cc= h_hat -1; 
      h_cc = 0; % set to zero if equal macro vol shocks across regimes
    % normalize
      Sigma2_0_cc = 1;
    
    
% output

          
        
     
end  
        
        
