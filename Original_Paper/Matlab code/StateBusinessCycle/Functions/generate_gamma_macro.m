 function  gamma_macro = generate_gamma_macro(y, x_t, PSI_i, SIG2_i, R00_, T00_, R00_4, T00_4,selectVar, nVars)
         
         psi_1 = PSI_i(1);
    
         
         if selectVar ~= nVars
             
            Ystar=y(2:end)-psi_1*y(1:end-1);
            Xstar=x_t(2:end)-psi_1*x_t(1:end-1);
            
            [Tstar, nVars] = size(Xstar);
    
            % priors
            
             R0 = R00_;
             T0 = T00_;
             
         elseif selectVar == nVars
            
             
             Ystar=y(8:end)-psi_1*y(7:end-1);
         
              XSTAR_1 =  x_t(8:end,1)    -psi_1*x_t(7:end-1,1);
              XSTAR_2 =  x_t(7:end-1,1)  -psi_1*x_t(6:end-2,1);
              XSTAR_3 =  x_t(6:end-2,1)  -psi_1*x_t(5:end-3,1);
              XSTAR_4 =  x_t(5:end-3,1)  -psi_1*x_t(4:end-4,1);
          
             Xstar = [XSTAR_1, XSTAR_2, XSTAR_3, XSTAR_4]; 
  
             [Tstar, nVars] = size(Xstar);
              
             R0 = R00_4;
             T0 = T00_4;
            
         end
         
         
         
        V = (R0 + (SIG2_i)^(-1)*(Xstar'*Xstar))\(eye(nVars));
        gamma =  V*(R0*T0 + SIG2_i^(-1)*(Xstar'*Ystar));
        C = chol(V);
    
        gamma_macro = gamma + C'*randn(nVars,1);
         
 end
