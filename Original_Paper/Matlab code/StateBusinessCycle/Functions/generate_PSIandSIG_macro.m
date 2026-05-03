  function [ psi_macro,  SIG2_i] = generate_PSIandSIG_macro(y,x_t,gamma_i, SIG2_i, R0_V, T0_V, V0_, D0_, selectVar, nVars)
    
     if selectVar ~= nVars
          
          Ystar = y - gamma_i*x_t;
          Xstar = Ystar(1:end-1); 
          Ystar = Ystar(2:end);
          
          Tstar = length(Ystar);
           
      elseif selectVar == nVars 
          
           Ystar= y(4:end) - gamma_i(1)*x_t(4:end)   - gamma_i(2)*x_t(3:end-1) ...
                              - gamma_i(3)*x_t(2:end-2) - gamma_i(4)*x_t(1:end-3);

          
          Xstar = Ystar(4:end-1);
          Ystar = Ystar(5:end); 
          
          Tstar = length(Ystar);
          
      end
      

        V = (R0_V + (SIG2_i)^(-1)*(Xstar'*Xstar))\(eye(1));
        PSI =  V*(R0_V*T0_V + SIG2_i^(-1)*Xstar'*Ystar);
        C = chol(V);
        
        psi_macro = PSI + C*randn(1,1); 
        
        
         e_mat = Ystar - Xstar*psi_macro;
         nn = Tstar + V0_;
         d = D0_ + e_mat'*e_mat;
         
         temp = chol(d\eye(1),'lower')*randn(1,nn);
         SIG2_i = (temp*temp')\eye(1);
        

    end
  
