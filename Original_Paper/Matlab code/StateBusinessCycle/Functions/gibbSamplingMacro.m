
function [gamma_macro, psi_macro, SIG2_i_macro] = gibbSamplingMacro(yy, x_t, param_macro_gibbs, priorsMacroGibbs,indexMonthly)


nVars = size(yy,2);

gamma_macro = param_macro_gibbs.gamma_macro;
SIG2_i_macro = param_macro_gibbs.SIG2_i_macro;
psi_macro  = param_macro_gibbs.psi_macro;
 
R0_V = priorsMacroGibbs.R0_V;
T0_V = priorsMacroGibbs.T0_V;
V0_ = priorsMacroGibbs.V0_;
D0_ = priorsMacroGibbs.D0_;
R00_ = priorsMacroGibbs.R00_;
T00_ = priorsMacroGibbs.T00_;
R00_4 = priorsMacroGibbs.R00_4;
T00_4 = priorsMacroGibbs.T00_4;
   

   if  indexMonthly ==1
 
   for selectVar = 1:nVars
       yy_select = yy(:,selectVar);
       
        indexData = ~isnan(yy_select);
        yy_select = yy_select(indexData);
        x_t_select = x_t(indexData);  
        
        
        SIG2_i  = SIG2_i_macro(selectVar);

    if selectVar == nVars
        
        gamma_i = gamma_macro(nVars:end);
        
    else
        gamma_i  = gamma_macro(selectVar);
    end

       [psi_macro_temp,  SIG2_i_temp] = generate_PSIandSIG_macro(yy_select,x_t_select,gamma_i, SIG2_i, R0_V, T0_V, V0_, D0_, selectVar, nVars);

        SIG2_i_macro(selectVar) = SIG2_i_temp;
        psi_macro(selectVar) = psi_macro_temp;
    
   end  
   

       gamma_macro_0 =[];
       for selectVar = 1:nVars
           
           
        yy_select = yy(:,selectVar);
           
        indexData = ~isnan(yy_select);
        yy_select = yy_select(indexData);
        x_t_select = x_t(indexData);  
           
        
       SIG2_i  = SIG2_i_macro(selectVar);
       PSI_i =  psi_macro(selectVar);
       
      
       gamma_macro_temp = generate_gamma_macro(yy_select, x_t_select, PSI_i, SIG2_i, R00_, T00_, R00_4, T00_4,selectVar, nVars);
       
       gamma_macro_0 = [gamma_macro_0;gamma_macro_temp];
         
       end
       
       gamma_macro = gamma_macro_0;
       
   else

       
    for selectVar = 1:nVars
       yy_select = yy(:,selectVar);
       
        indexData = ~isnan(yy_select);
        yy_select = yy_select(indexData);
        x_t_select = x_t(indexData);  
        
        
        SIG2_i  = SIG2_i_macro(selectVar);

        gamma_i  = gamma_macro(selectVar);

       [psi_macro_temp,  SIG2_i_temp] = generate_PSIandSIG_macro(yy_select,x_t_select,gamma_i, SIG2_i, R0_V, T0_V, V0_, D0_, 1, nVars);

        SIG2_i_macro(selectVar) = SIG2_i_temp;
        psi_macro(selectVar) = psi_macro_temp;
    
    end  
   
       gamma_macro_0 =[];
              
       for selectVar = 1:nVars
           
           
        yy_select = yy(:,selectVar);
           
        indexData = ~isnan(yy_select);
        yy_select = yy_select(indexData);
        x_t_select = x_t(indexData);  
           
        
       SIG2_i  = SIG2_i_macro(selectVar);
       PSI_i =  psi_macro(selectVar);
       
      
       gamma_macro_temp = generate_gamma_macro(yy_select, x_t_select, PSI_i, SIG2_i, R00_, T00_, R00_4, T00_4,1, nVars);
       
       gamma_macro_0 = [gamma_macro_0;gamma_macro_temp];
         
       end
       
       gamma_macro = gamma_macro_0;
       
       

   end
    
end
