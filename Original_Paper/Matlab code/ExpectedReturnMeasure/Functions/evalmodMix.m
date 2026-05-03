 function [loglh_tot, At_draw_tot, At_mat_tot,At_pred_tot, Kgain, modelInfo_1, modelInfo_2] = evalmodMix(para, YY,pi_t, indexMinimize)
 
 % specify parameters
     mu_l = para(1);   rho_l = para(2);   corr_s = para(3);  
     phi_1 = para(4);  phi_2 = para(5);   h = para(6);  sigma2_1 = para(7); 
 
 % parameters in state 1
    para_1 = [mu_l; rho_l; corr_s; phi_1; sigma2_1];
    
 % parameters in state 2   
    para_2 = [mu_l; rho_l; corr_s; phi_2; sigma2_1*(1+h)]; 
   
  [loglh_1, At_draw_1, At_mat_1, Kg_mat_1, At_pred_1] = evalmod(para_1, YY, indexMinimize);
  [loglh_2, At_draw_2, At_mat_2, Kg_mat_2, At_pred_2] = evalmod(para_2, YY, indexMinimize);

  loglh_tot = (1 - pi_t).*loglh_1 +  pi_t.*loglh_2;

  % prediction
  At_draw_tot = (1 - pi_t).*At_draw_1 +  pi_t.*At_draw_2;
  At_mat_tot = (1 - pi_t).*At_mat_1 +  pi_t.*At_mat_2; 
  At_pred_tot = (1 - pi_t).*At_pred_1 +  pi_t.*At_pred_2;  

  Kgain = [Kg_mat_1(1,1),Kg_mat_2(1,1)];
  
  modelInfo_1 = [loglh_1, At_draw_1(:,2), At_mat_1(:,2), At_pred_1(:,2)];
  modelInfo_2 = [loglh_2, At_draw_2(:,2), At_mat_2(:,2), At_pred_2(:,2)];


 end  
