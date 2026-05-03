
function [H0,H1,RR,F0,F1,Q] = coefficients(para)

% coefficients
    mu_l    = para(1);
    rho_l   = para(2);
    corr_s  = para(3);
    phi     = para(4);
    sigma2  = para(5);


sigma2_l  = (1-rho_l^2)*phi*sigma2;
sigma2_r  = (1-phi)*sigma2;
sigma2_lr = corr_s*sigma2*((1-rho_l^2)*phi*(1-phi))^(0.5);

H0 = 0;
H1 = [ 0,  1,  1 ];
RR  = 0;

% State-Transition Matrices
F0    = [mu_l*(1-rho_l);... 
         0;...  
         0];
   
F1    = [rho_l,   0,   0;...
           1,   0,   0;...
           0,   0,   0];

Q     = [sigma2_l,  0,   sigma2_lr;...
            0,  0,     0;...
          sigma2_lr,  0, sigma2_r];  
      
end
