function [loglh, z_t] = generate_xt_sv(yy_monthly, yy_quarterly, s_t, param_macro_MH, param_macro_gibbs,indexQuarter)

[Ystar,H0,H1,RR,F0_t,F1,Q_t,A_select,Ystar_m, Ystar_q] = get_coefficients_sv(yy_monthly, yy_quarterly, s_t, param_macro_MH, param_macro_gibbs);


A_last = A_select.A_last;
A_NotLast = A_select.A_NotLast;

[Tstar, nVars] = size(Ystar);


% Storage
mdim = size(F1,1);   
At_mat = zeros(Tstar,mdim);
At_pred  = zeros(Tstar,mdim);
Pt_mat = zeros(Tstar,mdim^2);
At_draw = zeros(Tstar,mdim);

% Initialize
loglh = 0;  
Pt = dlyap(F1, mean(Q_t,3));   


At = mean(F0_t,2);

 for t = 1:Tstar   
       
     if indexQuarter(t)==1 % end of the quarter
         
         A_t = A_last;
         H0_A = A_t*H0;
         H1_A = A_t*H1;
         RR_A = A_t*RR*A_t';
         nVars = size(H0_A,1);
         
          y_t = [Ystar_m(t,:)'; Ystar_q(t,:)'];  
          
     elseif indexQuarter(t)==0 % Not end of the quarter
         A_t = A_NotLast; 

         H0_A = A_t*H0;
         H1_A = A_t*H1;
         RR_A = A_t*RR*A_t';
         
         nVars = size(H0_A,1);
          y_t = Ystar_m(t,:)';

     end

           
        % See if we have missing Data
         indexData = ~isnan(y_t);
         nVars_t = sum(indexData);
         AdjM = eye(nVars);
         AdjM(~indexData,:) =[];
         
         H0_M = AdjM*H0_A;
         H1_M  = AdjM*H1_A;
         RR_M  = AdjM*RR_A*AdjM';
         y_t = y_t(indexData);
      
         
        At1  = At;
        Pt1  = Pt;
        
        F0 = F0_t(:,t);
        Q = Q_t(:,:,t);
       
        % Forecasting   
        alphahat = F0 + F1 * At1 ;
        Phat     = F1 * Pt1 * F1' + Q;
        Phat     = 0.5*(Phat+Phat');
      
        yhat = H0_M + H1_M*alphahat;
        nut  = y_t - yhat;
       
        Ft = H1_M*Phat*H1_M' + RR_M;  
        Ft = 0.5*(Ft+Ft');

        invFt = Ft\eye(nVars_t);
        
        loglh = loglh - 0.5*nVars_t*log(2*pi) -0.5*log(det(Ft))-0.5*nut'*invFt*nut;
        loglh = real(loglh);
        
        Phat_h1 = Phat*H1_M';
        
        % Updating 
        At = alphahat + (Phat_h1)*invFt*nut;
        Pt = Phat - (Phat_h1)*invFt*(Phat_h1)';
        At_mat(t,:)  = At';
        Pt_mat(t,:)  = reshape(Pt,1,mdim^2);
        At_pred(t,:) = alphahat';
        
        [u, s, ~] = svd(reshape(Pt,mdim,mdim));
        Pchol = u*sqrt(s);
        At_draw(t,:)   = At' + (Pchol*randn(mdim,1))';

 end 

% common component
Zt_draw = [flip(At_draw(2:3,2));At_draw(2:end,1)];  
Zt_pred = [ flip(At_pred(2:3,2));  At_pred(2:end,1)];
Zt_mat = [ flip(At_mat(2:3,2)); At_mat(2:end,1)];


z_t = [Zt_draw, Zt_mat, Zt_pred];
 
        
end

