function [loglh, At_draw, At_mat, Kg_mat, At_pred] = evalmod(para, YY, indexMinimize)
    
   [T,nv] = size(YY);
 
     
    [H0,H1,RR,F0,F1,Q] = coefficients(para);

 
    mdim = size(F1,1);    
    
    At = [para(1) 0 0]';  
    Pt = dlyap(F1, Q);   


    At_pred  = zeros(T,mdim);
    Pt_mat = zeros(T,mdim^2);      
    Kg_mat = zeros(T,mdim);    
      
    loglh = nan(T,1);
    constLogh = 0.5*nv*log(2*pi); 
 
    for t = 1:T
                
        At1  = At;
        Pt1  = Pt;

        alphahat = F0 + F1 * At1 ;
        Phat     = F1 * Pt1 * F1' + Q;
        Phat     = 0.5*(Phat+Phat');
      
        yhat = H0 + H1*alphahat;
        nut  = YY(t,:)' - yhat;
       
        Ft = H1*Phat*H1' + RR;  
        Ft = 0.5*(Ft+Ft');

        invFt = Ft\eye(nv);
        
        loglh(t) = real(- constLogh -0.5*log(det(Ft))-0.5*nut'*invFt*nut);

        Phat_h1 = Phat*H1';
        
        Kgain = ((Phat*H1')*invFt);
        Kg_mat(t,:) = Kgain'; 
        
        At = alphahat + (Phat_h1)*invFt*nut;
        Pt = Phat - (Phat_h1)*invFt*(Phat_h1)';
        At_mat(t,:)  = At';
        Pt_mat(t,:)  = reshape(Pt,1,mdim^2);
        At_pred(t,:) = alphahat';

    end 
   
    if indexMinimize==1

         loglh = -loglh;
         At_draw = zeros(T,mdim);
    else 
        At_draw = zeros(T,mdim);
        Pt_draw = zeros(T,mdim^2);


        [u, s, ~] = svd(reshape(Pt_mat(T,:),mdim,mdim));
        Pchol = u*sqrt(s);
        At_draw(T,:)   = At_mat(T,:)+(Pchol*randn(mdim,1))';
        Pt_draw(T,:,:) = reshape(Pchol,1,mdim^2); 
         At_draw_2(T,:)= At_mat(T,:);

         
        for i = 1:T-1

            Att  = At_mat(T-i,:)';
            Ptt  = reshape(Pt_mat(T-i,:),mdim,mdim);

            Phat = F1 * Ptt * F1' + Q;
            Phat = 0.5*(Phat+Phat');

            inv_Phat = Phat\eye(mdim); %inv(Phat);

            nut  = At_draw(T-i+1,:)'- F1*Att - F0;
            
           

            Amean = Att + (Ptt*F1')*inv_Phat*nut;
            Pmean = Ptt - (Ptt*F1')*inv_Phat*(Ptt*F1')';   

            [um, sm, ~] = svd(Pmean);
            Pmchol = um*sqrt(sm);
            At_draw(T-i,:)   = (Amean+Pmchol*randn(mdim,1))'; 
            Pt_draw(T-i,:,:) = reshape(Pmchol,1,mdim^2);
            At_draw_2(T-i,:)= Amean';

        end
    end


     
