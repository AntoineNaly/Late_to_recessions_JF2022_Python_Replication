function [retp1,retp2, At_draw_tot, At_mat_tot,At_pred_tot, Kgain, loglh_tot, modelInfo_1, modelInfo_2]= objfcnMixStates(para, YY,pi_t, indexMinimize, pshape,pmean,pstdd,pmask,pmaskinv,pfix,lubound)

parabd_ind1 = para > lubound(:,1);                   
parabd_ind2 = para < lubound(:,2);                
parabd_ind1 = prod( parabd_ind1 + pmask );
parabd_ind2 = prod( parabd_ind2 + pmask );


if ( parabd_ind1 > 0 ) && ( parabd_ind2 > 0 ) 

    % likelihood 
    modelpara = para.*pmaskinv + pfix.*pmask;
    [loglh_tot, At_draw_tot, At_mat_tot,At_pred_tot, Kgain,modelInfo_1, modelInfo_2] = evalmodMix(modelpara, YY,pi_t, indexMinimize);
    loglh = sum(loglh_tot);
     
    
    lnpY      = loglh;
    npara     = length(para);

    % Evaluate the Prior distribution 
    lnprio = 0;
    for i = 1:npara
    
        if pmask(i) == 1
    
            lnprio = lnprio;
    
        else
        
        if pshape(i) == 1     % BETA Prior
            
            a = (1-pmean(i))*pmean(i)^2/pstdd(i)^2 - pmean(i);
            b = a*(1/pmean(i) - 1);
            lnprio = lnprio + log( pdf('beta',para(i),a,b) );
            
        elseif pshape(i) == 2  % GAMMA PRIOR 
            b = pstdd(i)^2/pmean(i);
            a = pmean(i)/b;
            lnprio = lnprio + log( pdf('gam',para(i),a,b) );
      
        elseif pshape(i) == 3  % GAUSSIAN PRIOR 
            
            a = pmean(i);
            b = pstdd(i);
            lnprio = lnprio + log( pdf('norm',para(i),a,b) );

        elseif pshape(i) == 4  % INVGAMMA PRIOR 
            
            aux = para(i).^2;
            a = pmean(i);
            b = pstdd(i);
            lnprio = lnprio +log( aux^(-b-1)*exp(-0.5*b*a^2/aux^2) );
            
           log( aux^(-a-1)*exp(-b/aux) );

         elseif pshape(i) == 5  % uniform prior 
            
            a = pmean(i);
            b = pstdd(i);
            lnprio = lnprio + log( (para(i)-a)/(b-a) );

      end
    
        end    
    
    end

    if indexMinimize ==1  %  minimize the negative of the Posterior 
        
        retp1 = real( lnpY(1) - lnprio ); % lnpY(1) its already negative
        
    else 
        
         retp1 = real( lnpY(1) + lnprio );
          
    end

      retp2 = real( lnpY(1));

else                          
    retp1 = -1E20;
    retp2 = -1E20;
    At_draw = -1E20;
    At_mat  = -1E20;
    sGG     = -1E20;
    KG      = [0 0];
    
end

    
    
    
