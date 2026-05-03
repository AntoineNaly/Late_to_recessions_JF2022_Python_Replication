
clear all; close all; clc

path('Data',path);
path('Prior',path);
path('Functions',path);  



nsim     = 10000;
nburn    = 0*nsim;
cc0      = 0.0001; 
cc       =0.0005; 
counter  = 0;

load('returnData_1965_2019.mat')

riskFreeRate = marketRF;
NBERIndex = NBER_rec_index;

load('commonGrowthData_1965_2019.mat')

pi_t = pi_t_mean;
z_t = commonGrowth_mean;

pi_t(isnan(pi_t))=0;

%=========================================================================
%                      METROPOLIS-HASTINGS SAMPLER
%=========================================================================
disp('                                                                  ')
disp('   DYNAMIC FACTOR MODEL WITH CORRELATED ERRORS                    ')
disp('                                                                  ')
YY = marketReturn_excess;
T = length(YY);
sigma2_1_fix = std(YY(~logical(NBERIndex)))^2; 


%--------------------------------------------------------------------------
% Priors and parameter bounds 
%--------------------------------------------------------------------------
boundsParam
priorParam

load para.txt

npara = length(para);
para_old = para;



indexMinimize = 0;
fcn      = @(x1,x2) objfcnMixStates(x1,YY,pi_t, indexMinimize, pshape,pmean,pstdd,pmask,pmaskinv,pfix,lubound);



 [post_old, like_old, At_draw_tot_old, At_mat_tot_old,At_pred_tot_old, Kgain_old, loglh_tot_old, modelInfo_1_old, modelInfo_2_old] = fcn(para_old);  
 
 

  check =0;
  while check<1
    
    para_old = mvnrnd(para_old, cc0*sigscale,1)';
    para_old = para_old.*pmaskinv + pfix.*pmask;
    
  [post_old, like_old, At_draw_tot_old, At_mat_tot_old,At_pred_tot_old, Kgain_old, loglh_tot_old, modelInfo_1_old, modelInfo_2_old]  = fcn(para_old ); 
    
    if post_old > -1E6
        check = 1;
    end
  end   
 

%--------------------------------------------------------------------------     
%        Random Walk Metropolis Algorithm
%--------------------------------------------------------------------------

lbd = lubound;


        
    % storages 
    parasim  = zeros(nsim,npara);       % parameter draws
    likesim  = zeros(nsim,1);           % likelihood
    postsim  = zeros(nsim,1);           % posterior probability
    rej      = zeros(nsim,1);           % rej = 1: rejected
    X_sm_simul       =  zeros(T,3, nsim);
    X_up_simul     =  zeros(T,3, nsim);
    X_pred_simul     =  zeros(T,3, nsim);
    kg_sim   = zeros(nsim,2);
    logLikiMix = zeros(T, nsim);
    logLikiMod_1 = zeros(T, nsim);    
    logLikiMod_2 = zeros(T, nsim);
    X_up_mix = zeros(T, nsim);
    X_up_Mod_1 = zeros(T, nsim);    
    X_up_Mod_2 = zeros(T, nsim); 
    
    for indexSimul = 1:nsim
       %   indexSimul = 1
  

        
              % propose   
        genacc = 0;
        while genacc < 1
            para_new = mvnrnd(para_old, cc*sigscale,1)';
            para_new = para_new.*pmaskinv + pfix.*pmask;
            par1    = para_new;
            
            % check boundary conditions
            genacc  = all(par1 < lbd(:,2) & par1 > lbd(:,1));
        end
        
        
       % evaluate at the new candidate 
     [post_new, like_new, At_draw_tot_new, At_mat_tot_new,At_pred_tot_new, Kgain_new, loglh_tot_new, modelInfo_1_new, modelInfo_2_new]= fcn(para_new);            
           
        
        r = min([1 exp( post_new - post_old)]);   
        
        
        if rand > r 
           % reject proposed jump             
           rej(indexSimul) = 1;
        else
           % accept proposed jump    
            para_old    = para_new;  
           
            post_old  = post_new;
            like_old  = like_new;
            At_draw_tot_old   = At_draw_tot_new;
            At_mat_tot_old  = At_mat_tot_new;
            At_pred_tot_old = At_pred_tot_new;
            Kgain_old = Kgain_new;
            loglh_tot_old    = loglh_tot_new;
            modelInfo_1_old    = modelInfo_1_new;  
            modelInfo_2_old    = modelInfo_2_new;
        end
        
   %----------------------------------------------------------------------     
   % store simulated variables
   %----------------------------------------------------------------------  
    parasim(indexSimul,:) = para_old';
    likesim(indexSimul) = like_old;
    postsim(indexSimul) = post_old;    
    X_sm_simul(:,:,indexSimul)=At_draw_tot_old;
    X_up_simul(:,:,indexSimul)=At_mat_tot_old;
    X_pred_simul(:,:,indexSimul)=At_pred_tot_old;   
    kg_sim(indexSimul,:) = Kgain_old;
    
    logLikiMix(:,indexSimul) = loglh_tot_old;
    logLikiMod_1(:,indexSimul) = modelInfo_1_old(:,1);   
    logLikiMod_2(:,indexSimul) = modelInfo_2_old(:,1);  
   
    X_up_mix(:,indexSimul)   =  At_mat_tot_old(:,2);
    X_up_Mod_1(:,indexSimul) = modelInfo_1_old(:,3);   
    X_up_Mod_2(:,indexSimul) = modelInfo_2_old(:,3);           
  
    end 
    
    
%%


close all

%Plot expected returns
measure_expected_returns = median(squeeze(X_pred_simul(:,1,:)),2)*1200;
plot(measure_expected_returns)



