//+------------------------------------------------------------------+
//|                                             ApexPredator_ESN.mq5 |
//|                                                 William Nicholas |
//|                                             https://www.mql5.com |
//+------------------------------------------------------------------+
//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+


#include <ESNstochastic.mqh>
#include <ESN.mqh>
#include <Math\Stat\Normal.mqh>
#include <NeuroplascticNeuralNetwork.mqh>



#include <DoubleFilo.mqh>

#include<Trade\Trade.mqh>
#include <Trade\AccountInfo.mqh>
#include <Trade\PositionInfo.mqh>
#include <Trade\DealInfo.mqh>


DoubleFilo Filo(32);



CAccountInfo AccInfo;
CTrade trading;  
CPositionInfo m_position;
enum ENUM_LEARNING_STYLE{Reward, Punishment};

input double Lot                      = 1.0;
double  pip                           = _Point*10;
input ENUM_TIMEFRAMES Time_Frame      = PERIOD_M30;
input ENUM_LEARNING_STYLE LEARNING_STYLE = Reward;
input int SamplesOfGBM                = 10000;
input int Size_of_Esn                 = 1;
input bool continuation               = true;
input int Neurons_                    = 500;
input double Neuroplasticity_         = 0.001;
input double spartsity_               = 0.001;
input double noise_                   = 0.001;
input double spectral_radius_         = 2.8;
input double Hurst_Upper_Range        = .55;
input double Hurst_Lower_Range        = .45;
input double KappaThreshold           = 1;
input double Scale_Factor             = 1;
input int HurstAndKappaDepth          = 10;
input int MagicNumberBUY              = 66691; 
input int MagicNumberSELL             = 66692;
input bool Verbose_                   = false;
input int percent_of_acc_risk         = 1;
input int dxThreshold                 = 100;
input int NumberOfDreams              = 5;
input int Neurons                     = 200;
input double Neuroplasticity          =.001;
input double beta_1 = .9;
input double beta_2 = .999;
input double fitting = .001;
input double Learning_rate = .001;
input int    Max_iterations= 50000;
input int    StartHour =18;
input int    EndHour = 22;

input bool   Verbose = false;
input int RSI_param = 14; 
input int STO_param_1 =14;
input int STO_param_2 =3;
input int STO_param_3 =3;
input int Fast_EMA_param = 8;
input int Slow_EMA_param = 20;
input bool pred_Verbose = true;

double  NumberOfLosses = 0;
double  NumberOfWins = 0;
int min_tracker = -1;
double NN_Neuroplastic_b = 1.1 ;
double NN_Neuroplastic_a =1;
double ESN_Neuroplastic_b = 1.01 ;
double ESN_Neuroplastic_a =1;
double Lot_Martingale = 0;
int loadedErrorEsn =0;
double pred_prev = (SymbolInfoDouble(_Symbol,SYMBOL_BID) +SymbolInfoDouble(_Symbol,SYMBOL_ASK) )/2;

ENUM_TIMEFRAMES TFNN[9]={ PERIOD_M1,PERIOD_M5,PERIOD_M10,PERIOD_M15,PERIOD_M30,PERIOD_H1,PERIOD_H4};

NeuroplascticNeuralNetwork  *NNSlist[7]; 


matrix<double> shift_(Size_of_Esn,Size_of_Esn);
matrix<double> shift_b =shift_.Fill(0.0);
matrix<double> scaling_(Size_of_Esn,Size_of_Esn);
matrix<double> scaling_b =scaling_.Fill(1.0);
bool teacher_scaling_on_b = false;
bool input_scaling_on_b = false;
bool teacher_forcing_on_b = true;
bool teacher_shift_on_b = false;
bool input_shift_on_b = false;
double scaler_C =1;
double shift_C =0;

ESNstochastic ESN_vol(Size_of_Esn,Size_of_Esn,Neurons_,spectral_radius_,spartsity_,noise_,shift_b,input_shift_on_b,scaling_b,input_scaling_on_b,teacher_forcing_on_b,scaler_C,teacher_scaling_on_b,shift_C,teacher_shift_on_b,STO_Identity,STO_InverseIdentity,Verbose_,1,1.01);


ESNstochastic ESN_mu(Size_of_Esn,Size_of_Esn,Neurons_,spectral_radius_,spartsity_,noise_,shift_b,input_shift_on_b,scaling_b,input_scaling_on_b,teacher_forcing_on_b,scaler_C,teacher_scaling_on_b,shift_C,teacher_shift_on_b,STO_Identity,STO_InverseIdentity,Verbose_,1,1.01);


ESN ESN_Error_vol(Size_of_Esn,Size_of_Esn,Neurons_,spectral_radius_,spartsity_,noise_,shift_b,input_shift_on_b,scaling_b,input_scaling_on_b,teacher_forcing_on_b,scaler_C,teacher_scaling_on_b,shift_C,teacher_shift_on_b,Identity,InverseIdentity,Verbose_);


ESN ESN_Error_mu(Size_of_Esn,Size_of_Esn,Neurons_,spectral_radius_,spartsity_,noise_,shift_b,input_shift_on_b,scaling_b,input_scaling_on_b,teacher_forcing_on_b,scaler_C,teacher_scaling_on_b,shift_C,teacher_shift_on_b,Identity,InverseIdentity,Verbose_);




//+------------------------------------------------------------------+
double RSI_Norm(double rsi){

   double res = .5;

   if(rsi >70){ res = 0;}
   if(rsi <30){ res = 1;}
   return res;

}

double STO_Norm(double sto){

   double res = .5;

   if(sto >80){ res = 0;}
   if(sto <20){ res = 1;}
   return res;

}



double EMA_Norm(double fast, double slow){

   double res = .5;

   if(fast >slow){ res = 0;}
   if(fast <slow){ res = 1;}
   return res;

}
double iPrice(string symbol , ENUM_TIMEFRAMES TF , int shifts){



   return ( iClose(symbol,TF,shifts) +iOpen(symbol,TF,shifts))/2;

}

double iPricedx(string symbol , ENUM_TIMEFRAMES TF , int shifts){



   return iPrice( symbol,TF,shifts ) -iPrice( symbol,TF,shifts+1 ) ;

}



double Vol(int shifts,int depth_, ENUM_TIMEFRAMES TF){


      double sum =0;
      
     
      for( int i=0;i<depth_;i++){
            
      
            sum = sum  +   MathLog( iPrice(_Symbol,TF,shifts+0+i)/iPrice(_Symbol,TF,shifts+1+i))*MathLog( iPrice(_Symbol,TF,shifts+0+i)/iPrice(_Symbol,TF,shifts+1+i));
      
      }
      
      sum = (1/double(depth_))*sum;
      
     

      return MathSqrt(sum);
}

double Mu(int shifts,int depth_, ENUM_TIMEFRAMES TF){


      double sum =0;
      
      for( int i=0;i<depth_;i++){
      
      
            sum = sum+ MathLog( iPrice(_Symbol,TF,shifts+0+i)/iPrice(_Symbol,TF,shifts+1+i));
      
      }
      
      sum = (1/double(depth_))*sum;


      return sum;
}



double Scale_Vol(double val){


return MathExp(MathExp(val))-1; 

}


double unScale_Vol(double val){


return MathLog(MathLog(val+1)); 

}

double Scale_Mu(double val){


return MathExp(MathExp(val))-1; 

}

double unScale_Mu(double val){


return MathLog(MathLog(val+1)); 

}



double Sum_m(int n, ENUM_TIMEFRAMES TF){



   double sum =0;
   for(int i=0 ; i < n ; i++ ){
   
       
      sum = sum + iPrice(_Symbol,TF,i);
   
   }
   
   return sum/double(n);

}


double Z_Sum_t(int n,int t, ENUM_TIMEFRAMES TF){


   
   
   double sum =0;
   for(int i=0 ; i < t ; i++ ){
   
      sum = sum + iPrice(_Symbol,TF,i) -Sum_m(n,TF);
   
   }
  
   return sum/double(n);

}





double std(int n,int t, ENUM_TIMEFRAMES TF){


   
   
   double sum =0;
   for(int i=0 ; i < t ; i++ ){
   
      sum = sum + pow(iPrice(_Symbol,TF,i) -Sum_m(n,TF),2);
   
   }
  
   return MathSqrt( sum/double(t));

}







double RS_n(int n, ENUM_TIMEFRAMES TF){

         
         
         
         double R_max = 0;
         double R_min = 0;
         for(int t =1 ; t <= n; t++ ){
         
               double Zt =Z_Sum_t(n,t,TF);
               if( t == 0){
               R_max = Zt;
               R_min = Zt;
               }else{
                    
                    if( R_max < Zt){
                    
                    R_max = Zt;
                    }
                    
                    if( R_min > Zt){
                    
                    R_min = Zt;
                    }
                
                     
               
               
               
               }
         
         
         
         
         }
         
         
         double Rt = R_max - R_min;   



         double St = std(n,n,TF);
         
         if( Rt ==0 || St ==0){ return 0;}

      return Rt/St;
}




double Hurst(int depth_power_of_2, ENUM_TIMEFRAMES TF){



      int NumberOfpowersOf2 =  HurstAndKappaDepth;

      matrix Var(NumberOfpowersOf2-2,2);
      vector EofRtdivSt(NumberOfpowersOf2-2);
      
      for( int i = 2 ;  i<NumberOfpowersOf2; i++ ){
      
            int n = int(pow(2,i));
            
            double ln_n = MathLog(double(n));
            double ln_c = 1;
            double RS = RS_n(n,TF);
            
            Var[i-2][0]= ln_n;    
            Var[i-2][1]= ln_c;    
            EofRtdivSt[i-2]=RS;
            
      }

      
      vector res =Var.LstSq(EofRtdivSt);
      double H = res[1];
      return H;




}

  
double GeometricBrownianMotion(double sigma, double mu,double time, double curr_price, int samples, int steps_into_future){



      double pred =0;
      
      
      for( int sample = 0 ; sample < samples ; sample++){
      
      
      
               double price = curr_price;
      
      
      
               for( int steps = 0; steps <steps_into_future ; steps++){
               
                 int error;
                 
                  price = price*MathExp( (mu - (sigma*sigma)/ 2) * time  + sigma*MathRandomNormal(0,MathSqrt(time),error ));
               
               
               }
      
               
               pred = pred + price;
      
      
      
      }

      return pred/samples;


}   
 

double Sum_n(int n, ENUM_TIMEFRAMES TF){



   double sum =0;
   for(int i=0 ; i < n ; i++ ){
   
      if(iPrice(_Symbol,TF,0+i)==0 || iPrice(_Symbol,TF,1+i)==0){
         
         return 0;
      
      } 
      sum = sum +   MathAbs( MathLog( iPrice(_Symbol,TF,0+i)/iPrice(_Symbol,TF,1+i)));
   
   }
  
   return sum;

}
 
 
double E_Sum_n(int n, ENUM_TIMEFRAMES TF){



   double sum =0;
   for(int i=0 ; i < n ; i++ ){
   
       
      sum = sum + Sum_n( i,  TF);
   
   }
  
   return sum/double(n);

} 
 
 
double MeanAbsoluteDeviation(int n, ENUM_TIMEFRAMES TF){



      double sum = 0; 
      
      for( int i=1; i <=n ; i++){
       
    
         sum = sum + MathAbs(  Sum_n(i,TF)  -  E_Sum_n(i,TF));
      
      
      }

      return sum/double(n);

      
}


double Kappa(int n_0 , int n, ENUM_TIMEFRAMES TF){

   double Sum_n_n = Sum_n(n,TF);
   double Sum_n_0 = Sum_n(n_0,TF);
   double Mad_n_n = MeanAbsoluteDeviation(n,TF);
   double Mad_n_0 = MeanAbsoluteDeviation(n_0,TF);
   
   if( Sum_n_0 == 0 || Sum_n_n == 0 ||Mad_n_0 ==0 || Mad_n_n ==0){
   
   return 1; 
   }

   return 2- (MathLog(double(Sum_n_n))-MathLog(double(Sum_n_0)))/MathLog(Mad_n_n/Mad_n_0);
}





double Sum_n_error(int n, double &vals[]){



   double sum =0;
   for(int i=0 ; i < n ; i++ ){
         
      if(vals[n-i]==0 || vals[n-i-1]==0){
         
         return 0;
      
      }
       
      sum = sum + MathAbs( MathLog( vals[n-i]/vals[n-i-1])); 
   
   }
  
   return sum;

}
 
 
double E_Sum_n_error(int n, double &vals[]){



   double sum =0;
   for(int i=0 ; i < n ; i++ ){
   
       
      sum = sum + Sum_n_error( i,  vals);
   
   }
  
   return sum/double(n);

} 
 
 
double MeanAbsoluteDeviation_error(int n,double &vals[]){



      double sum = 0; 
      
      for( int i=1; i <=n ; i++){
       
    
         sum = sum + MathAbs(  Sum_n_error(i,vals)  -  E_Sum_n_error(i,vals));
      
      
      }

      return sum/double(n);

      
}


double Kappa_error(int n_0 , int n, double &vals[]){


   double Sum_n_n = Sum_n_error(n,vals);
   double Sum_n_0 = Sum_n_error(n_0,vals);
   double Mad_n_n = MeanAbsoluteDeviation_error(n,vals);
   double Mad_n_0 = MeanAbsoluteDeviation_error(n_0,vals);
   
   if( Sum_n_0 == 0 || Sum_n_n == 0 ||Mad_n_0 ==0 || Mad_n_n ==0 || (Mad_n_n==Mad_n_0) ){
   
   return 1; 
   }


   return 2- (MathLog(double(Sum_n_n))-MathLog(double(Sum_n_0)))/MathLog(Mad_n_n/Mad_n_0);
}

double calcLots(double slPoints,double kelly , double RiskPercent ){
   double risk = AccountInfoDouble(ACCOUNT_BALANCE) * RiskPercent / 100 * kelly;
   double ticksize = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_SIZE);
   double tickvalue = SymbolInfoDouble(_Symbol,SYMBOL_TRADE_TICK_VALUE);
   double lotstep = SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_STEP);
   double moneyPerLotstep = slPoints / ticksize * tickvalue * lotstep;
   double lots = NormalizeDouble(MathFloor(risk / moneyPerLotstep) * lotstep,2);
      
      
      if( lots < .01){
      lots = .01;
      
      }


   lots = MathMin(NormalizeDouble(lots,2),SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MAX));
   lots = MathMax(NormalizeDouble(lots,2),SymbolInfoDouble(_Symbol,SYMBOL_VOLUME_MIN));      
   
   

      
   return lots;
}



int OnInit()
  {
//---
   
   for( int p = 0 ; p<7 ; p++){
   
   
      NeuroplascticNeuralNetwork *NNS = new NeuroplascticNeuralNetwork(3,35,Neurons,1,fitting,Learning_rate,Verbose,beta_1,beta_2,Max_iterations,1,1.5);
   
      NNSlist[p] = NNS;
      
   
   
   
   }
   

    
//---
   return(INIT_SUCCEEDED);
  }
//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
//---
  for( int p = 6 ; p>=0 ; p--){
   
   
            delete NNSlist[p] ;
            }
            
            
         
     
      
   
  }
//+------------------------------------------------------------------+
//| Expert tick function                                             |
//+------------------------------------------------------------------+
void OnTick()
  {
//---

   
  
   matrix trainL[7];
   matrix predL[7];

   matrix corrL[7];

   int Data_depth = 5; 

 
   matrix corr_bias(7,1);
    
    
    
    
  
   
   
   
   for( int Gtime = 0 ; Gtime <7 ; Gtime++){
      
        
         
                     
                     
                     
                     
                     
                     
                     
            
                  
                     ENUM_TIMEFRAMES TheTimeFrame = TFNN[Gtime];
                     int TheTimeFrameInMin = PeriodSeconds(TheTimeFrame)/60;
                  
                  
                     vector in_train_RSI(7*Data_depth);
                     vector in_train_STO(7*Data_depth);
                     vector in_train_EMA(7*Data_depth);
                    
                     
                     for(int time= 0; time < 7; time++){
                     
                     
                        
                        
                        int RSI_handle = iRSI(_Symbol,TFNN[time],RSI_param,PRICE_CLOSE);
                        double RSI_1_Array[];
                        ArraySetAsSeries(RSI_1_Array,true);
                        CopyBuffer(RSI_handle,0,0,245,RSI_1_Array);
                        
                        
                        
                        
                        
                        ENUM_TIMEFRAMES TheTimeFrameInLoop = TFNN[time];
                        int TheTimeFrameInMinInLoop = PeriodSeconds(TheTimeFrameInLoop)/60;
                        
                        for( int place = 0 ; place < Data_depth ; place ++){
                        
                           in_train_RSI[time*Data_depth + (place)] =RSI_1_Array[int(MathCeil(double(TheTimeFrameInMin)/double(TheTimeFrameInMinInLoop)))+place]/100;
                            
                        } 
                        
                        
                       
                        int STO_handle = iStochastic(_Symbol,TFNN[time],STO_param_1,STO_param_2,STO_param_3,MODE_EMA,STO_LOWHIGH);
                        double STO_1_Array[];
                        ArraySetAsSeries(STO_1_Array,true);
                        CopyBuffer(STO_handle,0,0,245,STO_1_Array);
                        
                        
                       
                        
                        for( int place = 0 ; place < Data_depth ; place ++){
                           in_train_STO[time*Data_depth + (place)] =STO_1_Array[int(MathCeil(double(TheTimeFrameInMin)/double(TheTimeFrameInMinInLoop)))+place]/100; 
                           
                        }
                      
                      
                        int MA_fast_handle = iMA(_Symbol,TFNN[time],Fast_EMA_param,0,MODE_EMA,PRICE_CLOSE);
                        double MA_fast_1_Array[];
                        ArraySetAsSeries(MA_fast_1_Array,true);
                        CopyBuffer(MA_fast_handle,0,0,245,MA_fast_1_Array);
                        
                        
                        
                        int MA_slow_handle = iMA(_Symbol,TFNN[time],Slow_EMA_param,0,MODE_EMA,PRICE_CLOSE);
                        double MA_slow_1_Array[];
                        ArraySetAsSeries(MA_slow_1_Array,true);
                        CopyBuffer(MA_slow_handle,0,0,245,MA_slow_1_Array);
                        
                       
                        for( int place = 0 ; place < Data_depth ; place++){
                           
                           
                           double ma_fast =MA_fast_1_Array[int(MathCeil(double(TheTimeFrameInMin)/double(TheTimeFrameInMinInLoop))) +place];
                           double ma_slow =MA_slow_1_Array[int(MathCeil(double(TheTimeFrameInMin)/double(TheTimeFrameInMinInLoop)))+place];
                           
                           
                           in_train_EMA[time*Data_depth + (place)] = EMA_Norm(ma_fast,ma_slow); 
                           
                           }
                     
                         }
                   

                   
                   matrix in_train(3,7*Data_depth);
                   in_train.Row(in_train_RSI,0);
                   in_train.Row(in_train_STO,1);
                   in_train.Row(in_train_EMA,2);
                   trainL[Gtime] = in_train;
                    
      
               
         
         
      
      
      
      
      
                     
                     
                     
                     double RSI_Sum = 0;
                     double STO_Sum = 0;
                     double EMA_Sum = 0;
                  
                        
                     vector in_pred_RSI(7*Data_depth);
                     vector in_pred_STO(7*Data_depth);
                     vector in_pred_EMA(7*Data_depth);
                     
                     for(int time= 0; time < 7; time++){
                        int RSI_handle = iRSI(_Symbol,TFNN[time],RSI_param,PRICE_CLOSE);
                        double RSI_1_Array[];
                        ArraySetAsSeries(RSI_1_Array,true);
                        CopyBuffer(RSI_handle,0,0,245,RSI_1_Array);
                        for( int place = 0 ; place < Data_depth ; place ++){
                              in_pred_RSI[time*Data_depth + (place)] =RSI_1_Array[0 +place]/100;
                              RSI_Sum = in_pred_RSI[time*Data_depth + (place)] + RSI_Sum ;
                        }
                        
                        
                        int STO_handle = iStochastic(_Symbol,TFNN[time],STO_param_1,STO_param_2,STO_param_3,MODE_EMA,STO_LOWHIGH);
                        double STO_1_Array[];
                        ArraySetAsSeries(STO_1_Array,true);
                        CopyBuffer(STO_handle,0,0,245,STO_1_Array);
                        for( int place = 0 ; place < Data_depth ; place ++){
                           in_pred_STO[time*Data_depth + (place)] =STO_1_Array[0+place]/100;
                            STO_Sum = in_pred_STO[time*Data_depth + (place)] + STO_Sum ;
                           }
                        
                        
                        
                        int MA_fast_handle = iMA(_Symbol,TFNN[time],Fast_EMA_param,0,MODE_EMA,PRICE_CLOSE);
                        double MA_fast_1_Array[];
                        ArraySetAsSeries(MA_fast_1_Array,true);
                        CopyBuffer(MA_fast_handle,0,0,245,MA_fast_1_Array);
                        
                        
                        
                        int MA_slow_handle = iMA(_Symbol,TFNN[time],Slow_EMA_param,0,MODE_EMA,PRICE_CLOSE);
                        double MA_slow_1_Array[];
                        ArraySetAsSeries(MA_slow_1_Array,true);
                        CopyBuffer(MA_slow_handle,0,0,245,MA_slow_1_Array);
                        
                         
                              for( int place = 0 ; place < Data_depth ; place++){
                        double ma_fast =MA_fast_1_Array[0+place];
                        double ma_slow =MA_slow_1_Array[0+place];
                        
                        in_pred_EMA[time*Data_depth + (place)]= EMA_Norm(ma_fast,ma_slow); 
                       EMA_Sum = 1 + EMA_Norm(ma_fast,ma_slow) ;
                        }
                         
                     }
                  
                     
                     double temp_val_pred= (RSI_Sum/(7*Data_depth) + STO_Sum/(7*Data_depth) + EMA_Sum/(7*Data_depth))/(3);
                     //Print(temp_val_pred, " pred");
                     
                     
                         
                  
                    
                   
                    
                   
                   
                    
                  
                   
                   
                  matrix in_pred(3,7*Data_depth);
         
                  in_pred.Row(in_pred_RSI,0);
                  in_pred.Row(in_pred_STO,1);
                  in_pred.Row(in_pred_EMA,2);
                  
                  predL[Gtime] = in_pred;
                  
                  
                  
                  
           matrix corr(3,1);
      
         
         
         
         double corr_value = .5;
         {
            ENUM_TIMEFRAMES TheTimeFrame = TFNN[Gtime];
            int TheTimeFrameInMin = PeriodSeconds(TheTimeFrame)/60;
            double curr_price = (SymbolInfoDouble(_Symbol,SYMBOL_ASK)+ SymbolInfoDouble(_Symbol,SYMBOL_BID))/2 ;
            double prev_price = (iClose(_Symbol,PERIOD_M1,TheTimeFrameInMin)+iOpen(_Symbol,PERIOD_M1,TheTimeFrameInMin))/2;
            if( curr_price- prev_price >0){  corr_value = 1; }
            if( curr_price- prev_price <0){  corr_value = 0; }
             
         }
         corr= corr.Fill(corr_value);
        
        
         
         
         
         
        
         corrL[Gtime] = corr;
               
                   
                     
            }
        
         
           
     
        
 int tf_in_seconds = PeriodSeconds(Time_Frame);
     int tf_in_mins = tf_in_seconds/60;
     
     matrix<double> vol_(Size_of_Esn,Size_of_Esn);
     
      matrix<double> mu_(Size_of_Esn,Size_of_Esn);
      
           matrix<double> prev_vol_(Size_of_Esn,Size_of_Esn);
     
      matrix<double> prev_mu_(Size_of_Esn,Size_of_Esn);
     
     matrix<double> ones_(Size_of_Esn,Size_of_Esn);
     matrix<double> ones = ones_.Fill(1.0);
      
      
     ENUM_TIMEFRAMES Vol_Incriment_Period = PERIOD_M1;
     double in_Vol =  Scale_Vol( 100*Vol(0,tf_in_mins,Vol_Incriment_Period));
     double in_Mu  =  Scale_Mu(100*Mu(0,tf_in_mins,Vol_Incriment_Period ));
   
   double prev_in_Vol =   Scale_Vol( 100*Vol(1,tf_in_mins,Vol_Incriment_Period));
     double prev_in_Mu  =  Scale_Mu(100*Mu(1,tf_in_mins,Vol_Incriment_Period ));
   
   
   
      matrix<double> vol =vol_.Fill(in_Vol);
      matrix<double> mu = mu_.Fill(in_Mu);
     
      matrix<double> prev_vol =prev_vol_.Fill(prev_in_Vol);
      matrix<double> prev_mu = prev_mu_.Fill(prev_in_Mu);
      
      
     
 
   

 
      
    
     
     
     
     
   
   
   
   
   
   datetime    tm=TimeCurrent();
   MqlDateTime stm;
   TimeToStruct(tm,stm);
   
   
   
   bool TradeTracker = (m_position.SelectByMagic(_Symbol,MagicNumberBUY) == false) &&(m_position.SelectByMagic(_Symbol,MagicNumberSELL) == false);
    
   if(m_position.SelectByMagic(_Symbol,MagicNumberBUY) == true){
   
   
         ulong ticket = m_position.Ticket();
         
         
         if(  ulong(TimeCurrent()) -  ulong(m_position.Time()) > ulong(tf_in_seconds)){
         
         
          if(m_position.Profit() > 0){
         
         NumberOfLosses=0;
         NumberOfWins = NumberOfWins+1;
         
         }else{
         
         NumberOfLosses = NumberOfLosses+1;
         NumberOfWins = NumberOfWins-1;
         }
         trading.PositionClose(ticket,-1);
         }
   
   }
   
     
   if(m_position.SelectByMagic(_Symbol,MagicNumberSELL) == true){
   
         ulong ticket = m_position.Ticket();
       
         
         if(  ulong(TimeCurrent()) -  ulong(m_position.Time()) > ulong(tf_in_seconds) ){
         
         
           
         if(m_position.Profit() > 0){
         
         NumberOfLosses=0;
         NumberOfWins = NumberOfWins+1;
         }else{
         
         NumberOfLosses = NumberOfLosses+1;
         NumberOfWins = NumberOfWins-1;
         }
         trading.PositionClose(ticket,-1);
         }
   
   }
    
    Lot_Martingale =NormalizeDouble( pow(Scale_Factor,NumberOfLosses),2);
   
   if (  (stm.day_of_week==1 ||stm.day_of_week==2 || stm.day_of_week==3 || stm.day_of_week==4 )&&  (stm.hour >=StartHour  && stm.hour <= EndHour  && stm.min != min_tracker && stm.min%tf_in_mins==0  )  ){
         min_tracker = stm.min;
         double predictionNN =0;
         double LearnStyle = 0;
         if(LEARNING_STYLE == Reward){         
         
            LearnStyle= 1;
         
         }
         
         if(LEARNING_STYLE == Punishment){         
         
            LearnStyle= -1;
         
         }
         
         if(NumberOfWins>4){
         
         NumberOfWins =4;
         }
         if(NumberOfWins<0){
         
         NumberOfWins =0;
         }
         
         NN_Neuroplastic_b = 1.01 +  LearnStyle*Neuroplasticity*NumberOfWins;
         NN_Neuroplastic_a =1;
         
         
         for( int Btime= 0; Btime<(7) ; Btime++){
     
            for( int i =0 ; i <=NumberOfDreams ; i++){
            
               
               NNSlist[Btime].Train(trainL[Btime],corrL[Btime],NN_Neuroplastic_a,NN_Neuroplastic_b);
               matrix pred_M = NNSlist[Btime].Prediction(predL[Btime]);
               double pred_temp = pred_M.Mean();
               if( i == NumberOfDreams){
                  predictionNN = predictionNN + pred_temp;
               }
            }
         }
         
         predictionNN = predictionNN/(7);
        
         
     
     
      ESN_Neuroplastic_b = 1.01 + LearnStyle*Neuroplasticity_*NumberOfWins;
      ESN_Neuroplastic_a =1;
     
     
      ESN_vol.Fit(prev_vol,vol,ESN_Neuroplastic_a,ESN_Neuroplastic_b); 
      ESN_mu.Fit(prev_mu,mu,ESN_Neuroplastic_a,ESN_Neuroplastic_b);
   
     
      
     
     
     matrix<double> pred_vol = ESN_vol.predict(vol,continuation);
     matrix<double> pred_mu = ESN_mu.predict(mu,continuation);
    
    int power_of_2 = int(pow(2,HurstAndKappaDepth)); 
    
    
      
   

        
          
     double real_mu = unScale_Mu(pred_mu.Mean());
     double real_vol = unScale_Vol(pred_vol.Mean());
     
     int NumberOfTicksASecond =1;
     int n = tf_in_seconds*NumberOfTicksASecond; //5*60;
     int T =1;
     
     double dt = double(T)/double(n);
     
     double Ask = SymbolInfoDouble(_Symbol,SYMBOL_ASK);
     double Bid = SymbolInfoDouble(_Symbol,SYMBOL_BID);
     double curr_price = (Ask+Bid)/2;
     double  pred_price = curr_price;
     
     double H =Hurst(power_of_2,PERIOD_M1);
     if(H<Hurst_Lower_Range || H>Hurst_Upper_Range){
      pred_price =GeometricBrownianMotion(real_vol,real_mu,dt,curr_price,SamplesOfGBM,n);}
     
     double K = Kappa(1, 30,PERIOD_M1);
   
      
     
          
      double dx =pred_price-curr_price;
      double dx_prev = pred_prev -curr_price;  
      double actualpredError =0;
      if(H<Hurst_Lower_Range || H>Hurst_Upper_Range ){
      double tempvaldx =MathAbs(dx_prev)+1000 ;
      Filo.Insert(tempvaldx);
      
         
         
         
         
         
         Print(MathAbs(dx_prev)+1000, " head");
         for( int j = 0 ; j < ArraySize(Filo.m_arr); j++ ){
         
           
                Print(Filo.m_arr[j], "<---",j);
         }
         
          
         
         
         
             
             
             
            
              
            
            
             double KappaOfError = Kappa_error(1,30,Filo.m_arr);
             
             
              
             Print("Kappa Of Error-> ",KappaOfError);
              
              
             if( Filo.m_numberOfinserted >= 32){ 
              
              
             double muError  =0;
             double stdError =0;
             for( int i_=0 ; i_<30;i_++){
                  
                     double ind = MathLog(Filo.m_arr[i_]/Filo.m_arr[i_+1]);
                     stdError = stdError +pow(ind,2);
                     muError   =  muError +ind;
                  
             
             }
             
             Print(stdError);
             Print(muError);

             stdError = MathExp((MathSqrt(stdError/30)));
             muError  =  MathExp( ((muError/30)))+ .5; 
             
             
             double muErrortrain  =0;
             double stdErrortrain =0;
             for( int i_=1; i_<31;i_++){
                  
                     double ind = MathLog(Filo.m_arr[i_]/Filo.m_arr[i_+1]);
                     stdErrortrain = stdErrortrain +pow(ind,2);
                     muErrortrain   =muErrortrain + ind;
             
             }
             
             Print(stdErrortrain);
             Print(muErrortrain);
             stdErrortrain = MathExp(((MathSqrt(stdErrortrain/30))));
             muErrortrain  = MathExp(((muErrortrain/30))) + .5; 
             
             
             
     
     
            matrix<double> vol_error(Size_of_Esn,Size_of_Esn);
            matrix<double> mu_error(Size_of_Esn,Size_of_Esn);
           
            vol_error = vol_error.Fill(stdError);
            mu_error  = mu_error.Fill(muError);
           
            matrix<double> prev_vol_error(Size_of_Esn,Size_of_Esn) ;
            matrix<double> prev_mu_error(Size_of_Esn,Size_of_Esn) ;
            
            prev_mu_error = prev_mu_error.Fill(muErrortrain);
            prev_vol_error = prev_vol_error.Fill(stdErrortrain);
     
            ESN_Error_vol.Fit(prev_vol_error,vol_error); 
            ESN_Error_mu.Fit(prev_mu_error,mu_error);
   
     
      
     
     
           matrix<double> pred_vol_error = ESN_Error_vol.predict(vol_error,continuation);
           matrix<double> pred_mu_error = ESN_Error_mu.predict(mu_error,continuation);
                   
            double real_mu_error = MathLog(pred_mu_error.Mean()-.5);
            double real_vol_error = MathLog((pred_vol_error.Mean()));
            double currError = Filo.m_arr[0];
            Print( currError, "currError");
            loadedErrorEsn = loadedErrorEsn +1;
            if( loadedErrorEsn >10){
                loadedErrorEsn =11;
               double predError = GeometricBrownianMotion(real_vol_error,real_mu_error,dt,currError,SamplesOfGBM,n) -1000;
               actualpredError = predError; 
            }else{
            
            Print("Loading error ESN : ",double(loadedErrorEsn)/double(10));
            }
            
            }
             
         
      }
      
  pred_prev = pred_price;
      
     
    double Sl = int(1*MathAbs(  pred_price-curr_price))+1; 
      
      double Tp = int(1*MathAbs(  pred_price-curr_price))+1; 
          
         
         double b =Tp/Sl; 
          
          
         if(K >1.14){
         
         K = 1.14; 
         }
         
         if( H > .97){
         
         H = .97;
         
         }
         
         if( H < .01){
         
         H = .01;
         
         }
          
      
     double temp_HK = (   (1 - MathAbs(  (0.15 - K)*1.15))         ); 
        
     double p = 0;
     
     
     
     if(  temp_HK >1){
     
      p =1;
     
     }else{
     
      p= temp_HK;
     }
     
     
     double kelly =  1-temp_HK;//p - (1-p)/b;
     
     if(pred_Verbose){
      Print("------------ New Step-----------");
      Print("H-> ",H);
      Print("K-> ",K );
      Print("PredError-> ",actualpredError); 
      Print("pred_mu-> ",real_mu);
      Print("pred_vol-> ",real_vol);
      Print("pred_NN-> ",predictionNN);
      Print("pred_price-> " ,pred_price);
      Print("pred_dx-> " ,pred_price-curr_price);
      Print("kelly-> ", kelly);
      Print("Neuroplasticity-> ",ESN_Neuroplastic_a-ESN_Neuroplastic_a);}
          double Lot_Martingale =NormalizeDouble(100*calcLots( curr_price -Sl, kelly,percent_of_acc_risk) *NormalizeDouble( pow(Scale_Factor,NumberOfLosses),2),2);
          
          if(H>Hurst_Upper_Range){
          
          
          
            if( MathAbs(dx)-actualpredError >dxThreshold &&  predictionNN >.5 && pred_price > curr_price  &&  (H<Hurst_Lower_Range || H>Hurst_Upper_Range)  && K <KappaThreshold && TradeTracker    ){
              Print( "dx = ",dx," Bias ", predictionNN ," Hurst range = ",H<Hurst_Lower_Range || H>Hurst_Upper_Range," kappa range = : ", K , "kelly : ", kelly  , " Lot : ", Lot_Martingale);
            
            
            double TakeProfit = Tp-1;
            double StopLoss = Sl-1;
          
           MqlTradeRequest myrequest;
           MqlTradeResult myresult;
           ZeroMemory(myrequest);
           ZeroMemory(myresult);
           
             
           myrequest.type = ORDER_TYPE_BUY;
           myrequest.action = TRADE_ACTION_DEAL;
           myrequest.sl = SymbolInfoDouble(_Symbol,SYMBOL_BID) - StopLoss ;
           myrequest.tp =  SymbolInfoDouble(_Symbol,SYMBOL_ASK)+TakeProfit ;
           //myrequest.deviation =20;
           myrequest.symbol = _Symbol;
           myrequest.volume = Lot_Martingale;
           myrequest.type_filling = ORDER_FILLING_FOK;
           myrequest.price = SymbolInfoDouble(_Symbol,SYMBOL_ASK) ;
           myrequest.magic = MagicNumberBUY;
           
           bool sent = OrderSend(myrequest,myresult);
           
           
           }
            
            
            if(  MathAbs(dx)-actualpredError >dxThreshold &&  predictionNN<.5&&  pred_price < curr_price && K <KappaThreshold &&   ( H<Hurst_Lower_Range || H>Hurst_Upper_Range)  &&TradeTracker   ){
            
                    Print( "dx = ",dx," Bias ", predictionNN ," Hurst range = ",H<Hurst_Lower_Range || H>Hurst_Upper_Range," kappa range = : ", K , "kelly : ", kelly  , " Lot : ", Lot_Martingale);
            
            double TakeProfit = Tp-1;
            double StopLoss = Sl-1;
            
           MqlTradeRequest myrequesta;
           MqlTradeResult myresulta;
           ZeroMemory(myrequesta);
           ZeroMemory(myresulta);
           
             
           myrequesta.type = ORDER_TYPE_SELL;
           myrequesta.action = TRADE_ACTION_DEAL;
           myrequesta.sl = SymbolInfoDouble(_Symbol,SYMBOL_ASK) + StopLoss ;
           myrequesta.tp = SymbolInfoDouble(_Symbol,SYMBOL_BID)- TakeProfit; //  .00250;
           //myrequest.deviation =20;
           myrequesta.symbol = _Symbol;
           myrequesta.volume = Lot_Martingale;
           myrequesta.type_filling = ORDER_FILLING_FOK;
           myrequesta.price = SymbolInfoDouble(_Symbol,SYMBOL_BID);
           myrequesta.magic = MagicNumberSELL;
           
           bool sent =OrderSend(myrequesta,myresulta); 
            
            
            }
         
   
      }
   
   
   
      if(H<Hurst_Lower_Range){
          
          
          
            if( MathAbs(dx)-actualpredError >dxThreshold &&  predictionNN <.5 && pred_price < curr_price  &&  (H<Hurst_Lower_Range || H>Hurst_Upper_Range)  && K <KappaThreshold && TradeTracker    ){
              Print( "dx = ",dx," Bias ", predictionNN ," Hurst range = ",H<Hurst_Lower_Range || H>Hurst_Upper_Range," kappa range = : ", K , "kelly : ", kelly  , " Lot : ", Lot_Martingale);
            
            
            double TakeProfit = Tp;
            double StopLoss = Sl;
          
           MqlTradeRequest myrequest;
           MqlTradeResult myresult;
           ZeroMemory(myrequest);
           ZeroMemory(myresult);
           
             
           myrequest.type = ORDER_TYPE_BUY;
           myrequest.action = TRADE_ACTION_DEAL;
           myrequest.sl = SymbolInfoDouble(_Symbol,SYMBOL_BID) - StopLoss ;
           myrequest.tp =  SymbolInfoDouble(_Symbol,SYMBOL_ASK)+TakeProfit ;
           //myrequest.deviation =20;
           myrequest.symbol = _Symbol;
           myrequest.volume = Lot_Martingale;
           myrequest.type_filling = ORDER_FILLING_FOK;
           myrequest.price = SymbolInfoDouble(_Symbol,SYMBOL_ASK) ;
           myrequest.magic = MagicNumberBUY;
           
           bool sent = OrderSend(myrequest,myresult);
           
           
           }
            
            
            if( MathAbs(dx)-actualpredError >dxThreshold &&  predictionNN>.5&&  pred_price > curr_price && K <KappaThreshold &&   ( H<Hurst_Lower_Range || H>Hurst_Upper_Range)  &&TradeTracker   ){
            
                    Print( "dx = ",dx," Bias ", predictionNN ," Hurst range = ",H<Hurst_Lower_Range || H>Hurst_Upper_Range," kappa range = : ", K , "kelly : ", kelly  , " Lot : ", Lot_Martingale);
            
            double TakeProfit = Tp;
            double StopLoss = Sl;
            
           MqlTradeRequest myrequesta;
           MqlTradeResult myresulta;
           ZeroMemory(myrequesta);
           ZeroMemory(myresulta);
           
             
           myrequesta.type = ORDER_TYPE_SELL;
           myrequesta.action = TRADE_ACTION_DEAL;
           myrequesta.sl = SymbolInfoDouble(_Symbol,SYMBOL_ASK) + StopLoss ;
           myrequesta.tp = SymbolInfoDouble(_Symbol,SYMBOL_BID)- TakeProfit; //  .00250;
           //myrequest.deviation =20;
           myrequesta.symbol = _Symbol;
           myrequesta.volume = Lot_Martingale;
           myrequesta.type_filling = ORDER_FILLING_FOK;
           myrequesta.price = SymbolInfoDouble(_Symbol,SYMBOL_BID);
           myrequesta.magic = MagicNumberSELL;
           
           bool sent =OrderSend(myrequesta,myresulta); 
            
            
            }
         
   
      }
   
   
   
   }
   
   
     
     
     
     
     
     
     
     
     
     
     
     
  }





 
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
  {
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD){
      
     CDealInfo deal;
     deal.Ticket(trans.deal);
     HistorySelect(TimeCurrent() - PeriodSeconds(PERIOD_D1 ),TimeCurrent()+10);
     if(deal.Symbol()==_Symbol && (deal.Magic() == MagicNumberBUY ||deal.Magic() == MagicNumberSELL )){
       
       if(deal.Entry()==DEAL_ENTRY_OUT ){
       if(deal.Profit()  >0 ){
       
         // pas
         Print("--");
         Print(NumberOfLosses);
         NumberOfLosses =0;
         NumberOfWins = NumberOfWins+1;
       }else{
            Print("++");
            NumberOfLosses = NumberOfLosses+1;
            NumberOfWins = NumberOfWins-1;
            Print(NumberOfLosses);
       
         }
       }       
     
     }
      
   }
  }  
 