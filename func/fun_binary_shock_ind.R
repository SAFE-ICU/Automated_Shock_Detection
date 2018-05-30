binary_shock_ind <- function(x , y ){
  temp <- NA
  for(i in 1:length(x)){
    if(x[i] <=3 & y[i] > 2.3 & !is.na(y[i]) & !is.na(x[i])){
      temp[i] <- 1
    }else{
      if( 3< x[i] & x[i] <= 6 &  y[i] > 1.7 & !is.na(y[i]) &  !is.na(x[i])){
      temp[i] <- 1
    }else{
      if(6< x[i] & x[i] <= 12 &  y[i] > 1.5 & !is.na(y[i]) &  !is.na(x[i])){
        temp[i] <- 1
      }else{
        if(12< x[i] & x[i] <=36 &  y[i] > 1.2 & !is.na(y[i]) &  !is.na(x[i])){
          temp[i] <- 1
        }else{
          if(36< x[i] & x[i] <=72 &  y[i] > 1.157 & !is.na(y[i]) &  !is.na(x[i])){
            temp[i] <- 1
          }else{
            if(72< x[i] & x[i] <=144 &  y[i] > 0.95 & !is.na(y[i]) &  !is.na(x[i])){
              temp[i] <- 1
            }else{
              if(144 < x[i] &  y[i] > 0.77 & !is.na(y[i]) &  !is.na(x[i])){
                temp[i] <- 1  
              }else{
                if(!is.na(y[i]) & !is.na(x[i])){
                temp[i] <-  0 
                }
              } 
            }}}}}}
  }
  return(temp)
}

# binary_SI <- binary_shock_ind(for_modeling_wto_diff_with_30min$age_mo, for_modeling_wto_diff_with_30min$Shock_ind_30min)
# check <- cbind(for_modeling_wto_diff_with_30min$age_mo , for_modeling_wto_diff_with_30min$Shock_ind_30min, binary_SI)
# 
#   