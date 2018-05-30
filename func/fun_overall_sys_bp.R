fun_overall_sys_bp<- function(img_diff_SI_3){
  img_diff_SI_3$overall_sys_bp <- NA  
  for(i in 1:nrow(img_diff_SI_3)){
    if(!is.na(img_diff_SI_3$Median_Art.Sys[i]) &  img_diff_SI_3$Median_Art.Sys[i] > 20){
      img_diff_SI_3$overall_sys_bp[i] <- img_diff_SI_3$Median_Art.Sys[i]
      img_diff_SI_3$overall_dia_bp[i] <- img_diff_SI_3$Median_dia[i]
    }else{
      img_diff_SI_3$overall_sys_bp[i] <- img_diff_SI_3$Median_Sys[i]
      img_diff_SI_3$overall_dia_bp[i] <- img_diff_SI_3$Median_dia[i]
    }
  }
  return(img_diff_SI_3)
}


