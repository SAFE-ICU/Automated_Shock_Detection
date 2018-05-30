cp_max_thrsh <- function(perf_glmer){

  roc_y_glmer<- as.vector(do.call(cbind, perf_glmer@y.values))
  roc_x_glmer<- as.vector(do.call(rbind, perf_glmer@x.values))
  roc_alpha_glmer<- as.vector(do.call(cbind, perf_glmer@alpha.values))
  roc_Data_glmer <- data.frame(roc_x_glmer,roc_y_glmer,roc_alpha_glmer)
  roc_Data_glmer$diff <- (roc_Data_glmer$roc_y_glmer - roc_Data_glmer$roc_x_glmer)
  cp_max_height_glmer<- roc_Data_glmer$roc_alpha_glmer[which(roc_Data_glmer$diff==max(roc_Data_glmer$diff))]
  return(cp_max_height_glmer)
}