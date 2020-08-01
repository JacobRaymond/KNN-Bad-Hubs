library(ISLR)
library(FNN)
library(tidyverse)
library(caret)

####Import data####
df=Smarket

#Convert "Direction" to numeric" using one-hot encoding
df=mutate(df, Direction=ifelse(Direction=="Up", 1, 0))


#### PCA ####
pc=prcomp(df[-1])
df=cbind(df$Today, pc$x[,1:7])

#### Error Weighting ####

EwKNN=function(train, test, k=10){
  
  #Get the closest neighbours
  cn=get.knn(train[,-1], k)
  
  #Find the points part of N(x) for each observation
  whereNX=function(i){
    cn$nn.index  %>%
      as_tibble() %>%
      mutate(Obs=1:nrow(.))%>%
      filter_at(vars(-Obs), any_vars(. ==i))%>%
      .$Obs
  }
  NX=map(1:nrow(train[,-1]), whereNX)
  
  #Calculate error
  ErrorNX=vector()
  for(i in 1:length(NX)){
    if(length(NX[[i]])==0){
      ErrorNX[i]=NA
    }else{
      ErrorNX[i]=abs(as.numeric(train[i, 1])-train[NX[[i]], 1]) %>% 
        as_vector() %>% 
        mean()
    }
  }
  
  #Calculate weights
  Err=tibble(Obs=1:nrow(train), 
             Error=ErrorNX,
             y=as_vector(train[,1]),
             Instances=map_dbl(NX, function(x)(length(x)))) %>%
    mutate(err=scale(Error))%>%
    mutate(wx=exp(-err))
  
  #Remove data with a weight of 0
  train_na=train %>% 
    mutate(Obs=seq_len(nrow(.))) %>% 
    filter(row_number() %in% filter(Err, Instances!=0)$Obs)
  
  
  #Fit data to test set
  obs=split(test[,-1], seq(nrow(test)))
  
  #Predict new values
  knnpred=function(x){
    #Find closest neighbours
    cn_obs=get.knnx(data=train_na[,c(-1, -length(train_na))], query=x, k=k)
    
    #Predict
    Err %>% 
      mutate(numerator=y*wx) %>% 
      filter(Obs %in% train_na$Obs[cn_obs$nn.index]) %>% 
      summarize(sum(numerator)/sum(wx)) %>%
      as.double()
  }
  preds=map_dbl(obs, knnpred) 
  
  #MAE
  MAE=(preds-test[,1]) %>% abs() %>% as_vector() %>% mean()
  
  #Output of results
  output=list(preds, MAE)
  names(output)=c("Predictions", "MAE")
  output
}

#### Error Correction ####

EcKNN=function(train, test, k=10){
  
  #Get the closest neighbours
  cn=get.knn(train[,-1], k)
  
  #Find the points part of N(x) for each observation
  whereNX=function(i){
    cn$nn.index  %>%
      as_tibble() %>%
      mutate(Obs=1:nrow(.))%>%
      filter_at(vars(-Obs), any_vars(. ==i))%>%
      .$Obs
  }
  NX=map(1:nrow(train[,-1]), whereNX)
  
  #Calculate Corrected Labels
  CorrectedLabels=function(x){
    if(length(x)<1){
      NA
    }else{
      train[x,1] %>% 
        as_vector() %>% 
        mean()
    }
  }
  train_yc=train %>% 
    mutate(yc=map_dbl(NX, CorrectedLabels)) %>% 
    drop_na()
  
  #Fit data to test set
  obs=split(test[,-1], seq(nrow(test)))
  
  #Predict new values
  knnpred=function(x){
    
    #Find closest neighbours
    cn_obs=get.knnx(data=train_yc[,c(-1, -length(train_yc))], query=x, k=k)
    
    #Predict
    train_yc$yc[cn_obs$nn.index] %>% 
      mean()
  }
  preds=map_dbl(obs, knnpred) 
  
  #MAE
  MAE=(preds-test[,1]) %>% abs() %>% as_vector() %>% mean()
  
  #Output of results
  output=list(preds, MAE)
  names(output)=c("Predictions", "MAE")
  output
}

#### Error-Weighted Correction####

EcwKNN=function(train, test, k=10){
  
  #Get the closest neighbours
  cn=get.knn(train[,-1], k)
  
  #Find the points part of N(x) for each observation
  whereNX=function(i){
    cn$nn.index  %>%
      as_tibble() %>%
      mutate(Obs=1:nrow(.))%>%
      filter_at(vars(-Obs), any_vars(. ==i))%>%
      .$Obs
  }
  NX=map(1:nrow(train[,-1]), whereNX)
  
  #Calculate error
  ErrorNX=vector()
  for(i in 1:length(NX)){
    if(length(NX[[i]])==0){
      ErrorNX[i]=NA
    }else{
      ErrorNX[i]=(as.numeric(train[i, 1])-train[NX[[i]], 1])^2 %>% 
        as_vector() %>% 
        mean()
    }
  }
  
  #Calculate weights
  Err=tibble(Obs=1:nrow(train), 
             Error=ErrorNX,
             y=as_vector(train[,1]),
             Instances=map_dbl(NX, function(x)(length(x)))) %>%
    mutate(err=scale(Error))%>%
    mutate(wx=exp(-err))
  
  #Calculate Corrected Labels
  CorrectedLabels=function(x){
    if(length(x)<1){
      NA
    }else{
      train[x,1] %>% 
        as_vector() %>% 
        mean()
    }
  }
  
  #Calculate weights
  Err=tibble(Obs=1:nrow(train), 
             Error=ErrorNX,
             y=as_vector(train[,1]),
             Instances=map_dbl(NX, function(x)(length(x))),
             yc=map_dbl(NX, CorrectedLabels)) %>%
    mutate(err=scale(Error))%>%
    mutate(wx=exp(-err))
  
  #Remove data with a weight of 0
  train_wyc=train %>% 
    mutate(Obs=seq_len(nrow(.))) %>% 
    filter(row_number() %in% filter(Err, Instances!=0)$Obs)
  
  #Fit data to test set
  obs=split(test[,-1], seq(nrow(test)))
  
  #Predict new values
  knnpred=function(x){
    
    #Find closest neighbours
    cn_obs=get.knnx(data=train_wyc[,c(-1, -length(train_wyc))], query=x, k=k)
    
    #Predict
    Err %>% 
      mutate(numerator=yc*wx) %>% 
      filter(Obs %in% train_wyc$Obs[cn_obs$nn.index]) %>% 
      summarize(sum(numerator)/sum(wx)) %>%
      as.double()
  }
  preds=map_dbl(obs, knnpred) 
  
  #MAE
  MAE=(preds-test[,1]) %>% abs() %>% as_vector() %>% mean()
  
  #Output of results
  output=list(preds, MAE)
  names(output)=c("Predictions", "MAE")
  output
}

#### Simulation ####

set.seed(20831748)

#Create folds
folds=nrow(df) %>% 
  seq_len() %>% 
  sample()

knn.MAE=vector()
knn.time=vector()
EwKNN.MAE=vector()
EwKNN.time=vector()
EcKNN.MAE=vector()
EcKNN.time=vector()
EcwKNN.MAE=vector()
EcwKNN.time=vector()
knn.MAE10=vector()
knn.time10=vector()
EwKNN.MAE10=vector()
EwKNN.time10=vector()
EcKNN.MAE10=vector()
EcKNN.time10=vector()
EcwKNN.MAE10=vector()
EcwKNN.time10=vector()
LinReg.MAE=vector()
rf.MAE=vector()

for(i in 1:10){
  
  #Training and Test set
  test=df[folds[(125*i-124): (125*i)], ] %>% 
    as_tibble()
  train=df%>% as_tibble() %>% anti_join(test)
  
  # KNN (k=5)
  Start=Sys.time()
  knn.pred=knn.reg(train[,-1], test=test[,-1], y=as_vector(train[,1]), k=5)
  knn.MAE[i]=(knn.pred$pred-test[,1]) %>% abs() %>% as_vector() %>% mean()
  End=Sys.time()
  knn.time[i]=End-Start
  
  # KNN (k=10)
  Start=Sys.time()
  knn.pred=knn.reg(train[,-1], test=test[,-1], y=as_vector(train[,1]), k=10)
  knn.MAE10[i]=(knn.pred$pred-test[,1]) %>% abs() %>% as_vector() %>% mean()
  End=Sys.time()
  knn.time10[i]=End-Start
  
  #EwKNN (k=5)
  Start=Sys.time()
  EwKNN.MAE[i]=EwKNN(train, test, k=5)$MAE
  End=Sys.time()
  EwKNN.time[i]=End-Start
  
  #EcKNN (k=5)
  Start<-Sys.time()
  EcKNN.MAE[i]=EcKNN(train, test, k=5)$MAE
  End=Sys.time()
  EcKNN.time[i]=End-Start
  
  #EcwKNN (k=5)
  Start=Sys.time()
  EcwKNN.MAE[i]=EcwKNN(train, test, k=5)$MAE
  End=Sys.time()
  EcwKNN.time[i]=End-Start
  
  # KNN (k=10)
  Start=Sys.time()
  knn.pred=knn.reg(train[,-1], test=test[,-1], y=as_vector(train[,1]), k=10)
  knn.MAE10[i]=(knn.pred$pred-test[,1]) %>% abs() %>% as_vector() %>% mean()
  End=Sys.time()
  knn.time10[i]=End-Start
  
  #EwKNN (k=10)
  Start=Sys.time()
  EwKNN.MAE10[i]=EwKNN(train, test, k=10)$MAE
  End=Sys.time()
  EwKNN.time10[i]=End-Start
  
  #EcKNN (k=10)
  Start=Sys.time()
  EcKNN.MAE10[i]=EcKNN(train, test, k=10)$MAE
  End=Sys.time()
  EcKNN.time10[i]=End-Start
  
  #EcwKNN (k=10)
  Start=Sys.time()
  EcwKNN.MAE10[i]=EcwKNN(train, test, k=10)$MAE
  End=Sys.time()
  EcwKNN.time10[i]=End-Start
  
  #Linear Regression
  y=train[,1] %>% as_vector()
  LinReg=lm(y~., data=train[,-1])
  LinReg.pred=predict(LinReg, newdata=test[-1])
  LinReg.MAE[i]=(LinReg.pred-test[,1]) %>% abs() %>% as_vector() %>% mean()
  
  #Random forest
  tunegrid = expand.grid(.mtry = (1:7)) #7 possible parameters
  rf_mod = train(V1 ~ ., 
                 data = train,
                 ntree = 500,
                 method = 'rf',
                 tuneGrid = tunegrid,
                 trControl=trainControl(method = "cv", number=5, verboseIter=F))
  rf.pred=predict(rf_mod, newdata=test[-1])
  rf.MAE[i]=(rf.pred-test[,1]) %>% abs() %>% as_vector() %>% mean()
}

#### MAE ####
res=matrix(c("KNN, k=5", signif(mean(knn.MAE), digits = 3), signif(sd(knn.MAE), digits = 3), signif(sum(knn.time), digits = 1), 
             "KNN, k=10", signif(mean(knn.MAE10), digits = 3),signif(sd(knn.MAE10), digits = 3),signif(sum(knn.time10, digits = 1)),
             "EwKNN, k=5", signif(mean(EwKNN.MAE), digits = 3), signif(sd(EwKNN.MAE), digits = 3),signif(sum(EwKNN.time, digits = 1)), 
             "EwKNN, k=10", signif(mean(EwKNN.MAE10), digits = 3),signif(sd(EwKNN.MAE10), digits = 3),signif(sum(EwKNN.time10, digits = 1)),
             "EcKNN, k=5", signif(mean(EcKNN.MAE), digits = 3), signif(sd(EcKNN.MAE), digits = 3), signif(sum(EcKNN.time, digits = 1)),
             "EcKNN, k=10", signif(mean(EcKNN.MAE10), digits = 3), signif(sd(EcKNN.MAE10), digits = 3), signif(sum(EcKNN.time10, digits = 1)),
             "EcwKNN, k=5", signif(mean(EcwKNN.MAE), digits = 3), signif(sd(EcwKNN.MAE), digits = 3), signif(sum(EcwKNN.time, digits = 1)),
             "EcwKNN, k=10", signif(mean(EcwKNN.MAE10), digits = 3), signif(sd(EcwKNN.MAE10), digits = 3), signif(sum(EcwKNN.time10, digits = 1)),
             "Linear Regression", signif(mean(LinReg.MAE), digits = 3), signif(sd(LinReg.MAE), digits = 3), "-",
             "Random Forests", signif(mean(rf.MAE), digits = 3), signif(sd(rf.MAE), digits = 3), "-"), nrow = 4)
rownames(res)=c("Model", "MAE", "Standard Deviation", "Duration (Secs.)")
res=t(res)

knitr::kable(res, booktabs=T, align="c", caption ="Results of the Simulation Study", label="Res")