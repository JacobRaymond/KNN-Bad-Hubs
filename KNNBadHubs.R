library(ISLR)
library(FNN)
library(tidyverse)

####Import data####
df<-Smarket

#Convert "Direction" to numeric" using one-hot encoding
df=mutate(df, Direction=ifelse(Direction=="Up", 1, 0))

####Hub analysis####

#Remove output
Today=df$Today
df=select(df, -Today)

#Get the closest neighbours
cn=get.knn(df, k = 3)

#Number of instances
cn$nn.index %>%
  table()%>%
  as.data.frame()%>%
  ggplot()+
    geom_histogram(aes(Freq))+
    labs(x="N(x)", y="Count")+
  scale_x_continuous(breaks=seq_len(10))

#Find the points part of N(x) for each observation
whereNX=function(i){
  cn$nn.index  %>%
    as_tibble() %>%
    mutate(Obs=1:nrow(.))%>%
    filter_at(vars(-Obs), any_vars(. ==i))%>%
    .$Obs
}
NX=map(1:nrow(df), whereNX)

#Calculate error
ErrorNX=vector()
for(i in 1:length(NX)){
  if(length(NX[[i]])==0){
    ErrorNX[i]=NA
  }else{
    ErrorNX[i]=mean((Today[i]-Today[NX[[i]]])^2)
  }
}
Err=tibble(Obs=1:nrow(df), 
            Error=ErrorNX, 
            Instances=map_dbl(NX, function(x)(length(x)))) %>%
  drop_na()

#High-error cases
Err %>% 
  top_frac(wt=Error, n=0.25) %>%
  ggplot()+geom_bar(aes(Instances))+labs(x="N(x)", y="Count")+
  scale_x_continuous(breaks=seq_len(10))

#Number of high error points
Err %>% 
  top_frac(wt=Error, n=0.25) %>% nrow()

#Proportion of bad hubs
Err %>% 
  top_frac(wt=Error, n=0.25) %>% 
  filter(Instances >5) %>% nrow() /Err %>% 
  top_frac(wt=Error, n=0.25) %>%  nrow()

df=cbind(Today, df)

#### PCA ####
pc=prcomp(df[-Today])
(pc$sdev^2/sum(pc$sdev^2)) %>% cumsum() #7 PC
df=cbind(df$Today, pc$x[,1:7])

#### Error-based Weighing ####

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


#### Simulation ####

# Training and Test Sets
train= df%>% as_tibble() %>% sample_frac(size = 0.8)
test=df%>% as_tibble() %>% anti_join(train)


# KNN
knn.pred=knn.reg(train[,-1], test=test[,-1], y=as_vector(train[,1]), k=10)
(knn.pred$pred-test[,1]) %>% abs() %>% as_vector() %>% mean()

#EwKNN
EwKNN(train, test)$MAE

#Linear Regression
y=train[,1] %>% as_vector()
LinReg=lm(y~., data=train[,-1])
LinReg.pred=predict(LinReg, newdata=test[-1])
(LinReg.pred-test[,1]) %>% abs() %>% as_vector() %>% mean()
