### Import and Preprocess data 

rm(list=ls())
getwd()
setwd("/Users/vaidy105/Documents/R/NPML")
library(stringr)
library(caret)
library(reshape)
library(plotly)
library(dplyr)
library(tree)
library(randomForest)

#read in data
data<-read.csv("PosParental.csv")
data<-data[,apply(data,2,function(x){sum(!is.na(x))>0})]
data[is.na(data)] <- 0

formulations<-read.csv("formulations.csv")
lipids<-read.csv("lipid_properties.csv")

##Rearrange data into new format 

#Compile output variable into one column 

y<-list()

for (b in 1:16){
  y1<-data[b,7:18]
  y=c(y,y1)
}

y<-as.data.frame(y)
y<-t(y)
y<-as.data.frame(y)
colnames(y)<-c('y')

#Create empty dataframe 
df<-data.frame(Size=double(),Zeta=double(),PDI=double(),D=double(),DSPC=double(),
               DSPG=double(),DSPE=double(),POPE=double(),
               POPC=double(),POPG=double(),Cholesterol=double(),Sitosterol=double(),
               DOPE=double())

#Populate physicochemical properties (do not include concentration)
for (x in 1:16){
  n=1+(x-1)*12
  m=x*12
  df[n:m,1]=data[x,3]
  df[n:m,2]=data[x,4]
  df[n:m,3]=data[x,5]
  df[n:m,4]=data[x,6]
  df[n:m,5]=formulations[x,2]
  df[n:m,6]=formulations[x,3]
  df[n:m,7]=formulations[x,4]
  df[n:m,8]=formulations[x,5]
  df[n:m,9]=formulations[x,6]
  df[n:m,10]=formulations[x,7]
  df[n:m,11]=formulations[x,8]
  df[n:m,12]=formulations[x,9]
  df[n:m,13]=formulations[x,10]
}


#Normalize Features with Min-Max Scaling from 0 to 1 
process <-preProcess(as.data.frame(df),method=c("range"))
data_norm<-predict(process,as.data.frame(df))

#Normalize with standard scaling 
#data_norm<-as.data.frame(scale(df))

#Attach output variable 
data_norm<-cbind(data_norm,y)

#Remove empty rows for formulations that have less than 12 replicates 
data_norm<-data_norm[data_norm$y != 0,]


### Segment Dataset
set.seed(100)
train<-sample(1:nrow(data_norm),nrow(data_norm)*.8 )
SLCOE.test<-data_norm[-train,]
SLCOE.train<-data_norm[train,]


### Algorithm Tune (tuneRF)
x= SLCOE.train[,1:9]
y=SLCOE.train[,10]
bestmtry <- tuneRF(x, y, stepFactor=1.5, improve=1e-5, ntree=500)
print(bestmtry)
#mtry=6 seem to be best option 

### Tune with Caret 

# Create model with default paramters
control <- trainControl(method="repeatedcv", number=10, repeats=3)
seed <- 100 
metric <- "RMSE"
set.seed(seed)
mtry <- sqrt(ncol(x))
tunegrid <- expand.grid(.mtry=mtry)
rf_default <- train(y~., data=SLCOE.train, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_default)

#Random Search
control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
mtry <- sqrt(ncol(x))
rf_random <- train(y~., data=SLCOE.train, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

## Random Forest 

RMSE=data.frame(RMSE.train=double(),RMSE.test=double())

feature_importance_values <-list()
#set.seed(44)

for (i in 1:100 ){
  set.seed(i)
  train<-sample(1:nrow(data_norm),nrow(data_norm)*.8 )
  SLCOE.test<-data_norm[-train,]
  SLCOE.train<-data_norm[train,]
  rf.SLCOE<-randomForest(y~.,data=data_norm,subset=train,mtry=4, importance=TRUE,ntree=100)
  train_predictions <- predict(rf.SLCOE, newdata = SLCOE.train)
  test_predictions <- predict(rf.SLCOE, newdata = SLCOE.test)
  RMSE[i,1] <- sqrt(mean((train_predictions - SLCOE.train$y)^2))
  RMSE[i,2] <- sqrt(mean((test_predictions - SLCOE.test$y)^2))
  
  importance_values <-importance(rf.SLCOE)
  feature_importance_values[[i]]<-importance_values
}

features_summary<-do.call(data.frame,feature_importance_values)
Nodepurity_features <- features_summary[, seq(2, ncol(features_summary), by = 2)]  # Select even columns
MSE_features <- features_summary[, seq(1, ncol(features_summary), by = 2)]   # Select odd columns


averaged_importance_values<-data.frame(AvgNodePurity=double(),AvgMSE=double())


for (h in 1:nrow(importance_values)){
  averaged_importance_values[h,1]= rowMeans(Nodepurity_features[h,])
  averaged_importance_values[h,2]= rowMeans(MSE_features[h,])
}

rownames(averaged_importance_values)<-row.names(features_summary)
averaged_importance_values <- averaged_importance_values[order(averaged_importance_values$AvgMSE, decreasing = TRUE), ]

barplot(averaged_importance_values$AvgMSE,col=rgb(0.2,0.4,0.6,0.6),
        names.arg=rownames(averaged_importance_values),
        xlab="features",
        ylab="% Inc MSE",
        #ylim=c(0,25))
)


mean(RMSE$RMSE.train)
sd(RMSE$RMSE.train)
mean(RMSE$RMSE.test)
sd(RMSE$RMSE.test)
