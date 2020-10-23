
#--------------------first chunk(no output at all)-----------------
library(rpart)
library(tree)
library(randomForest)
library(xgboost)
library(ranger)
library(broom)
library(ggplot2)
library(pROC)
library(car)
library(dplyr)
library(gridExtra)
library(glmnet)
library(MASS)
#reproduce
set.seed(100) 
#load data
white_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
whitewine <- read.csv(white_url, header = TRUE, sep = ";")
#data cleaning
whitewine <-  unique(whitewine)#delete repeat
which(is.na(whitewine))#missing data
#remove outlier
linear_0 <- lm(quality ~ . , whitewine) 
nt=nrow(whitewine)
p=11 
#detect outlier 
alpha=0.05
stud.del.resids= rstudent(linear_0)
whitewine = whitewine[-which(stud.del.resids[] > qt(1-alpha/(2*nt),nt-p-1)),]#outlier point Y-2295
#detect multicollinearity
vif(linear_0)
#solve inbalanced issue(response)
category=function(a){
  if(a<=4){return(0)}
  if(a==5){return(1)}
  if(a==6){return(2)}
  if(a==7){return(3)}
  if(a>7){return(4)}
}
quality2=sapply(1:nrow(whitewine),function(i) category(whitewine$quality[i]))
whitewine$quality = quality2
#split train,test dataset
ntrain = floor(nt*0.8)
ii = sample(1:nrow(whitewine),ntrain)
trainDf = whitewine[ii,-c(4,8)] #delete density&residual_sugar
testDf = whitewine[-ii,-c(4,8)]
attach(whitewine)
                
#--------------------second chunk(just show result of data)-----------------
knitr::kable(whitewine[1:3,],digits=2) 
                
#--------------------third chunk(don't show message and code)-----------------


# add all interactions abd second order
linearinterract=model.matrix(quality ~(.)^2,data=trainDf)[,-1]
linearinterract1=model.matrix(quality ~(.)^2,data=testDf)[,-1]

#grid search
hyper_gridglm = expand.grid(alpha = seq(0,1,by = 0.5),
                         nfolds = seq(5,10,by=1),
                         OOBerror = 0)

# perform grid search
for(i in 1:nrow(hyper_gridglm)) {
  
  fit.cv <- cv.glmnet(linearinterract, 
                      trainDf[,10], 
                      type.measure="deviance", 
                      alpha=hyper_gridglm$alpha[i],
                      nfolds=hyper_gridglm$nfolds[i],
                      standardize=TRUE,
                      family="multinomial")
  # add OOB error to grid
  hyper_gridglm$OOBerror[i] <- sqrt(fit.cv$cvm)
}
cat("parameters of best model: " )
hyper_gridglm[which.min(hyper_gridglm$OOBerror),]

#use the best setting from grid search above, find best lambda for regularized lm
optimal_RLM = cv.glmnet(linearinterract, 
                        trainDf[,10], 
                        type.measure="deviance", 
                        alpha=hyper_gridglm$alpha[which.min(hyper_gridglm$OOBerror)],
                        nfolds=hyper_gridglm$nfolds[which.min(hyper_gridglm$OOBerror)],
                        standardize=TRUE,
                        family="multinomial")

#apply to ridge regularization and lambda from optimal_RLM
fit.ridge.best <- glmnet(linearinterract, trainDf[,10], family = "multinomial", alpha = hyper_gridglm$alpha[which.min(hyper_gridglm$OOBerror)],
                         lambda = optimal_RLM$lambda.min)
fit.ridge.bestse <- glmnet(linearinterract, trainDf[,10], family = "multinomial", alpha =  hyper_gridglm$alpha[which.min(hyper_gridglm$OOBerror)],
                         lambda = optimal_RLM$lambda.1se)
fit.ridge.pred <- predict(fit.ridge.best, linearinterract1, type = "class")
fit.ridge.pred1 <- predict(fit.ridge.bestse, linearinterract1, type = "class")

#--------------------fourth chunk(just show plot and table)-----------------

phatL = list() #store the test phat for the different methods here
phatL$glmmin = fit.ridge.pred #for lambda.min
phatL$glmse = fit.ridge.pred1 #for lambda.1se


#compare
acc = cbind(mean(phatL$glmmin == testDf[,10]),mean(phatL$glmse == testDf[,10]))
rownames(acc)='accuracy'
colnames(acc)=c('lambda.min','lambda.1se')
knitr::kable(acc)##so, choose lambda.min as best lambda

#choose good one--lambda.min
table(phatL$glmmin,testDf[,10]) # confusion matrix

#plot for changing in oob error

ggplot(hyper_gridglm, aes(x=seq(1,nrow(hyper_gridglm)),y=OOBerror)) +
  geom_point() +
  geom_vline(xintercept =which.min(hyper_gridglm$OOBerror),colour='red')+
  ylab("out of sample error")+
  xlab("different settings")+
  ggtitle("oob error in GLM") +
  # guides(fill=F)+
  scale_fill_gradient(low="grey", high="blue")


#plot roc curve
qualityte = one_hot(as.data.table(factor(testDf$quality))) # original factor of testDf
qualitypre=factor(phatL$glmmin,levels=0:4) # reset level in case some category doesn't show up in the prediction
qualitypre=one_hot(as.data.table(qualitypre))
nfit = 5
par(mai=c(.8, 1.2, .5, .5))
plot(roc(qualityte[[1]], qualitypre[[1]]),xlab='Specificity', ylab='Sensitivity'
     ,xlim=c(0.5,0),ylim=c(0,1),col=1,main = "regularized linear models")
for(i in 2:nfit-1) {
  roc=roc(qualityte[[i]], qualitypre[[i]])
  lines(roc,col=i) 
}
legend(0, 1, c("1", "2", "3","4","5"), col = c(1, 2, 3, 4, 5), lwd = c(1,1,1,1,1),bg="transparent",cex=0.6)
#auc
roc.multiglm = multiclass.roc(testDf$quality,as.numeric(phatL$glmmin))
auc = auc(roc.multiglm)

#lift
#obtain prob for each class(lift)
probglm<- as.data.frame(predict(fit.ridge.best, linearinterract1, type = "response"))
lift.many.plot = function(phat.list, y.list,...) {
  if(is.factor(y.list)) y.list = as.numeric(y.list)-1
  n = nrow(y.list)
  
  par(mai=c(.8, 1.2, .5, .5))
  plot(c(0,1), c(0,1), type="n", xlab='% tried', ylab='% of successes', ...)
  abline(0, 1, lty=2)
  ii = (1:n) / n
  for(i in 1:length(phat.list)) {
    oo = order(-phat.list[[i]])
    sy = cumsum(y.list[[i]][oo])/sum(y.list[[i]]==1)
    lines(ii,sy,type="l",lwd=2,col=i)
  }
}
phatLLglm = vector("list",nfit)
for(i in 1:nfit) {
  phatLLglm[[i]] = probglm[,i] }
lift.many.plot(phatLLglm,qualityte)
legend(0, 1, c("1", "2", "3","4","5"), col = c(1, 2, 3,4,5), lwd = c(1,1,1,1,1),bg="transparent",cex=0.7)













