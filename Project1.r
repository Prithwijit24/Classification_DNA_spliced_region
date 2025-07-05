D = readRDS("A:/Mentorship - Data Mining/spliced1.RDS")
#View(D)
dim(D)
unique(D$class)
Predictor_D = D[,-1]
View(head(D))

sum(is.na(D))
#Dummy used.....
############################################
#EDA on the response variable
library(dplyr)
New_data = D %>% group_by(class) %>% summarise(count = n())
New_data

library(ggplot2)
ggplot(data = New_data,mapping = aes(class,count,color = class,fill = class))+
  geom_col()+
  geom_text(mapping = aes(label = count),color = "black",position = position_stack(vjust = 0.5))+
  ggtitle(paste("Pie diagram on the response : class"))+
  coord_polar()





##########################################
#One-Hot-Encoding
library(caret)
dummy = dummyVars("~.",data = D[,-1])
D1 = data.frame(predict(dummy,newdata = D[,-1]))
D1=cbind(D[,1],D1)
value = c("class",colnames(D1)[-1])
colnames(D1) = value
#View(D1)

D2 = D1[,-seq(5,241,by = 4)]
#View(D2)
dim(D2)



#train & Test Splitting
    #80% Train
    #20% Test
set.seed(1318916)
data_80_ = floor((nrow(D2)*0.8))
train_index_ = sample(nrow(D2),data_80_)
D2_train = D2[train_index_,]
D2_test = D2[-train_index_,]
dim(D2_train)
dim(D2_test)
dim(D2)








##########################################
#EDA on the training data-set
library(dplyr)
New_data_train = D2_train %>% group_by(class) %>% summarise(count = n())
New_data

library(ggplot2)
g_train = ggplot(data = New_data_train,mapping = aes(class,count,color = class,fill = class))+
          geom_col()+
          geom_text(mapping = aes(label = count),color = "black",position = position_stack(vjust = 0.5))+
          ggtitle(paste("Pie diagram on the response of train data : class"))+
          coord_polar()

#EDA on the test data-set
library(dplyr)
New_data_test = D2_test %>% group_by(class) %>% summarise(count = n())
New_data

library(ggplot2)
g_test = ggplot(data = New_data_test,mapping = aes(class,count,color = class,fill = class))+
        geom_col()+
        geom_text(mapping = aes(label = count),color = "black",position = position_stack(vjust = 0.5))+
        ggtitle(paste("Pie diagram on the response of test data : class"))+
        coord_polar()

par(mfrow = c(1,2))
plot(g_train)
plot(g_test)



###Does Not work

#####################################################
#Multiple Correspondence Analysis(MCA)
library(FactoMineR)
mca = MCA(D[,-1])
#plot(mca)
mca$eig
#####################################################
###Does not work





###################################################################################
#Data Reduction
library(elasticnet)
library(sparsepca)
result = spca(Reduced_data5[,-1],k = 100,max_iter = 200)
summary(result)
dim(result$scores)

D3 = data.frame(class,result$scores)
dim(D3)









#Naive Bayes Estimators

library(e1071)
library(caTools)
#Naive Bayes Classifier building on the Training Data-Set
NB_Classifier2 = naiveBayes(class~.,data = D2_train)

#Prediction on the Test Data-Set
Class_NB_Pred2 = predict(NB_Classifier2,newdata = D2_test)


#Confusion Matrix
Conf_Mat2 = table(D2_test$class,Class_NB_Pred2)

#Model Evaluation
Model2 = confusionMatrix(Conf_Mat2)
Model2






#############################################
#K-Nearest Neighbor Classifier


set.seed(74823292)
start = Sys.time()
CR_Valid = train(factor(class)~.,data = D2_train,method = "kknn",
                 trControl = trainControl(method = "repeatedcv",number = 5,repeats = 1),
                 tuneGrid = data.frame(kmax = 400:405,distance = 1,kernel = "rectangular"))
end = Sys.time()
end-start
CR_Valid
PRE_CR_Valid = predict(CR_Valid,newdata = D2_test)
PRE_CR_Valid
plot(CR_Valid)
CM = table(PRE_CR_Valid,D2_test$class)
confusionMatrix(CM)

##Google Colab


#####################################################
#Logistic regression on D2
set.seed(97321830)
Logistic2 = nnet::multinom(factor(class)~.,data = D2_train)
Logistic2_fit = predict(Logistic2,newdata = D2_test)
Logistic2_fit
confusionMatrix(table(Logistic2_fit,D2_test$class))



##Penalized Logistic Regression
library(glmnet)
library(Matrix)
PEN_Logistic = glmnet(class~.,data = D2_train,family = "mutinomial")

plot(PEN_Logistic)
PEN_Logistic_fit = predict(PEN_Logistic,newdata = D2_test,s = "lambda.min",type = "class")
confusionMatrix(table(PEN_Logistic_fit,D2_test$class))










##Decision Tree
#Decision Tree
library(rpart)
library(rpart.plot)
library(tree)
library(ISLR)
Dtree = rpart(factor(class)~., D2_train)
rpart.plot(Dtree,type=5)
Decesion_tree_predict = predict(Dtree, D2_test, type='class')
confusionMatrix(table(Decesion_tree_predict,D2_test$class))
set.seed(123567)
train_control <- trainControl(method = "repeatedcv",number = 5, repeats = 2)
#Choose optimum Complexity parameter
printcp(Dtree)
plotcp(Dtree)
#Postpruning
# Prune the model based on the optimal cp value
tree_pruned <- prune(Dtree, cp = 0.01 )
rpart.plot(tree_pruned, type=5)
# Compute the accuracy of the pruned tree
pred_51 <- predict(tree_pruned, D2_test, type = "class")
confusionMatrix(table(Decesion_tree_predict,D2_test$class))


##Random Forest
set.seed(55677667)
crtl = trainControl(method = "repeatedcv",number = 5,repeats = 5)
Random_Forest = train(factor(class)~.,data = D2_train,method = "rf",
                      metric = "Accuracy",tunelenth = 10)
Random_Forest_fit = predict(Random_Forest,newdata = D2_test)
confusionMatrix(table(Random_Forest_fit,D2_test$class))






## Neural Network
set.seed(7648831)
crl = trainControl(method = "repeatedcv",number = 5,repeats = 5)
NNET = train(factor(class)~.,data = D2_train,method = "nnet",trControl = crl,maxit = 1000,
             tunegrid =expand.grid(size = seq(2, 27,by =5),decay = 10^seq(-9,0,by = 1)),metric = "Accuracy")
NNET_fit = predict(NNET,newdata = D2_test)


saveRDS(NNET,file = "NNET.rds")       #Saving in RDS file
NNET = readRDS("A:/Mentorship - Data Mining/NNET.rds")


confusionMatrix(table(NNET_fit,D2_test$class))
ggplot(NNET)

NeuralNetTools::plotnet(NNET,max_sp = TRUE, y_names = c("EI","IE","N"),segments = ("lwd" = 0.1),
                        circle_col = "yellow",bord_col = "darkblue",
                        pos_col = "darkgreen")















#################################################################
##Data Reduction Using Dutta
count <- function(vec,k)
{
  n <- length(vec)
  tab <- numeric(2**k)
  
  for( i in 1:(n-k+1) )
  {
    temp <- strtoi( paste(vec[i:(i+k-1)], collapse = ""), base = 2) + 1
    tab[temp] <- tab[temp] + 1
  }
  return(tab)
}




























