Practical Machine Learning - Prediction Assignment Writeup
==========================================================

For this assignment I analyzed the provided data to determine what activity an individual perform.
To do this I made use of caret and randomForest, this allowed me to generate correct answers for
each of the 20 test data cases provided in this assignment.  I made use of a seed value for 
consistent results.


```{r}
library(Hmisc)
library(caret)
library(randomForest)
library(foreach)
library(doParallel)
set.seed(2048)
options(warn=-1)
```

First, I loaded the data both from the provided training and test data provided by COURSERA.
Some values contained a "#DIV/0!" that I replaced with an NA value.

```{r}
training_data <- read.csv("pml-training.csv", na.strings=c("#DIV/0!") )
evaluation_data <- read.csv("pml-testing.csv", na.strings=c("#DIV/0!") )
```

I also casted all columns 8 to the end to be numeric.

```{r}
for(i in c(8:ncol(training_data)-1)) {training_data[,i] = as.numeric(as.character(training_data[,i]))}

for(i in c(8:ncol(evaluation_data)-1)) {evaluation_data[,i] = as.numeric(as.character(evaluation_data[,i]))}
```

Some columns were mostly blank.  These did not contribute well to the prediction.  I chose a feature
set that only included complete columns.  We also remove user name, timestamps and windows.  

Determine and display out feature set.

```{r}
feature_set <- colnames(training_data[colSums(is.na(training_data)) == 0])[-(1:7)]
model_data <- training_data[feature_set]
feature_set
```

We now have the model data built from our feature set.

```{r}
idx <- createDataPartition(y=model_data$classe, p=0.75, list=FALSE )
training <- model_data[idx,]
testing <- model_data[-idx,]
```

We now build 5 random forests with 150 trees each. We make use of parallel processing to build this
model. I found several examples of how to perform parallel processing with random forests in R, this
provided a great speedup.

```{r}
registerDoParallel()
x <- training[-ncol(training)]
y <- training$classe

rf <- foreach(ntree=rep(150, 6), .combine=randomForest::combine, .packages='randomForest') %dopar% {
randomForest(x, y, ntree=ntree) 
}
```

Provide error reports for both training and test data.
```{r}
predictions1 <- predict(rf, newdata=training)
confusionMatrix(predictions1,training$classe)


predictions2 <- predict(rf, newdata=testing)
confusionMatrix(predictions2,testing$classe)
```

Conclusions and Test Data Submit
--------------------------------

As can be seen from the confusion matrix this model is very accurate.  I did experiment with PCA 
and other models, but did not get as good of accuracy. Because my test data was around 99% 
accurate I expected nearly all of the submitted test cases to be correct.  It turned out they 
were all correct.

Prepare the submission. (using COURSERA provided code)

```{r}
pml_write_files = function(x){
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}


x <- evaluation_data
x <- x[feature_set[feature_set!='classe']]
answers <- predict(rf, newdata=x)

answers

pml_write_files(answers)
```
