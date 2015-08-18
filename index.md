---
title: "Predicting the exercise manner - Project Writeup"
author: "Shami Gupta"
date: "Tuesday, August 18, 2015"
output: html_document
---

##Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways.

##Data Source

The training data for this project are available at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available at https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 

The authorities of the data source has been very generous in allowing their data to be used for this kind of assignment and hence it is worth to mention.

##Model Outline

1. Download the data while converting the missing values as NA
2. Perform exploratory analysis for basic data pattern
3. Create Training and Test (Validation) data from training data for cross validation testing
4. Feature Selection - Subset Data per relevance 
5. Train the model - Random Forest is selected to be used here
6. Model Validation against Training set accuracy
7. Model Validation against Validation set accuracy (Out of sample) - Cross Validation
8. Run the model on the Test set to predict the excercise manner
9. Run the supplied code to create the prediction output

##Data Processing

####Data download


```r
setInternet2(TRUE)
trainurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(trainurl, "train.csv")
train <- read.csv("train.csv",na.strings=c("", "NA", "#DIV/0!"))

testurl <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(testurl, "test.csv")
test <- read.csv("test.csv",na.strings=c("", "NA", "#DIV/0!"))
```

####Perform exploratory study for basic data pattern


```r
dim(train)
```

```
## [1] 19622   160
```


```r
table(train$classe)
```

```
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
```

The data is having 160 features - not all features may be relevant for machine learning

####Create Training and Test (Validation) data from training data for cross validation

Have decided 60% for training the model and 40% for the cross validation testing


```r
library(caret)
set.seed(90909090)
trainset <- createDataPartition(train$classe, p = 0.6, list = FALSE)
Training <- train[trainset, ]
Validation <- train[-trainset, ]
```

####Feature Selection 

Identify zero variance features


```r
nzvcol <- names(Training)[nearZeroVar(Training)]
```

Identify features with low volume of meaningful data (50% or more is NA)

```r
cntlength <- sapply(Training, function(x){ sum(is.na(x)/nrow(Training))})
nullcol <- names(cntlength[cntlength >= 0.5])
```

Identify feature with descriptions indicating low relevance

```r
descol <- c("X", "user_name", "raw_timestamp_part_1",
            "raw_timestamp_part_2", "cvtd_timestamp")
```

Exclude non important features 

```r
skipcols <- unique(c(nullcol, nzvcol, descol))
Training <- Training[,setdiff(names(Training),skipcols)]
dim(Training)
```

```
## [1] 11776    54
```

####Train the model - Random Forest is selected to be used here

Will be using random forest as the model as implemented in the randomForest package by
Breiman's random forest algorithm (based on Breiman and Cutler's original Fortran code) for
classification and regression.


```r
library(randomForest)
rfModel <- randomForest(classe ~ ., data = Training, importance = TRUE, ntrees = 5)
```

####Model Validation against Training set accuracy


```r
ptraining <- predict(rfModel, Training)
print(confusionMatrix(ptraining, Training$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3348    0    0    0    0
##          B    0 2279    0    0    0
##          C    0    0 2054    0    0
##          D    0    0    0 1930    0
##          E    0    0    0    0 2165
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1838
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

Obviously our model performs excellent against the training set, but we need to cross validate the performance against the held out set and see if we have avoided overfitting.

####Model Validation against Validation set accuracy (Out of sample) - Cross Validation

```r
pvalidation <- predict(rfModel, Validation)
print(confusionMatrix(pvalidation, Validation$classe))
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2232    3    0    0    0
##          B    0 1513   16    0    0
##          C    0    2 1352   16    0
##          D    0    0    0 1270    4
##          E    0    0    0    0 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9948          
##                  95% CI : (0.9929, 0.9962)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9934          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9967   0.9883   0.9876   0.9972
## Specificity            0.9995   0.9975   0.9972   0.9994   1.0000
## Pos Pred Value         0.9987   0.9895   0.9869   0.9969   1.0000
## Neg Pred Value         1.0000   0.9992   0.9975   0.9976   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2845   0.1928   0.1723   0.1619   0.1833
## Detection Prevalence   0.2849   0.1949   0.1746   0.1624   0.1833
## Balanced Accuracy      0.9997   0.9971   0.9928   0.9935   0.9986
```

The cross validation accuracy is 99.48% and the out-of-sample error is therefore 0.52% so the model performs reasonably well.

####Run the model on the Test set to predict the excercise manner


```r
ptest <- predict(rfModel, test)
ptest
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

####Finally, run the supplied code to create the prediction output set for submission requirement


```r
answers <- as.vector(ptest)
pml_write_files = function(x) {
  n = length(x)
  for (i in 1:n) {
    filename = paste0("problem_id_", i, ".txt")
    write.table(x[i], file = filename, quote = FALSE,
                row.names = FALSE,
                col.names = FALSE)
  }
}
pml_write_files(answers)
```

