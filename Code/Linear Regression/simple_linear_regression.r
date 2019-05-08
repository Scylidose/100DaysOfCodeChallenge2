  # Importing the data-set
  dataset = read.csv("/Users/Scylidose/Desktop/ML Folder/Linear Regression/breslow.csv")
  
  # Splitting the dataset into the Training set and the Test set
  # install.packages('caTools')
  library(caTools)
  set.seed(123)
  split = sample.split(dataset$age, SplitRatio = 2/3)
  training_set = subset(dataset, split == TRUE)
  test_set = subset(dataset, split == FALSE)
  
  # Feature Scaling
  # training_set[, 3:6] = scale(training_set[, 3:6])
  # test_set[, 3:6] = scale(test_set[, 3:6])
  
  # fitting Simple Linear Regression to the Training set
  regressor = lm(formula = age ~ ns, 
                 data = training_set)
  
  # Predicting the Test set results
  y_pred = predict(regressor, newdata = test_set)
  
  # Visualising the Training set results
  # install.packages('ggplot2')
  library(ggplot2)
  ggplot() + geom_point(aes(x = training_set$ns, y = training_set$age),
                        colour = 'red') + 
              geom_line(aes(x = training_set$ns, y = predict(regressor, newdata = test_set)),
                        colour = 'blue') + 
              ggtitle('The number of smoker vs Age (Training Set)') +
                xlab('The number of smoker') + 
                  ylab('Age')
  
  # Visualising the Test set results
  ggplot() + geom_point(aes(x = test_set$ns, y = test_set$age),
                        colour = 'red') + 
    geom_line(aes(x = training_set$ns, y = predict(regressor, newdata = test_set)),
              colour = 'blue') + 
    ggtitle('The number of smoker vs Age (Test Set)') +
    xlab('The number of smoker') + 
    ylab('Age')