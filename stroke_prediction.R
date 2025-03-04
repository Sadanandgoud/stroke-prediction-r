
# Stroke Prediction Model Using R

# Install necessary packages
install.packages(c("tidyverse", "caret", "e1071", "randomForest", "xgboost", "pROC", "rpart"))

# Load necessary libraries
library(tidyverse)
library(caret)
library(e1071)       # Support Vector Machine
library(randomForest) # Random Forest
library(xgboost)      # Extreme Gradient Boosting
library(pROC)         # ROC Curves
library(rpart)        # Decision Tree

# Load dataset
stroke_data <- read.csv("healthcare-dataset-stroke-data.csv")

# Convert categorical variables to factors
stroke_data$gender <- as.factor(stroke_data$gender)
stroke_data$smoking_status <- as.factor(stroke_data$smoking_status)
stroke_data$hypertension <- as.factor(stroke_data$hypertension)
stroke_data$heart_disease <- as.factor(stroke_data$heart_disease)
stroke_data$ever_married <- as.factor(stroke_data$ever_married)
stroke_data$work_type <- as.factor(stroke_data$work_type)

# Handle missing values
stroke_data$bmi <- as.numeric(stroke_data$bmi)
stroke_data$bmi[is.na(stroke_data$bmi)] <- median(stroke_data$bmi, na.rm = TRUE)

# Split data into training (80%) and testing (20%)
set.seed(123)
trainIndex <- createDataPartition(stroke_data$stroke, p = 0.8, list = FALSE)
train <- stroke_data[trainIndex, ]
test <- stroke_data[-trainIndex, ]

# Train logistic regression model
log_model <- glm(stroke ~ ., data = train, family = binomial)
log_predictions <- predict(log_model, test, type = "response")
test$log_predicted <- ifelse(log_predictions > 0.5, 1, 0)

# Train decision tree model
tree_model <- rpart(stroke ~ ., data = train, method = "class")
tree_predictions <- predict(tree_model, test, type = "class")

# Train random forest model
rf_model <- randomForest(stroke ~ ., data = train, ntree = 100)
rf_predictions <- predict(rf_model, test)

# Convert categorical variables for XGBoost
categorical_cols <- c("gender", "ever_married", "work_type", "smoking_status")
train_xgb <- train %>% mutate(across(all_of(categorical_cols), ~ as.numeric(as.factor(.)))) %>% select(-stroke)
test_xgb <- test %>% mutate(across(all_of(categorical_cols), ~ as.numeric(as.factor(.)))) %>% select(-stroke)

# Convert target variable to numeric
train_labels <- as.numeric(train$stroke) - 1
test_labels <- as.numeric(test$stroke) - 1

# Convert into XGBoost matrix format
xgb_train <- xgb.DMatrix(data = as.matrix(train_xgb), label = train_labels)
xgb_test <- xgb.DMatrix(data = as.matrix(test_xgb), label = test_labels)

# Train XGBoost Model
xgb_model <- xgboost(data = xgb_train, nrounds = 100, objective = "binary:logistic")
xgb_predictions <- predict(xgb_model, xgb_test)
test$xgb_predicted <- ifelse(xgb_predictions > 0.5, 1, 0)

# Evaluate model performance
confusionMatrix(factor(test$log_predicted), factor(test$stroke))
confusionMatrix(factor(tree_predictions), factor(test$stroke))
confusionMatrix(factor(rf_predictions), factor(test$stroke))
confusionMatrix(factor(test$xgb_predicted), factor(test$stroke))
