install.packages(c("quantmod", "PerformanceAnalytics", "caret", "glmnet", "xgboost", "nnet", "rpart"))
library(quantmod)
library(PerformanceAnalytics)
library(caret)
library(glmnet)
library(xgboost)
library(nnet)
library(rpart)
symbols <- c("AAPL", "MSFT", "GOOG", "AMZN", "META")
getSymbols(symbols, from = "2018-01-01", to = "2023-12-31")
prices <- do.call(merge, lapply(symbols, function(sym) Cl(get(sym))))
returns <- na.omit(ROC(prices, type = "discrete"))
# Create lagged returns as features
features <- na.omit(merge(
  lag(returns, 1),
  lag(returns, 2),
  lag(returns, 3)
))
colnames(features) <- paste0(rep(symbols, each = 3), "_Lag", rep(1:3, times = length(symbols)))
target <- returns[, "AAPL"]  # Predicting AAPL next-day return
dataset <- na.omit(merge(features, target))
colnames(dataset)[ncol(dataset)] <- "target"
train_idx <- 1:floor(0.8 * nrow(dataset))
train_data <- dataset[train_idx, ]
test_data <- dataset[-train_idx, ]
x_train <- as.matrix(train_data[, -ncol(train_data)])
y_train <- train_data$target

lasso_model <- cv.glmnet(x_train, y_train, alpha = 1)
lasso_pred <- predict(lasso_model, as.matrix(test_data[, -ncol(test_data)]), s = "lambda.min")
tree_model <- rpart(target ~ ., data = as.data.frame(train_data), method = "anova")
tree_pred <- predict(tree_model, newdata = as.data.frame(test_data))
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dtest <- xgb.DMatrix(data = as.matrix(test_data[, -ncol(test_data)]))
xgb_model <- xgboost(data = dtrain, nrounds = 50, objective = "reg:squarederror", verbose = 0)
xgb_pred <- predict(xgb_model, dtest)
nn_model <- nnet(target ~ ., data = as.data.frame(train_data), size = 5, linout = TRUE, maxit = 500, trace = FALSE)
nn_pred <- predict(nn_model, newdata = as.data.frame(test_data))
# Create return series for each model
model_preds <- list(
  LASSO = lasso_pred,
  Tree = tree_pred,
  XGBoost = xgb_pred,
  NeuralNet = nn_pred
)

results <- list()
actual <- test_data$target

for (model in names(model_preds)) {
  signal <- ifelse(model_preds[[model]] > 0, 1, 0)  # Long if prediction > 0
  strategy_return <- actual * signal
  results[[model]] <- xts(strategy_return, order.by = index(test_data))
}
returns_df <- do.call(merge, results)
colnames(returns_df) <- names(results)
charts.PerformanceSummary(returns_df)
table.Performance(returns_df)