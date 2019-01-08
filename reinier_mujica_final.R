
# Limpiamos el workspace de R
```{r}
rm(list=ls())
setwd("d:/MADM/Analisis/statics_final_task/")
```

# Cargamos los datos
```{r}
datos <- read.csv2('HI.csv', dec=".", sep = ",")

# eliminamos la primera columna que es solo un contador de filas
datos["X"] = NULL

# eliminamos posibles valores NA dentro del dataset
datos = na.omit(datos)
attach(datos)
colnames(datos)
```

# Dividimos el dataset en TRAIN y TEST

```{r}
seed = 1991
set.seed(seed)

# generamos los datos de TRAIN con el 70% del dataset
train.size = round(dim(datos)[1] * 0.7)
train.indexs = sample(1:dim(datos)[1], train.size)
train.data = datos[train.indexs, ]

test.indexs = -train.indexs
test.data = datos[test.indexs, ]
test.size = dim(test.data)[1]

```

# Realizamos algunas Visualizaciones
```{r}
library(ggplot2)
ggplot(data = datos) +
  geom_bar(mapping = aes(x = region))

```

```{r}
ggplot(data = datos) +
  geom_bar(mapping = aes(x = race))

```


```{r}
ggplot(data = datos) +
  geom_bar(mapping = aes(x = hispanic))

```

# Aplicamos un MCO a los datos 

```{r}
set.seed(seed)
mco.fit = glm(whrswk ~ ., data = train.data)
mco.pred = predict(mco.fit, newdata = test.data)

error.mco <- mean((test.data[, "whrswk"] - mco.pred) ^ 2)

summary(mco.fit)
```


# Aplicamos métodos mas avanzados

# Usando Random Forest

```{r}
library(randomForest)
library(MASS)
set.seed(seed)
Ntree = 500

rf.fit=randomForest(whrswk~.,data=train.data, ntree = Ntree)
rf.fit
rf.pred = predict(rf.fit, test.data)
error.rf = with(test.data, mean((whrswk - rf.pred)^2))

```

```{r}
set.seed(seed)
Ntree = 200

dimension = dim(datos)[2] - 1
oob.error = double(dimension)
error.rf.full =double(dimension)

for(mtry in 1:dimension) {
  fit = randomForest(whrswk~., data = train.data, mtry = mtry, ntree = Ntree)
  oob.error[mtry] = fit$mse[Ntree]
  pred = predict(fit, test.data)
  error.rf.full[mtry] = with(test.data, mean((whrswk - pred) ^ 2))
  message(mtry)
}

matplot(1:mtry, cbind(error.rf.full, oob.error), pch=19, col=c("red", "blue"), type="b", ylab="Mean Squared Error")
legend("topright",legend=c("Test","OOB"),pch=19,col=c("red","blue"))
```

```{r}
# nos quedamos con el menor error de los Random Forests
error.rf = min(c(error.rf, min(error.rf.full)))
```


# BOOSTING

```{r}
library(gbm)
Ntree = 1000 
set.seed(seed)

boosting.fit = gbm(whrswk~., data=train.data, distribution = "gaussian" , n.trees = Ntree, shrinkage=0.01, interaction.depth = 4)
summary(boosting.fit)
```

```{r}
n.trees=seq(from=100, to=Ntree, by=100)
boosting.pred = predict(boosting.fit, newdata = test.data, n.trees=n.trees)

error.boosting = with(test.data, apply((boosting.pred - whrswk) ^ 2, 2, mean))

plot(n.trees, error.boosting, pch=19, ylab="Mean Squared Error", xlab="# Trees",main="Boosting Test Error")

error.boosting = min(error.boosting)
```

```{r}
CV = 5
Ntree = 1000
set.seed(seed)

boosting.cv = gbm(whrswk~., data=train.data, distribution = "gaussian", n.trees=Ntree, cv.folds = CV, shrinkage = 0.01, interaction.depth = 4)

best_iter <- gbm.perf(boosting.cv, method="cv")
```

```{r}
summary(boosting.cv, n.trees = best_iter) 
```

```{r}

n.trees = seq(from=100, to=best_iter, by=100)
predmat = predict(boosting.fit, newdata = test.data, n.trees = n.trees)

error.boosting_cv = with(test.data, apply((predmat - whrswk) ^ 2, 2, mean))
plot(n.trees, error.boosting_cv, pch=19, ylab = "Mean Squared Error", xlab = "# Trees", main = "Boosting Test Error")
error.boosting_cv = min(error.boosting_cv)
``` 

# BOOSTING CON CARET
```{r}
library(caret)
set.seed(seed)
Ntree = 1000

gbmGrid <- expand.grid(interaction.depth = 4,
                       n.trees = seq(200, Ntree, 200),
                       shrinkage = c(0.01, 0.05),
                       n.minobsinnode = 10)
fitControl <- trainControl(method = 'cv', number = 5, summaryFunction = defaultSummary)

out <- capture.output(gradientBoosting.fit <- train(whrswk~., data = train.data, method = 'gbm', trControl=fitControl, tuneGrid=gbmGrid, metric='RMSE'))

plot(gradientBoosting.fit)

```

```{r}
gradientBoosting.fit$bestTune
```

```{r}
res <- gradientBoosting.fit$results
RMSE <- subset(res[5])
```

```{r}
gradientBoosting.pred <- predict(gradientBoosting.fit, test.data)
error.gradientBoosting = (mean((gradientBoosting.pred - test.data$whrswk) ^ 2))
```

# Comparamos y gráficamos los resultados
```{r}
results <- matrix(NA, nrow = 1, ncol = 5)
colnames(results) <- c("MCO","Random Forests", "Boosting", "Boosting con 5-CV", "Gradient Boosting")
rownames(results) <- c("Error")
results <- as.table(results)

results["Error", "MCO"] = error.mco
results["Error", "Random Forests"] = error.rf
results["Error", "Boosting"] = error.boosting
results["Error", "Boosting con 5-CV"] = error.boosting_cv
results["Error", "Gradient Boosting"] = error.gradientBoosting

results

```
```{r}
gradientBoosting.fit$bestTune
```
