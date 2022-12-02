library(ggplot2)
library(lattice)
library(MASS)
library(tidyverse)
library(tibble)
library(caret)
library(dplyr)
library(lubridate)
library(glmnet)
library(FSinR)
library(ggcorrplot)
library(prcomp) #pca
library(ggfortify) #plotting pca results
library(latex2exp)

#---------------------  FUNCTIONS  ---------------------
# Remove columns with more than a threshold (60%) missing data
remove_col_large_missing_val = function(ds, thres){
  #ds = ds_data
  #thres=0.6
  
  df_missing_val = enframe(colSums(is.na(ds)))
  col_missing_val = df_missing_val[(df_missing_val$value)/dim(ds)[1]>=thres, ]
  ds = remove_col(ds, vec=c(col_missing_val$name))
  return(ds)
}


# Remove columns from a dataframe
remove_col = function(ds, vec){
  
  print(paste("# of Columns before removal: ", dim(ds)[2]))
  print("Columns to be removed: ")
  print(vec)
  
  ds = ds[,!(names(ds) %in% vec)]
  print(paste("# of Columns after removal: ", dim(ds)[2]))
  
  return(ds)
}


# Remove columns from a dataframe with zero variance
remove_zerovariance_cols = function(ds){
  idx_near_zero_var = nearZeroVar(ds)
  cols_to_remove = NULL
  
  for(i in 1:length(idx_near_zero_var)){
    
    if(length(unique(ds[,idx_near_zero_var[i]]))== 1){
      cols_to_remove = c(cols_to_remove,idx_near_zero_var[i])
    }
  }
  
  ds = remove_col(ds, names(ds)[cols_to_remove])
  return(ds)
}

# UpSampling the minority class
UpSampling = function(df,y){
  tbl = sort(table(df[y]))
  y_minority_val = names(tbl)[1]
  y_minority_count = as.numeric(tbl)[1]
  y_majority_count = as.numeric(tbl)[2]
  
  num_rows_toadd = y_majority_count-y_minority_count
  df_minority = df[df[y]==1,]
  df_minority_added = sample_n(df_minority, num_rows_toadd, replace=TRUE)
  
  return(rbind(df,df_minority_added))
}

# DownSampling the majority class
DownSampling = function(df,y){
  #df=df_sim
  #y="y_sim_initial"
  tbl = sort(table(df[y]))
  
  y_majority_val = names(tbl)[2]
  y_minority_count = as.numeric(tbl)[1]
  y_majority_count = as.numeric(tbl)[2]
  
  df_majority = df[df[y]==0,]
  df_majority_downsized = sample_n(df_majority, y_minority_count, replace=FALSE)
  
  return(rbind(df[df[y]==1,],df_majority_downsized))
}

# Lasso Model
lasso_model = function(X, y, family_type){
  set.seed(81)
  grid=10^seq(10, -2,length=100) #grid of lambda
  
  # using 5-fold cross-validation
  cv.lasso = cv.glmnet(x=X, y=y, alpha=1, 
                       family=family_type, nfolds=5)
  
  plot(cv.lasso)
  bestlam=cv.lasso$lambda.1se
  
  # using lambda.1se
  lasso.fit = glmnet(x=X, y=y, alpha=1, 
                     family=family_type, lambda=bestlam)
  return(lasso.fit)
  
}

#------------------------------------------------------------

ds_data = read.csv("C:\\Users\\sengu\\Dropbox\\PC\\Desktop\\SFSU_Sem4\\Math748\\CapstoneProject\\MendeleyData\\DataCoSupplyChainDataset.csv", 
                 header=TRUE, sep=",")

#View(ds_data)


#-----------
# Remove columns from the dataframe with more than 60% of the cells missing data
#-----------
ds = remove_col_large_missing_val(ds_data, thres=0.6)
#names(ds)

#-------------
# Filter the zero/near-zero variance
#-------------
ds = remove_zerovariance_cols(ds)

#-------------
# Remove Column Product Images
#-------------
ds = remove_col(ds, "Product_Image")

#-------------
# Add Columns
#-------------
ds["Order_month"] = format(as.Date(ds$order_date, format="%m/%d/%Y"),"%m")
unique(ds$Order_month)
ds["Order_day"] = weekdays(as.Date(ds$order_date, format="%m/%d/%Y"))
unique(ds$Order_day)
ds["Shipping_month"] = format(as.Date(ds$shipping_date, format="%m/%d/%Y"),"%m")
unique(ds$Shipping_month)
ds["Shipping_day"] = weekdays(as.Date(ds$shipping_date, format="%m/%d/%Y"))
unique(ds$Shipping_day)
#View(ds)
#summary(ds)

#-------------
# Delete rows with Missing values
#-------------
print(paste("number of rows with NA: ",sum(apply(ds,1,anyNA))))
ds_cleaned = na.omit(ds)
print("Dimension after missing rows are deleted: ")
print(dim(ds_cleaned))


################  FRAUD DETECTION ###############
#-------------
# Add a binary column for fraud/not fraud
#-------------
unique(ds_cleaned$Order_Status)

ds_fraud = ds_cleaned%>%
  dplyr::mutate(response_fraud = ifelse(Order_Status=="SUSPECTED_FRAUD",1,0))%>%
  dplyr::select(-Order_Status,-order_date,-shipping_date)
ds_fraud = as.data.frame(ds_fraud)

#-----------
# Convert data set to numeric
#----------- 
ds_fraud_mat = data.matrix(data.frame(unclass(ds_fraud))) 
ds_fraud_frm = as.data.frame(ds_fraud_mat)
contrasts(as.factor(ds_fraud$Type))


#-----------
######################### FEATURE SELECTION ########################
#----------- 
summary(ds_fraud)
categorical_var = c(1,2,3,6,7,8,9,10,11,15,16,17,18,19,20,23,24,25,31,38,39,41,42,44,45,46,47,48,49)
quantitative_car = c(4,5,13,21,22,26,27,28,29,30,32,33,34,35,36,37,40,43,49) #12,14


#----------- 
# Filter Method: Using filterVarImp() function
#----------- 
#use roc_curve area as score
roc_imp = filterVarImp(x = ds_fraud_frm, y = ds_fraud_frm$response_fraud)

#sort the score in decreasing order
roc_imp = data.frame(cbind(variable = rownames(roc_imp), score = roc_imp[,1]))
roc_imp$score = as.double(roc_imp$score)
roc_imp[order(roc_imp$score,decreasing = TRUE),]
var1 = roc_imp$variable[roc_imp$score>2]
filter_method1_var = var1[-15]
filter_method1_var

#-----------
# Filter Method: using correlation coefficients on quantitative variables
#----------- 
#View(ds_fraud_frm[,quantitative_car])
cor_matrix = data.frame(cor(ds_fraud_frm[,quantitative_car]))
dim(cor_matrix)

summary(abs(cor_matrix[,dim(cor_matrix)[2]]))
ggcorrplot(cor_matrix)+ggtitle("Heat Map of Quantitative Variables against Transaction Type (Fraud/Not Fraud)")
colnames(ds_fraud_frm[,quantitative_car])[which(cor_matrix[,dim(cor_matrix)[2]]>=0.05)]

#-----------
# Filter Method: using correlation coefficients on categorical variables
#----------- 
cor_matrix = data.frame(cor(ds_fraud_frm[,categorical_var]))
dim(cor_matrix)

summary(abs(cor_matrix[,dim(cor_matrix)[2]]))
corrplot = ggcorrplot(cor_matrix)+
  ggtitle("Heat Map of Categorical Variables against Transaction Type (Fraud/Not Fraud)")
corrplot
colnames(ds_fraud_frm[,quantitative_car])[which(cor_matrix[,dim(cor_matrix)[2]]>=0.05)]

#-------------
# Splitting the data into Training and Test sets
#-------------
split_val = 0.7
total_len = dim(ds_fraud_frm)[1]

set.seed(25)
train_idx = sample(total_len, total_len*split_val)
train_ds = ds_fraud_frm[train_idx,]
#dim(train_ds)

test_ds = ds_fraud_frm[-train_idx,]
#dim(test_ds)

#-------------
# Up-sampling the minority class "Suspected_Fraud"
#-------------
train_ds_Up = UpSampling(train_ds, "response_fraud")
table(train_ds$response_fraud)

#-------------
# Down-sampling the majority classes to the minority class "Suspected_Fraud" frequency
#-------------
table(train_ds$response_fraud)
train_ds_dwn = DownSampling(train_ds, "response_fraud")
table(train_ds_dwn$response_fraud)
dim(train_ds_dwn)

unique(train_ds_dwn$response_fraud)

#-------------
# WRAPPER FEATURE SELECTION: Step-wise
#-------------
# Fit the full model 
full.model = lm(response_fraud ~., data = train_ds_dwn)
# Stepwise regression model
stepmodel_summary = summary(stepAIC(full.model, direction = "both", 
                      trace = FALSE))

wrapper_var = names(which(stepmodel_summary$coefficients[,4]<0.05))[-1]
wrapper_var

#-------------
# EMBEDDED FEATURE SELECTION: Lasso
#-------------
X = model.matrix(response_fraud~.,train_ds_dwn)[,-1]

# Fit Lasso Model
lasso.fit = lasso_model(X, train_ds_dwn$response_fraud, "binomial")
Lasso_var = colnames(train_ds_dwn)[abs(coef(lasso.fit)[,1])>0]
Lasso_var  # significant variables identified using Lasso

# Lasso on Test data
lasso_pred = predict(lasso.fit, newx=as.matrix(test_ds[,-49]), type="response")
dim(lasso_pred)
lasso_class = rep(0,length(lasso_pred))
lasso_class[lasso_pred>=0.5] = 1
table(lasso_class, test_ds$response_fraud)

# Calculate Test Error
TestErr_Lasso = mean(lasso_class != test_ds$response_fraud)
TestErr_Lasso

#-------------
# Principle Component Analysis PCA
#-------------
#PCA on predictors
table(ds_fraud_frm$response_fraud)
ds_fraud_dwn = DownSampling(ds_fraud_frm, "response_fraud")
table(ds_fraud_dwn$response_fraud)
dim(ds_fraud_dwn)

fraud_data.pca = prcomp(ds_fraud_dwn, scale=T)
fraud.pcs_summary = summary(fraud_data.pca)


#elbow point of scree plot
plot(fraud_data.pca$sdev^2, xlab = TeX("Principal Components  ($\\lambda$)"),
     ylab = "Proportion of Variance Explained",
     type = "b", main="Scree plot to identify the Elbow point",
     lwd=2, bg = 1, col = rainbow(50),
     xlim=c(1,dim(ds_fraud_dwn)[2]))

# Threshold the proportion of max variance explained to 90%
imp_pca.fraud = fraud.pcs_summary$importance
pca_len.fraud = length(which(imp_pca.fraud[3,]<=0.9))
pca_len.fraud

set.seed(81)
pca.ds_fraud = data.frame(fraud_data.pca$x[,1:pca_len.fraud])
pca_fraud.glmfit = glm(ds_fraud_dwn$response_fraud~., 
                   data=pca.ds_fraud, family=binomial(link="logit"))

y.pred = predict(pca_fraud.glmfit, newdata = scores[-train,])
y.test = Hitters$Salary[-train]
sqrt(mean((y.pred-y.test)^2)) #RMSE

#regular lm
lm.fit0 <- lm(Salary~.,data=Hitters,subset=train)
y.pred0 <- predict(lm.fit0,newdata = Hitters[-train,])
y.test <- Hitters$Salary[-train]
sqrt(mean((y.pred0-y.test)^2)) 


#-------------
# Logistic Regression
#-------------
#
#y_resp = train_ds_dwn$response_fraud
#
#w = ifelse(y_resp==1,(1/table(y_resp)[1])*0.999,(1/table(y_resp)[2])*0.001)
View(ds_fraud)
#unique(ds_fraud$Days_for_shipping_real,ds_fraud$Shipping_day)
ds_fraud%>%
  dplyr::group_by(Delivery_Status)%>%
  dplyr::filter(response_fraud==1)%>%
  dplyr::summarise(n())

set.seed(81)
log.fit=glm(response_fraud~., data=train_ds_dwn, family=binomial(link="logit"))
summary(log.fit)
signicoef_log.fit = data.frame(summary(log.fit)$coef[summary(log.fit)$coef[,4] <= .05, 1])
colnames(signicoef_log.fit) = c("Coefficients of Significant Features")
signicoef_log.fit

View(ds_fraud)
unique(ds_fraud$Days_for_shipping_real)
table(ds_fraud_frm$Days_for_shipping_real)
unique(ds_fraud_frm$Delivery_Status)
unique(ds_fraud$Delivery_Status)
unique(ds_fraud_frm$Customer_Segment)
table(ds_fraud$Customer_Segment)
unique(ds_fraud$Order_Region)
unique(ds_fraud$Product_Name)
unique(ds_fraud$Shipping_Mode)

contrasts(as.factor(ds_fraud$Days_for_shipping_real))
contrasts(as.factor(ds_fraud$Delivery_Status))
contrasts(as.factor(ds_fraud$Order_Region))
contrasts(as.factor(ds_fraud$Product_Name))
contrasts(as.factor(ds_fraud$Shipping_Mode))


log.prob = predict(log.fit, newdata = test_ds, scale=1, type="response")
log.prob[1:10]
log.pred = rep(0,length(log.prob))
log.pred[log.prob>0.5] = 1

table(log.pred, test_ds$response_fraud)
mean(log.pred == test_ds$response_fraud)


##########################################################################
#-------------
# Data Visualization
#-------------
# Bar plot for Transaction Type
plot = ggplot(data=ds, aes(x=Type, fill=Type)) + 
  geom_bar(stat="count") + theme_minimal() +  
  geom_text(aes(label =..count..), stat="count") +
  ggtitle("# of transactions per Transaction-Type") +
  theme(plot.title = element_text(hjust = 0.5))
plot

# Bar plot for Department Name
plot = ggplot(data=ds, aes(x=Department_Name, fill=Department_Name)) + 
  geom_bar(stat="count") + theme_minimal() +  
  geom_text(aes(label =..count..), stat="count") +
  ggtitle("# of Transactions per Department") +
  theme(plot.title = element_text(hjust = 0.5))
plot
unique(ds[ds$Department_Name=='Fan Shop',43])

# Bar plot for Order Status
plot = ggplot(data=ds, aes(x=Order_Status, fill=Order_Status)) + 
  geom_bar(stat="count") + theme_minimal() +  
  geom_text(aes(label =..count..), stat="count") +
  ggtitle("# of Transactions per Order Status") +
  theme(plot.title = element_text(hjust = 0.5))
plot

# Pie plot for Market
mrkt = ds%>%
  dplyr::group_by(Market)%>%
  dplyr::summarise(freq=n())
mrkt

plot = ggplot(mrkt, aes(x = Market, fill = Market, y=freq)) +
  geom_col() +
  geom_label(aes(label = freq),
             show.legend = FALSE) +
  coord_polar(theta = "y") +
  ggtitle("# of Transactions per Market") +
  theme(plot.title = element_text(hjust = 0.5))
plot

# Bar plot for Delivery Status
plot = ggplot(data=ds, aes(x=Delivery_Status, fill=Delivery_Status)) + 
  geom_bar(stat="count") + theme_minimal() +  
  geom_text(aes(label =..count..), stat="count") +
  ggtitle("# of Transactions per Delivery Status") +
  theme(plot.title = element_text(hjust = 0.5))
plot

# Bar plot for Customer Segment
plot = ggplot(data=ds, aes(x=Customer_Segment, fill=Customer_Segment)) + 
  geom_bar(stat="count") + theme_minimal() +  
  geom_text(aes(label =..count..), stat="count") +
  ggtitle("# of Transactions per Customer Segment") +
  theme(plot.title = element_text(hjust = 0.5))
plot

#-------------------------------------------------

#----------------
# Analyze Order Status and Market
#----------------
OrdrStat_vs_Market = ds%>%
  dplyr::group_by(Order_Status, Market)%>%
  dplyr::summarise(freq = n())
#View(OrdrStat_vs_Market)


max_freq_OrderStat_Market = OrdrStat_vs_Market%>%
  dplyr::group_by(Order_Status)%>%
  dplyr::filter(freq==max(freq))
names(max_freq_OrderStat_Market)[3] = "Max_freq"
max_freq_OrderStat_Market

min_freq_OrderStat_Market = OrdrStat_vs_Market%>%
  dplyr::group_by(Order_Status)%>%
  dplyr::filter(freq==min(freq))
names(min_freq_OrderStat_Market)[3] = "Min_freq"
min_freq_OrderStat_Market

#----------------
# Analyze Order Status and Order Month
#----------------
OrdrStat_vs_OrdrMonth = ds%>%
  dplyr::group_by(Order_Status, Order_month)%>%
  dplyr::summarise(freq = n())
View(OrdrStat_vs_OrdrMonth)


max_freq_OrderStat_OrdrMonth = OrdrStat_vs_OrdrMonth%>%
  dplyr::group_by(Order_Status)%>%
  dplyr::filter(freq==max(freq))
names(max_freq_OrderStat_OrdrMonth)[3] = "Max_freq"
max_freq_OrderStat_OrdrMonth

min_freq_OrderStat_OrdrMonth = OrdrStat_vs_OrdrMonth%>%
  dplyr::group_by(Order_Status)%>%
  dplyr::filter(freq==min(freq))
names(min_freq_OrderStat_OrdrMonth)[3] = "Min_freq"
min_freq_OrderStat_OrdrMonth

#----------------
# Analyze Order Status and Weekday
#----------------
OrdrStat_vs_OrdrWeekday = ds%>%
  dplyr::group_by(Order_Status, Order_day)%>%
  dplyr::summarise(freq = n())
View(OrdrStat_vs_OrdrWeekday)


max_freq_OrderStat_OrdrWeekday = OrdrStat_vs_OrdrWeekday%>%
  dplyr::group_by(Order_Status)%>%
  dplyr::filter(freq==max(freq))
names(max_freq_OrderStat_OrdrWeekday)[3] = "Max_freq"
max_freq_OrderStat_OrdrWeekday

min_freq_OrderStat_OrdrWeekday = OrdrStat_vs_OrdrWeekday%>%
  dplyr::group_by(Order_Status)%>%
  dplyr::filter(freq==min(freq))
names(min_freq_OrderStat_OrdrWeekday)[3] = "Min_freq"
min_freq_OrderStat_OrdrWeekday

#----------------
# Analyze Order Status and Customer Segment
#----------------
CustomerSeg_vs_OrdrStat = ds%>%
  dplyr::group_by(Customer_Segment, Order_Status)%>%
  dplyr::summarise(freq = n())
View(CustomerSeg_vs_OrdrStat)


max_freq_CustomerSeg_OrdrStat = CustomerSeg_vs_OrdrStat%>%
  dplyr::group_by(Customer_Segment)%>%
  dplyr::filter(freq==max(freq))
names(max_freq_CustomerSeg_OrdrStat)[3] = "Max_freq"
max_freq_CustomerSeg_OrdrStat

min_freq_CustomerSeg_OrdrStat = CustomerSeg_vs_OrdrStat%>%
  dplyr::group_by(Customer_Segment)%>%
  dplyr::filter(freq==min(freq))
names(min_freq_CustomerSeg_OrdrStat)[3] = "Min_freq"
min_freq_CustomerSeg_OrdrStat


##########################################################################
#when predictors should be removed, a vector of integers is
#returned that indicates which columns should be removed
cor.matrix = cor(ds_data[,1:9],use="pairwise.complete.obs")
corrplot(cor.matrix, order="hclust")

highCorr <- findCorrelation(cor.matrix,cutoff=0.75)
length(highCorr)
highCorr.names <- names(housing)[highCorr] #columns with high correlation
highCorr.names

ds_descrip = read.csv("C:\\Users\\sengu\\Dropbox\\PC\\Desktop\\SFSU_Sem4\\Math748\\CapstoneProject\\MendeleyData\\DescriptionDataCoSupplyChain.csv", 
                   header=TRUE, sep=",")
View(ds_descrip)
summary(ds_descrip)


ds_log = read.csv("C:\\Users\\sengu\\Dropbox\\PC\\Desktop\\SFSU_Sem4\\Math748\\CapstoneProject\\MendeleyData\\tokenized_access_logs.csv", 
                   header=TRUE, sep=",")
View(ds_log)
summary(ds_log)