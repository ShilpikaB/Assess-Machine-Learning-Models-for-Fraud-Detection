library(ggplot2)
library(lattice)
library(MASS)
library(tidyverse)
library(tibble)
library(caret)
library(dplyr)
library(lubridate)
library(glmnet)

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
# Convert everything to numeric
#----------- 
ds_fraud_mat = data.matrix(data.frame(unclass(ds_fraud))) 
ds_fraud_frm = as.data.frame(ds_fraud_mat)
contrasts(as.factor(ds_fraud$Type))

#-----------
# Standardized features 
#-----------  
#y_resp = train_ds_dwn$response_fraud
#ds_fraud_scaled = apply(ds_fraud, 2, function(y_resp) (y_resp - mean(y_resp)) / sd(y_resp) ^ as.logical(sd(y_resp)))


#-------------
# Splitting the data into Training and Test sets
#-------------
split_val = 0.7
total_len = dim(ds_fraud_frm)[1]

train_idx = sample(total_len, total_len*split_val)
train_ds = ds_fraud_frm[train_idx,]
#dim(train_ds)

test_ds = ds_fraud_frm[-train_idx,]
#dim(test_ds)




#-------------
# Up-sampling the minority class "Suspected_Fraud"
#-------------
train_ds_Up = UpSampling(train_ds, "response_fraud")
table(train_ds_Up$response_fraud)

#-------------
# Down-sampling the majority classes to the minority class "Suspected_Fraud" frequency
#-------------
train_ds_dwn = DownSampling(train_ds, "response_fraud")
table(train_ds_dwn$response_fraud)
dim(train_ds_dwn)

unique(train_ds_dwn$response_fraud)

#correlations
d = ds_cleaned[,c(2,3,4,5,10,11,12,13,14,15)]
cm = cor(d, method="pearson")
library(corrplot)
library(RColorBrewer)
col<-colorRampPalette(c("#BB4444","#EE9988","#FFFFFF","#77AADD","#4477AA"))
corrplot(cm,method="color",col=col(50),type="upper",order="hclust",
         addCoef.col="black",tl.col="black",tl.srt=45,number.cex=0.7)

#-------------
# Logistic Regression
#-------------
#
y_resp = train_ds_dwn$response_fraud
#
w = ifelse(y_resp==1,(1/table(y_resp)[1])*0.999,(1/table(y_resp)[2])*0.001)

log.fit=glm(response_fraud~., data=train_ds_dwn, family=binomial(link="logit"), weights=w)
#summary(log.fit)

log.prob = predict(log.fit, newdata = test_ds, scale=1, type="response")
log.prob[1:10]
log.pred = rep(0,length(log.prob))
log.pred[log.prob>0.5] = 1

table(log.pred, test_ds$response_fraud)
mean(log.pred == test_ds$response_fraud)

#-------------
# Lasso Regression
#-------------
X = model.matrix(response_fraud~.,train_ds_dwn)[,-1]
colnames(X)
dim(X)

set.seed(81)
grid=10^seq(10, -2,length=100) #grid of lambda
cv.lasso = cv.glmnet(x=X, y=train_ds_dwn$response_fraud, alpha=1, 
                     family=binomial(link="logit"), nfolds=5)

plot(cv.lasso)
bestlam=cv.lasso$lambda.1se
bestlam

lasso.fit = glmnet(x=X, y=train_ds_dwn$response_fraud, alpha=1, family=binomial(link="logit"),
                   lambda=bestlam)
lasso.fit

lasso_pred = predict(lasso.fit, newdata=test_ds, type="coefficients")
sqrt(mean((lasso.pred-y.test)^2)) #lower than OLS error:193253
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]

X = train_ds_dwn%>%
  dplyr::select(-Order_Status)
X.bm = as.matrix(X)
# lasso, default
par(mfrow=c(1,2))
fit.lasso = biglasso(X.bm, y=X.bm$response_fraud, family = 'gaussian')
plot(fit.lasso, log.l = TRUE, main = 'lasso')
fit.lasso


grid=10^seq(10,-2,length=100) #grid of lambda
lasso_fraud.mod=glmnet(train_fraud, train_ds_dwn$response_fraud, 
                 alpha=1,lambda=grid, family="binomial")
plot(lasso_fraud.mod)#coefficient plot
set.seed(1)
cv.out=cv.glmnet(train_fraud, train_ds_dwn$response_fraud,alpha=1, family="binomial")
plot(cv.out)#CV error plot
bestlam=cv.out$lambda.min #cv.out$lambda.1se
lasso.pred=predict(lasso.mod,s=bestlam,newx=x[test,])
sqrt(mean((lasso.pred-y.test)^2)) #lower than OLS error:193253
out=glmnet(x,y,alpha=1,lambda=grid)
lasso.coef=predict(out,type="coefficients",s=bestlam)[1:20,]
lasso.coef
lasso.coef[lasso.coef!=0]

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