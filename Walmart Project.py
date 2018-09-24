
# coding: utf-8

# In[1]:


import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis
import scipy.stats as stat
get_ipython().magic('matplotlib inline')
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_boston
from sklearn.cross_validation import train_test_split


# In[2]:


walmartDf = pd.read_csv('train.csv')
walmartDf.head()


# In[3]:


walmartFDF = pd.read_csv('features.csv')
walmartFDF.head()


# In[4]:


walmartFDF.MarkDown1.isnull().sum()


# In[5]:


walmartFDF.describe()


# In[6]:


walmartFDF.CPI.isnull().sum()


# In[7]:


walmartDF = pd.merge(walmartDf,walmartFDF,on=['Store','Date'], how='inner')
walmartDF.head()


# In[8]:


walmartCorr = walmartDF.corr(method='pearson')
walmartCorr


# In[9]:


a,b = plt.subplots(figsize=(6,6))
b.matshow(walmartCorr)

# In[10]:


walmartDF.describe()


# In[11]:


walmartDF.shape


# In[12]:


walmartDF.head()


# In[13]:


salesByStore = walmartDF.groupby('Store')['Weekly_Sales'].apply(lambda x : np.sum(x)) 
salesByStore.head()


# In[14]:


salesByStoreDate = walmartDF.groupby(['Store','Date'])['Weekly_Sales'].apply(lambda x : np.sum(x))
salesByStoreDate1 = salesByStoreDate.reset_index()
salesByStoreDate1.head()


# In[15]:


walmartDF.Store.unique()


# In[16]:


salesByStoreDept = walmartDF.groupby(['Store','Dept'])['Weekly_Sales'].apply(lambda x : np.sum(x)) 
salesByStoreDept.reset_index().head()


# In[17]:


salesByStoreDF1 = pd.DataFrame(salesByStore)


# In[18]:


plt.plot(salesByStoreDF1.index, salesByStoreDF1['Weekly_Sales'])
#plt.xticks(range(salesByStoreDF1.index),salesByStoreDF1.index)
plt.show()


# In[19]:


salesByStoreDF = pd.DataFrame(salesByStoreDate1)
salesByStoreDF.head()


# In[20]:


salesByStoreDF[salesByStoreDF.Weekly_Sales == salesByStoreDF.Weekly_Sales.max()]


# In[21]:


walmartDF.tail()


# In[22]:


walmartFDF['DateObj'] = walmartFDF['Date'].apply(lambda x : dt.datetime.strptime(x,'%m/%d/%Y') )
walmartFDF.head()


# In[23]:


salesByStoreDateDF = pd.DataFrame(salesByStoreDate1)
salesByStoreDateDF.info()


# In[24]:


walmartDFFinal_1 = pd.merge(walmartFDF,salesByStoreDateDF,on=['Store','Date'],how='inner')


# In[25]:


walmartDFFinal_1.head()


# In[26]:


#walmartDF.head()
plt.plot(walmartDF[(walmartDF.Store==1)]['Date'],walmartDF[(walmartDF.Store==1)]['Weekly_Sales'],'ro')
plt.xticks(rotation=90)
plt.show()


# In[27]:


plt.plot(walmartDFFinal_1[walmartDFFinal_1['Store']==1]['DateObj'],walmartDFFinal_1[walmartDFFinal_1['Store']==1]['Weekly_Sales'],'ro')
plt.xticks(rotation=90)
plt.show()


# In[28]:


plt.plot(walmartDFFinal_1['DateObj'],walmartDFFinal_1['Weekly_Sales'],'ro')
plt.xticks(rotation=90)
plt.show()


# In[29]:


walmartDFFinal_1.columns


# In[30]:


len(walmartDFFinal_1.Unemployment.unique())


# In[31]:


x = list(range(2)) + list(range(3,5))
print (x)


# In[32]:


walmartDFFinal_1.iloc[1:3:,3:5]


# In[33]:


walmartDFFinal_AllStores = walmartDFFinal_1.iloc[:,list(range(2,4)) + list(range(9,14))]
walmartDFFinal_Store = walmartDFFinal_1.iloc[:,list(range(4)) + list(range(9,14))]
walmartDFFinal_AllStores.head()


# In[34]:


walmartDFFinal_AllStores[walmartDFFinal_AllStores.IsHoliday == False]['IsHoliday'].count()


# In[35]:



walmartDFFinal_AllStores[walmartDFFinal_AllStores.IsHoliday == True]['IsHoliday'].count()


# In[36]:


def ConvertHoliday(x):    
    if x == False:        
        returnVal = 0
    else:
        returnVal = 1
    return returnVal


# In[37]:


walmartDFFinal_3 = map(ConvertHoliday,walmartDFFinal_AllStores.IsHoliday)
falsecount = truecount = 0
for i in walmartDFFinal_3:
    #print (i)
    if i == 0:
        falsecount += 1
    else:
        truecount += 1

print (falsecount, truecount)


# In[38]:


walmartDFFinal_AllStores = walmartDFFinal_AllStores.replace({True:1,False:0})
walmartDFFinal_AllStores.head()


# In[39]:


walmartDFFinal_AllStores.DateObj.isnull().sum()


# In[40]:


len(walmartDFFinal_AllStores.Unemployment.unique())


# In[41]:


del walmartDFFinal_AllStores['DateObj']


# In[42]:


walmartDFFinal_AllStores.describe()


# In[43]:


walmartDFFinal_AllStoresTrain,walmartDFFinal_AllStoresTest = train_test_split(walmartDFFinal_AllStores,test_size=0.2,random_state=43)
walmartDFFinal_AllStoresTest.shape


# In[44]:


#walmartDFTrain.shape[-1]
walmartDF_AllStoresTrain_X = walmartDFFinal_AllStoresTrain.iloc[:,range(walmartDFFinal_AllStoresTrain.shape[-1]-1)]
walmartDF_AllStoresTrain_X.head()


# In[45]:


walmartDF_AllStoresTrain_Y = walmartDFFinal_AllStoresTrain.iloc[:,-1]
walmartDF_AllStoresTrain_Y.head()


# In[46]:


walmartDF_AllStoresTest_X = walmartDFFinal_AllStoresTest.iloc[:,range(walmartDFFinal_AllStoresTest.shape[-1]-1)]
walmartDF_AllStoresTest_X.head()


# In[47]:


walmartDF_AllStoresTest_Y = walmartDFFinal_AllStoresTest.iloc[:,-1]
walmartDF_AllStoresTest_Y.head()


# In[48]:


from sklearn import linear_model


# In[49]:


reg = linear_model.LinearRegression(normalize=True)


# In[50]:


#Calculating model for all stores
model_AllStores = reg.fit(walmartDF_AllStoresTrain_X,walmartDF_AllStoresTrain_Y)


# In[51]:


model_AllStores.coef_


# In[52]:


walmartTestDF = pd.read_csv('test.csv')
walmartTestDF.head()


# In[53]:


Yhat_AllStores = model_AllStores.predict(walmartDF_AllStoresTest_X)
Yhat_AllStores


# In[54]:


plt.plot(walmartDF_AllStoresTest_Y,Yhat_AllStores,'ro')
plt.plot(walmartDF_AllStoresTest_Y,walmartDF_AllStoresTest_Y,'b-')
plt.show()


# In[55]:


#Calculate Error
def calculateError(model,testX,testY):
    yhat = model.predict(testX)
    MPSE = np.mean(abs(yhat-testY)/testY)
    MSSE = np.mean(np.square(yhat-testY))
    return MPSE, MSSE


# In[56]:


mpse, msse = calculateError(model_AllStores, walmartDF_AllStoresTest_X, walmartDF_AllStoresTest_Y)
print (mpse, msse)
#Huge error in model, needs regularization


# In[57]:


predicted = model_AllStores.predict(walmartDF_AllStoresTest_X)


# In[58]:


walmartDF_AllStoresTrain_X.describe()


# In[59]:


#Using statsmodel api to see p-values of IV and do feature selection.
import statsmodels.api as sm
walmartDF_AllStoresTrain_X2 = sm.add_constant(walmartDF_AllStoresTrain_X)
est = sm.OLS(walmartDF_AllStoresTrain_Y, walmartDF_AllStoresTrain_X2)
est2 = est.fit()
print(est2.summary())
#from this result, it seems Fuel_price should be removed as p-value is greater than most common threshold of 0.05 but need
#check with adjusted R-square before blindly removing it.


# In[60]:


walmartDF_AllStoresTrain_X.corr()


# In[61]:


from sklearn import metrics
print (metrics.r2_score(walmartDF_AllStoresTest_Y, predicted))


# In[63]:


walmartDF_AllStoresTrain_X.columns


# In[64]:


#Attempting Lasso regularization
lasso = linear_model.Lasso(alpha=0.5, max_iter=1000)


# In[65]:


#walmartDF_AllStoresTrain_X.shape
model_AllStores_Lasso = lasso.fit(walmartDF_AllStoresTrain_X,walmartDF_AllStoresTrain_Y)
Yhat_AllStores_Lasso = model_AllStores_Lasso.predict(walmartDF_AllStoresTest_X)
Yhat_AllStores_Lasso


# In[66]:


plt.plot(walmartDF_AllStoresTest_Y,Yhat_AllStores_Lasso,'ro')
plt.plot(walmartDF_AllStoresTest_Y,walmartDF_AllStoresTest_Y,'b-')
plt.show()


# In[67]:


mpse, msse = calculateError(model_AllStores_Lasso, walmartDF_AllStoresTest_X, walmartDF_AllStoresTest_Y)
print (mpse, msse)


# In[68]:


#Still huge error, perform cross-tabulation
Store_dummies = pd.get_dummies(walmartDFFinal_Store.Store,prefix='Store').iloc[:,1:]
walmartDFFinal_AllStoresCross = pd.concat([walmartDFFinal_Store,Store_dummies],axis=1)
walmartDFFinal_AllStoresCross.head()


# In[69]:


walmartDFFinal_AllStoresCross_1 = walmartDFFinal_AllStoresCross.replace({True:1,False:0})
walmartDFFinal_AllStoresCross_1.head()


# In[70]:


walmartDFFinal_AllStoresCross_Train, walmartDFFinal_AllStoresCross_Test = train_test_split(
    walmartDFFinal_AllStoresCross_1,test_size=0.3,random_state=42)


# In[71]:


walmartDFFinal_AllStoresCross_Test.shape


# In[72]:


wmDFFinal_AllStoresCross_Train_X = walmartDFFinal_AllStoresCross_Train.iloc[:,list(range(2,7)) + list(range(9,walmartDFFinal_AllStoresCross_Train.shape[1]))]
wmDFFinal_AllStoresCross_Train_X.head()


# In[73]:


wmDFFinal_AllStoresCross_Train_Y = walmartDFFinal_AllStoresCross_Train.Weekly_Sales
wmDFFinal_AllStoresCross_Train_Y.head()


# In[74]:


wmDFFinal_AllStoresCross_Test_X = walmartDFFinal_AllStoresCross_Test.iloc[:,list(range(2,7)) + list(range(9,walmartDFFinal_AllStoresCross_Test.shape[1]))]
wmDFFinal_AllStoresCross_Test_X.head()


# In[75]:


wmDFFinal_AllStoresCross_Test_Y = walmartDFFinal_AllStoresCross_Test.Weekly_Sales
wmDFFinal_AllStoresCross_Test_Y.head()


# In[76]:


wmDFFinal_AllStoresCross_Train_X.shape


# In[77]:


#Attempting backward elimination approach for getting accuracy
alpha = 0.05
df = wmDFFinal_AllStoresCross_Train_X
df['Ones'] = np.ones(wmDFFinal_AllStoresCross_Train_X.shape[0],)
dfcolumns = df.columns.values


# In[78]:


dfcolumns = list(dfcolumns[(dfcolumns.shape[0])-1:(dfcolumns.shape[0])]) + list(dfcolumns[:(dfcolumns.shape[0])-1])
dfcolumns


# In[79]:


wmdf = df[dfcolumns]
wmdf.shape


# In[80]:


from statsmodels.formula.api import OLS


model_ols = OLS(endog=wmDFFinal_AllStoresCross_Train_Y, exog=wmdf).fit()
model_ols.summary()
#keep removing feature with highest p>alpha till all features are having p<alpha.


# In[82]:


del wmDFFinal_AllStoresCross_Train_X['Ones']
wmDFFinal_AllStoresCross_Train_X.head()


# In[83]:


#First performing regression without Lasso
model_AllStores_Cross = reg.fit(wmDFFinal_AllStoresCross_Train_X,wmDFFinal_AllStoresCross_Train_Y)


# In[84]:


Yhat_AllStores_Cross = model_AllStores_Cross.predict(wmDFFinal_AllStoresCross_Test_X)


# In[85]:


model_AllStores_Cross.score(wmDFFinal_AllStoresCross_Train_X,wmDFFinal_AllStoresCross_Train_Y)


# In[86]:


#find error
mape, msse = calculateError(model_AllStores_Cross,wmDFFinal_AllStoresCross_Test_X,wmDFFinal_AllStoresCross_Test_Y)
print (mape, msse)


# In[87]:


plt.plot(wmDFFinal_AllStoresCross_Test_Y, Yhat_AllStores_Cross,'ro')
plt.plot(wmDFFinal_AllStoresCross_Test_Y,wmDFFinal_AllStoresCross_Test_Y,'b-')
plt.show()


# In[88]:


reg.coef_


# In[89]:


#Performing regression with lasso
lasso = linear_model.Lasso(alpha=1,max_iter=1000)
model_AllStores_Cross_Lasso = lasso.fit(wmDFFinal_AllStoresCross_Train_X,wmDFFinal_AllStoresCross_Train_Y)
model_AllStores_Cross_Lasso.coef_


# In[90]:


Yhat_AllStores_Cross_Lasso = model_AllStores_Cross_Lasso.predict(wmDFFinal_AllStoresCross_Test_X)


# In[91]:


#find error
mape, msse = calculateError(model_AllStores_Cross_Lasso,wmDFFinal_AllStoresCross_Test_X,wmDFFinal_AllStoresCross_Test_Y)
print (mape, msse)
#Lasso needs to be tuned as accuracy has gone down


# In[92]:


plt.plot(wmDFFinal_AllStoresCross_Test_Y, Yhat_AllStores_Cross_Lasso,'ro')
plt.plot(wmDFFinal_AllStoresCross_Test_Y,wmDFFinal_AllStoresCross_Test_Y,'b-')
plt.show()


# In[93]:


#Finding out out best value of alpha
modelErrors = np.empty(20)
for i, alpha1 in enumerate(np.linspace(1,20,20)):
    lasso1 = linear_model.Lasso(alpha=alpha1, max_iter=1000)
    model1 = lasso1.fit(wmDFFinal_AllStoresCross_Train_X,wmDFFinal_AllStoresCross_Train_Y)
    mape, msse = calculateError(model1, wmDFFinal_AllStoresCross_Test_X,wmDFFinal_AllStoresCross_Test_Y)
    #modelErrors[i] = 1-mape
    modelErrors[i] = msse


# In[94]:


#print (modelErrors*100)
print (modelErrors)


# In[95]:


from sklearn.metrics import mean_squared_error


# In[96]:


modelErrors = np.empty(50)
i=0
for alpha1 in (np.linspace(0.1,20,50)):
    lasso1 = linear_model.Lasso(alpha=alpha1)
    model1 = lasso1.fit(wmDFFinal_AllStoresCross_Train_X,wmDFFinal_AllStoresCross_Train_Y)    
    modelErrors[i] = mean_squared_error(wmDFFinal_AllStoresCross_Test_Y,model1.predict(wmDFFinal_AllStoresCross_Test_X))
    i += 1
      


# In[97]:


print (modelErrors)


# In[98]:


plt.plot(np.linspace(0.1,20,50), modelErrors)
plt.show()


# In[99]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures


# In[100]:


scalar = StandardScaler()


# In[101]:


wmDFFinal_AllStoresCross_Train_X.columns


# In[102]:


X = scalar.fit_transform(wmDFFinal_AllStoresCross_Train_X)
X.shape


# In[103]:


poly = PolynomialFeatures(2)
poly_trans = poly.fit_transform(X)


# In[104]:


poly.fit(poly_trans, wmDFFinal_AllStoresCross_Train_Y)


# In[105]:


polymodel = linear_model.LinearRegression()
polymodel.fit(poly_trans,wmDFFinal_AllStoresCross_Train_Y)


# In[106]:


X_test = scalar.transform(wmDFFinal_AllStoresCross_Test_X)
X_test.shape


# In[107]:


poly_test = PolynomialFeatures(2)
poly_trans_test = poly_test.fit_transform(X_test)


# In[108]:


mape, msse = calculateError(polymodel, poly_trans_test,wmDFFinal_AllStoresCross_Test_Y)
print (mape, msse)


# In[109]:


wmDFFinal_AllStoresCross_Test_Y.shape


# In[110]:


plt.plot(wmDFFinal_AllStoresCross_Test_Y, wmDFFinal_AllStoresCross_Test_Y,'b-')
plt.plot(wmDFFinal_AllStoresCross_Test_Y, polymodel.predict(poly_trans_test),'ro')
plt.figure(figsize=(8,8))
plt.show()


# In[111]:


#Using SVR
from sklearn.svm import SVR


# In[112]:


svr = SVR()
svrmodel = svr.fit(X,wmDFFinal_AllStoresCross_Train_Y)


# In[113]:


mape, msse = calculateError(svrmodel, X_test, wmDFFinal_AllStoresCross_Test_Y)
print (mape, msse)
#Poor prediction as Y was not scaled. Scaling doesnt happen in SVR automatically unlike in Linear/Polynomial Regression


# In[114]:


scalar1 = StandardScaler()


# In[115]:


Y = scalar1.fit_transform(wmDFFinal_AllStoresCross_Train_Y.reshape(-1,1))


# In[116]:


Y_test = scalar1.transform(wmDFFinal_AllStoresCross_Test_Y.reshape(-1,1))


# In[117]:


svr1 = SVR()
svrmodel1 = svr1.fit(X,Y)


# In[118]:


mape, msse = calculateError(svrmodel1, X_test, Y_test)
print (mape, msse)
#Still poor prediction


# In[119]:


plt.plot(wmDFFinal_AllStoresCross_Test_Y, wmDFFinal_AllStoresCross_Test_Y,'b-')
plt.plot(wmDFFinal_AllStoresCross_Test_Y, scalar1.inverse_transform(svrmodel1.predict(X_test)),'ro')
plt.figure(figsize=(8,8))
plt.show()


# In[120]:


#Using Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor


# In[121]:


dtree = DecisionTreeRegressor(random_state=10)


# In[122]:


#First checking without scaling
dtree.fit(wmDFFinal_AllStoresCross_Train_X,wmDFFinal_AllStoresCross_Train_Y)


# In[123]:


mape, msse = calculateError(dtree, wmDFFinal_AllStoresCross_Test_X, wmDFFinal_AllStoresCross_Test_Y)
print (mape, msse)


# In[124]:


plt.plot(wmDFFinal_AllStoresCross_Test_Y, wmDFFinal_AllStoresCross_Test_Y,'b-')
plt.plot(wmDFFinal_AllStoresCross_Test_Y, dtree.predict(wmDFFinal_AllStoresCross_Test_X),'ro')
plt.figure(figsize=(8,8))
plt.show()


# In[125]:


#Using scaling in DTR


# In[126]:


dtree1 = DecisionTreeRegressor(random_state=10)


# In[127]:


dtree1.fit(X,wmDFFinal_AllStoresCross_Train_Y)


# In[128]:


mape, msse = calculateError(dtree1, X_test, wmDFFinal_AllStoresCross_Test_Y)
print (mape, msse)
#Error decreased after scaling


# In[129]:


#Using RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor


# In[130]:


#trying without scaling first
rf = RandomForestRegressor(n_estimators=500,random_state=10)


# In[131]:


rf.fit(wmDFFinal_AllStoresCross_Train_X,wmDFFinal_AllStoresCross_Train_Y)


# In[132]:


mape, msse = calculateError(rf, wmDFFinal_AllStoresCross_Test_X, wmDFFinal_AllStoresCross_Test_Y)
print (mape, msse)
#Accuracy increased compared to Polynomial and DTR


# In[133]:


#trying with scaling
rf1 = RandomForestRegressor(n_estimators=500,random_state=10)
rf1.fit(X,wmDFFinal_AllStoresCross_Train_Y)
mape, msse = calculateError(rf1, X_test, wmDFFinal_AllStoresCross_Test_Y)
print (mape, msse)
#Accuracy increased after scaling


# In[134]:


plt.plot(wmDFFinal_AllStoresCross_Test_Y, wmDFFinal_AllStoresCross_Test_Y,'b-')
plt.plot(wmDFFinal_AllStoresCross_Test_Y, rf1.predict(X_test),'ro')
plt.figure(figsize=(8,8))
plt.show()


# In[135]:


#trying with scaling in Y also
rf2 = RandomForestRegressor(n_estimators=500,random_state=10)
rf2.fit(X,Y)
mape, msse = calculateError(rf2, X_test, Y_test)
print (mape, msse)
#Accuracy came down drastically with scaling in Y. Not recommended


# In[136]:


#Using k-fold cross validation to better understand model performance
from sklearn.model_selection import cross_val_score


# In[137]:


accuracies = cross_val_score(estimator=rf,X=wmDFFinal_AllStoresCross_Train_X, y=wmDFFinal_AllStoresCross_Train_Y,cv=10)
accuracies


# In[138]:


accuracies.mean()


# In[139]:


#Using GridSearchCV for tuning the model with best parameters
from sklearn.model_selection import GridSearchCV


# In[140]:


rf3 = RandomForestRegressor(n_estimators=100,random_state=10)
#rf3.fit(X,wmDFFinal_AllStoresCross_Train_Y)
params = [{'n_estimators':[300,400], 'min_samples_split':[2,3], 'min_samples_leaf':[1,2]}          
         ]


# In[141]:


grid_search = GridSearchCV(estimator=rf3,param_grid=params,cv=10,n_jobs=-1)


# In[142]:


grid_search = grid_search.fit(X,wmDFFinal_AllStoresCross_Train_Y)


# In[143]:


grid_search.best_params_


# In[144]:


grid_search.best_score_


# In[145]:


pd.DataFrame(X).head()


# In[146]:


#Attempting Gradient Descent without regularization


# In[147]:


Y_gradient = np.array([wmDFFinal_AllStoresCross_Train_Y]).transpose()
Y_gradient.shape


# In[148]:


Y_gradient.transpose().shape


# In[149]:


X


# In[150]:


X_gradient1 = np.ones(X.shape[0])
X_gradient2 = np.c_[X_gradient1,X]
X_gradient2


# In[151]:


X_gradient = X_gradient2.transpose
X_gradient().shape


# In[152]:


thetanew = np.random.randint(0,50,size=(X_gradient().shape[0],1))
thetanew.shape


# In[153]:


X_gradient().shape


iteration = 1000


# In[267]:


thetanew = np.random.randint(0,50,size=(X_gradient().shape[0],1))
theta = thetanew
errorLog = np.empty(iteration)
alpha = 0.85
for i in range(iteration):
    thetaTX = np.dot(theta.transpose(),X_gradient())
    diff = np.subtract(thetaTX,Y_gradient.transpose())    
    derivative = np.dot(diff,X_gradient().transpose())
    theta = np.subtract(thetanew, np.multiply((alpha/X_gradient().shape[1]),derivative.transpose()))
    thetaTXNew = np.dot(theta.transpose(),X_gradient())
    costerror = np.multiply(np.divide(1,np.multiply(2,X_gradient().shape[0])), 
                            np.square(np.subtract(thetaTXNew,Y_gradient.transpose())))
    #print (costerror.sum())
    errorLog[i] = costerror.sum()
    #Checking convergence
    #if(i>0 and (errorLog[i-1] - errorLog[i])<0.0001):
     #   break
    thetanew = theta


# In[268]:


plt.plot(range(iteration),errorLog)
plt.show()


# In[230]:


errorLog[iteration-2] - errorLog[iteration-1]


# In[231]:


def calculateErrorGradientDescent(theta,testX,testY):
    yhat = np.dot(theta.transpose(),testX())
    MPSE = np.mean(np.divide(np.subtract(yhat,testY.transpose()),testY.transpose()))
    return MPSE


# In[232]:


Y_test_gradient = np.array([wmDFFinal_AllStoresCross_Test_Y]).transpose()
Y_test_gradient.shape


# In[233]:


Y_test_gradient.transpose().shape


# In[234]:


X_test_gradient1 = np.ones(X_test.shape[0])
X_test_gradient2 = np.c_[X_test_gradient1,X_test]
X_test_gradient2


# In[235]:


X_test_gradient = X_test_gradient2.transpose
X_test_gradient().shape


# In[236]:


mpse = calculateErrorGradientDescent(theta,X_test_gradient,Y_test_gradient)
print (1-mpse)


# In[237]:


from numpy.linalg import inv


# In[258]:


def calculateThetaNormal(trainX,trainY):
    XTX = np.dot(trainX.transpose(), trainX)
    InvXT = np.dot(inv(XTX),trainX.transpose())
    thetaNormal = np.dot(InvXT,trainY)
    return thetaNormal


# In[259]:


Y_gradient.shape


# In[261]:


#Using normal equation
thetaNormal = calculateThetaNormal(X_gradient().transpose(), Y_gradient)
thetaNormal.shape


# In[262]:


thetaNormal


# In[263]:


theta


# In[264]:


mpse = calculateErrorGradientDescent(thetaNormal,X_test_gradient,Y_test_gradient)
print (1-mpse)

