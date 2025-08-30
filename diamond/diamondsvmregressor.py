import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("10-diamonds.csv")


df = df.drop(["Unnamed: 0"], axis=1)
df.info()
print(df.describe())

df = df.drop(df[df["x"]==0].index)
df = df.drop(df[df["y"]==0].index)
df = df.drop(df[df["z"]==0].index)

print(df.describe())

#sns.pairplot(df)
#plt.show()

#sns.scatterplot(x=df["x"], y=df["price"])
#plt.show()

#sns.scatterplot(x=df["y"], y=df["price"])
#plt.show()

#sns.scatterplot(x=df["z"], y=df["price"])
#plt.show()

#sns.scatterplot(x=df["table"], y=df["price"])
#plt.show()

#sns.scatterplot(x=df["depth"], y=df["price"])
#plt.show()

#sns.scatterplot(x=df["z"], y=df["price"])
#plt.show()

X= df.drop(["price"],axis =1)
y= df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.25, random_state=15)
label_encoder = LabelEncoder()
for col in ['cut', 'color', 'clarity']:
    X_train[col] = label_encoder.fit_transform(X_train[col])
    X_test[col] = label_encoder.transform(X_test[col])

print(X_train.head())
print(X_train.describe())
print(X_train.info())

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
linreg=LinearRegression()
linreg.fit(X_train_scaled,y_train)
y_pred=linreg.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred)
#plt.show()

from sklearn.svm import SVR
svr=SVR()
svr.fit(X_train_scaled, y_train)
y_pred=svr.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred)
plt.show()

from sklearn.model_selection import GridSearchCV

# defining parameter range
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear']}

grid = GridSearchCV(SVR(), param_grid, refit = True,verbose = 3,n_jobs=-1) # n_jobs=-1 means use all processor
grid.fit(X_train_scaled, y_train)

grid.best_params_

y_pred=grid.predict(X_test_scaled)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)
print("Mean absolute error", mae)
print("R2 Score", score)
plt.scatter(y_test,y_pred)
plt.show()