import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy as np

#对listing.csv数据集画图
df = pd.read_csv('listings.csv')
sns.regplot(x='latitude',y = 'longitude', data = df)
plt.show()

#单个特征构造模型
#lm是实例化一个线性回归模型
lm = linear_model.LinearRegression()
features = ['latitude']
X = df[features]
y = df['longitude']
print(X.shape,y.shape)
#model在这里存储训练好的模型
model = lm.fit(X,y)
print(model.intercept_,model.coef_)
print("--------------------------------")
print(" ")

#交叉验证评估模型
scores = -cross_val_score(lm,X,y,cv=5,scoring='neg_mean_absolute_error')
print(scores)
print(np.mean(scores))
scores2 = -cross_val_score(lm,X,y,cv=5,scoring='neg_mean_squared_error')
print(np.mean(scores2))
