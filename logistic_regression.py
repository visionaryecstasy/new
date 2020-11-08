from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns',1000)
pd.set_option('display.width',1000)
pd.set_option('display.max_colwidth',1000)

df = pd.read_csv('listings.csv')
sns.regplot(x='latitude',y = 'longitude', data = df)
plt.show()

le = LabelEncoder()
le.fit(df['room_type'])
y=le.transform(df['room_type'])
print(y)

model_log = linear_model.LogisticRegression()
features=['longitude','latitude','price']
X = df[features]
model_log.fit(X,y)
#model_log = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
 #         intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
 #         penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
  #        verbose=0, warm_start=False)

features = ['longitude','latitude','price']
scores = cross_val_score(model_log,X, y, cv=5,scoring='accuracy')
print(np.mean(scores))

#特征越多，识别概率越高