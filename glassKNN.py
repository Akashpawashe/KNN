
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

glass = pd.read_csv("D:\\excelR\\Data science notes\\KNN\\asgmnt\\glass.csv")

# Measure of dispersion
np.var(glass)
np.std(glass)
# Skewness and kurtosis
from scipy.stats import skew, kurtosis
skew(glass )
kurtosis(glass)

# Histogram 
plt.hist(glass['RI']);plt.title('Histogram of RI');plt.xlabel('RI');plt.ylabel('Frequency')
plt.hist(glass['Na'], color='yellow');plt.title('Histogram of Na');plt.xlabel('Na');plt.ylabel('Frequency')




from sklearn.model_selection import train_test_split
train,test = train_test_split(glass,test_size = 0.2) 

from sklearn.neighbors import KNeighborsClassifier as KNC 
neigh = KNC(n_neighbors= 3)
 
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9]) # 94 %

# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9]) # 100%

neigh = KNC(n_neighbors=5)
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])

neigh = KNC(n_neighbors=15)
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values 
 
for i in range(3,50,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc = np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])


import matplotlib.pyplot as plt # library to do visualizations 

# train accuracy plot 
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")

plt.legend(["train","test"])



