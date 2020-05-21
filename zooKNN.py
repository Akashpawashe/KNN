

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

zoo = pd.read_csv("D:\\excelR\\Data science notes\\KNN\\asgmnt\\Zoo.csv")
zoo.info()
zoo.shape
zoo.head()
zoo.describe()
zoo.columns
zoo.isnull().sum() # no missing values
zoo.drop_duplicates(keep='first',inplace= True) # no
zoo.corr()
zoo.type.value_counts()

import seaborn as sns
sns.countplot(zoo['hair']).set_title('Countplot of Hair')
sns.countplot(zoo['hair']).set_title('Countplot of Hair')
sns.countplot(zoo['feathers']).set_title('Countplot of Feathers')
sns.countplot(zoo['eggs']).set_title('Countplot of Eggs')
sns.countplot(zoo['milk']).set_title('Countplot of Milk')
sns.countplot(zoo['airborne']).set_title('Countplot of Airborne')
sns.countplot(zoo['aquatic']).set_title('Countplot of Aquatic')
sns.countplot(zoo['predator']).set_title('Countplot of Predator')
sns.countplot(zoo['toothed']).set_title('Countplot of Toothed')
sns.countplot(zoo['backbone']).set_title('Countplot of Backbone')
sns.countplot(zoo['breathes']).set_title('Countplot of Breathes')
sns.countplot(zoo['venomous']).set_title('Countplot of Venomous')
sns.countplot(zoo['fins']).set_title('Countplot of Fins')
sns.countplot(zoo['legs']).set_title('Countplot of Legs')
sns.countplot(zoo['tail']).set_title('Countplot of Tail')
sns.countplot(zoo['domestic']).set_title('Countplot of Domestic')
sns.countplot(zoo['catsize']).set_title('Countplot of Catsize')
sns.countplot(zoo['type']).set_title('Countplot of Type of Animals')

# Boxplot

sns.boxplot(zoo['feathers'], orient='v').set_title('Boxplot of Feathers')
sns.boxplot(zoo['airborne'], orient='v').set_title('Boxplot of Airborne')
sns.boxplot(zoo['backbone'], orient='v').set_title('Boxplot of Backbone')
sns.boxplot(zoo['breathes'], orient='v').set_title('Boxplot of Breathes')
sns.boxplot(zoo['venomous'], orient='v').set_title('Boxplot of Venomous')
sns.boxplot(zoo['fins'], orient='v').set_title('Boxplot of Fins')
sns.boxplot(zoo['legs'], orient='v').set_title('Boxplot of Legs')
sns.boxplot(zoo['domestic'], orient='v').set_title('Boxplot of Domestic')
sns.boxplot(zoo['type'], orient='v').set_title('Boxplot of Type of type')
sns.boxplot(zoo['eggs'], orient='v').set_title('Boxplot of eggs')
sns.boxplot(zoo['milk'], orient='v').set_title('Boxplot of milk')
sns.boxplot(zoo['aquatic'], orient='v').set_title('Boxplot of aquatic')
sns.boxplot(zoo['predator'], orient='v').set_title('Boxplot of predator')
sns.boxplot(zoo['toothed'], orient='v').set_title('Boxplot of toothed')
sns.boxplot(zoo['tail'], orient='v').set_title('Boxplot of tail')
sns.boxplot(zoo['catsize'], orient='v').set_title('Boxplot of catsize')
sns.boxplot(zoo['hair'], orient='v').set_title('Boxplot of hair')

# Heatmap
sns.heatmap(zoo.corr(), annot=True)

from sklearn.model_selection import train_test_split
train,test = train_test_split(zoo,test_size = 0.2, random_state=0)  

from sklearn.neighbors import KNeighborsClassifier as KNC
# for 3 nearest neighbours 
neigh = KNC(n_neighbors= 5)
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])

# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17]) 
train_acc # 96.25%
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17]) 
test_acc ##  100%

# for 3 nearest neighbours
neigh = KNC(n_neighbors=3)
# fitting with training data
neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
train_acc ## 97.5%
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
test_acc ## 100%
# creating empty list variable 
#acc = []
acc = []
# storing the accuracy values 
for i in range(3,25,2):  ####changes should be done in range considering data
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:17],train.iloc[:,17])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:17])==train.iloc[:,17])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:17])==test.iloc[:,17])
    acc.append([train_acc,test_acc])
 
# train accuracy plot 
plt.plot(np.arange(3,25,2),[i[0] for i in acc],"bo-")
plt.plot(np.arange(3,25,2),[i[1] for i in acc],"ro-")
plt.legend(["train","test"])
