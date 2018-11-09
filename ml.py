import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
df = pd.read_csv("Flaveria.csv")
val = pd.get_dummies(df)
y=val.values[:, 0]
x=val.values[:,1:10]
trainX, testX, trainY, testY = train_test_split(x, y, test_size=0.11, random_state=39)
knn = KNeighborsRegressor(algorithm='auto', leaf_size=9, metric='minkowski', metric_params=None, n_jobs=1, n_neighbors=4, p=2, weights='uniform')
knn.fit(trainX, trainY)
knn.predict(testX)
print("Score KNeighborsRegressor testset: ", knn.score(testX, testY))
