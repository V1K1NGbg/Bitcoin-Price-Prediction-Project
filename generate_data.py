import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model


def test_data(real, pre):    
    total = 0

    for i in range(0, len(pre)):
        d = abs(pre[i] - real[i])
        # print(f"{pre[i]} - {real[i]} = {d}")

        total += d
    
    print(total/len(pre))


# reg = ensemble.GradientBoostingRegressor(n_estimators = 600, max_depth = 7, min_samples_split = 5, learning_rate = 0.7, loss = 'squared_error')
# reg = ensemble.GradientBoostingRegressor(n_estimators = 400, max_depth = 2, min_samples_split = 3,learning_rate = 0.07, loss = 'squared_error')
# reg = ensemble.GradientBoostingRegressor(n_estimators = 1000, max_depth = 15, min_samples_split = 9, learning_rate = 0.2, loss = 'squared_error')

reg = linear_model.LinearRegression()



# conv_dates = [0 if ("2011" in values or "2012" in values or "2013" in values or "2014" in values or "2015" in values or "2016" in values) else 1 for values in data.date ]

# print(data.head())
# data = data.drop(["Ask1#", "Ask2#", "Ask3#", "Ask4#", "Ask5#", "Ask6#", "Ask7#", "Ask8#", "Ask9#", "Ask10#", "Bid1#", "Bid2#", "Bid3#", "Bid4#", "Bid5#", "Bid6#", "Bid7#", "Bid8#", "Bid9#", "Bid10#"], axis=1)


# conv_dates = [0 if ("2011" in values or "2012" in values or "2013" in values or "2014" in values or "2015" in values or "2016" in values) else 1 for values in data.date ]

datatest = pd.read_excel("TrainingSetBTCChallenge.xlsx")

train2 = datatest['TARGET']
train1 = datatest.drop('TARGET', axis=1)

# train1 = data.drop('lat', axis=1)
# train1 = data.drop('long', axis=1)

# train2 = list(map(lambda p: np.log2(p), train2))

# print(train1, len(train2), train2[0])

reg.fit(train1, train2)

x_train, x_test, y_train, y_test = train_test_split(
    train1, train2, test_size=0.10)

# y_train = list(map(lambda p: np.log2(p), y_train))

# x_pred = list(map(lambda p: 2**p, clf.predict(x_test)))

# print(type(predlist))
# print(y_test, x_pred)

# print(clf.get_params())
print(reg.score(x_test,y_test))


datatest = pd.read_excel("TestSetBTCChallenge.xlsx")

results = reg.predict(datatest)

datatest["TARGET"] = results
datatest.to_excel('result.xlsx')

# print(datatest.head())
# print(reg.get_params())
# print(reg.score())
# print(data_accuracy(y_test, x_pred))

