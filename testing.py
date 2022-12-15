import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn import ensemble


def test_data(real, pre):    
    total = 0

    for i in range(0, len(pre)):
        d = abs(pre[i] - real[i])
        # print(f"{pre[i]} - {real[i]} = {d}")

        total += d
    
    print(total/len(pre))


# reg = ensemble.GradientBoostingRegressor(n_estimators = 1100, max_depth = 15, min_samples_split = 9,learning_rate = 0.5, loss = 'squared_error')
# reg = ensemble.GradientBoostingRegressor(n_estimators = 1000, max_depth = 15, min_samples_split = 9, learning_rate = 0.2, loss = 'squared_error')
# reg = ensemble.GradientBoostingRegressor(n_estimators = 600, max_depth = 7, min_samples_split = 5, learning_rate = 0.7, loss = 'squared_error')

reg = linear_model.LinearRegression()

# reg = ensemble.AdaBoostRegressor()

# reg = ensemble.GradientBoostingRegressor()

# reg = ensemble.BaggingRegressor()

# reg = ensemble.ExtraTreesRegressor()

# reg = ensemble.HistGradientBoostingRegressor()

# reg = ensemble.RandomForestRegressor()

data = pd.read_excel("TrainingSetBTCChallenge.xlsx")

# conv_dates = [0 if ("2011" in values or "2012" in values or "2013" in values or "2014" in values or "2015" in values or "2016" in values) else 1 for values in data.date ]

# print(data.head())
# data = data.drop(["Ask1#", "Ask2#", "Ask3#", "Ask4#", "Ask5#", "Ask6#", "Ask7#", "Ask8#", "Ask9#", "Ask10#", "Bid1#", "Bid2#", "Bid3#", "Bid4#", "Bid5#", "Bid6#", "Bid7#", "Bid8#", "Bid9#", "Bid10#"], axis=1)


labels = data['TARGET']
train1 = data.drop('TARGET', axis=1)

x_train, x_test, y_train, y_test = train_test_split(
    train1, labels, test_size=0.10)

# y_train = list(map(lambda p: np.log2(p), y_train))

reg.fit(x_train, y_train)

# x_pred = list(map(lambda p: 2**p, clf.predict(x_test)))

x_pred = reg.predict(x_test)
predlist = x_pred.tolist()
anslist = y_test.to_numpy().tolist()
# print(type(predlist))
# print(y_test, x_pred)

# print(clf.get_params())
print(reg.score(x_test,y_test))
print(test_data(anslist, predlist))