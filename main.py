import numpy as np
import pandas as pd
import warnings
import scipy.stats as stats
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

warnings.filterwarnings('ignore')

# data import
train = pd.read_csv("datasets/train.csv")
test = pd.read_csv("datasets/test.csv")
transactions = pd.read_csv("datasets/transactions.csv").sort_values(["store_nbr", "date"])
holiday = pd.read_csv("datasets/holidays_events.csv")
oil = pd.read_csv("datasets/oil.csv")
stores = pd.read_csv("datasets/stores.csv")

train = train.merge(stores, on='store_nbr')
train = train.merge(oil, on='date', how='left')
holiday = holiday.rename(columns={'type': 'holiday_type'})
train = train.merge(holiday, on='date', how='left')

# data cleaning
train.isnull().sum()
missing_percentages = train.isnull().sum()/len(train) * 100
columns_to_delete = missing_percentages[missing_percentages > 30].index
train = train.drop(columns=columns_to_delete)
train = train.drop_duplicates()

# whether the types of stores affect the sales using the ANOVA
grouped = train.groupby('type')['sales']
f_statistic, p_value = stats.f_oneway(*[grouped.get_group(type) for type in grouped.groups])
if p_value < 0.05:
    print("types affect the sales")
else:
    print("types doesn't affect the sales")



# visualization
plt.scatter(train['onpromotion'], train['sales'])
plt.xlabel('Promotion')
plt.ylabel('Sales')
plt.title('Promotion vs Sales')
plt.show()

plt.scatter(train['type'], train['sales'])
plt.ylabel('sales')
plt.xlabel('type')
plt.show()

# grid search 确定p,q,r值
train['date']= pd.to_datetime(train['date'])
train = train.groupby('date').sum().reset_index()
train_origin = train['sales'].values.astype('float64')


def evaluate_arima_model(X, arima_order):
    train_size = int(len(X) * 0.66)
    train_set, test_set = X[0:train_size], X[train_size:]
    history = train_set
    predictions = list()
    for t in range(len(test_set)):
        model = ARIMA(history, order=arima_order)
        model_fit = model.fit()
        yhat = model_fit.forecast()[0]
        predictions.append(yhat)
        history = np.append(history, test_set[t])
    rmse = sqrt(mean_squared_error(test_set, predictions))
    return rmse

# evaluate combinations of p, d and q values for an ARIMA model


def evaluate_models(dataset, p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p,d,q)
                try:
                    rmse = evaluate_arima_model(dataset, order)
                    if rmse < best_score:
                        best_score, best_cfg = rmse, order
                    print('ARIMA%s RMSE=%.3f' % (order, rmse))
                except:
                    continue
    print('Best ARIMA%s RMSE=%.3f' % (best_cfg, best_score))


p_values = [0, 1, 2]
d_values = range(0, 3)
q_values = range(0, 3)
evaluate_models(train_origin, p_values, d_values, q_values)
