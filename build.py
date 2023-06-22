import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


# Importing the dataset
df = pd.read_csv('deployed_df')
x = df.drop('Price',axis=1)
y = df['Price']


# Splitting the dataset into the Training set and Test set

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)


# Train Cat Boost
from catboost import CatBoostRegressor
model_cat = CatBoostRegressor()

# Fit the model
model_cat.fit(x_train, y_train)

# Predicting the Test set results
cat_pred = model_cat.predict(x_test)

# Evaluating the model
cat_r2 = r2_score(y_test, cat_pred)
print(f"The r2 Score: {round(cat_r2*100,2)}%")

import pickle
# Saving model
pickle.dump(model_cat, open('model_cat.pkl','wb'))
