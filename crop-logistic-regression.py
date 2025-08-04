# All required libraries are imported here for you.
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Load the dataset
crops = pd.read_csv("soil_measures.csv")

# Write your code here
missing_data = crops.isna().sum()
print(missing_data)

test = crops['crop'].unique()
print(test)

X = crops.iloc[:, :-1]
y = crops.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.3, random_state = 42
)

feature_performance = {}

for feature in ["N", "P", "K", "ph"]:
    log_reg = LogisticRegression(multi_class="multinomial")
    log_reg.fit(X_train[[feature]], y_train)
    y_pred = log_reg.predict(X_test[[feature]])
    feature_importance = metrics.f1_score(y_test, y_pred, average="weighted")
    feature_performance[feature] = feature_importance
    print(f"F1-score for {feature}: {feature_performance}")

best_key = max(feature_performance, key=feature_performance.get)
best_value = feature_performance[best_key]

best_predictive_feature = {best_key: best_value}
print(best_predictive_feature)





