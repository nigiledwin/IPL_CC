from joblib import load
import pandas as pd

import pathlib
curr_dir = pathlib.Path(__file__)
home_dir = curr_dir.parent.parent.parent
data_path =  home_dir.as_posix() + '/models'


model_path = data_path + "/model_xgb_10oversruns_.joblib"
model = load(model_path)
print(model)

TARGET = "1st_10overruns"
data_path_test =  home_dir.as_posix() + '/data/processed'
test_features = pd.read_csv(data_path_test + "/test_df_10overruns.csv")
X_test= test_features.drop(TARGET, axis=1)
y_test= test_features[TARGET]
y_pred=model.predict(X_test)
data = {'y_pred': y_pred, 'y_test': y_test}
df = pd.DataFrame(data)

print(df)

