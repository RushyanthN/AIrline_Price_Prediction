import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from feature_engine import encoding

def load_and_preprocess_data():
    bus_df = pd.read_csv(r"C:\Users\rushy\Desktop\Summer\Info_H515\Flight_Info\business.csv")
    eco_df = pd.read_csv(r"C:\Users\rushy\Desktop\Summer\Info_H515\Flight_Info\economy.csv")
    eco_df['class'] = "Economy"
    bus_df['class'] = "Business"
    eco_df = eco_df[(eco_df['airline'] == "Vistara") | (eco_df['airline'] == "Air India")]
    df = pd.concat([eco_df, bus_df], ignore_index=True)
    df = df.drop(columns=['ch_code', 'num_code'], axis=1)

    df = df[(df['from'] == "Delhi") & (df['to'] == "Mumbai")]
    df = df.drop(columns=['from', 'to'], axis=1)
    df = df.drop_duplicates(keep=False)

    df['date'] = pd.to_datetime(df['date'], format="%d-%m-%Y")
    df['day_of_week'] = df['date'].dt.day_name()

    def convert_to_time_window(dep_time):
        hour = int(dep_time.split(':')[0])
        if 0 <= hour < 4:
            return 0
        elif 4 <= hour < 8:
            return 1
        elif 8 <= hour < 12:
            return 2
        elif 12 <= hour < 16:
            return 3
        elif 16 <= hour < 20:
            return 4
        else:
            return 5

    df['dep_time_window'] = df['dep_time'].apply(convert_to_time_window)
    df['arr_time__window'] = df['arr_time'].apply(convert_to_time_window)

    def convert_to_hours(time_taken):
        hours, minutes = time_taken.split('h ')
        hours = int(hours)
        minutes = int(minutes[:-1])
        return hours + minutes / 60

    df['Duration_in_hours'] = df['time_taken'].apply(convert_to_hours).round(2)

    df['stop'] = df['stop'].replace('1-stop', 1, regex=True)
    df['stop'] = df['stop'].replace('non-stop', 0, regex=True)
    df['stop'] = df['stop'].replace('2+-stop', 2, regex=True)

    df['price'] = df['price'].str.replace(',', '').astype(float)

    df = df.drop(columns=['dep_time', 'time_taken', 'arr_time', 'date'], axis=1)

    var_num = df.select_dtypes(include='number').columns.to_list()
    var_cat = df.select_dtypes(exclude='number').columns.to_list()

    onehot = encoding.OneHotEncoder(variables=var_cat)
    onehot.fit(df)
    df_model = onehot.transform(df)
    df_model['log_price'] = np.log(df_model['price']).round(2)
    df_model = df_model.drop(columns=['price'], axis=1)

    df_economy = df_model[df_model['class_Economy'] == 1].drop(columns=['class_Business'])
    df_business = df_model[df_model['class_Business'] == 1].drop(columns=['class_Economy'])

    return df_economy, df_business, onehot , df_model

best_params = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 200
}

def train_xgboost_model(df):
    X = df.drop(columns='log_price', axis=1)
    y = df['log_price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Convert object data types to float
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = X_train[col].astype('float')
            X_test[col] = X_test[col].astype('float')

    model = XGBRegressor(**best_params,random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE: {mse}")

    return model

def predict_price(model, input_df):
    predicted_log_price = model.predict(input_df)
    predicted_price = np.exp(predicted_log_price).round(2)
    return predicted_price[0]