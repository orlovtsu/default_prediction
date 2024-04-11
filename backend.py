import xgboost as xgb
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load

class ModelPredictor:
    def __init__(self, model_path):
        """Initialize the model predictor with a given model path."""
        self.model = xgb.Booster()
        self.model.load_model(model_path)

    def predict(self, input_df):
        """Make a prediction using the input dataframe."""
        dmatrix = xgb.DMatrix(input_df)
        prediction = self.model.predict(dmatrix)
        print(prediction)
        return prediction[0]

def normalize_data(input_df):
    """Normalize the input dataframe and apply one-hot encoding to categorical features."""
    scaler = StandardScaler()
    normalized_df = input_df.copy()
    features_to_normalize = ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AGE', 'CNT_APPLICATIONS']
    categorical_features = ['NAME_FAMILY_STATUS', 'NAME_EDUCATION_TYPE']

    # Load column names from file
    with open('column_names.txt', 'r') as f:
        column_names = f.read().split(',')
    full_features_df = pd.DataFrame(columns=column_names, index=[0])

    # Load the saved scaler and normalize the numeric features
    loaded_scaler = load('scaler.joblib')
    normalized_df[features_to_normalize] = loaded_scaler.transform(normalized_df[features_to_normalize])

    # Apply one-hot encoding to categorical features
    one_hot_categories = pd.concat([pd.get_dummies(normalized_df[col], prefix=col) for col in categorical_features], axis=1)

    # Update the full features dataframe with the one-hot encoded columns
    for column in one_hot_categories.columns:
        if column in full_features_df.columns:
            full_features_df[column] = one_hot_categories[column].values

    # Combine the normalized numeric features and one-hot encoded categorical features
    normalized_df = normalized_df[features_to_normalize + ['EXT_SOURCE_2']].join(full_features_df)
    normalized_df = normalized_df.fillna(False)
    print(input_df.iloc[0])
    print(normalized_df.iloc[0])
    return normalized_df