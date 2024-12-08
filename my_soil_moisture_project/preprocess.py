import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    
    # Print column names to debug
    print("Columns in the dataset:", data.columns)
    
    # Preprocess data
    data = data.dropna()  # Drop rows with missing values
    
    # Define features (X) and target variable (y)
    X = data[['moisture0']]  # Use the relevant feature column
    y = data['irrgation']  # Assuming 'irrgation' is the target variable
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
