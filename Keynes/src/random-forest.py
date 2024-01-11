# Random Forests and Decision Trees in Pricing Optimization

# Igor Mol <igor.mol@makes.ai>

# In pricing optimization, Random Forests and Decision Trees are utilized to 
# model the relationship between product prices and relevant features. 
# Decision Trees serve as the basic building blocks, capturing the 
# hierarchical decision-making process based on input features. Random 
# Forests, on the other hand, utilize an ensemble of Decision Trees to 
# enhance predictive accuracy and robustness. For instance, in the 
# 'objective_function' here, a Random Forest is trained with various 
# 'price' values to maximize the R-squared metric, providing an optimized 
# price parameter for pricing strategies, considering the interplay 
# between product attributes and pricing.

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

# extract_features_and_target:
# This function extracts features and target variable from a DataFrame.
# Parameters:
#   - df: DataFrame containing the data
# Returns:
#   - X: DataFrame with features (order_item_id, price, product_category_name_english)
#   - y: Series representing the target variable (month from shipping_limit_date)

def extract_features_and_target(df):
    # Extracting features (order_item_id, price, product_category_name_english)
    X = df[['order_item_id', 'price', 'product_category_name_english']]
    
    # Extracting target variable by converting shipping_limit_date to month
    y = pd.to_datetime(df['shipping_limit_date']).dt.month
    
    # Returning features and target
    return X, y

# training_testing_subsets:
# This function splits input features and target variable into training and tes-
# ting subsets.
# Parameters:
#   - X: DataFrame of features
#   - y: Series of target variable
#   - test_size: Proportion of data to be used for testing (default is 0.2)
#   - random_state: Seed for reproducibility (default is None)
# Returns:
#   - X_train: DataFrame of training features
#   - X_test: DataFrame of testing features
#   - y_train: Series of training target variable
#   - y_test: Series of testing target variable

def training_testing_subsets(X, y, test_size=0.2, random_state=None):
    # Setting seed for reproducibility
    np.random.seed(random_state)
    
    # Shuffle indices
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    # Calculate the number of samples for testing
    test_samples = int(test_size * len(X))
    
    # Split indices into training and testing sets
    train_indices = indices[test_samples:]
    test_indices = indices[:test_samples]
    
    # Create training and testing subsets
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    
    # Return training and testing subsets
    return X_train, X_test, y_train, y_test

# train_random_forest:
# This function trains a Random Forest ensemble on the input features and target
# variable.
# Parameters:
#   - X: DataFrame of features
#   - y: Series of target variable
#   - n_estimators: Number of decision trees in the ensemble (default is 10)
#   - random_state: Seed for reproducibility (default is None)
# Returns:
#   - estimators: List of trained decision trees in the Random Forest ensemble

def train_random_forest(X, y, n_estimators=10, random_state=None):
    # List to store the trained decision trees
    estimators = []
    
    # Loop to train each decision tree in the ensemble
    for _ in range(n_estimators):
        # Randomly sample with replacement from the data
        indices = np.random.choice(len(X), size=len(X), replace=True)
        X_sampled = X.iloc[indices]
        y_sampled = y.iloc[indices]
        
        # Create and train a decision tree
        tree = DecisionTreeRegressor(random_state=random_state)
        tree.fit(X_sampled, y_sampled)
        
        # Append the trained decision tree to the list
        estimators.append(tree)
    
    # Return the list of trained decision trees
    return estimators

# predict_random_forest:
# This function predicts the target variable using a Random Forest ensemble.
# Parameters:
#   - X: DataFrame of features for prediction
#   - estimators: List of trained decision trees in the Random Forest ensemble
# Returns:
#   - predictions: Array of predicted values for the target variable

def predict_random_forest(X, estimators):
    # Initializing an array to store predictions
    predictions = np.zeros(len(X))
    
    # Accumulating predictions from each decision tree in the ensemble
    for tree in estimators:
        predictions += tree.predict(X)
    
    # Calculating the average prediction across all trees
    return predictions / len(estimators)

# objective_function:
# This function defines an objective for optimization, aiming to maximize R-squared.
# Parameters:
#   - params: List of parameters to be optimized (in this case, only 'price' is considered)
#   - X_train: DataFrame of training features
#   - X_test: DataFrame of testing features
#   - y_train: Series of training target variable
#   - y_test: Series of testing target variable
# Returns:
#   - Negative R-squared as the objective value (to maximize R-squared)

def objective_function(params, X_train, X_test, y_train, y_test):
    # Extracting the 'price' parameter from the input
    price = params[0]
    
    # Setting the 'price' parameter for training features
    X_train['price'] = price
    
    # Training a Random Forest model on the modified training data
    model = train_random_forest(X_train, y_train, n_estimators=10, random_state=42)
    
    # Making predictions on the testing data
    y_pred = predict_random_forest(X_test, model)
    
    # Calculating R-squared as the objective to be maximized
    residual_sum_of_squares = ((y_test - y_pred) ** 2).sum()
    total_sum_of_squares = ((y_test - y_test.mean()) ** 2).sum()
    r_squared = 1 - (residual_sum_of_squares / total_sum_of_squares)
    
    # Returning negative R-squared since we aim to maximize it in optimization
    return -r_squared

# run_optimization:
# This function runs an optimization process to find the optimal 'price' parameter.
# Parameters:
#   - X_train: DataFrame of training features
#   - X_test: DataFrame of testing features
#   - y_train: Series of training target variable
#   - y_test: Series of testing target variable
# Returns:
#   - optimal_price: The optimized 'price' parameter that maximizes R-squared

def run_optimization(X_train, X_test, y_train, y_test):
    # Initial guess for the optimization
    initial_guess = [X_train['price'].mean()]
    
    # Running the optimization using Nelder-Mead method
    result = minimize(objective_function, initial_guess, args=(X_train.copy(), X_test.copy(), y_train.copy(), y_test.copy()), method='Nelder-Mead')
    
    # Extracting the optimized 'price' parameter from the result
    optimal_price = result.x[0]
    
    # Returning the optimal 'price' parameter
    return optimal_price

def main():

    X, y = extract_features_and_target(df)
    X_train, X_test, y_train, y_test = training_testing_subsets(X, y, test_size=0.2, random_state=42)
    optimal_price = run_optimization(X_train, X_test, y_train, y_test)
    
    print("Optimal Price:", optimal_price)

if __name__ == "__main__":
    main()
