# Pricing Optimization Problem with Genetic Algorithm

# Igor Mol <igor.mol@makes.ai>

# In this solution to pricing optimization, the genetic algorithm aims to find
# the optimal set of pricing parameters. The create_toolbox() function initializes the
# DEAP Toolbox for the genetic algorithm. It defines a search space by specifying the
# lower_bound and upper_bound for pricing parameters. The algorithm minimizes a fitness
# function, configured to evaluate pricing strategies using training and testing data.
# The genetic operations, including crossover (mate) and mutation, are performed on
# individuals representing pricing parameter sets. The selection process is based on
# tournament selection. This toolbox is instrumental in running the genetic algorithm
# to evolve pricing strategies that enhance performance on the given pricing optimization
# problem.

import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms

# load_data() loads and preprocess data from a DataFrame with inventory info.
# It assumes that 'df' is the input DataFrame containing relevant information.

# Extracting features and the target variable:
# - 'order_item_id', 'price', and 'product_category_name_english' are selected as features.
# - 'shipping_limit_date' is converted to datetime, and its month is extracted as the target variable.
#   (The target variable represents the month for simplicity.)

def load_data():
    # Assuming df is your DataFrame
    # Extract features and target variable
    X = df[['order_item_id', 'price', 'product_category_name_english']].values
    y = pd.to_datetime(df['shipping_limit_date']).dt.month.values  # Extracting the month for simplicity
    return X, y

# split_data() is used to divide the dataset into training and testing sets.
# It takes two parameters, X (features) and y (target variable), and uses the train_test_split
# function from scikit-learn to perform the splitting.

# Parameters:
# - X: Features of the dataset.
# - y: Target variable corresponding to the features.

# Returns:
# - X_train: Features of the training set.
# - X_test: Features of the testing set.
# - y_train: Target variable values of the training set.
# - y_test: Target variable values of the testing set.

def split_data(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


# linear_regression() performs linear regression using the normal equation.
# It takes training features (X_train), corresponding target variable (y_train),
# and testing features (X_test) as input.

# Parameters:
# - X_train: Features of the training set.
# - y_train: Target variable values of the training set.
# - X_test: Features of the testing set.

# Returns:
# - r_squared: Negative R-squared value, a measure of model performance on the test set.

def linear_regression(X_train, y_train, X_test):
    # Add a bias term to X_train
    X_train_bias = np.c_[np.ones((X_train.shape[0], 1)), X_train]

    # Use the normal equation to find the optimal weights
    theta = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train

    # Add a bias term to X_test
    X_test_bias = np.c_[np.ones((X_test.shape[0], 1)), X_test]

    # Predict the target variable on the test data
    y_pred = X_test_bias @ theta[1:]

    # Calculate the negative R-squared value (minimize negative R-squared)
    r_squared = -1.0 - ((y_test - y_pred) ** 2).sum() / ((y_test - y_test.mean()) ** 2).sum()

    return r_squared,


# create_toolbox()sets up a genetic algorithm toolbox for optimization problem.
# It creates a fitness class for minimization and an individual class using the
# DEAP library.

# Parameters:
# - lower_bound: The lower bound of the search space for genetic algorithm parameters.
# - upper_bound: The upper bound of the search space for genetic algorithm parameters.

# Returns:
# - toolbox: DEAP Toolbox configured for the genetic algorithm.

def create_toolbox(lower_bound, upper_bound):
    # Create a fitness class for minimization if not exists
    if not hasattr(creator, "FitnessMin"):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    
    # Create an individual class if not exists
    if not hasattr(creator, "Individual"):
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

    # Create the genetic algorithm toolbox
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.uniform, lower_bound, upper_bound)
    toolbox.register("individual", tools.initIterate, creator.Individual, lambda: np.array([toolbox.attr_float()]))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", objective_function, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Define the bounds for the price parameter
    lower_bound = df['price'].min()
    upper_bound = df['price'].max()

    toolbox = create_toolbox(lower_bound, upper_bound)

    # Run the genetic algorithm
    population_size = 10
    num_generations = 100
    population = toolbox.population(n=population_size)
    algorithm_result = algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=2*population_size,
                                                  cxpb=0.7, mutpb=0.2, ngen=num_generations, stats=None,
                                                  halloffame=None, verbose=True)

    # Display the optimal price
    optimal_price_ga = algorithm_result[0][0]
    print("Optimal Price (Genetic Algorithm):", optimal_price_ga[0])

if __name__ == "__main__":
    main()
