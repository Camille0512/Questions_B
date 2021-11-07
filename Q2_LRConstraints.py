import numpy as np
from math import exp
import pandas as pd
from copy import deepcopy
from collections import Counter
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from Q1_MinimumPrice import print_info

pd.options.display.max_columns = None


@print_info("EDA Process")
def eda_process(df: pd.DataFrame):
    print("Check basic information:\n", df.info())
    print("Describe data:\n", df.describe())
    print("Positive & Negative Rate:\n", Counter(target))
    print("Correlation matrix:\n", all_data.corr())
    print("\nCorrelation with target:\n", all_data.corr().sort_values("target")["target"])


@print_info("Check multi-collinearity")
def multi_collinearity_vif(df: pd.DataFrame, vif_val=10):
    """
    Check multi-collinearity
    :params df: The whole data frame without target variable.
    :params vif_val: The set VIF value threshold for the multi-collinearity judgement.
    :return: The selected columns.
    """
    vif_df = pd.DataFrame()
    cols = list(df.columns)
    vif_df["features"] = cols
    vif_df["VIF"] = [variance_inflation_factor(df.values, col) for col in range(len(cols))]
    vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)
    while vif_df.loc[0, "VIF"] > vif_val:
        print(vif_df.loc[0, "features"])
        cols.remove(vif_df.loc[0, "features"])
        vif_df = vif_df.loc[1:]
        vif_df["VIF"] = [variance_inflation_factor(df[cols].values, col) for col in range(len(cols))]
        vif_df = vif_df.sort_values("VIF", ascending=False).reset_index(drop=True)
    print("Final VIF data frame:\n", vif_df)
    return cols


@print_info("Split data set")
def split_data_set(df: pd.DataFrame, tar: pd.Series, cols: list, train_size=0.7, random_state=1234):
    """
    Split dataset into train & test.
    :param df: The data frame with all explained variable.
    :param tar: The target data.
    :param cols: The selected column name list.
    :param train_size: The train set size, in proportion format.
    :param random_state: Random seed for splitting data.
    :return: The train & test sets of explained and target variables.
    """
    final_data = df[cols].copy()
    X_train, X_test, Y_train, Y_test = train_test_split(final_data, np.array(tar), train_size=train_size,
                                                        random_state=random_state)
    print("Shape\ntrain: x {}  y {}\ntest: x {}  y {}".format(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape))
    Y_train, Y_test = np.array(Y_train), np.array(Y_test)
    return X_train, X_test, Y_train, Y_test


@print_info("Standardize data")
def standardize_data(X_train: np.array, X_test: np.array):
    """
    Standardize input train & test data.
    :param X_train: The training data set, also used to define the parameters for standardizing test data set.
    :param X_test: The test data set.
    :return: Standardized train & test data sets.
    """
    z_scale = StandardScaler()
    X_train = z_scale.fit_transform(X_train)
    X_test = z_scale.transform(X_test)
    return X_train, X_test


def initialize_with_zeros(dim):
    """
    Initialize the coefficient and intercept as 0 values.
    :params dim: The dimention of the input explained variables.
    :return: Initial coefficients and intercept.
    """
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


def sigmoid(z):
    """
    The sigmoid function.
    :params z: The value to be put in the sigmoid function. Calcualted by wx+b.
    :return: The transformation of sigmoid function, values from [0, 1].
    """
    z = z.astype(np.float128)
    return 1 / (1 + np.exp(-1 * z))


def constraint(dw: np.array, constraints: str):
    """
    Fill in the constraint for the optimization algorithm.
    :params dw: The gradient array of the coefficients.
    :params constraint: The constraint name.
    :return: The gradient array of the coefficient with consideration of the given constraint.
    """
    if constraints == "non_neg":
            dw[dw > 0] = 0
    elif constraints == "ordered_desc":
        dw_shape, dw0 = dw.shape, dw.reshape(1, -1)[0].copy()
        dw, prev = [], None
        for pw in dw0:
            if prev is None:
                prev = pw
            else:
                if pw > prev:
                    pw = prev
                else:
                    prev = pw
            dw.append(pw)
        dw = np.array(dw).reshape(dw_shape)
    else:
        raise ValueError("Please input correct constraint name.")
    return dw


def propagate(w, b, X, Y, constraints=None):
    """
    Propagation process.
    :params w: The coefficients.
    :params b: The intercept.
    :params X: The input explained variables.
    :params Y: The target variable.
    :params constraints: Whether to include constraints.
    :return: The gradient dictionary and cost value.
    """
    m, dummy = X.shape[1], 0.000001
    A = sigmoid(np.dot(w.T, X) + b).reshape(1, -1)
    cost = (np.dot(np.log(A + dummy), Y.reshape(-1, 1)) + np.dot(np.log(1 - A + dummy), 1 - Y.reshape(-1, 1))) / m * (-1)
    dw = (np.dot(X, A.reshape(-1, 1) - Y.reshape(-1, 1))) / m
    if constraints is not None:
        dw = constraint(dw, constraints)
    db = sum(A.reshape(-1, 1) - Y.reshape(-1, 1))[0] / m
    cost = np.squeeze(np.array(cost))
    grads = {
        "dw": dw,
        "db": db
    }
    return grads, cost


def optimize(w, b, X, Y, constraints=None, num_iter=100, learning_rate=0.009):
    """
    Optimization function with coefficients non-negative constraint.
    :params w: The coefficients.
    :params b: The intercept.
    :params X: The input explained variables.
    :params Y: The target variable.
    :params constraints: Whether to include constraints.
    :params num_iter: The number of iterations.
    :params learning_rate: The learning rate.
    :return: The Logistic Regression parameters dictionary, gradient dictionary, and cost list.
    """
    w = deepcopy(w)
    b = deepcopy(b)
    costs, dw, db = [], None, None
    for i in range(num_iter):
        grads, cost = propagate(w, b, X, Y, constraints)
        dw = grads["dw"]
        db = grads["db"]
        w = w - learning_rate * dw
        b = b - learning_rate * db # Intercept is not regarded as one of the coefficients.
        if i % num_iter == 0:
            costs.append(cost)
    params = {
        "w": w,
        "b": b
    }
    grads = {
        "dw": dw,
        "db": db
    }
    return params, grads, costs


def predict(w, b, X):
    """
    Get the prediction result of Logistic Regression.
    :params w: The coefficients.
    :params b: The intercept.
    :params X: The input explained variables.
    :return: Prediction result.
    """
    m = X.shape[1]
    Y_hat = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    A = sigmoid(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_hat[0, i] = 1
        else:
            Y_hat[0, i] = 0
    return Y_hat


@print_info("Calling logistic regression model")
def model(X_train, Y_train, X_test, Y_test, constraints=None, num_iter=2000, learning_rate=0.5, print_cost=False):
    """
    Build the Logistic Regression model.
    :params X_train: The training data set.
    :params Y_train: The training target values.
    :params X_test: The testing data set.
    :params Y_test: The testing target values.
    :params constraints: Whether to include constraints.
    :params num_iter: Iteration times.
    :params learning_rate: The learning rate, decide the step width in each iteration.
    :params print_cost: Whether to print the costs.
    :return: The dictionary of prediction result as well as other parameters.
    """
    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, constraints=constraints, num_iter=num_iter, learning_rate=learning_rate)
    w = params["w"].copy()
    b = params["b"].copy()
    Y_hat_test = predict(w, b, X_test)
    Y_hat_train = predict(w, b, X_train)
    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_hat_test - Y_test)) * 100))
    d = {
        "costs": costs,
        "Y_hat_test": Y_hat_test,
        "Y_hat_train": Y_hat_train,
        "w": w,
        "b": b,
        "learning_rate": learning_rate,
        "num_iterations": num_iter
    }
    return d


@print_info("Coefficient of the equation")
def coefficient_parameters(res_d: dict, cols: list):
    """
    Get the coefficients information of the LR equation.
    :param res_d: The dictionary with optimized linear regression.
    :param cols: The selected column names.
    :return: The coefficient data frame.
    """
    coef = list(res_d["w"].reshape(1, -1)[0])
    coef.append(res_d["b"])
    if "intercept" not in cols: cols.append("intercept")
    coef_df = pd.DataFrame({
        "Features": cols,
        "coefficient": coef
    })
    return coef_df


if __name__ == "__main__":
    # Load data
    data, target = load_breast_cancer(return_X_y=True, as_frame=True)
    all_data = data.join(target)

    # EDA
    eda_process(all_data)

    # Feature selection
    cols_sel = multi_collinearity_vif(data)
    print("Correlation between selected features and target:\n",
          all_data[cols_sel + ["target"]].corr().sort_values("target")["target"])
    cols_sel = ["smoothness error", "symmetry error", "concave points error", "worst fractal dimension",
                "perimeter error"]

    # Split dataset
    x_train, x_test, y_train, y_test = split_data_set(data, target, cols_sel)

    # Standardize data
    x_train, x_test = standardize_data(x_train, x_test)

    # Logistic Regression without constraints
    d = model(x_train.T, y_train, x_test.T, y_test)
    coef_data = coefficient_parameters(d, cols_sel)
    print("Logistic Regression without constraints:\n", coef_data)

    # Logistic Regression with non-negative coefficient constraint
    d = model(x_train.T, y_train, x_test.T, y_test, constraints="non_neg")
    coef_data_nn = coefficient_parameters(d, cols_sel)
    print("Logistic Regression with non-negative coefficient constraint:\n", coef_data_nn)

    # Logistic Regression with descending ordered coefficient constraint
    d = model(x_train.T, y_train, x_test.T, y_test, "ordered_desc")
    coef_data_do = coefficient_parameters(d, cols_sel)
    print("Logistic Regression with descending ordered coefficient constraint:\n", coef_data_do)