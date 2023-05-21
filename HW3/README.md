# **HW3**
## **Introduction**

The code performs the following steps:

1. Reads the input data from CSV files.
1. Randomly splits the data into training, validation, and test sets.
1. Applies MLR to predict the target variable on the test set.
1. Applies BLR to predict the target variable on the test set.
1. Applies XGBoost Regression to predict the target variable on the test set.
1. Calculates the mean squared error (MSE) for each prediction.
1. Generates plots to visualize the MLR and BLR predictions.

## **Methods**
The code consists of several methods that perform specific tasks. Here is a brief explanation of each method:

- 'read_data()': Reads the feature and target data from CSV files.
- 'random_split(x, y)': Randomly splits the data into training, validation, and test sets.
- 'calMSE(predict, actual)': Calculates the mean squared error between predicted and actual values.
- 'expand(x)': Expands a columns front of a basis matrix.
- 'get_polynomial_basis(x)': Generates a polynomial basis matrix from the feature set.
- 'MLR(x_train, y_train, x_test, lamda)': Performs Maximum likelihood Linear Regression.
- 'posterior(phi, t, alpha, beta)': Calculates the posterior distribution of the training set for Bayesian Linear Regression.
- 'posterior_predictive(phi_test, m_N, S_N, beta)': Predicts the posterior distribution for Bayesian Linear Regression.
- 'fit(phi, y, alpha_0, beta_0, max_iter, rtol)': Maximizes the evidence function to obtain the optimal alpha and beta for Bayesian Linear Regression.
- 'get_sample(phi_test, m_N, S_N)': Randomly generates samples from the posterior distribution of Bayesian Linear Regression.
- 'BLR(x_train, y_train, x_test)': Performs Bayesian Linear Regression.
- 'plot_OLS(ax, x, y, c, ls, label)': Plots the ordinary least square line.
- 'plot_scatter(ax, x, y, label)': Plots scatter points.
- 'plot_MLR(x, y_truth, y_MLR)': Plots the MLR prediction.
- 'plot_BLR(x, y_truth, y_BLR, y_samples)': Plots the BLR prediction with posterior samples.
- 'xgbr(x_train, y_train, x_test)': Performs XGBoost Regression.
- 'main()': The main function that orchestrates the execution of the code.