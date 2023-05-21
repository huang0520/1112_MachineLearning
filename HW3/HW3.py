import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb


# Set random state
rng = np.random.default_rng(100)


def read_data():
    """
    Read the data

    Returns:
        feature data, target data
    """
    x_read = pd.read_csv("./exercise.csv")
    y_read = pd.read_csv("./calories.csv")

    x_read = x_read.drop(columns=["User_ID", "Gender"])
    y_read = y_read.drop(columns=["User_ID"])

    return x_read, y_read


def random_split(x, y):
    """
    Randomly split data to train, test, validate set

    Args:
        x : feature set
        y : target set

    Returns:
        features of train, test, validate, target of train, test, validate
    """
    idx_rng = rng.permutation(len(x))

    x_rng = x.iloc[idx_rng].values
    y_rng = y.iloc[idx_rng].values

    train_split = int(0.7 * len(x))
    validate_split = int(0.8 * len(x))

    x_train, x_validate, x_test = np.split(x_rng, [train_split, validate_split])
    y_train, y_validate, y_test = np.split(y_rng, [train_split, validate_split])
    return x_train, x_validate, x_test, y_train, y_validate, y_test


def calMSE(predict: np.ndarray, actual: np.ndarray):
    """
    Calculate the mean square error

    Args:
        predict : prediction target
        actual : actual target

    Returns:
        MSE
    """
    return np.mean((predict.ravel() - actual.ravel()) ** 2)


def expand(x):
    """
    Expand a columns front of basis matrix

    Args:
        x : basis matrix

    Returns:
        expanded basis matrix
    """
    return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)


def get_polynomial_basis(x):
    """
    Get the polynomial basis of features

    Args:
        x: feature set

    Return:
        The polynomial basis matrix
    """

    def poly_basis_function(x):
        """
        Get polynomial function of one feature set.
        [x1, x2, x1^2, x2^2, x1*x2]

        Args:
            x : feature set

        Returns:
            set of polynomial basis function
        """
        x = x.reshape(-1, 1)
        tmp = x @ x.T
        i, j = np.tril_indices_from(tmp)
        return tmp[i, j].ravel()

    power_x = np.apply_along_axis(poly_basis_function, 1, x)
    return expand(np.concatenate([x, power_x], axis=1))


def MLR(x_train, y_train, x_test, lamda):
    """
    Maximum likelihood and least square regression

    Args:
        x_train : training feature set
        y_train : training target
        x_test : test feature set
        lamda : regularization coefficient

    Returns:
        test target prediction
    """
    x_phi = get_polynomial_basis(x_train)
    weight = (
        la.inv(x_phi.T @ x_phi + lamda * np.identity(x_phi.shape[1]))
        @ x_phi.T
        @ y_train
    )

    y_phi = get_polynomial_basis(x_test)
    return y_phi @ weight


def posterior(phi, t, alpha, beta):
    """
    Calculate the posterior distribution of training set

    Args:
        phi : basis matrix of training feature
        t : target of training set
        alpha : initial alpha
        beta : initial beta

    Returns:
        mean and variance of posterior distribution
    """
    S_N_inv = alpha * np.eye(phi.shape[1]) + beta * phi.T @ phi
    S_N = la.inv(S_N_inv)
    m_N = beta * S_N @ phi.T @ t

    return m_N, S_N


def posterior_predictive(phi_test, m_N, S_N, beta):
    """
    Predict the posterior distribution

    Args:
        phi_test : basis matrix of testing feature
        m_N : mean of training posterior distribution
        S_N : variance of training posterior distribution
        beta  beta

    Returns:
        mean and variance of test target prediction distribution
    """
    y = phi_test @ m_N
    y_var = 1 / beta + phi_test @ S_N @ phi_test.T

    return y, y_var


def fit(phi, y, alpha_0=1e-5, beta_0=1e-5, max_iter=200, rtol=1e-5):
    """
    Maximizing the evidence function to get the optimal alpha and beta.

    Args:
        Phi: basis matrix.
        y: training target.
        alpha_0: initial value for alpha.
        beta_0: initial value for beta.
        max_iter: maximum number of iterations.
        rtol: convergence criterion.

    Returns:
        alpha, beta, posterior mean, posterior covariance.
    """
    N, M = phi.shape

    eigenvalues_0 = la.eigvalsh(phi.T @ phi)

    beta = beta_0
    alpha = alpha_0

    for i in range(max_iter):
        beta_prev = beta
        alpha_prev = alpha

        eigenvalues = eigenvalues_0 * beta

        m_N, S_N = posterior(phi, y, alpha, beta)

        gamma = np.sum(eigenvalues / (eigenvalues + alpha))
        alpha = gamma / np.sum(m_N**2)

        beta_inv = 1 / (N - gamma) * np.sum((y - phi @ m_N) ** 2)
        beta = 1 / beta_inv

        if np.isclose(alpha_prev, alpha, rtol=rtol) and np.isclose(
            beta_prev, beta, rtol=rtol
        ):
            return alpha, beta, m_N, S_N

    return alpha, beta, m_N, S_N


def get_sample(phi_test, m_N, S_N):
    """
    According to the training set posterior distribution
    randomly generator sample lines

    Args:
        phi_test : basis matrix of test feature
        m_N : mean of posterior distribution
        S_N : Covariance of posterior distribution

    Returns:
       Sampled targets
    """
    w_samples = rng.multivariate_normal(m_N.ravel(), S_N, 10, "ignore").T

    return phi_test @ w_samples


def BLR(x_train, y_train, x_test):
    """
    Bayesian linear regression

    Args:
        x_train : training features
        y_train : training target
        x_test : testing feature

    Returns:
        mean and samples of prediction
    """
    phi_train = get_polynomial_basis(x_train)
    phi_test = get_polynomial_basis(x_test)

    alpha, beta, m_N, S_N = fit(phi_train, y_train, rtol=10**-5)
    y, y_var = posterior_predictive(phi_test, m_N, S_N, beta)

    y_samples = get_sample(phi_test, m_N, S_N)

    return y, y_samples


def plot_OLS(ax, x, y, c="black", ls="-", label=None):
    """
    Plot the ordinary least square line

    Args:
        x
        y
    """
    dx = np.linspace(x.min(), x.max(), 100)
    m, n = np.polyfit(x, y, 1)
    ax.plot(dx, m * dx + n, c=c, ls=ls, label=label)


def plot_scatter(ax, x, y, label):
    ax.scatter(x, y, s=10, edgecolors="w", linewidths=0.1, label=label)


def plot_MLR(x, y_truth, y_MLR):
    fig, ax = plt.subplots()
    plot_scatter(ax, x, y_truth, "Observation")
    plot_OLS(ax, x, y_MLR, ls="--", label="OLS fit")
    ax.set_title("MLR Predict")
    ax.set_xlabel("Duration (min)")
    ax.set_ylabel("Calories")
    ax.legend()
    plt.show()


def plot_BLR(x, y_truth, y_BLR, y_samples):
    fig, ax = plt.subplots()
    plot_scatter(ax, x, y_truth, "Observation")

    for sample in y_samples.T:
        plot_OLS(ax, x, sample, c="r", label="Bayesian Posterior Fits")
    plot_OLS(ax, x, y_BLR, ls="--", label="OLS Fit")

    handles, labels = ax.get_legend_handles_labels()
    new_handles, new_labels = [], []

    for handle, label in zip(handles, labels):
        if label not in new_labels:
            new_labels.append(label)
            new_handles.append(handle)

    ax.legend(new_handles, new_labels)
    ax.set_xlabel("Duration (min)")
    ax.set_ylabel("Calories")
    ax.set_title("Posterior Predictions with Limited Observations")
    plt.show()


def xgbr(x_train, y_train, x_test):
    xgbr = xgb.XGBRegressor()
    xgbr.fit(x_train, y_train)

    return xgbr.predict(x_test)


def main():
    x_read, y_read = read_data()
    x_train, x_validate, x_test, y_train, y_validate, y_test = random_split(
        x_read, y_read
    )

    # Predict through MLR
    y_MLR = MLR(x_train, y_train, x_test, 0.1)

    # Predict through BLR
    y_BLR, y_samples = BLR(x_train, y_train, x_test)

    # Predict through XGBoost
    y_XGBR = xgbr(x_train, y_train, x_test)

    print("========== MSE ==========")
    print(f"{'MLR:' : <10}{calMSE(y_MLR, y_test)}")
    print(f"{'BLR:' : <10}{calMSE(y_BLR, y_test)}")
    print(f"{'XGBoost:' : <10}{calMSE(y_XGBR, y_test)}")
    print("=========================")

    plot_MLR(x_test[:, 3], y_test, y_MLR)
    plot_BLR(x_test[:, 3], y_test, y_BLR, y_samples)


if __name__ == "__main__":
    main()
