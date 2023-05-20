import pandas as pd
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import seaborn as sns


# Set random state
rng = np.random.default_rng(10)


def read_data():
    """
    Read the data

    Returns:
        feature data, label data
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
        y : label set

    Returns:
        features of train, test, validate, label of train, test, validate
    """
    idx_rng = rng.permutation(len(x))

    x_rng = x.iloc[idx_rng].values
    y_rng = y.iloc[idx_rng].values

    train_split = int(0.7 * len(x))
    validate_split = int(0.8 * len(x))

    x_train, x_validate, x_test = np.split(x_rng, [train_split, validate_split])
    y_train, y_validate, y_test = np.split(y_rng, [train_split, validate_split])
    return x_train, x_validate, x_test, y_train, y_validate, y_test


def normalize(x):
    """
    Normalize the array

    Args:
        x : numpy array

    Returns:
        normalized array
    """
    x_maxs = np.max(x, axis=0)
    x_mins = np.min(x, axis=0)

    return (x - x_mins) / (x_maxs - x_mins)


def calMSE(predict: np.ndarray, actual: np.ndarray):
    return np.mean((predict - actual) ** 2)


def expand(x):
    """
    Expand a columns front of phi matrix

    Args:
        x : phi

    Returns:
        expanded phi
    """
    return np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)


def get_gaussian_matrix(x, n_centroid=10):
    """
    Calculate the gaussian basis matrix.

    Args:
        x : feature set
        n_centroid : Number of centroid (mu). Defaults to 10.

    Returns:
        gaussian basis matrix
    """

    def gaussian_basis_function(x, mu, sigma):
        return np.exp(-0.5 * (x - mu) ** 2 / sigma**2)

    def cal_pairwise_dis(x):
        x_cross = x @ x.T
        x_norm = np.repeat(
            np.diag(x_cross).reshape([1, -1]), x.shape[0], axis=0
        )
        return np.median(x_norm + x_norm.T - 2 * x_cross)

    # Using pairwise distance as sigma
    std = cal_pairwise_dis(x)

    x_maxs = np.max(x, axis=0)
    x_mins = np.min(x, axis=0)

    centroids = np.linspace(x_maxs, x_mins, n_centroid, axis=0)

    phi = [gaussian_basis_function(x, centroid, std) for centroid in centroids]
    phi = np.concatenate(phi, axis=1)

    return expand(phi)


def plot_OLS(x, y):
    """
    Plot the ordinary least square line

    Args:
        x
        y
    """
    dx = np.linspace(x.min(), x.max(), 100)
    m, n = np.polyfit(x, y, 1)
    plt.plot(dx, m * dx + n, c="r", ls="dashed")


def MLR(x_train, y_train, x_test, lamda, n_centroid=10):
    x_basisMatrix = get_gaussian_matrix(x_train, n_centroid)
    weight = (
        la.inv(
            x_basisMatrix.T @ x_basisMatrix
            + lamda * np.identity(x_basisMatrix.shape[1])
        )
        @ x_basisMatrix.T
        @ y_train
    )

    y_basisMatrix = get_gaussian_matrix(x_test, n_centroid)
    return y_basisMatrix @ weight


def posterior(phi, t, alpha, beta, return_inverse=False):
    S_N_inv = alpha * np.eye(phi.shape[1]) + beta * phi.T @ phi
    S_N = la.inv(S_N_inv)
    m_N = beta * S_N @ phi.T @ t

    return (m_N, S_N) if not return_inverse else (m_N, S_N, S_N_inv)


def posterior_predictive(phi_test, m_N, S_N, beta):
    y = phi_test @ m_N
    y_var = 1 / beta + phi_test @ S_N @ phi_test.T

    return y, y_var


def log_marginal_likelihood(phi, t, alpha, beta):
    """Computes the log of the marginal likelihood."""
    N, M = phi.shape

    m_N, _, S_N_inv = posterior(phi, t, alpha, beta, return_inverse=True)

    E_D = beta * np.sum((t - phi @ m_N) ** 2)
    E_W = alpha * np.sum(m_N**2)

    score = (
        M * np.log(alpha)
        + N * np.log(beta)
        - E_D
        - E_W
        - np.log(la.det(S_N_inv))
        - N * np.log(2 * np.pi)
    )

    return 0.5 * score


def fit(phi, t, alpha_0=1e-5, beta_0=1e-5, max_iter=200, rtol=1e-5):
    N, M = phi.shape

    eigenvalues_0 = la.eigvalsh(phi.T @ phi)

    beta = beta_0
    alpha = alpha_0

    for i in range(max_iter):
        beta_prev = beta
        alpha_prev = alpha

        eigenvalues = eigenvalues_0 * beta

        m_N, S_N = posterior(phi, t, alpha, beta)

        gamma = np.sum(eigenvalues / (eigenvalues + alpha))
        alpha = gamma / np.sum(m_N**2)

        beta_inv = 1 / (N - gamma) * np.sum((t - phi @ m_N) ** 2)
        beta = 1 / beta_inv

        if np.isclose(alpha_prev, alpha, rtol=rtol) and np.isclose(
            beta_prev, beta, rtol=rtol
        ):
            print(f"Convergence after {i + 1} iterations.")
            return alpha, beta, m_N, S_N

    print(f"Stopped after {max_iter} iterations.")
    return alpha, beta, m_N, S_N


def BLR(x_train, y_train, x_test, n_centroid=10):
    phi_train = get_gaussian_matrix(normalize(x_train), n_centroid)
    phi_test = get_gaussian_matrix(normalize(x_test), n_centroid)

    # m_N, S_N = posterior(phi_train, y_train, alpha, beta)
    alpha, beta, m_N, S_N = fit(phi_train, y_train, rtol=10**-5)
    y, y_var = posterior_predictive(phi_test, m_N, S_N, beta)

    return m_N, S_N, y, y_var


def main():
    x_read, y_read = read_data()
    x_train, x_validate, x_test, y_train, y_validate, y_test = random_split(
        x_read, y_read
    )

    # Predict through MLR
    y_MLR_pred = MLR(normalize(x_train), y_train, normalize(x_test), 0.1)

    # Predict through BLR
    m_N, S_N, y_BLR_mean, y_BLR_var = BLR(x_train, y_train, x_validate, 10)

    print(f"MSE of MLR: {calMSE(y_MLR_pred, y_test)}")
    print(f"MSE of BLR: {calMSE(y_BLR_mean, y_validate)}")


if __name__ == "__main__":
    main()
