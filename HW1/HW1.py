import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# The structure of MAP model
class MAP:
    def __init__(self, target: np.ndarray, feature: np.ndarray) -> None:
        self.__cal_param(target, feature)
        pass

    # Calculate the parameter
    def __cal_param(self, target: np.ndarray, feature: np.ndarray):
        num_feature = feature.shape[1]
        self.__num_classes = np.unique(target).shape[0]

        mean = np.zeros((self.__num_classes, num_feature))
        var = np.zeros((self.__num_classes, num_feature))
        prior = np.zeros(self.__num_classes)

        # Calculate parameter for likelihood & Prior
        for i in np.unique(target):
            # Extract subset of current class
            subset = feature[target == i]
            # Means of features of class
            mean[i] = np.mean(subset, axis=0)
            # Variation of features of class
            var[i] = np.var(subset, axis=0)
            # Prior of class
            prior[i] = len(subset) / len(feature)

        self.__mean = mean
        self.__var = var
        self.__prior = prior
        pass

    # Prediction
    def pred(self, feature: np.ndarray):
        pred = np.zeros(feature.shape[0]).astype(int)

        for i, x in enumerate(feature):
            pred[i] = self.__pred_one(x)

        return pred

    def __pred_one(self, x):
        posterior = np.log(self.__prior) + self.__log_likelihood(x)

        return np.argmax(posterior)

    def __log_likelihood(self, x):
        tmp = (
            -((x - self.__mean) ** 2) / (2 * self.__var)
            - np.log(self.__var) / 2
        )

        return np.sum(tmp, axis=1)


# Split the training set and test set
def split_train_test(df: pd.DataFrame):
    grouped = df.groupby(df["target"])
    test_df = pd.DataFrame()

    # Random select the data in three type as test set
    for type in grouped.groups.keys():
        tmp = grouped.get_group(type).sample(
            n=config["test_size"] // len(grouped), random_state=config["seed"]
        )
        test_df = pd.concat([test_df, tmp])

    # The remaining of data is training set
    train_df = df.drop(test_df.index)

    return train_df, test_df


# Split target and feature
def split_target_feature(df: pd.DataFrame):
    x = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    return y, x


def cal_accuracy(pred, test):
    num_corr = np.count_nonzero([np.array(pred == test)])
    return num_corr / len(test)


def plot_pred(y, x):
    pca = PCA(n_components=2)
    x_pca = pca.fit_transform(x)

    type = ["Type 0", "Type 1", "Type 2"]
    for i in np.unique(y):
        plt.scatter(x_pca[y == i, 0], x_pca[y == i, 1], label=type[i])
    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.legend()
    plt.show()


def main():
    # Read dataset and split training, test set
    read = pd.read_csv(config["file"])
    train, test = split_train_test(read)

    # Output csv of training and test set
    train.to_csv("./train.csv", index=False)
    test.to_csv("./test.csv", index=False)

    # Split target and feature
    y_train, x_train = split_target_feature(train)
    y_test, x_test = split_target_feature(test)

    map = MAP(y_train, x_train)

    # Predict the result
    y_pred = map.pred(x_test)

    # Show the accuracy of prediction
    print(f"Random seed: {config['seed']}")
    print(f"Accuracy: {cal_accuracy(y_pred, y_test) * 100}%")

    plot_pred(y_pred, x_test)


# Main function
if __name__ == "__main__":
    # configuration
    config = {
        "file": "./Wine.csv",
        # "seed": np.random.randint(10000),
        "seed": 2425,
        "test_size": 60,
    }

    main()
