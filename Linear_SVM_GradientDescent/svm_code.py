import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def generate_cancer_data():
    np.random.seed(42)
    benign = np.random.multivariate_normal(mean=[2, 2], cov=[[1, 0.5], [0.5, 1]], size=50)
    malignant = np.random.multivariate_normal(mean=[6, 6], cov=[[1, 0.5], [0.5, 1]], size=50)
    benign_labels = np.zeros(len(benign))
    malignant_labels = np.ones(len(malignant))
    data = np.vstack((benign, malignant))
    labels = np.hstack((benign_labels, malignant_labels))
    return pd.DataFrame(data, columns=["Tumor_Size", "Tumor_Density"]), pd.Series(labels, name="labels")


def plot_data(data, labels, w, b):
    plt.scatter(data["Tumor_Size"], data["Tumor_Density"], c=labels, cmap="bwr", alpha=0.8)
    plt.xlabel("Tumor_Size")
    plt.ylabel("Tumor_Density")
    plt.title("Cancer Data")
    x_values = np.linspace(data["Tumor_Size"].min(), data["Tumor_Size"].max(), 100)
    y_values = -(w[0]*x_values - b)/w[1]
    plt.plot(x_values, y_values, color="black", label="Decision Boundary or HyperPlane")
    plt.grid()
    plt.show()


def train(X, y, lr=0.01, epochs=1000, lambda_param=0.01):
    nsamples, nfeatures = X.shape
    w = np.zeros(nfeatures)
    b = 0
    for epoch in range(epochs):
        for i, xi in enumerate(X):
            if y[i]*(np.dot(w, xi)-b) >= 1:
                w -= lr*(2*lambda_param*w)
            else:
                w -= lr*(2*lambda_param*w-np.dot(y[i], xi))
                b -= lr*y[i]
    return w, b


def predict(newdata, w, b):
    pred = np.sign(np.dot(newdata, w)-b)
    return pred


def main():
    X, y = generate_cancer_data()
    y = y.apply(lambda x: -1 if x == 0 else 1)
    w, b = train(X.values, y.values)
    new_data = [8.35, 7.2]
    new_data = [8, 8]
    new_data = np.array(new_data)
    print(new_data)
    pred = predict(new_data, w, b)
    print("Prediction:", pred)
    plot_data(X, y, w, b)


main()
