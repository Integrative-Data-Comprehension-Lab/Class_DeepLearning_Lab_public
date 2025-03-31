import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer

def load_data():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    
    selected_features = ["mean radius", "mean texture"]
    X = df[selected_features].values
    y = df["target"].values.reshape(-1, 1)  # Make y a column vector
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test

def test_init(model):
    num_features = 10
    w, b = model.weights, model.bias
    assert type(w) == np.ndarray, f"Wrong type for weights. {type(w)} != np.ndarray"
    assert w.shape == (num_features, 1), f"Wrong shape for w. {w.shape} != {(num_features, 1)}"
    assert np.allclose(w, np.zeros((num_features,1))), f"Wrong initialization values for weights"

def test_sigmoid(model):
    sigmoid_output = model.sigmoid(np.array([0, 2]))
    assert type(sigmoid_output) == np.ndarray, "Wrong type. Expected np.ndarray"
    assert np.allclose(sigmoid_output, [0.5, 0.88079708]), f"Wrong value. {sigmoid_output} != [0.5, 0.88079708]"
    sigmoid_output = model.sigmoid(1)
    assert np.allclose(sigmoid_output, 0.7310585), f"Wrong value. {sigmoid_output} != 0.7310585"

def test_propagate(model):
    model.weights, model.bias = np.array([[1.], [2.], [-1]]), 2.5, 
    X = np.array([[1., 3., 3.], [2., 4., 4.], [-1., -3.2, -3.2], [0., 1., -3.5]])
    Y = np.array([[1], [1], [0], [0]])
    expected_dw = np.array([[-0.03909333], [ 0.12501464], [-0.99960809]])
    expected_db = np.float64(0.288106326429569)
    expected_cost = np.array(2.0424567983978403)
    cost, dw, db = model.propagate(X, Y)
    assert type(dw) == np.ndarray, f"Wrong type for 'dw'. {type(dw)} != np.ndarray"
    assert dw.shape == model.weights.shape, f"Wrong shape for 'dw'. {dw.shape} != {model.weights.shape}"
    assert np.allclose(dw, expected_dw), f"Wrong values for 'dw'. {dw} != {expected_dw}"
    assert np.allclose(db, expected_db), f"Wrong values for 'db'. {db} != {expected_db}"
    assert np.allclose(cost, expected_cost), f"Wrong values for cost. {cost} != {expected_cost}"

def test_optimize(model):
    model.weights, model.bias, X, Y = np.array([[1.], [2.]]), 2., np.array([[1., 2., -1.], [3., 4., -3.2]]).T, np.array([[1, 0, 1]]).T
    expected_cost = [5.80154532, 0.31057104]
    costs = model.optimize(X, Y, num_iterations=101, learning_rate=0.1)
    assert type(costs) == list, "Wrong type for costs. It must be a list"
    assert len(costs) == 2, f"Wrong length for costs. {len(costs)} != 2"
    assert np.allclose(costs, expected_cost), f"Wrong values for costs. {costs} != {expected_cost}"

def test_predict(model):
    model.weights = np.array([[0.3], [0.5], [-0.2]])
    model.bias = -0.33333
    X = np.array([[1., -0.3, 1.5],[2, 0, 1], [0, -1.5, 2]]).T
    pred, prob = model.predict(X)
    assert type(pred) == np.ndarray, f"Wrong type for pred. {type(pred)} != np.ndarray"
    assert pred.shape == (X.shape[0], 1), f"Wrong shape for pred. {pred.shape} != {(X.shape[1], 1)}"
    assert np.bitwise_not(np.allclose(pred, [[1.], [1.], [1]])), f"Perhaps you forget to add b in the calculation of A"
    assert np.allclose(pred, [[1.], [0.], [1]]), f"Wrong values for pred. {pred} != {[[1.], [0.], [1.]]}"

def test_logistic_regression(model):
    np.random.seed(0)

    expected_output = {'costs': [np.array(0.69314718)], 
                    'Y_prediction_test': np.array([[1., 1., 0.]]).T, 
                    'Y_prediction_train': np.array([[1., 1., 0., 1., 0., 0., 1.]]).T, 
                    'w': np.array([[ 0.08639757],
                            [-0.08231268],
                            [-0.11798927],
                            [ 0.12866053]]), 
                    'b': -0.03983236094816321}

    b, Y, X = 1.5, np.array([[1, 0, 0, 1, 0, 0, 1]]).T, np.random.randn(4, 7).T,
    X_test = np.random.randn(4, 3).T
    y_test = np.array([[0, 1, 0]]).T
    costs = model.optimize(X, Y, num_iterations=50, learning_rate=0.01)
    Y_preds_train, Y_prob_train = model.predict(X)
    Y_preds_test, Y_prob_test = model.predict(X_test)

    assert np.allclose(costs, expected_output['costs']), f"Wrong values for costs. {costs} != {expected_output['costs']}"
    assert np.allclose(model.weights, expected_output['w']), f"Wrong values for model.weights. {model.weights} != {expected_output['w']}"
    assert np.allclose(model.bias, expected_output['b']), f"Wrong values for model.bias. {model.bias} != {expected_output['b']}"
    assert np.allclose(Y_preds_test, expected_output['Y_prediction_test']), f"Wrong values for Y_prediction_test. {Y_preds_test} != {expected_output['Y_prediction_test']}"
    assert np.allclose(Y_preds_train, expected_output['Y_prediction_train']), f"Wrong values for Y_prediction_train. {Y_preds_train} != {expected_output['Y_prediction_train']}"


def visualize_decision_boundary(model, X, Y):
    """
    Plots the decision boundary for a trained logistic regression model.

    Args:
        model: Trained LogisticRegression model.
        X: Input data, shape (num_samples, num_features).
        Y: True labels, shape (num_samples, 1).
    """
    # Define the grid range
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Flatten the grid and predict
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    pred, prob = model.predict(grid_points)
    Z = prob.reshape(xx.shape)

    # Plot the decision boundary with probability shading
    plt.contourf(xx, yy, Z, levels=100, cmap="coolwarm", alpha=0.7)
    plt.colorbar(label="Probability of Class 1")
    
    # Overlay the data points
    plt.scatter(X[:, 0], X[:, 1], c=Y.ravel(), edgecolors='k', cmap="coolwarm", s=25)

    # Labels and title
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.title("Decision Boundary with Probability Contours")
    
    plt.show()

def visualize_costs(costs, learning_rate):
    """
    Plots the cost values over iterations.

    Args:
        costs: List of cost values.
    """
    plt.plot(costs)
    plt.ylabel('Cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title(f"Cost over Iterations (Learning rate = {learning_rate})")
    plt.show()