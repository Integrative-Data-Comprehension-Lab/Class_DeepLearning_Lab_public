import pytest
import numpy as np
from logistic_regression import LogisticRegression  # Assuming student's implementation is in logistic_regression.py
import inspect

# Fixture to create a model instance with a specified number of features
@pytest.fixture
def model():
    return LogisticRegression(num_features=3)  # Using 3 features for testing

def test_init_score_1(model):
    """Check if weights and bias are correctly initialized."""
    num_features = 3
    w, b = model.weights, model.bias
    assert isinstance(w, np.ndarray), f"Wrong type for weights. Expected np.ndarray, got {type(w)}"
    assert w.shape == (num_features, 1), f"Wrong shape for w. Expected {(num_features, 1)}, got {w.shape}"
    assert np.allclose(w, np.zeros((num_features, 1))), "Weights should be initialized to zeros"
    assert b == 0, "Bias should be initialized to zero"

def test_sigmoid_score_1(model):
    """Test sigmoid function with known inputs."""
    sigmoid_output = model.sigmoid(np.array([0, 2]))
    assert isinstance(sigmoid_output, np.ndarray), "Sigmoid output should be a numpy array"
    assert np.allclose(sigmoid_output, [0.5, 0.88079708]), f"Incorrect sigmoid output. {sigmoid_output} != [0.5, 0.88079708]"
    
    scalar_output = model.sigmoid(1)
    assert np.allclose(scalar_output, 0.7310585), f"Sigmoid(1) incorrect: {scalar_output} != 0.7310585"

    source_code = inspect.getsource(model.sigmoid)
    code_lines = [line.strip() for line in source_code.split("\n")]
    assert all([not (line.startswith('for ') and ' in ' in line) and not ('[' in line and ']' in line and ' for ' in line and ' in ' in line) for line in code_lines]), "do NOT use for loop in `sigmoid` implementation"

def test_propagate_score_3(model):
    """Test propagation (cost and gradients)."""
    model.weights = np.array([[1.], [2.], [-1.]])
    model.bias = 2.5
    X = np.array([[1., 3., 3.], [2., 4., 4.], [-1., -3.2, -3.2], [0., 1., -3.5]])
    Y = np.array([[1], [1], [0], [0]])

    expected_dw = np.array([[-0.03909333], [ 0.12501464], [-0.99960809]])
    expected_db = np.float64(0.288106326429569)
    expected_cost = np.array(2.0424567983978403)

    cost, dw, db = model.propagate(X, Y)

    assert isinstance(dw, np.ndarray), "dw should be a numpy array"
    assert dw.shape == model.weights.shape, f"Wrong shape for 'dw'. {dw.shape} != {model.weights.shape}"
    assert np.allclose(dw, expected_dw), f"Wrong values for 'dw'. {dw} != {expected_dw}"
    assert np.allclose(db, expected_db), f"Wrong values for 'db'. {db} != {expected_db}"
    assert np.allclose(cost, expected_cost), f"Wrong values for cost. {cost} != {expected_cost}"

    source_code = inspect.getsource(model.propagate)
    code_lines = [line.strip() for line in source_code.split("\n")]
    assert all([not (line.startswith('for ') and ' in ' in line) and not ('[' in line and ']' in line and ' for ' in line and ' in ' in line) for line in code_lines]), "do NOT use for loop in `propagate` implementation"

def test_optimize_score_2(model):
    """Test the optimization function updates weights and bias correctly."""
    model.weights = np.array([[1.], [2.]])
    model.bias = 2.
    X = np.array([[1., 2., -1.], [3., 4., -3.2]]).T
    Y = np.array([[1, 0, 1]]).T

    expected_cost = [5.80154532, 0.31057104]
    costs = model.optimize(X, Y, num_iterations=101, learning_rate=0.1)

    assert isinstance(costs, list), "Costs should be a list"
    assert len(costs) == 2, f"Wrong length for costs. {len(costs)} != 2"
    assert np.allclose(costs, expected_cost), f"Wrong values for costs. {costs} != {expected_cost}"

def test_predict_score_2(model):
    """Test the predict function for correct classification."""
    model.weights = np.array([[0.3], [0.5], [-0.2]])
    model.bias = -0.33333
    X = np.array([[1., -0.3, 1.5], [2, 0, 1], [0, -1.5, 2]]).T

    pred, prob = model.predict(X)

    assert isinstance(pred, np.ndarray), "Prediction should be a numpy array"
    assert pred.shape == (X.shape[0], 1), f"Incorrect prediction shape: {pred.shape} != {(X.shape[0], 1)}"
    assert np.bitwise_not(np.allclose(pred, [[1.], [1.], [1.]])), f"Perhaps you forget to add b in the calculation of A"
    assert np.allclose(pred, [[1.], [0.], [1.]]), f"Incorrect predictions: {pred} != {[[1.], [0.], [1.]]}"

    source_code = inspect.getsource(model.predict)
    code_lines = [line.strip() for line in source_code.split("\n")]
    assert all([not (line.startswith('for ') and ' in ' in line) and not ('[' in line and ']' in line and ' for ' in line and ' in ' in line) for line in code_lines]), "do NOT use for loop in `predict` implementation"

def test_lr_score_1():
    """Test the full logistic regression pipeline."""
    np.random.seed(0)

    model = LogisticRegression(num_features = 4)
    expected_output = {
        'costs': [np.array(0.69314718)], 
        'Y_prediction_test': np.array([[1., 1., 0.]]).T, 
        'Y_prediction_train': np.array([[1., 1., 0., 1., 0., 0., 1.]]).T, 
        'w': np.array([[ 0.08639757], [-0.08231268], [-0.11798927], [ 0.12866053]]), 
        'b': -0.03983236094816321
    }

    Y = np.array([[1, 0, 0, 1, 0, 0, 1]]).T
    X = np.random.randn(4, 7).T
    X_test = np.random.randn(4, 3).T

    costs = model.optimize(X, Y, num_iterations=50, learning_rate=0.01)
    Y_preds_train, _ = model.predict(X)
    Y_preds_test, _ = model.predict(X_test)

    assert np.allclose(costs, expected_output['costs']), f"Wrong values for costs. {costs} != {expected_output['costs']}"
    assert np.allclose(model.weights, expected_output['w']), f"Wrong values for model.weights. {model.weights} != {expected_output['w']}"
    assert np.allclose(model.bias, expected_output['b']), f"Wrong values for model.bias. {model.bias} != {expected_output['b']}"
    assert np.allclose(Y_preds_test, expected_output['Y_prediction_test']), f"Wrong values for test predictions. {Y_preds_test} != {expected_output['Y_prediction_test']}"
    assert np.allclose(Y_preds_train, expected_output['Y_prediction_train']), f"Wrong values for train predictions. {Y_preds_train} != {expected_output['Y_prediction_train']}"

