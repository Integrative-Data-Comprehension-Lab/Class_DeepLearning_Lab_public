import pytest
import time
import inspect
import numpy as np

from python_essentials import Matrix, Timer
 

@pytest.fixture
def sample_matrix():
    return Matrix([
        [1, 2, 3, 4, 5],   
        [6, 7, 8, 9, 10],  
        [11, 12, 13, 14, 15], 
        [16, 17, 18, 19, 20], 
        [21, 22, 23, 24, 25], 
        [26, 27, 28, 29, 30], 
        [31, 32, 33, 34, 35]
    ])

def test_deep_copy_score_1():
    original_data = [
        [1, 2, 3, 4, 5],   
        [6, 7, 8, 9, 10],  
        [11, 12, 13, 14, 15], 
        [16, 17, 18, 19, 20], 
        [21, 22, 23, 24, 25], 
    ]

    mat = Matrix(original_data)
    
    original_data[2][3] = 999
    expected_val = 14
    assert mat[2][3] == expected_val, f"Deep copy test failed! Expected {expected_val}, but got {mat[2][3]}"


def test_getitem_score_1(sample_matrix):
    assert sample_matrix[3] == [16, 17, 18, 19, 20], "Row indexing test failed"
    assert sample_matrix[4, 1] == 22, "Element-wise indexing test failed"


def test_call_score_2(sample_matrix):
    """스칼라 곱 연산 테스트"""
    sample_matrix(2)
    
    expected_result = [
        [2, 4, 6, 8, 10],   
        [12, 14, 16, 18, 20],  
        [22, 24, 26, 28, 30], 
        [32, 34, 36, 38, 40], 
        [42, 44, 46, 48, 50], 
        [52, 54, 56, 58, 60], 
        [62, 64, 66, 68, 70]
    ]

    assert sample_matrix.data == expected_result, (
        f"__call__ test failed! Expected {expected_result}, but got {sample_matrix.data}"
    )

    source_code = inspect.getsource(sample_matrix.__call__)
    code_line = [line.strip() for line in source_code.split("\n") if line.strip().startswith("self.data")][-1]
    assert '[' in code_line and ']' in code_line and 'for' in code_line, "__call__ implementation must be single line using list comprehension!"


def test_filter_score_1(sample_matrix):
    expected_result = [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    actual_result = sample_matrix.get_elements_greater_than(17)

    assert set(actual_result) == set(expected_result), (
        f"Filter elements test failed! Expected elements {set(expected_result)}, but got {set(actual_result)}"
    )

    assert len(actual_result) == len(expected_result), (
        f"Filter elements test failed! Expected {len(expected_result)} elements, but got {len(actual_result)}"
    )


def test_get_n_rows_score_1(sample_matrix):
    """행 부분 선택 기능 테스트"""
    expected_result = [
        [1, 2, 3, 4, 5],   
        [6, 7, 8, 9, 10],  
        [11, 12, 13, 14, 15], 
        [16, 17, 18, 19, 20]
    ]
    assert sample_matrix.get_first_n_rows(4) == expected_result, (
        f"First N rows test failed! Expected {expected_result}, but got {sample_matrix.get_first_n_rows(4)}"
    )


def test_get_n_columns_score_2(sample_matrix):
    """열 부분 선택 기능 테스트"""
    expected_result = [
        [1, 2, 3],
        [6, 7, 8],
        [11, 12, 13],
        [16, 17, 18],
        [21, 22, 23],
        [26, 27, 28],
        [31, 32, 33]
    ]
    assert sample_matrix.get_first_n_columns(3) == expected_result, (
        f"First N columns test failed! Expected {expected_result}, but got {sample_matrix.get_first_n_columns(3)}"
    )

    source_code = inspect.getsource(sample_matrix.get_first_n_columns)
    code_line = [line.strip() for line in source_code.split("\n") if line.strip().startswith("sliced_data")][-1]
    assert '[' in code_line and ']' in code_line and 'for' in code_line, "get_first_n_columns implementation must be single line using list comprehension!"

def test_timer_score_2():
    with Timer() as t:
        time.sleep(0.5)

    assert np.isclose(t.elapsed_time, 0.5, atol=1e-2), f"Timer test failed! Expected ~0.5 seconds, but got {t.elapsed_time:.6f} seconds"

    with Timer() as t:
        time.sleep(0.2)

    assert np.isclose(t.elapsed_time, 0.2, atol=1e-2), f"Timer test failed! Expected ~0.5 seconds, but got {t.elapsed_time:.6f} seconds"



