# (c) Copyright Riverlane 2020-2025.
from __future__ import annotations

import math
from itertools import product

import numpy as np
import pytest

from deltakit_explorer.codes._logicals import css_code_compute_logicals


def _binary_matrix(rows: list[list[int]], n_cols: int) -> np.ndarray:
    if len(rows) == 0:
        return np.zeros((0, n_cols), dtype=np.uint8)
    return np.asarray(rows, dtype=np.uint8)


def _binary_row_space(matrix: np.ndarray) -> set[tuple[int, ...]]:
    num_rows, num_cols = matrix.shape

    if num_rows == 0:
        return {tuple([0] * num_cols)}

    vectors: set[tuple[int, ...]] = set()
    for coefficients in product((0, 1), repeat=num_rows):
        vector = np.zeros(num_cols, dtype=np.uint8)
        for coefficient, row in zip(coefficients, matrix, strict=True):
            if coefficient:
                vector ^= row
        vectors.add(tuple(int(value) for value in vector))

    return vectors


def _binary_null_space(matrix: np.ndarray) -> set[tuple[int, ...]]:
    _, num_cols = matrix.shape

    vectors: set[tuple[int, ...]] = set()
    for entries in product((0, 1), repeat=num_cols):
        vector = np.asarray(entries, dtype=np.uint8)
        if np.all((matrix @ vector) % 2 == 0):
            vectors.add(entries)

    return vectors


def _dimension(space: set[tuple[int, ...]]) -> int:
    return int(math.log2(len(space)))


def _assert_logical_basis(
    logicals: np.ndarray, kernel_checks: np.ndarray, image_checks: np.ndarray
) -> None:
    assert logicals.ndim == 2
    assert logicals.shape[1] == kernel_checks.shape[1] == image_checks.shape[1]
    assert np.all(np.isin(logicals, [0, 1]))

    null_space = _binary_null_space(kernel_checks)
    image_space = _binary_row_space(image_checks)
    generated_space = _binary_row_space(np.vstack([image_checks, logicals]))

    # CSS logicals should extend the image space into a basis for the kernel.
    assert generated_space == null_space

    expected_num_logicals = _dimension(null_space) - _dimension(image_space)
    assert logicals.shape[0] == expected_num_logicals


@pytest.mark.parametrize(
    ("hx", "hz"),
    [
        (_binary_matrix([], 3), _binary_matrix([], 3)),
        (_binary_matrix([[1, 1, 0], [0, 1, 1]], 3), _binary_matrix([], 3)),
        (_binary_matrix(np.eye(3, dtype=np.uint8).tolist(), 3), _binary_matrix([], 3)),
        (
            _binary_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]], 3),
            _binary_matrix([], 3),
        ),
        (
            _binary_matrix([[1, 1, 1, 1]], 4),
            _binary_matrix([[1, 1, 0, 0], [0, 0, 1, 1]], 4),
        ),
    ],
)
def test_css_code_compute_logicals_returns_valid_css_logical_bases(
    hx: np.ndarray, hz: np.ndarray
):
    assert np.all((hx @ hz.T) % 2 == 0)

    lx, lz = css_code_compute_logicals(hx.astype(float), hz.astype(float))

    _assert_logical_basis(np.asarray(lx, dtype=np.uint8), hz, hx)
    _assert_logical_basis(np.asarray(lz, dtype=np.uint8), hx, hz)


def test_css_code_compute_logicals_is_invariant_to_redundant_rows():
    hx = _binary_matrix([[1, 1, 0], [0, 1, 1]], 3)
    redundant_hx = _binary_matrix([[1, 1, 0], [0, 1, 1], [1, 0, 1]], 3)
    hz = _binary_matrix([], 3)

    lx, lz = css_code_compute_logicals(hx.astype(float), hz.astype(float))
    redundant_lx, redundant_lz = css_code_compute_logicals(
        redundant_hx.astype(float), hz.astype(float)
    )

    assert _binary_row_space(np.asarray(lx, dtype=np.uint8)) == _binary_row_space(
        np.asarray(redundant_lx, dtype=np.uint8)
    )
    assert _binary_row_space(np.asarray(lz, dtype=np.uint8)) == _binary_row_space(
        np.asarray(redundant_lz, dtype=np.uint8)
    )


def test_css_code_compute_logicals_preserves_logical_dimension_for_known_rank_checks():
    # Adapted from the explicit GF(2) row-reduction fixture in galois/tests/fields/test_linalg.py.
    hx = _binary_matrix(
        [
            [1, 0, 1, 0, 1, 0, 1, 0],
            [0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 0, 1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
        8,
    )
    hz = _binary_matrix([], 8)

    rank_hx = _dimension(_binary_row_space(hx))
    assert rank_hx == 4

    lx, lz = css_code_compute_logicals(hx.astype(float), hz.astype(float))

    assert lx.shape == (8 - rank_hx, 8)
    assert lz.shape == (8 - rank_hx, 8)
    _assert_logical_basis(np.asarray(lx, dtype=np.uint8), hz, hx)
    _assert_logical_basis(np.asarray(lz, dtype=np.uint8), hx, hz)
