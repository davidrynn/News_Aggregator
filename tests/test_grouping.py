import numpy as np
import pytest

from newsagg import grouping


@pytest.mark.parametrize("title, expected", [
    ("Florida officer accused of taking over home – NBC Connecticut", "Florida officer accused of taking over home"),
    ("HOLDING HER BACK? | The Manila Times", "HOLDING HER BACK? | The Manila Times"),  # too short once stripped
    ("Senate passes budget bill after long debate", "Senate passes budget bill after long debate"),
])
def test_clean_title(title, expected):
    assert grouping.clean_title(title) == expected


def test_group_vectors_uses_leaders_not_chains():
    # b is close to a, c is close to b but not to a: c must start its own group.
    angle = np.radians([0, 25, 50])
    vectors = np.stack([np.cos(angle), np.sin(angle)], axis=1)  # unit vectors
    assert grouping.group_vectors(vectors, threshold=0.85) == [[0, 1], [2]]
