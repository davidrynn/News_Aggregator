import pytest

from newsagg.runlock import AlreadyRunning, run_lock


def test_second_run_is_refused_while_first_holds_lock(tmp_path):
    path = tmp_path / "run.lock"
    with run_lock(path):
        with pytest.raises(AlreadyRunning):
            with run_lock(path):
                pass
    with run_lock(path):        # released after the first run ends
        pass


def test_lock_released_when_run_fails(tmp_path):
    path = tmp_path / "run.lock"
    with pytest.raises(ValueError):
        with run_lock(path):
            raise ValueError("crash")
    with run_lock(path):
        pass
