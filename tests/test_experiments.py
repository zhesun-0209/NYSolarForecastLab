#!/usr/bin/env python3
"""Tests for experiment-grid and result-file bookkeeping."""

import pandas as pd

from nysolarforecastlab.experiments import (
    check_plant_completion,
    check_plant_status,
    find_latest_result_file,
    generate_all_configs,
    get_result_file_path,
)


def test_generate_all_configs_full_grid_count():
    configs = generate_all_configs("data/Project171.csv")

    assert len(configs) == 284
    assert sum(config["model"] == "Linear" for config in configs) == 4


def test_generate_all_configs_test_mode_is_model_specific():
    configs = generate_all_configs("data/Project171.csv", test_mode=True, test_model="Linear")

    assert len(configs) == 2
    assert {config["model"] for config in configs} == {"Linear"}


def test_result_status_counts_success_rows_only(tmp_path):
    result_path = get_result_file_path(str(tmp_path), "171")
    pd.DataFrame(
        [
            {"experiment_name": "Linear_NWP_noTE", "status": "SUCCESS"},
            {"experiment_name": "LSTM_low_PV_noTE", "status": "FAILED"},
        ]
    ).to_csv(result_path, index=False)

    status = check_plant_status("171", str(tmp_path))
    assert status["completed"] == 1
    assert status["status"] == "IN_PROGRESS"
    assert status["result_file"] == result_path

    is_complete, completed, found_path = check_plant_completion("171", str(tmp_path))
    assert not is_complete
    assert completed == 1
    assert found_path == result_path


def test_legacy_result_file_is_discoverable(tmp_path):
    legacy_path = tmp_path / "results_171.csv"
    pd.DataFrame(
        [
            {"experiment_name": "Linear_NWP_noTE", "status": "SUCCESS"},
        ]
    ).to_csv(legacy_path, index=False)

    assert get_result_file_path(str(tmp_path), "171").endswith("results_171_all.csv")
    assert find_latest_result_file(str(tmp_path), "171") == str(legacy_path)

    is_complete, completed, found_path = check_plant_completion("171", str(tmp_path))
    assert not is_complete
    assert completed == 1
    assert found_path == str(legacy_path)


def test_canonical_result_file_is_preferred_over_legacy(tmp_path):
    canonical_path = get_result_file_path(str(tmp_path), "171")
    legacy_path = tmp_path / "results_171.csv"

    pd.DataFrame([{"experiment_name": "Linear_NWP_noTE", "status": "SUCCESS"}]).to_csv(
        legacy_path, index=False
    )
    pd.DataFrame([{"experiment_name": "Linear_NWP_TE", "status": "SUCCESS"}]).to_csv(
        canonical_path, index=False
    )

    assert find_latest_result_file(str(tmp_path), "171") == canonical_path
