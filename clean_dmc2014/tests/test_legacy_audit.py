import numpy as np
import pandas as pd

import dmc2014.legacy_audit as legacy_audit
from dmc2014.legacy_audit import prepare_legacy_audit_data, score_predictions


def test_prepare_legacy_audit_data_seals_competition_labels():
    frame = pd.DataFrame(
        {
            "order_item_id": [1, 2, 1],
            "order_date": pd.to_datetime(["2013-03-01", "2013-03-02", "2013-04-01"]),
            "delivery_date": pd.to_datetime(["2013-03-03", "2013-03-04", "2013-04-03"]),
            "user_dob": pd.to_datetime(["1980-01-01"] * 3),
            "user_reg_date": pd.to_datetime(["2012-01-01"] * 3),
            "order_id": ["a", "b", "c"],
            "item_id": [10, 11, 12],
            "size": [1, 2, 3],
            "item_color": [1, 2, 3],
            "brand_id": [4, 5, 6],
            "user_id": [7, 8, 9],
            "user_title": [0, 0, 0],
            "user_state": [0, 0, 0],
            "mode_item_id": [10, 11, 12],
            "mode_size": [1, 2, 3],
            "mode_brand_id": [4, 5, 6],
            "mode_item_color": [1, 2, 3],
            "numeric_feature": [1.0, 2.0, 3.0],
            "return": [0.0, 1.0, np.nan],
        }
    )
    data = prepare_legacy_audit_data(frame)
    assert data.train_x.shape[0] == 2
    assert data.competition_x.shape[0] == 1
    assert "return" not in data.train_x
    assert "return" not in data.competition_x
    assert data.train_y.tolist() == [0, 1]
    assert data.train_x.columns.tolist() == data.competition_x.columns.tolist()


def test_legacy_catboost_settings_use_dynamic_workers(monkeypatch):
    monkeypatch.setattr(legacy_audit, "resolve_workers", lambda profile: 13)
    assert legacy_audit.legacy_catboost_settings()["thread_count"] == 13
    assert legacy_audit.legacy_catboost_settings({"thread_count": 3})["thread_count"] == 3


def test_score_predictions_is_exact_mistake_count():
    result = score_predictions(np.array([0, 1, 1, 0]), np.array([0, 0, 1, 1]))
    assert result.points == 2
    assert result.accuracy == 0.5
