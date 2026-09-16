import numpy as np
import pandas as pd

from dmc2014_fast_history import prepare_training_features_loo


def test_prepare_training_features_loo_excludes_each_rows_own_target():
    frame = pd.DataFrame(
        {
            "orderItemID": [1, 2, 3],
            "orderDate": ["2012-04-10", "2012-05-10", "2012-06-10"],
            "customerID": [9, 9, 8],
            "returnShipment": [1, 0, 1],
        }
    )

    features, target, categorical = prepare_training_features_loo(
        frame,
        group_specs=[("customerID",)],
        smoothing=0.0,
    )

    # Customer 9 has two rows. Each row should see only the other row's label.
    assert target.tolist() == [1, 0, 1]
    assert features["hist_customerID_count"].tolist() == [1, 1, 0]
    assert np.allclose(features["hist_customerID_return_rate"], [0.0, 1.0, 2 / 3])
    assert "customerID" in categorical
