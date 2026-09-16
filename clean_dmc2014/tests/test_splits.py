import pandas as pd

from dmc2014.splits import TemporalFold, default_folds, split_fold


def test_default_folds_are_january_february_march_2013():
    folds = default_folds()
    assert [fold.name for fold in folds] == ["2013-01", "2013-02", "2013-03"]
    assert [fold.valid_start for fold in folds] == [
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2013-02-01"),
        pd.Timestamp("2013-03-01"),
    ]


def test_split_fold_is_strictly_chronological():
    frame = pd.DataFrame(
        {
            "orderDate": pd.to_datetime(
                ["2012-12-31", "2013-01-01", "2013-01-31", "2013-02-01"]
            ),
            "returnShipment": [0, 1, 0, 1],
        }
    )
    fold = TemporalFold(
        name="2013-01",
        train_end=pd.Timestamp("2013-01-01"),
        valid_start=pd.Timestamp("2013-01-01"),
        valid_end=pd.Timestamp("2013-02-01"),
    )
    train, valid = split_fold(frame, fold)
    assert train["orderDate"].tolist() == [pd.Timestamp("2012-12-31")]
    assert valid["orderDate"].tolist() == [
        pd.Timestamp("2013-01-01"),
        pd.Timestamp("2013-01-31"),
    ]
    assert train["orderDate"].max() < fold.valid_start
    assert valid["orderDate"].min() >= fold.valid_start
    assert valid["orderDate"].max() < fold.valid_end
