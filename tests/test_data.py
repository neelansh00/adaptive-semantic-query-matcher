import pandas as pd

from src.utils.data import EXPECTED_COLUMNS, TARGET, clean_pairs, identity_key, invalid_pair_mask


def test_raw_schema_and_target(raw_df):
    assert list(raw_df.columns) == EXPECTED_COLUMNS
    assert len(raw_df) == 404_290
    assert set(raw_df[TARGET].unique()) == {0, 1}
    assert raw_df["id"].is_unique


def test_invalid_pairs_are_the_three_known_rows(raw_df):
    assert sorted(raw_df.loc[invalid_pair_mask(raw_df), "id"]) == [105780, 201841, 363362]


def test_clean_pairs_keeps_text_untouched():
    df = pd.DataFrame({"question1": ["  Hi there? ", "", "n/a", "NA"],
                       "question2": ["x", "y", "z", "w"]})
    out = clean_pairs(df)
    assert out["question1"].tolist() == ["  Hi there? ", "NA"]  # "NA" is a real string, not a placeholder


def test_identity_key_is_conservative():
    assert identity_key("What is  AI? ") == identity_key("what is ai?")
    assert identity_key("What is AI?") != identity_key("What is A.I.?")
