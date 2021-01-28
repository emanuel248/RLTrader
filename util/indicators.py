from ta import add_all_ta_features

# using
# https://technical-analysis-library-in-python.readthedocs.io/en/latest
def add_indicators(df):
    df = add_all_ta_features(
        df, open="Open", high="High", low="Low", close="Close", volume="Volume BTC", fillna=True)

    return df
