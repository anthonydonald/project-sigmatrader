import os

import pandas as pd

from sigmatrader.splitter import RelativeSplitter


def test_splitter():
    # GIVEN
    test_df = pd.DataFrame([i for i in range(100)])
    splitter = RelativeSplitter(train=0.7, valid=0.1, test=0.2)

    # WHEN
    datasplits = splitter.split(test_df)

    # THEN
    assert [j for j in datasplits["train"].values] == [i for i in range(70)]
    assert [j for j in datasplits["valid"].values] == [i for i in range(70, 80)]
    assert [j for j in datasplits["test"].values] == [i for i in range(80, 100)]
