import numpy as np
import pandas as pd

from src.features.lendingclub import parse_emp_length, parse_percent, parse_term


def test_lendingclub_term_parsing():
    assert parse_term(" 36 months") == 36
    assert parse_term("60 months") == 60
    assert np.isnan(parse_term(None))


def test_lendingclub_emp_length_parsing():
    assert parse_emp_length("10+ years") == 10
    assert parse_emp_length("< 1 year") == 0
    assert parse_emp_length("3 years") == 3
    assert np.isnan(parse_emp_length("n/a"))


def test_lendingclub_percent_parsing():
    assert parse_percent("42.3%") == 42.3
    assert parse_percent(12.5) == 12.5
    assert pd.isna(parse_percent(""))
