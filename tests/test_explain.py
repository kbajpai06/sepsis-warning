import numpy as np

from sepsis.explain import describe, parse_feature, top_drivers


def test_parse_feature():
    assert parse_feature("MAP_slope6") == ("MAP", "slope", 6)
    assert parse_feature("O2Sat_std12") == ("O2Sat", "std", 12)
    assert parse_feature("shock_index_mean6") == ("shock_index", "mean", 6)
    assert parse_feature("Lactate_age") == ("Lactate", "staleness", None)
    assert parse_feature("HR") == ("HR", "level", None)


def test_describe_directional_text():
    assert describe("MAP_slope6", -3.0, None).startswith("Rapid drop in mean arterial pressure")
    assert describe("Lactate", 4.0, 1.5).startswith("Elevated lactate")


def test_top_drivers_aggregates_by_variable_and_ignores_static():
    names = ["Lactate", "Lactate_mean6", "MAP_slope6", "HR", "Age"]
    shap_row = np.array([1.0, 0.8, 0.5, -0.2, 5.0])           # Age is huge but must be ignored
    values = dict(zip(names, [4.0, 3.5, -2.0, 70.0, 80.0]))
    out = top_drivers(shap_row, names, values, {"Lactate": 1.5}, k=3)
    assert [d["variable"] for d in out] == ["lactate", "mean arterial pressure"]   # HR net-negative -> excluded
    assert out[0]["shap"] == 1.8
