import pandas as pd

from data_utils import load_csv, filter_by_species


def test_load_csv(tmp_path):
    csv_file = tmp_path / "test.csv"

    csv_file.write_text(
        "Espèce,Elevation\n"
        "Spruce,2500\n"
        "Pine,1800\n"
    )

    df = load_csv(csv_file)

    assert len(df) == 2
    assert list(df.columns) == ["Espèce", "Elevation"]


def test_filter_by_species():
    df = pd.DataFrame({
        "Espèce": ["Spruce", "Pine", "Spruce"],
        "Elevation": [2500, 1800, 2200]
    })

    result = filter_by_species(df, ["Spruce"])

    assert len(result) == 2
    assert (result["Espèce"] == "Spruce").all()


def test_filter_multiple_species():
    df = pd.DataFrame({
        "Espèce": ["Spruce", "Pine", "Fir"],
        "Elevation": [2500, 1800, 2100]
    })

    result = filter_by_species(df, ["Spruce", "Fir"])

    assert len(result) == 2
    assert set(result["Espèce"]) == {"Spruce", "Fir"}