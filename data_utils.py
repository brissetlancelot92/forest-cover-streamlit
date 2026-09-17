import pandas as pd


def load_csv(path):
    """Charge un fichier CSV dans un DataFrame."""
    return pd.read_csv(path)


def filter_by_species(df, species):
    """Filtre un DataFrame selon les espèces sélectionnées."""
    return df[df["Espèce"].isin(species)].copy()