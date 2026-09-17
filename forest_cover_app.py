import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Forest Cover Type", page_icon="🌲", layout="wide")

TRAIN_PATH = "train (1).csv"
TEST_PATH = "test-full (1).csv"
TARGET = "Cover_Type"
RANDOM_STATE = 42

COVER_NAMES = {1: "Spruce/Fir", 2: "Lodgepole Pine", 3: "Ponderosa Pine", 4: "Cottonwood/Willow",
               5: "Aspen", 6: "Douglas-fir", 7: "Krummholz"}
# Une couleur fixe par espèce, identique dans tous les graphiques
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
COLOR_MAP = dict(zip(COVER_NAMES.values(), PALETTE))
SPECIES_ORDER = {"Espèce": list(COVER_NAMES.values()), "Espèce prédite": list(COVER_NAMES.values())}

NUM_COLS = ["Elevation", "Aspect", "Slope", "Horizontal_Distance_To_Hydrology",
            "Vertical_Distance_To_Hydrology", "Horizontal_Distance_To_Roadways",
            "Hillshade_9am", "Hillshade_Noon", "Hillshade_3pm", "Horizontal_Distance_To_Fire_Points"]
LABELS = {
    "Elevation": "Altitude (m)",
    "Aspect": "Orientation (°)",
    "Slope": "Pente (°)",
    "Horizontal_Distance_To_Hydrology": "Distance horizontale à l'eau (m)",
    "Vertical_Distance_To_Hydrology": "Distance verticale à l'eau (m)",
    "Horizontal_Distance_To_Roadways": "Distance aux routes (m)",
    "Hillshade_9am": "Ombrage 9h (0-255)",
    "Hillshade_Noon": "Ombrage midi (0-255)",
    "Hillshade_3pm": "Ombrage 15h (0-255)",
    "Horizontal_Distance_To_Fire_Points": "Distance aux départs de feu (m)",
}
WILDERNESS_NAMES = {"Wilderness_Area1": "Rawah", "Wilderness_Area2": "Neota",
                    "Wilderness_Area3": "Comanche Peak", "Wilderness_Area4": "Cache la Poudre"}

MODELS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
    "KNN (k=5)": lambda: KNeighborsClassifier(n_neighbors=5),
    "Régression logistique": lambda: LogisticRegression(max_iter=2000),
}

st.markdown("""
<style>
.hero {padding: 1.4rem 1.6rem; border-radius: 14px; margin-bottom: 1rem;
       background: linear-gradient(120deg, #0f3d2e 0%, #1c6b4a 55%, #2f8f5b 100%); color: #fff;}
.hero h1 {color: #fff; margin: 0; font-size: 2.1rem;}
.hero p {color: #e3f2ea; margin: .35rem 0 0 0; font-size: 1.05rem;}
</style>
""", unsafe_allow_html=True)


# ---------- Données ----------
def decode(df):
    """Ajoute des colonnes lisibles : nom de la zone naturelle et numéro du type de sol."""
    wild = [c for c in df if c.startswith("Wilderness_Area")]
    soil = [c for c in df if c.startswith("Soil_Type")]
    df = df.copy()
    df["Zone"] = df[wild].idxmax(axis=1).map(WILDERNESS_NAMES)
    df["Sol"] = df[soil].idxmax(axis=1).str.replace("Soil_Type", "").astype(int)
    return df


@st.cache_data
def load_train():
    df = pd.read_csv(TRAIN_PATH)
    const_cols = [c for c in df if df[c].nunique() == 1]
    df = df.drop(columns=["Id"] + const_cols)
    df = decode(df)
    df["Espèce"] = df[TARGET].map(COVER_NAMES)
    return df, const_cols


@st.cache_data
def load_test_sample(n, seed):
    df = pd.read_csv(TEST_PATH)
    return decode(df.sample(n=n, random_state=seed)), len(df)


def feature_cols(df):
    return NUM_COLS + [c for c in df if c.startswith(("Wilderness_Area", "Soil_Type"))]


def make_pipeline(name):
    prep = ColumnTransformer([("scale", StandardScaler(), NUM_COLS)], remainder="passthrough")
    return Pipeline([("prep", prep), ("model", MODELS[name]())])


# ---------- Modèles ----------
@st.cache_data
def evaluate_models():
    train, _ = load_train()
    X, y = train[feature_cols(train)], train[TARGET]
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)
    rows = []
    for name in MODELS:
        pipe = make_pipeline(name).fit(X_tr, y_tr)
        pred_tr, pred_val = pipe.predict(X_tr), pipe.predict(X_val)
        rows.append({"Modèle": name,
                     "Accuracy train": accuracy_score(y_tr, pred_tr),
                     "Accuracy val": accuracy_score(y_val, pred_val),
                     "F1 macro val": f1_score(y_val, pred_val, average="macro")})
    return pd.DataFrame(rows).set_index("Modèle").sort_values("F1 macro val", ascending=False)


@st.cache_resource
def fit_full_model(name):
    """Modèle final, entraîné sur la totalité du train."""
    train, _ = load_train()
    return make_pipeline(name).fit(train[feature_cols(train)], train[TARGET])


@st.cache_data
def predict_test(name, n, seed):
    test, n_total = load_test_sample(n, seed)
    model = fit_full_model(name)
    X = test[feature_cols(load_train()[0])]
    proba = model.predict_proba(X)
    test = test.assign(**{"Espèce prédite": pd.Series(model.classes_[proba.argmax(1)], index=test.index).map(COVER_NAMES),
                          "Confiance": proba.max(1)})
    return test, n_total


# ---------- Sidebar ----------
with st.sidebar:
    st.header("⚙️ Paramètres")
    model_name = st.selectbox("Modèle de prédiction", list(MODELS), index=0,
                              help="Random Forest = meilleur modèle du notebook")
    n_test = st.select_slider("Parcelles du test à prédire", options=[2_000, 5_000, 10_000, 25_000, 50_000],
                              value=10_000)
    seed = st.number_input("Graine d'échantillonnage", 0, 9999, RANDOM_STATE)
    st.divider()
    st.caption("Dataset : *Forest Cover Type* (Roosevelt National Forest, Colorado).")

st.markdown("""
<div class="hero">
  <h1>🌲 Forest Cover Type</h1>
  <p>Prédire l'espèce d'arbre dominante d'une parcelle à partir de sa topographie, de l'eau et du sol.</p>
</div>
""", unsafe_allow_html=True)


# ---------- 1. Chargement ----------
st.header("1. Chargement des données")
with st.spinner("Chargement du train…"):
    train, const_cols = load_train()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Parcelles (train)", f"{len(train):,}".replace(",", " "))
c2.metric("Variables explicatives", len(feature_cols(train)))
c3.metric("Espèces", train[TARGET].nunique())
c4.metric("Valeurs manquantes", int(train.isna().sum().sum()))

if st.checkbox("🔍 Explorer les données (EDA)"):
    with st.container(border=True):
        st.subheader("Échantillon")
        s1, s2 = st.columns([1, 2])
        n_rows = s1.slider("Nombre de lignes", 5, 200, 20, step=5)
        species_filter = s2.multiselect("Espèces", list(COVER_NAMES.values()), default=list(COVER_NAMES.values()))
        view = train[train["Espèce"].isin(species_filter)]
        st.dataframe(view[["Espèce", "Zone", "Sol"] + NUM_COLS].sample(min(n_rows, len(view)), random_state=seed),
                     use_container_width=True, hide_index=True)

        st.subheader("Visualisations interactives")
        eda_view = st.radio("Vue", ["📊 Distribution", "🔗 Relation entre variables", "🗺️ Zones & sols"],
                            horizontal=True, label_visibility="collapsed")

        if eda_view == "📊 Distribution":
            d1, d2, d3 = st.columns([2, 1, 1])
            var = d1.selectbox("Variable", NUM_COLS, format_func=LABELS.get)
            kind = d2.radio("Graphique", ["Histogramme", "Boîte", "Violon"], horizontal=True)
            nbins = d3.slider("Nombre de classes", 10, 100, 40, disabled=kind != "Histogramme")
            lo, hi = int(train[var].min()), int(train[var].max())
            rng = st.slider(f"Plage de {LABELS[var].lower()}", lo, hi, (lo, hi))
            sub = view[view[var].between(*rng)]
            common = dict(color="Espèce", color_discrete_map=COLOR_MAP, category_orders=SPECIES_ORDER,
                          labels={var: LABELS[var]})
            if kind == "Histogramme":
                fig = px.histogram(sub, x=var, nbins=nbins, barmode="overlay", opacity=0.6, **common)
            elif kind == "Boîte":
                fig = px.box(sub, x="Espèce", y=var, **common)
            else:
                fig = px.violin(sub, x="Espèce", y=var, box=True, **common)
            fig.update_layout(height=450, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"{len(sub):,} parcelles affichées".replace(",", " "))

        elif eda_view == "🔗 Relation entre variables":
            r1, r2, r3 = st.columns(3)
            x = r1.selectbox("Axe X", NUM_COLS, index=3, format_func=LABELS.get)
            y = r2.selectbox("Axe Y", NUM_COLS, index=0, format_func=LABELS.get)
            n_pts = r3.slider("Points affichés", 500, len(view), min(3000, len(view)), step=500)
            fig = px.scatter(view.sample(min(n_pts, len(view)), random_state=seed), x=x, y=y, color="Espèce",
                             color_discrete_map=COLOR_MAP, category_orders=SPECIES_ORDER, opacity=0.6,
                             labels=LABELS, hover_data=["Zone", "Sol"])
            fig.update_traces(marker_size=6)
            fig.update_layout(height=500, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

            if st.checkbox("Afficher la matrice de corrélation"):
                corr = train[NUM_COLS].corr().rename(index=LABELS, columns=LABELS)
                fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)

        else:
            z1, z2 = st.columns(2)
            zone_counts = view.groupby(["Zone", "Espèce"]).size().reset_index(name="Parcelles")
            fig = px.bar(zone_counts, x="Zone", y="Parcelles", color="Espèce", color_discrete_map=COLOR_MAP,
                         category_orders=SPECIES_ORDER, title="Espèces par zone naturelle")
            fig.update_layout(height=450, legend_title_text="", barmode="stack")
            z1.plotly_chart(fig, use_container_width=True)

            top_n = z2.slider("Types de sol les plus fréquents", 5, 38, 15)
            top_soils = view["Sol"].value_counts().head(top_n).index
            soil_tab = pd.crosstab(view.loc[view["Sol"].isin(top_soils), "Sol"],
                                   view.loc[view["Sol"].isin(top_soils), "Espèce"], normalize="index")
            fig = px.imshow(soil_tab, color_continuous_scale="Greens", aspect="auto", text_auto=".0%",
                            labels=dict(x="", y="Type de sol", color="Part"),
                            title="Part de chaque espèce par type de sol")
            fig.update_yaxes(type="category")
            fig.update_layout(height=450)
            z2.plotly_chart(fig, use_container_width=True)


# ---------- 2. Démarche ----------
st.header("2. Démarche")
left, right = st.columns([1.1, 1])
with left:
    st.markdown(f"""
**Données.** Chaque ligne est une parcelle de 30 × 30 m décrite par 10 variables topographiques
(altitude, pente, orientation, ombrage, distances à l'eau, aux routes et aux départs de feu),
4 zones naturelles et 40 types de sol encodés en one-hot. La cible `Cover_Type` compte 7 espèces,
parfaitement équilibrées dans le train (2 160 parcelles chacune).

**Nettoyage.** Aucune valeur manquante ni doublon ; chaque parcelle a exactement une zone et un sol.
On retire `Id` et les colonnes constantes ({", ".join(const_cols)}).

**Pré-traitement.** Les 10 variables continues sont standardisées (`StandardScaler`) dans un `Pipeline`,
ajusté uniquement sur la partie entraînement ; les variables binaires restent telles quelles.

**Modélisation.** Split stratifié 80 / 20, puis comparaison de trois classifieurs :
régression logistique multinomiale, Random Forest (300 arbres) et KNN (k = 5).
Le **Random Forest** l'emporte : il capte les effets non linéaires, notamment l'altitude.
Son accuracy de 100 % sur le train signale toutefois du sur-apprentissage.

**Prédiction.** Le modèle choisi est ré-entraîné sur tout le train, puis appliqué au fichier `test-full`
(non labellisé) : la section 3 explore ces prédictions.
""")
with right:
    with st.spinner("Évaluation des modèles…"):
        scores = evaluate_models()
    st.dataframe(scores.style.format("{:.3f}").highlight_max(axis=0, color="#1baf7a55"),
                 use_container_width=True)
    melted = scores[["Accuracy val", "F1 macro val"]].reset_index().melt(id_vars="Modèle", var_name="Métrique",
                                                                         value_name="Score")
    fig = px.bar(melted, x="Score", y="Modèle", color="Métrique", barmode="group", orientation="h",
                 text_auto=".3f", range_x=[0, 1], color_discrete_sequence=PALETTE[:2])
    fig.update_layout(height=320, legend=dict(orientation="h", y=-0.25, title_text=""), yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


# ---------- 3. Prédictions ----------
st.header("3. Prédictions par espèce sur le jeu de test")
with st.spinner(f"Entraînement du modèle {model_name} et prédiction…"):
    preds, n_test_total = predict_test(model_name, n_test, seed)
st.caption(f"Modèle **{model_name}** · {n_test:,} parcelles tirées au hasard parmi les {n_test_total:,} du test."
           .replace(",", " "))

with st.container(border=True):
    st.markdown("**Filtres**")
    f1, f2, f3 = st.columns(3)
    elev = f1.slider("Altitude (m)", int(preds.Elevation.min()), int(preds.Elevation.max()),
                     (int(preds.Elevation.min()), int(preds.Elevation.max())), step=10)
    hmax = int(preds.Horizontal_Distance_To_Hydrology.max())
    hydro = f2.slider("Distance horizontale à l'eau (m)", 0, hmax, (0, hmax), step=10)
    slope = f3.slider("Pente (°)", 0, int(preds.Slope.max()), (0, int(preds.Slope.max())))
    g1, g2 = st.columns([2, 1])
    zones = g1.multiselect("Zones naturelles", list(WILDERNESS_NAMES.values()), default=list(WILDERNESS_NAMES.values()))
    min_conf = g2.slider("Confiance minimale", 0.0, 1.0, 0.0, step=0.05)

filt = preds[preds.Elevation.between(*elev) & preds.Horizontal_Distance_To_Hydrology.between(*hydro)
             & preds.Slope.between(*slope) & preds.Zone.isin(zones) & (preds.Confiance >= min_conf)]

if filt.empty:
    st.warning("Aucune parcelle ne correspond aux filtres.")
    st.stop()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Parcelles filtrées", f"{len(filt):,}".replace(",", " "), f"{len(filt) / len(preds):.0%} de l'échantillon",
          delta_color="off")
dominant = filt["Espèce prédite"].value_counts()
k2.metric("Espèce dominante", dominant.index[0], f"{dominant.iloc[0] / len(filt):.0%}", delta_color="off")
k3.metric("Confiance moyenne", f"{filt.Confiance.mean():.0%}")
k4.metric("Altitude médiane", f"{filt.Elevation.median():.0f} m")

p1, p2 = st.columns([1, 1.6])
with p1:
    counts = filt["Espèce prédite"].value_counts().reindex(COVER_NAMES.values(), fill_value=0).reset_index()
    counts.columns = ["Espèce prédite", "Parcelles"]
    fig = px.bar(counts, x="Parcelles", y="Espèce prédite", color="Espèce prédite", orientation="h",
                 color_discrete_map=COLOR_MAP, text_auto=True, title="Répartition des espèces prédites")
    fig.update_layout(height=430, showlegend=False, yaxis_title="", yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with p2:
    b1, b2 = st.columns([2, 1])
    grad_var = b1.selectbox("Gradient étudié", NUM_COLS, format_func=LABELS.get)
    n_bins = b2.slider("Tranches", 4, 30, 12)
    binned = filt.assign(Tranche=pd.cut(filt[grad_var], bins=n_bins))
    prop = (binned.groupby("Tranche", observed=True)["Espèce prédite"].value_counts(normalize=True)
            .rename("Part").reset_index())
    prop["Milieu"] = prop["Tranche"].apply(lambda iv: iv.mid).astype(float)
    fig = px.area(prop, x="Milieu", y="Part", color="Espèce prédite", color_discrete_map=COLOR_MAP,
                  category_orders=SPECIES_ORDER, labels={"Milieu": LABELS[grad_var]},
                  title=f"Composition prédite selon : {LABELS[grad_var].lower()}")
    fig.update_layout(height=380, yaxis_tickformat=".0%", legend_title_text="", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Carte des prédictions dans l'espace topographique")
m1, m2, m3 = st.columns(3)
mx = m1.selectbox("Axe X ", NUM_COLS, index=3, format_func=LABELS.get)
my = m2.selectbox("Axe Y ", NUM_COLS, index=0, format_func=LABELS.get)
highlight = m3.multiselect("Espèces affichées", list(COVER_NAMES.values()), default=list(COVER_NAMES.values()))
plot_df = filt[filt["Espèce prédite"].isin(highlight)]
fig = px.scatter(plot_df.sample(min(len(plot_df), 8000), random_state=seed), x=mx, y=my, color="Espèce prédite",
                 color_discrete_map=COLOR_MAP, category_orders=SPECIES_ORDER, opacity=0.6, labels=LABELS,
                 hover_data={"Confiance": ":.0%", "Zone": True, "Sol": True})
fig.update_traces(marker_size=5)
fig.update_layout(height=520, legend_title_text="")
st.plotly_chart(fig, use_container_width=True)

st.download_button("⬇️ Télécharger les prédictions filtrées (CSV)",
                   filt[["Id", "Espèce prédite", "Confiance", "Zone", "Sol"] + NUM_COLS].to_csv(index=False),
                   file_name="predictions_forest_cover.csv", mime="text/csv")


# ---------- 4. Simulateur ----------
st.header("4. Simulateur : quelle espèce pour ma parcelle ?")
st.caption("Réglez les caractéristiques d'une parcelle fictive : la prédiction se met à jour.")
ref = train[NUM_COLS]
sim_cols = st.columns(3)
values = {}
for i, col in enumerate(NUM_COLS):
    values[col] = sim_cols[i % 3].slider(LABELS[col], int(ref[col].min()), int(ref[col].max()),
                                         int(ref[col].median()), key=f"sim_{col}")
w1, w2 = st.columns(2)
zone = w1.selectbox("Zone naturelle", list(WILDERNESS_NAMES.values()))
soil_options = sorted(train["Sol"].unique())
soil = w2.selectbox("Type de sol", soil_options, index=soil_options.index(int(train["Sol"].mode()[0])))

row = pd.DataFrame(0, index=[0], columns=feature_cols(train))
for col, v in values.items():
    row[col] = v
row[{v: k for k, v in WILDERNESS_NAMES.items()}[zone]] = 1
row[f"Soil_Type{soil}"] = 1

model = fit_full_model(model_name)
proba = pd.Series(model.predict_proba(row)[0], index=model.classes_).rename(COVER_NAMES)
best = proba.idxmax()

r1, r2 = st.columns([1, 2])
with r1:
    st.metric("Espèce prédite", best, f"probabilité {proba.max():.0%}", delta_color="off")
    st.markdown(f"<div style='height:14px;border-radius:7px;background:{COLOR_MAP[best]}'></div>",
                unsafe_allow_html=True)
with r2:
    pdf = proba.reset_index()
    pdf.columns = ["Espèce", "Probabilité"]
    fig = px.bar(pdf, x="Probabilité", y="Espèce", orientation="h", color="Espèce", color_discrete_map=COLOR_MAP,
                 text_auto=".0%", range_x=[0, 1])
    fig.update_layout(height=300, showlegend=False, xaxis_tickformat=".0%", yaxis_title="",
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)
