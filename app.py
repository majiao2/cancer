# Author: Your Name
# CancerLens: a wood-and-paper styled cancer-patient data explorer.
# This application is an educational project, not a medical device.

from io import BytesIO
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


# ---------------------------------------------------------
# Page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="CancerLens | Patient Data Atelier",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Wood, ink, and paper visual system
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=IM+Fell+English:ital@0;1&family=Manrope:wght@400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --paper: #efe6d5;
        --paper-light: #f8f3e9;
        --ink: #2f261e;
        --walnut: #503624;
        --wood: #795538;
        --brass: #b48a4d;
        --moss: #65745a;
        --line: rgba(77, 55, 35, .20);
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 12% 5%, rgba(185, 146, 90, .12), transparent 34%),
            radial-gradient(ellipse at 90% 85%, rgba(101, 116, 90, .10), transparent 33%),
            repeating-linear-gradient(0deg, rgba(80, 54, 36, .018) 0px, rgba(80, 54, 36, .018) 1px, transparent 1px, transparent 5px),
            var(--paper);
    }

    [data-testid="stSidebar"] {
        background:
            repeating-linear-gradient(90deg, rgba(255,255,255,.025) 0px, rgba(255,255,255,.025) 2px, transparent 2px, transparent 9px),
            linear-gradient(160deg, #3a2a20, #533923 55%, #38281e);
        border-right: 1px solid rgba(255,255,255,.12);
    }

    [data-testid="stSidebar"] * {
        color: #f4ead7;
    }

    [data-testid="stSidebar"] input {
        color: #2f261e !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3.5rem;
        max-width: 1450px;
    }

    h1, h2, h3, .serif-title {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        color: var(--walnut);
        letter-spacing: -.4px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 22px;
        padding: 40px 44px;
        color: #f4ead7;
        background:
            repeating-linear-gradient(2deg, rgba(255,255,255,.025) 0px, rgba(255,255,255,.025) 2px, transparent 2px, transparent 9px),
            linear-gradient(125deg, #38291f 0%, #68472d 52%, #354033 100%);
        box-shadow: 0 14px 34px rgba(54, 39, 26, .17);
        margin-bottom: 22px;
    }

    .hero:after {
        content: "✦";
        position: absolute;
        right: 34px;
        top: 14px;
        font-family: 'Cormorant Garamond', serif;
        font-size: 100px;
        color: rgba(238, 213, 164, .13);
        transform: rotate(12deg);
    }

    .hero .eyebrow {
        font: 500 11px 'DM Mono', monospace;
        letter-spacing: 2.3px;
        color: #dbc291;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #fff8eb;
        font-size: clamp(42px, 5vw, 62px);
        line-height: .98;
        margin: 13px 0 12px;
        font-weight: 600;
    }

    .hero p {
        color: #e6dac7;
        max-width: 800px;
        font-size: 15px;
        line-height: 1.8;
    }

    .metric-card {
        background: rgba(255, 251, 242, .82);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 17px 19px;
        min-height: 102px;
        box-shadow: 0 5px 14px rgba(74, 54, 34, .06);
    }

    .metric-label {
        font: 11px 'DM Mono', monospace;
        color: #766653;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .metric-value {
        font-family: 'Cormorant Garamond', Georgia, serif;
        font-size: 32px;
        font-weight: 700;
        margin-top: 5px;
        color: var(--walnut);
    }

    .paper-card {
        background: rgba(255, 251, 242, .72);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
    }

    .note {
        border-left: 3px solid var(--brass);
        background: rgba(255, 250, 238, .78);
        padding: 13px 16px;
        border-radius: 0 10px 10px 0;
        color: #514333;
        font-size: 13px;
    }

    .stButton > button {
        border-radius: 10px;
        border: 1px solid #68472d;
        background: linear-gradient(130deg, #68472d, #493326);
        color: #fff8eb;
        font-weight: 700;
    }

    .stButton > button:hover {
        border-color: #b48a4d;
        color: #fff8eb;
        box-shadow: 0 4px 14px rgba(80, 54, 36, .18);
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 251, 242, .68);
        border: 1px solid rgba(77, 55, 35, .14);
        border-radius: 14px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }

    [data-testid="stExpander"] {
        background: rgba(255, 251, 242, .55);
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    div[data-testid="stTabs"] button {
        font-weight: 700;
        color: var(--walnut);
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# App helpers
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_uploaded_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Load CSV or Excel data from the upload widget."""
    buffer = BytesIO(file_bytes)

    if file_name.lower().endswith(".csv"):
        return pd.read_csv(buffer)

    return pd.read_excel(buffer)


def normalize_name(value: str) -> str:
    """Normalize a column name for matching."""
    return "".join(
        character.lower()
        for character in str(value)
        if character.isalnum()
    )


def find_target_column(frame: pd.DataFrame):
    """Detect a common classification target name."""
    target_names = {
        "level",
        "cancerlevel",
        "target",
        "diagnosis",
        "label",
        "class",
        "outcome",
    }

    for column in frame.columns:
        if normalize_name(column) in target_names:
            return column

    return None


def find_local_dataset():
    """Load a bundled dataset when present."""
    candidates = [
        Path("data/cancer patient data sets.xlsx"),
        Path("data/cancer_patient_data.xlsx"),
        Path("cancer patient data sets.xlsx"),
        Path("data/cancer_patient_data.csv"),
    ]

    for path in candidates:
        if not path.exists():
            continue

        try:
            if path.suffix.lower() == ".csv":
                return pd.read_csv(path)
            return pd.read_excel(path)
        except Exception:
            continue

    return None


def show_metric(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_axis(ax):
    """Apply the app's ink-and-paper chart styling."""
    ax.set_facecolor("#fbf7ee")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#c9b99f")
    ax.spines["bottom"].set_color("#c9b99f")
    ax.tick_params(colors="#554635")
    ax.grid(axis="y", alpha=0.16, color="#806647")


def render_page_title(eyebrow: str, title: str, subtitle: str):
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------
# Sidebar, navigation, and data input
# ---------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="font-family:'Cormorant Garamond',serif;font-size:30px;
                font-weight:700;letter-spacing:1px;color:#f6ead4;">
        CANCERLENS
    </div>
    <div style="font-family:'IM Fell English',serif;font-style:italic;
                color:#d9c29c;margin-top:-4px;">
        An atlas of patterns and questions
    </div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.write("")

page = st.sidebar.radio(
    "OPEN A CHAPTER",
    [
        "Welcome",
        "Explore the Data",
        "Model Studio",
        "Patient Q&A",
        "Field Notes",
    ],
)

uploaded_file = st.sidebar.file_uploader(
    "Place your dataset",
    type=["xlsx", "xls", "csv"],
    help="Upload the cancer-patient Excel workbook or a CSV file.",
)

data = None

if uploaded_file is not None:
    try:
        data = read_uploaded_data(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
    except Exception as error:
        st.sidebar.error(f"Unable to read this file: {error}")

if data is None:
    data = find_local_dataset()

if data is None:
    st.sidebar.info("Upload the workbook to unlock the data chapters.")

if data is not None:
    data.columns = [str(column).strip() for column in data.columns]
    target_column = find_target_column(data)
    st.sidebar.caption(f"{len(data):,} records · {data.shape[1]} fields")
else:
    target_column = None


# ---------------------------------------------------------
# Welcome page
# ---------------------------------------------------------
if page == "Welcome":
    render_page_title(
        "A DATA ATLAS · CANCER PATIENT DATA",
        "Read the patterns.<br>Keep the human in view.",
        "An interactive learning space for exploring a cancer-patient "
        "dataset, comparing recorded features, and testing baseline "
        "classification models.",
    )

    if data is None:
        st.markdown(
            """
            <div class="note">
                Upload your Excel or CSV file using the sidebar to begin.
                The dataset source listed for this project is
                <a href="https://www.kaggle.com/datasets/rishidamarla/cancer-patients-data">
                Cancer Patients Dataset on Kaggle</a>.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.subheader("A note before we begin")
        st.write(
            "This app is designed for data-science education. It can show "
            "patterns present in a dataset, but it cannot determine what "
            "those patterns mean for an individual person."
        )
        st.stop()

    missing_count = int(data.isna().sum().sum())
    feature_count = data.shape[1] - (1 if target_column else 0)

    metric_cols = st.columns(4)
    with metric_cols[0]:
        show_metric("RECORDS", f"{len(data):,}")
    with metric_cols[1]:
        show_metric("FIELDS", f"{data.shape[1]:,}")
    with metric_cols[2]:
        show_metric("MISSING CELLS", f"{missing_count:,}")
    with metric_cols[3]:
        show_metric("TARGET FIELD", str(target_column or "Not found"))

    st.write("")
    overview_tab, ethics_tab, source_tab = st.tabs(
        ["The collection", "Responsible reading", "Source & scope"]
    )

    with overview_tab:
        st.subheader("What is in this collection?")
        st.write(
            "Each row is a record from the supplied workbook. Depending "
            "on the file, its fields may describe age, gender, risk-factor "
            "scores, symptoms, and a recorded severity label."
        )
        st.dataframe(
            data.head(12),
            use_container_width=True,
            hide_index=True,
        )

    with ethics_tab:
        st.markdown(
            """
            <div class="note">
                Associations in this dataset do not prove causation.
                The dataset should not be treated as a representative
                clinical sample, a personal risk calculator, or a source
                of diagnosis or treatment advice.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with source_tab:
        st.write(
            "The project uses the user-provided workbook associated with "
            "the Cancer Patients Dataset published on Kaggle."
        )
        st.link_button(
            "Open the Kaggle dataset page",
            "https://www.kaggle.com/datasets/rishidamarla/cancer-patients-data",
        )

    if target_column:
        st.subheader("Recorded label distribution")
        counts = (
            data[target_column]
            .fillna("Missing")
            .astype(str)
            .value_counts()
        )

        fig, ax = plt.subplots(figsize=(9, 3.8))
        colors = ["#795538", "#9b7043", "#65745a", "#b48a4d", "#9c6550"]
        ax.bar(counts.index, counts.values, color=colors[:len(counts)])
        ax.set_ylabel("Records")
        ax.set_title("How the target labels are represented")
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)


# ---------------------------------------------------------
# Explore the data
# ---------------------------------------------------------
elif page == "Explore the Data":
    if data is None:
        st.warning("Upload the dataset first.")
        st.stop()

    render_page_title(
        "CHAPTER I · EXPLORATION",
        "Explore the collection",
        "Use the controls to narrow the cohort, inspect distributions, "
        "and compare features across recorded labels.",
    )

    numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [
        column for column in numeric_columns
        if column != target_column
    ]

    labels = []
    chosen_range = None
    range_field = "None"

    with st.expander("Open the filter desk", expanded=True):
        filter_left, filter_right = st.columns(2)

        with filter_left:
            if target_column:
                available_labels = sorted(
                    data[target_column].dropna().astype(str).unique()
                )
                labels = st.multiselect(
                    f"Keep {target_column} labels",
                    available_labels,
                    default=available_labels,
                )

        with filter_right:
            range_choices = ["None"] + numeric_features
            range_field = st.selectbox(
                "Choose a numeric range filter",
                range_choices,
            )

            if (
                range_field != "None"
                and data[range_field].notna().any()
            ):
                low = float(data[range_field].min())
                high = float(data[range_field].max())

                if low < high:
                    chosen_range = st.slider(
                        f"Select the {range_field} interval",
                        min_value=low,
                        max_value=high,
                        value=(low, high),
                    )

    filtered = data.copy()

    if target_column and labels:
        filtered = filtered[
            filtered[target_column].astype(str).isin(labels)
        ]

    if chosen_range is not None:
        filtered = filtered[
            filtered[range_field].between(
                chosen_range[0],
                chosen_range[1],
            )
        ]

    st.caption(
        f"Current selection: **{len(filtered):,}** of "
        f"**{len(data):,}** records"
    )

    distribution_tab, comparison_tab, quality_tab = st.tabs(
        ["Distribution", "Group comparison", "Data quality"]
    )

    with distribution_tab:
        if numeric_features:
            selected_feature = st.selectbox(
                "Choose a field",
                numeric_features,
                key="distribution_feature",
            )

            fig, ax = plt.subplots(figsize=(9, 4.2))
            sns.histplot(
                filtered[selected_feature].dropna(),
                bins="auto",
                kde=True,
                color="#795538",
                ax=ax,
            )
            ax.set_title(f"Distribution of {selected_feature}")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Records")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No numeric fields were detected.")

    with comparison_tab:
        if target_column and numeric_features:
            compare_feature = st.selectbox(
                "Choose a field to compare",
                numeric_features,
                key="comparison_feature",
            )
            comparison = filtered[
                [target_column, compare_feature]
            ].dropna()

            if not comparison.empty:
                order = (
                    comparison.groupby(target_column)[compare_feature]
                    .median()
                    .sort_values()
                    .index
                )

                fig, ax = plt.subplots(figsize=(9, 4.2))
                sns.boxplot(
                    data=comparison,
                    x=target_column,
                    y=compare_feature,
                    order=order,
                    color="#c5a878",
                    ax=ax,
                )
                ax.set_title(
                    f"{compare_feature} across recorded labels"
                )
                ax.tick_params(axis="x", rotation=15)
                style_axis(ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No matching records are available for this view.")
        else:
            st.info("A target field and numeric features are needed.")

    with quality_tab:
        st.subheader("Missing values")
        missing = (
            filtered.isna()
            .mean()
            .mul(100)
            .sort_values(ascending=False)
            .head(20)
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(missing.index, missing.values, color="#9b7043")
        ax.set_ylabel("Missing values (%)")
        ax.tick_params(axis="x", rotation=35)
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        st.subheader("Filtered records")
        st.dataframe(
            filtered,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download this filtered selection",
            filtered.to_csv(index=False).encode("utf-8-sig"),
            file_name="cancer_patient_filtered.csv",
            mime="text/csv",
        )


# ---------------------------------------------------------
# Model studio
# ---------------------------------------------------------
elif page == "Model Studio":
    if data is None:
        st.warning("Upload the dataset first.")
        st.stop()

    render_page_title(
        "CHAPTER II · MODELING",
        "Model studio",
        "Compare baseline classifiers and inspect held-out predictions. "
        "This is a classroom exercise, not clinical validation.",
    )

    if not target_column:
        st.warning(
            "No target was detected. The app looks for Level, Target, "
            "Diagnosis, Label, Class, or Outcome."
        )
        st.stop()

    if data[target_column].nunique(dropna=True) < 2:
        st.warning("The target needs at least two distinct labels.")
        st.stop()

    st.markdown(
        """
        <div class="note">
            Model output reflects patterns in this dataset only. It must
            not be used to diagnose cancer, estimate personal risk, or
            guide treatment.
        </div>
        """,
        unsafe_allow_html=True,
    )

    id_fields = {
        column for column in data.columns
        if normalize_name(column)
        in {"index", "id", "patientid", "unnamed0"}
    }

    candidate_features = [
        column for column in data.columns
        if column != target_column and column not in id_fields
    ]

    with st.container(border=True):
        st.subheader("Set up an experiment")
        selected_features = st.multiselect(
            "Prediction features",
            candidate_features,
            default=candidate_features,
        )

        control_left, control_middle, control_right = st.columns(3)

        with control_left:
            model_name = st.selectbox(
                "Classifier",
                ["Random Forest", "Logistic Regression"],
            )

        with control_middle:
            test_percent = st.slider(
                "Held-out test set (%)",
                min_value=15,
                max_value=40,
                value=25,
                step=5,
            )

        with control_right:
            seed = st.number_input(
                "Reproducibility seed",
                min_value=0,
                max_value=9999,
                value=42,
                step=1,
            )

        run_model = st.button(
            "Run experiment",
            type="primary",
            disabled=not selected_features,
        )

    if run_model:
        model_data = data[
            selected_features + [target_column]
        ].dropna(subset=[target_column])

        X = model_data[selected_features]
        y_text = model_data[target_column].astype(str)

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_text)
        class_counts = pd.Series(y).value_counts()

        stratify_values = (
            y
            if len(class_counts) > 1 and class_counts.min() >= 2
            else None
        )

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_percent / 100,
                random_state=int(seed),
                stratify=stratify_values,
            )

            numeric_fields = X.select_dtypes(
                include=np.number
            ).columns.tolist()
            categorical_fields = [
                column for column in selected_features
                if column not in numeric_fields
            ]

            numeric_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            )

            categorical_pipeline = Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(strategy="most_frequent"),
                    ),
                    (
                        "encoder",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]
            )

            transformers = []

            if numeric_fields:
                transformers.append(
                    ("numeric", numeric_pipeline, numeric_fields)
                )

            if categorical_fields:
                transformers.append(
                    (
                        "categorical",
                        categorical_pipeline,
                        categorical_fields,
                    )
                )

            preprocessing = ColumnTransformer(
                transformers=transformers
            )

            if model_name == "Random Forest":
                classifier = RandomForestClassifier(
                    n_estimators=250,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=int(seed),
                )
            else:
                classifier = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=int(seed),
                )

            model = Pipeline(
                steps=[
                    ("preprocessing", preprocessing),
                    ("classifier", classifier),
                ]
            )

            with st.status(
                "Preparing features and training the model...",
                expanded=False,
            ) as status:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                status.update(
                    label="Experiment complete",
                    state="complete",
                    expanded=False,
                )

            accuracy = accuracy_score(y_test, predictions)
            result_cols = st.columns(3)

            with result_cols[0]:
                show_metric("HELD-OUT ACCURACY", f"{accuracy:.1%}")
            with result_cols[1]:
                show_metric("TRAIN RECORDS", f"{len(X_train):,}")
            with result_cols[2]:
                show_metric("TEST RECORDS", f"{len(X_test):,}")

            matrix_tab, report_tab, importance_tab = st.tabs(
                ["Confusion matrix", "Class report", "Feature importance"]
            )

            all_class_ids = np.arange(len(label_encoder.classes_))

            with matrix_tab:
                matrix = confusion_matrix(
                    y_test,
                    predictions,
                    labels=all_class_ids,
                )

                fig, ax = plt.subplots(figsize=(7, 5))
                ConfusionMatrixDisplay(
                    matrix,
                    display_labels=label_encoder.classes_,
                ).plot(
                    ax=ax,
                    cmap="YlGn",
                    colorbar=False,
                    xticks_rotation=20,
                )
                ax.set_title(f"{model_name} · held-out records")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                if len(label_encoder.classes_) == 2:
                    probabilities = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, probabilities)
                    fpr, tpr, _ = roc_curve(y_test, probabilities)

                    st.metric("Binary ROC AUC", f"{auc:.3f}")

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(
                        fpr,
                        tpr,
                        color="#795538",
                        linewidth=2,
                        label=f"ROC AUC = {auc:.3f}",
                    )
                    ax.plot(
                        [0, 1],
                        [0, 1],
                        "--",
                        color="#a89b84",
                        label="Random baseline",
                    )
                    ax.set_xlabel("False positive rate")
                    ax.set_ylabel("True positive rate")
                    ax.legend()
                    style_axis(ax)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            with report_tab:
                report = classification_report(
                    y_test,
                    predictions,
                    labels=all_class_ids,
                    target_names=[
                        str(label) for label in label_encoder.classes_
                    ],
                    zero_division=0,
                    output_dict=True,
                )
                st.dataframe(
                    pd.DataFrame(report).transpose().round(3),
                    use_container_width=True,
                )

            with importance_tab:
                if model_name == "Random Forest":
                    try:
                        classifier_model = model.named_steps["classifier"]
                        preprocessor = model.named_steps["preprocessing"]
                        names = preprocessor.get_feature_names_out()

                        importances = pd.Series(
                            classifier_model.feature_importances_,
                            index=names,
                        ).sort_values(ascending=False).head(15)

                        fig, ax = plt.subplots(figsize=(9, 5))
                        importances.sort_values().plot(
                            kind="barh",
                            color="#9b7043",
                            ax=ax,
                        )
                        ax.set_xlabel("Random Forest importance")
                        ax.set_ylabel("")
                        style_axis(ax)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception:
                        st.info(
                            "Feature importance is unavailable for this "
                            "feature combination."
                        )
                else:
                    st.info(
                        "Built-in feature importance is shown for the "
                        "Random Forest model. Logistic Regression is "
                        "available as a comparison baseline."
                    )

            st.caption(
                "A single train/test split is a limited performance estimate. "
                "Results can vary with the split, class balance, and selected "
                "features."
            )

        except ValueError as error:
            st.error(
                "The model could not be trained with this setup. Check the "
                "target classes, test-set size, and selected fields. "
                f"Details: {error}"
            )


# ---------------------------------------------------------
# Patient Q&A chat
# ---------------------------------------------------------
elif page == "Patient Q&A":
    render_page_title(
        "CHAPTER III · CONVERSATION",
        "A quiet place for questions",
        "Ask for general health education or help understanding this "
        "project's dataset. The assistant cannot diagnose, assess your "
        "personal risk, or recommend treatment.",
    )

    st.markdown(
        """
        <div class="note">
            Do not enter names, contact details, medical record numbers,
            or other identifying health information. For urgent symptoms,
            contact local emergency services or a qualified clinician.
            AI responses can be inaccurate and are not medical advice.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    with st.expander("What can this guide help with?", expanded=False):
        st.markdown(
            """
            - Explain general cancer-related terms in plain language.
            - Suggest questions a person may want to discuss with a clinician.
            - Explain what the charts and model metrics in this app mean.
            - Summarize the dataset's fields and project workflow.

            It cannot interpret an individual's symptoms, diagnose a disease,
            choose a treatment, or replace a healthcare professional.
            """
        )

    key_from_secrets = None
    try:
        key_from_secrets = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        key_from_secrets = None

    api_key = st.sidebar.text_input(
        "OpenAI API key",
        value="",
        type="password",
        help=(
            "Your key is kept only in this browser session by this app. "
            "API usage may incur charges."
        ),
        placeholder="Paste your API key to enable chat",
    )

    if not api_key:
        api_key = key_from_secrets

    model_name = st.sidebar.text_input(
        "API model",
        value="gpt-4.1-mini",
        help="Enter a model available to your OpenAI API project.",
    )

    st.sidebar.caption(
        "The key is not written into this app's source code. "
        "API usage may incur charges."
    )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_top_left, chat_top_right = st.columns([4, 1])

    with chat_top_left:
        st.subheader("Patient education companion")

    with chat_top_right:
        if st.button("Clear conversation"):
            st.session_state.chat_history = []
            st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"],
            avatar="✦" if message["role"] == "assistant" else "●",
        ):
            st.markdown(message["content"])

    if not api_key:
        st.info(
            "Enter your OpenAI API key in the sidebar to start a conversation."
        )

    user_question = st.chat_input(
        "Ask a general question about cancer or this dataset..."
    )

    if user_question:
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )

        with st.chat_message("user", avatar="●"):
            st.markdown(user_question)

        if not api_key:
            st.warning(
                "Add an API key in the sidebar, then submit your question again."
            )
        else:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=api_key)

                conversation = [
                    {
                        "role": item["role"],
                        "content": item["content"],
                    }
                    for item in st.session_state.chat_history[-12:]
                ]

                dataset_context = ""
                if data is not None:
                    target_summary = ""
                    if target_column:
                        target_summary = (
                            "\nRecorded target field: "
                            f"{target_column}."
                        )

                    dataset_context = (
                        "\n\nApp dataset context: "
                        f"{len(data)} records and {len(data.columns)} fields. "
                        "Field names: "
                        + ", ".join(map(str, data.columns))
                        + target_summary
                    )

                instructions = (
                    "You are a compassionate, careful health-information "
                    "education assistant inside a student data-science app. "
                    "Respond in clear English. You may explain general "
                    "medical concepts and this app's dataset or charts. "
                    "Never diagnose, estimate an individual's cancer risk, "
                    "interpret a person's symptoms as a diagnosis, or "
                    "recommend medication or treatment. Encourage the user "
                    "to discuss personal medical concerns with a licensed "
                    "healthcare professional. For possible emergencies or "
                    "severe symptoms, advise contacting local emergency "
                    "services immediately. Do not ask for identifying or "
                    "sensitive personal health details. State uncertainty "
                    "when appropriate. The dataset is educational and is "
                    "not clinically validated."
                    + dataset_context
                )

                with st.chat_message("assistant", avatar="✦"):
                    with st.spinner("Preparing a careful response..."):
                        response = client.responses.create(
                            model=model_name.strip(),
                            instructions=instructions,
                            input=conversation,
                        )
                        answer = response.output_text.strip()

                    if not answer:
                        answer = (
                            "I could not produce a text response. "
                            "Please try asking in a different way."
                        )

                    st.markdown(answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(
                    "The chat request failed. Check that the API key and "
                    "model name are valid and that your API project has "
                    f"access. Details: {error}"
                )


# ---------------------------------------------------------
# Field notes and limitations
# ---------------------------------------------------------
elif page == "Field Notes":
    render_page_title(
        "APPENDIX · FIELD NOTES",
        "Source, methods, and boundaries",
        "A compact guide to the data and the choices made in this app.",
    )

    source_tab, method_tab, inventory_tab = st.tabs(
        ["Source & questions", "Methods & limits", "Field inventory"]
    )

    with source_tab:
        st.markdown(
            """
            ### Dataset source

            This project uses the workbook supplied for the
            [Cancer Patients Dataset on Kaggle](https://www.kaggle.com/datasets/rishidamarla/cancer-patients-data).

            ### Questions explored

            1. How are the recorded target labels distributed?
            2. How do numeric features vary across recorded label groups?
            3. How do baseline classifiers perform on held-out records?
            """
        )

    with method_tab:
        st.markdown(
            """
            ### Methods

            Pandas and NumPy support data inspection and filtering.
            Matplotlib and Seaborn create the visualizations. The model
            pipeline imputes missing values, scales numeric fields, encodes
            categorical fields, and evaluates Random Forest or Logistic
            Regression classifiers on a reproducible train/test split.

            ### Limitations

            The app does not establish causality, estimate an individual's
            cancer risk, diagnose cancer, or recommend treatment. Dataset
            provenance, sampling, and label quality may limit generalization.
            Model results are a classroom demonstration, not clinical
            validation.
            """
        )

    with inventory_tab:
        if data is None:
            st.info("Upload a dataset to see its field inventory.")
        else:
            inventory = pd.DataFrame(
                {
                    "Field": data.columns,
                    "Data type": [
                        str(data[column].dtype)
                        for column in data.columns
                    ],
                    "Missing values": [
                        int(data[column].isna().sum())
                        for column in data.columns
                    ],
                    "Unique values": [
                        int(data[column].nunique(dropna=True))
                        for column in data.columns
                    ],
                }
            )

            st.dataframe(
                inventory,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download source data as CSV",
                data.to_csv(index=False).encode("utf-8-sig"),
                file_name="cancer_patient_data.csv",
                mime="text/csv",
            )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "CANCERLENS · A student data-science project · "
    "Streamlit, pandas, NumPy, Matplotlib, Seaborn, scikit-learn, OpenAI API"
)