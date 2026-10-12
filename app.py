# Author: Your Name
# CancerLens — educational data exploration and patient Q&A application.
# This app does not provide diagnosis, risk assessment, or treatment advice.

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


# ------------------------------------------------------------
# App setup and visual system
# ------------------------------------------------------------
st.set_page_config(
    page_title="CancerLens | Patient Data Atelier",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@400;500;600;700&family=IM+Fell+English:ital@0;1&family=Manrope:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --paper: #eee5d4;
        --paper-light: #faf5eb;
        --ink: #28231d;
        --walnut: #503725;
        --wood: #765337;
        --brass: #bd965b;
        --moss: #69765b;
        --line: rgba(75, 53, 34, .18);
    }

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
        color: var(--ink);
    }

    .stApp {
        background:
            radial-gradient(ellipse at 8% 4%, rgba(180, 140, 79, .17), transparent 32%),
            radial-gradient(ellipse at 93% 88%, rgba(88, 102, 72, .12), transparent 32%),
            repeating-linear-gradient(0deg,
                rgba(79, 54, 34, .022) 0px,
                rgba(79, 54, 34, .022) 1px,
                transparent 1px,
                transparent 5px),
            var(--paper);
    }

    [data-testid="stSidebar"] {
        background:
            repeating-linear-gradient(90deg,
                rgba(255,255,255,.025) 0px,
                rgba(255,255,255,.025) 2px,
                transparent 2px,
                transparent 10px),
            linear-gradient(155deg, #34251c, #60432c 58%, #3b3025);
        border-right: 1px solid rgba(255,255,255,.12);
    }

    [data-testid="stSidebar"] * { color: #f6ead5; }
    [data-testid="stSidebar"] input { color: #2d261d !important; }

    .block-container {
        max-width: 1480px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Cormorant Garamond', Georgia, serif !important;
        color: var(--walnut);
        letter-spacing: -.45px;
    }

    .masthead {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 3px 2px 15px;
        border-bottom: 1px solid var(--line);
        margin-bottom: 18px;
    }

    .masthead-name {
        font-family: 'Cormorant Garamond', Georgia, serif;
        color: var(--walnut);
        font-size: 24px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .masthead-note {
        color: #806d54;
        font: 10px 'DM Mono', monospace;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 38px 42px;
        margin: 10px 0 20px;
        border: 1px solid rgba(255,255,255,.16);
        border-radius: 22px;
        color: #fff8ec;
        background:
            repeating-linear-gradient(2deg,
                rgba(255,255,255,.025) 0px,
                rgba(255,255,255,.025) 2px,
                transparent 2px,
                transparent 10px),
            linear-gradient(125deg, #30241c 0%, #68482f 57%, #354033 100%);
        box-shadow: 0 16px 38px rgba(53, 39, 25, .19);
    }

    .hero:after {
        content: "✦";
        position: absolute;
        right: 38px;
        top: 4px;
        font: 130px 'Cormorant Garamond', Georgia, serif;
        color: rgba(236, 211, 164, .14);
        transform: rotate(12deg);
    }

    .hero-kicker {
        color: #ddc495;
        font: 500 10px 'DM Mono', monospace;
        letter-spacing: 2.3px;
        text-transform: uppercase;
    }

    .hero-title {
        color: #fff8ec;
        font: 600 clamp(42px, 5vw, 64px)/.98 'Cormorant Garamond', Georgia, serif;
        letter-spacing: -1px;
        margin: 12px 0;
    }

    .hero-copy {
        color: #e7dac6;
        max-width: 760px;
        font-size: 14px;
        line-height: 1.8;
    }

    .metric-tile {
        background: rgba(255, 251, 242, .83);
        border: 1px solid var(--line);
        border-radius: 15px;
        padding: 16px 18px;
        min-height: 96px;
        box-shadow: 0 5px 15px rgba(74, 54, 34, .06);
    }

    .metric-label {
        color: #766653;
        font: 10px 'DM Mono', monospace;
        letter-spacing: 1.1px;
        text-transform: uppercase;
    }

    .metric-value {
        color: var(--walnut);
        font: 700 31px 'Cormorant Garamond', Georgia, serif;
        margin-top: 5px;
    }

    .paper-panel {
        background: rgba(255, 251, 242, .75);
        border: 1px solid var(--line);
        border-radius: 16px;
        padding: 18px 20px;
        box-shadow: 0 7px 20px rgba(74, 54, 34, .045);
    }

    .note {
        background: rgba(255, 250, 238, .84);
        border-left: 3px solid var(--brass);
        border-radius: 0 10px 10px 0;
        padding: 13px 16px;
        color: #514333;
        font-size: 13px;
    }

    .stButton > button {
        border: 1px solid #65462f;
        border-radius: 10px;
        color: #fff8eb;
        background: linear-gradient(130deg, #795538, #493326);
        font-weight: 700;
    }

    .stButton > button:hover {
        color: #fff8eb;
        border-color: #bd965b;
        box-shadow: 0 5px 15px rgba(80, 54, 36, .17);
    }

    [data-testid="stChatMessage"] {
        background: rgba(255, 251, 242, .76);
        border: 1px solid rgba(77, 55, 35, .14);
        border-radius: 14px;
        margin-bottom: 10px;
    }

    [data-testid="stExpander"] {
        background: rgba(255, 251, 242, .55);
        border: 1px solid var(--line);
        border-radius: 12px;
    }

    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------
# Data and app helpers
# ------------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_excel_bytes(file_bytes: bytes, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(BytesIO(file_bytes), sheet_name=sheet_name)


@st.cache_data(show_spinner=False)
def read_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def get_excel_sheets(file_bytes: bytes) -> list[str]:
    workbook = pd.ExcelFile(BytesIO(file_bytes))
    return workbook.sheet_names


def normalize_name(value: str) -> str:
    return "".join(
        character.lower()
        for character in str(value)
        if character.isalnum()
    )


def detect_target(frame: pd.DataFrame):
    accepted = {
        "level",
        "cancerlevel",
        "target",
        "diagnosis",
        "label",
        "class",
        "outcome",
    }
    for column in frame.columns:
        if normalize_name(column) in accepted:
            return column
    return None


def metric_tile(label: str, value: str):
    st.markdown(
        f"""
        <div class="metric-tile">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def style_axis(ax):
    ax.set_facecolor("#fbf7ee")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cabb9f")
    ax.spines["bottom"].set_color("#cabb9f")
    ax.tick_params(colors="#554635")
    ax.grid(axis="y", alpha=.17, color="#806647")


def local_dataset():
    candidates = [
        Path("data/cancer patient data sets.xlsx"),
        Path("data/cancer_patient_data.xlsx"),
        Path("cancer patient data sets.xlsx"),
        Path("data/cancer_patient_data.csv"),
    ]

    for path in candidates:
        if path.exists():
            try:
                if path.suffix.lower() == ".csv":
                    return pd.read_csv(path)
                return pd.read_excel(path)
            except Exception:
                continue

    return None


# ------------------------------------------------------------
# Sidebar navigation
# ------------------------------------------------------------
st.sidebar.markdown(
    """
    <div style="font:700 30px 'Cormorant Garamond',serif;
                letter-spacing:2px;color:#f6ead5;">CANCERLENS</div>
    <div style="font:italic 14px 'IM Fell English',serif;
                color:#ddc69f;">An atlas of patterns and questions</div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.write("")

page = st.sidebar.radio(
    "CHOOSE A CHAPTER",
    [
        "Welcome",
        "Explore the Data",
        "Model Studio",
        "Patient Q&A",
        "Field Notes",
    ],
)

# API key field is shown only when the chat page is selected.
api_key = ""
api_model = "gpt-4.1-mini"

if page == "Patient Q&A":
    st.sidebar.divider()
    st.sidebar.markdown("### Conversation settings")
    api_key = st.sidebar.text_input(
        "OpenAI API key",
        type="password",
        placeholder="Paste your API key",
        help="The key is used for this session and is not written into app.py.",
    )
    api_model = st.sidebar.text_input(
        "API model",
        value="gpt-4.1-mini",
        help="Enter a model enabled for your OpenAI API project.",
    )
    st.sidebar.caption("API usage may incur charges.")

# ------------------------------------------------------------
# Prominent upload desk
# ------------------------------------------------------------
st.markdown(
    """
    <div class="masthead">
        <div class="masthead-name">CANCERLENS</div>
        <div class="masthead-note">Patient data · Exploratory atlas · 01</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    upload_left, upload_right = st.columns([3, 1])

    with upload_left:
        st.markdown("### Dataset desk")
        st.caption(
            "Upload an Excel workbook or CSV. Excel workbooks with multiple "
            "sheets will show a sheet selector."
        )

    with upload_right:
        st.caption("Accepted files")
        st.markdown("`XLSX` · `XLS` · `CSV`")

    uploaded_file = st.file_uploader(
        "Choose your patient dataset",
        type=["xlsx", "xls", "csv"],
        key="patient_dataset_upload",
        help="After selection, the app will confirm the filename and load the data.",
        label_visibility="visible",
    )

data = None
data_source = "No file loaded"
upload_error = None

if uploaded_file is not None:
    try:
        file_bytes = uploaded_file.getvalue()
        file_name = uploaded_file.name.lower()

        if file_name.endswith(".csv"):
            data = read_csv_bytes(file_bytes)
            data_source = uploaded_file.name
        else:
            sheets = get_excel_sheets(file_bytes)

            if not sheets:
                raise ValueError("This workbook contains no readable sheets.")

            selected_sheet = st.selectbox(
                "Choose a worksheet",
                options=sheets,
                key="excel_sheet_choice",
            )
            data = read_excel_bytes(file_bytes, selected_sheet)
            data_source = f"{uploaded_file.name} · {selected_sheet}"

        if data is not None and data.empty:
            raise ValueError("The selected file or worksheet contains no rows.")

    except Exception as error:
        data = None
        upload_error = str(error)

elif "dataset_override" in st.session_state:
    data = st.session_state.dataset_override
    data_source = st.session_state.get("dataset_source", "Loaded dataset")

else:
    data = local_dataset()
    if data is not None:
        data_source = "Bundled project file"

if upload_error:
    st.error(
        "The file could not be loaded. For Excel, confirm that the workbook "
        "is not password-protected and that `openpyxl` is installed. "
        f"Details: {upload_error}"
    )
elif data is not None:
    data.columns = [str(column).strip() for column in data.columns]
    target_column = detect_target(data)

    st.success(
        f"Dataset ready: **{data_source}** · "
        f"**{len(data):,} rows** · **{len(data.columns)} fields**"
    )
    with st.expander("Preview loaded data", expanded=False):
        st.dataframe(data.head(8), use_container_width=True, hide_index=True)
else:
    target_column = None
    st.info(
        "No dataset is loaded yet. Choose an Excel or CSV file above. "
        "The page navigation remains available while you upload."
    )


# ------------------------------------------------------------
# Welcome dashboard
# ------------------------------------------------------------
if page == "Welcome":
    st.markdown(
        """
        <div class="hero">
            <div class="hero-kicker">A DATA ATLAS · CANCER PATIENT DATA</div>
            <div class="hero-title">Patterns in the record.<br>Questions for the future.</div>
            <div class="hero-copy">
                A visual studio for cohort exploration, feature comparison,
                baseline classification, and general health education.
                Built for learning from data — never for individual diagnosis.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if data is None:
        st.markdown(
            """
            <div class="note">
                Start at the Dataset desk above. Once the file loads, the
                overview, charts, model studio, and field inventory will
                use the uploaded data.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    missing_total = int(data.isna().sum().sum())
    target_count = (
        data[target_column].nunique(dropna=True)
        if target_column
        else 0
    )

    cols = st.columns(5)
    for col, label, value in [
        (cols[0], "RECORDS", f"{len(data):,}"),
        (cols[1], "FIELDS", f"{len(data.columns)}"),
        (cols[2], "MISSING CELLS", f"{missing_total:,}"),
        (cols[3], "TARGET", str(target_column or "Not found")),
        (cols[4], "LABEL GROUPS", f"{target_count}" if target_count else "—"),
    ]:
        with col:
            metric_tile(label, value)

    st.write("")

    summary_tab, labels_tab, preview_tab = st.tabs(
        ["Collection profile", "Label landscape", "Record preview"]
    )

    with summary_tab:
        st.subheader("A quick reading of the collection")
        left, right = st.columns([1.4, 1])

        with left:
            st.markdown(
                """
                <div class="paper-panel">
                    <b>What this workspace does</b><br><br>
                    It summarizes the supplied workbook, lets you filter
                    records, compares numeric features across recorded
                    labels, and evaluates baseline classifiers on held-out
                    records.
                    <br><br>
                    <b>How to explore</b><br><br>
                    Start with the label landscape, use the feature filters,
                    then open Model Studio to inspect confusion patterns
                    and per-class metrics.
                </div>
                """,
                unsafe_allow_html=True,
            )

        with right:
            st.metric("Current data source", data_source)
            st.progress(
                min(missing_total / max(data.size, 1), 1.0),
                text="Share of cells that are missing",
            )

        st.write("")
        st.markdown(
            """
            <div class="note">
                Patterns in an observational dataset do not establish
                causation. This app does not estimate individual risk,
                diagnose cancer, or recommend treatment.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with labels_tab:
        if target_column:
            counts = data[target_column].fillna("Missing").astype(str).value_counts()
            fig, ax = plt.subplots(figsize=(9, 4))
            palette = ["#65452e", "#967044", "#69765b", "#bd965b", "#9b6650"]
            ax.bar(counts.index, counts.values, color=palette[:len(counts)])
            ax.set_title("Recorded target-label distribution")
            ax.set_ylabel("Records")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.warning(
                "No common target field was detected. The app searches for "
                "Level, Target, Diagnosis, Label, Class, or Outcome."
            )

    with preview_tab:
        st.dataframe(data.head(25), use_container_width=True, hide_index=True)
        st.download_button(
            "Download loaded data as CSV",
            data.to_csv(index=False).encode("utf-8-sig"),
            file_name="cancer_patient_data.csv",
            mime="text/csv",
        )


# ------------------------------------------------------------
# Interactive EDA
# ------------------------------------------------------------
elif page == "Explore the Data":
    if data is None:
        st.warning("Upload a dataset at the Dataset desk above.")
        st.stop()

    st.markdown("## Explore the collection")
    st.caption(
        "Use the filters to shape a cohort, then compare its distributions."
    )

    numeric_cols = data.select_dtypes(include=np.number).columns.tolist()
    numeric_features = [
        column for column in numeric_cols if column != target_column
    ]

    with st.popover("Open the filter controls", use_container_width=True):
        selected_labels = []
        filter_field = "None"
        numeric_range = None

        if target_column:
            available = sorted(data[target_column].dropna().astype(str).unique())
            selected_labels = st.multiselect(
                f"Include {target_column} labels",
                available,
                default=available,
            )

        if numeric_features:
            filter_field = st.selectbox(
                "Numeric filter field",
                ["None"] + numeric_features,
            )

            if filter_field != "None":
                low = float(data[filter_field].min())
                high = float(data[filter_field].max())
                if low < high:
                    numeric_range = st.slider(
                        f"{filter_field} interval",
                        min_value=low,
                        max_value=high,
                        value=(low, high),
                    )

    filtered = data.copy()

    if target_column and selected_labels:
        filtered = filtered[
            filtered[target_column].astype(str).isin(selected_labels)
        ]

    if numeric_range is not None:
        filtered = filtered[
            filtered[filter_field].between(*numeric_range)
        ]

    st.metric("Records in current cohort", f"{len(filtered):,}")

    chart_tab, compare_tab, quality_tab = st.tabs(
        ["Distribution", "Compare groups", "Missingness"]
    )

    with chart_tab:
        if numeric_features:
            field = st.selectbox("Choose a numeric field", numeric_features)
            fig, ax = plt.subplots(figsize=(9, 4.2))
            sns.histplot(
                filtered[field].dropna(),
                bins="auto",
                kde=True,
                color="#795538",
                ax=ax,
            )
            ax.set_title(f"Distribution of {field}")
            ax.set_ylabel("Records")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    with compare_tab:
        if target_column and numeric_features:
            field = st.selectbox(
                "Feature to compare",
                numeric_features,
                key="boxplot_field",
            )
            plot_data = filtered[[target_column, field]].dropna()

            if not plot_data.empty:
                fig, ax = plt.subplots(figsize=(9, 4.2))
                sns.violinplot(
                    data=plot_data,
                    x=target_column,
                    y=field,
                    inner="quartile",
                    color="#bd965b",
                    cut=0,
                    ax=ax,
                )
                ax.set_title(f"{field} by recorded label")
                ax.tick_params(axis="x", rotation=15)
                style_axis(ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No records match the selected filters.")

    with quality_tab:
        missing = filtered.isna().mean().mul(100).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(missing.index, missing.values, color="#967044")
        ax.set_ylabel("Missing (%)")
        ax.tick_params(axis="x", rotation=35)
        style_axis(ax)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Inspect filtered records"):
            st.dataframe(filtered, use_container_width=True, hide_index=True)


# ------------------------------------------------------------
# Machine-learning studio
# ------------------------------------------------------------
elif page == "Model Studio":
    if data is None:
        st.warning("Upload a dataset at the Dataset desk above.")
        st.stop()

    st.markdown("## Model Studio")
    st.caption(
        "Train a baseline classifier and examine its held-out predictions."
    )

    if not target_column:
        st.warning("A target field such as `Level` or `Diagnosis` was not found.")
        st.stop()

    id_fields = {
        column for column in data.columns
        if normalize_name(column) in {"id", "patientid", "index", "unnamed0"}
    }
    features = [
        column for column in data.columns
        if column != target_column and column not in id_fields
    ]

    with st.container(border=True):
        st.subheader("Experiment controls")
        chosen_features = st.multiselect(
            "Prediction fields",
            features,
            default=features,
        )
        c1, c2 = st.columns(2)

        with c1:
            classifier_name = st.selectbox(
                "Classifier",
                ["Random Forest", "Logistic Regression"],
            )

        with c2:
            test_size = st.slider(
                "Test-set percentage",
                min_value=15,
                max_value=40,
                value=25,
                step=5,
            )

        run = st.button(
            "Run the experiment",
            type="primary",
            disabled=not chosen_features,
        )

    if run:
        model_data = data[chosen_features + [target_column]].dropna(
            subset=[target_column]
        )
        X = model_data[chosen_features]
        encoder = LabelEncoder()
        y = encoder.fit_transform(model_data[target_column].astype(str))

        counts = pd.Series(y).value_counts()
        stratify = y if len(counts) > 1 and counts.min() >= 2 else None

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=test_size / 100,
                random_state=42,
                stratify=stratify,
            )

            numeric = X.select_dtypes(include=np.number).columns.tolist()
            categorical = [field for field in chosen_features if field not in numeric]

            transforms = []
            if numeric:
                transforms.append(
                    (
                        "numeric",
                        Pipeline(
                            [
                                ("imputer", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        numeric,
                    )
                )

            if categorical:
                transforms.append(
                    (
                        "categorical",
                        Pipeline(
                            [
                                (
                                    "imputer",
                                    SimpleImputer(strategy="most_frequent"),
                                ),
                                (
                                    "encode",
                                    OneHotEncoder(handle_unknown="ignore"),
                                ),
                            ]
                        ),
                        categorical,
                    )
                )

            prep = ColumnTransformer(transforms)

            if classifier_name == "Random Forest":
                classifier = RandomForestClassifier(
                    n_estimators=250,
                    class_weight="balanced",
                    min_samples_leaf=2,
                    random_state=42,
                )
            else:
                classifier = LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                )

            model = Pipeline(
                [
                    ("prep", prep),
                    ("classifier", classifier),
                ]
            )

            with st.status("Training and evaluating...", expanded=False) as status:
                model.fit(X_train, y_train)
                predictions = model.predict(X_test)
                status.update(label="Experiment complete", state="complete")

            score = accuracy_score(y_test, predictions)
            m1, m2, m3 = st.columns(3)

            with m1:
                st.metric("Held-out accuracy", f"{score:.1%}")
            with m2:
                st.metric("Training records", f"{len(X_train):,}")
            with m3:
                st.metric("Test records", f"{len(X_test):,}")

            cm_tab, report_tab, detail_tab = st.tabs(
                ["Confusion matrix", "Class report", "Model detail"]
            )

            labels = np.arange(len(encoder.classes_))

            with cm_tab:
                matrix = confusion_matrix(y_test, predictions, labels=labels)
                fig, ax = plt.subplots(figsize=(7, 5))
                ConfusionMatrixDisplay(
                    matrix,
                    display_labels=encoder.classes_,
                ).plot(
                    ax=ax,
                    cmap="YlGn",
                    colorbar=False,
                    xticks_rotation=15,
                )
                ax.set_title(f"{classifier_name} · held-out records")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                if len(encoder.classes_) == 2:
                    probs = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, probs)
                    fpr, tpr, _ = roc_curve(y_test, probs)

                    st.metric("Binary ROC AUC", f"{auc:.3f}")
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(fpr, tpr, color="#795538", label=f"AUC = {auc:.3f}")
                    ax.plot([0, 1], [0, 1], "--", color="#a99a80")
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
                    labels=labels,
                    target_names=[str(x) for x in encoder.classes_],
                    zero_division=0,
                    output_dict=True,
                )
                st.dataframe(
                    pd.DataFrame(report).transpose().round(3),
                    use_container_width=True,
                )

            with detail_tab:
                if classifier_name == "Random Forest":
                    try:
                        names = model.named_steps["prep"].get_feature_names_out()
                        values = model.named_steps["classifier"].feature_importances_
                        importance = pd.Series(values, index=names).nlargest(15)

                        fig, ax = plt.subplots(figsize=(9, 5))
                        importance.sort_values().plot(
                            kind="barh",
                            color="#967044",
                            ax=ax,
                        )
                        ax.set_xlabel("Feature importance")
                        style_axis(ax)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception:
                        st.info("Feature importance is unavailable for this dataset.")
                else:
                    st.write(
                        "Logistic Regression is included as a comparison "
                        "baseline. The Random Forest option displays "
                        "built-in feature importance."
                    )

            st.caption(
                "A single train/test split is only a limited estimate. "
                "This model is for coursework, not clinical use."
            )

        except ValueError as error:
            st.error(
                "The model could not run with this split. Check the class "
                f"counts and selected fields. Details: {error}"
            )


# ------------------------------------------------------------
# Patient Q&A
# ------------------------------------------------------------
elif page == "Patient Q&A":
    st.markdown("## Patient Q&A")
    st.caption(
        "General health education and help understanding this app's charts."
    )

    st.markdown(
        """
        <div class="note">
            Do not enter names, medical record numbers, contact details,
            or identifiable health information. This assistant cannot diagnose,
            assess personal risk, or recommend treatment. For urgent symptoms,
            contact local emergency services or a qualified clinician.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Topics this assistant can discuss"):
        st.markdown(
            """
            - Explain general medical terms in plain English.
            - Suggest questions to discuss with a licensed clinician.
            - Explain charts, model metrics, and dataset fields.
            - Summarize the analysis workflow and its limitations.
            """
        )

    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""

    active_key = api_key or secret_key

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    chat_left, chat_right = st.columns([4, 1])

    with chat_left:
        st.markdown("### The reading room")

    with chat_right:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()

    for message in st.session_state.chat_history:
        with st.chat_message(
            message["role"],
            avatar="✦" if message["role"] == "assistant" else "●",
        ):
            st.markdown(message["content"])

    if not active_key:
        st.info("Add your OpenAI API key in the sidebar to enable chat.")

    question = st.chat_input(
        "Ask a general question about cancer or this app..."
    )

    if question:
        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user", avatar="●"):
            st.markdown(question)

        if not active_key:
            st.warning("Enter an API key in the sidebar, then ask again.")
        else:
            try:
                from openai import OpenAI

                client = OpenAI(api_key=active_key)
                recent_messages = [
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                    for message in st.session_state.chat_history[-12:]
                ]

                context = ""
                if data is not None:
                    context = (
                        f"\nDataset has {len(data)} rows and "
                        f"{len(data.columns)} fields. "
                        "Field names: "
                        + ", ".join(map(str, data.columns))
                    )
                    if target_column:
                        context += f"\nDetected target: {target_column}."

                instructions = (
                    "You are a careful, compassionate health-information "
                    "education assistant inside a student data-science app. "
                    "Always respond in clear English. You may explain general "
                    "health concepts and this app's dataset, charts, and model "
                    "metrics. Never diagnose, estimate an individual's cancer "
                    "risk, interpret personal symptoms as a diagnosis, or "
                    "recommend medication or treatment. Encourage users to "
                    "discuss personal concerns with a licensed clinician. For "
                    "possible emergencies or severe symptoms, advise contacting "
                    "local emergency services immediately. Do not request "
                    "identifying or sensitive personal health details. State "
                    "uncertainty when appropriate. The dataset is educational "
                    "and not clinically validated."
                    + context
                )

                with st.chat_message("assistant", avatar="✦"):
                    with st.spinner("Composing a careful response..."):
                        response = client.responses.create(
                            model=api_model.strip(),
                            instructions=instructions,
                            input=recent_messages,
                        )
                        answer = response.output_text.strip()

                    if not answer:
                        answer = "I could not prepare a text response. Please try again."

                    st.markdown(answer)

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as error:
                st.error(
                    "The request failed. Check the API key, model name, and "
                    f"API project access. Details: {error}"
                )


# ------------------------------------------------------------
# Field notes
# ------------------------------------------------------------
elif page == "Field Notes":
    st.markdown("## Field Notes")

    source_tab, methods_tab, inventory_tab = st.tabs(
        ["Source & questions", "Methods & limits", "Field inventory"]
    )

    with source_tab:
        st.markdown(
            """
            ### Dataset source

            This app uses the supplied workbook associated with the
            [Cancer Patients Dataset on Kaggle](https://www.kaggle.com/datasets/rishidamarla/cancer-patients-data).

            ### Questions

            1. How are the recorded labels distributed?
            2. How do numeric features vary across label groups?
            3. How do baseline classifiers perform on held-out records?
            """
        )

    with methods_tab:
        st.markdown(
            """
            Pandas and NumPy support inspection and filtering. Matplotlib
            and Seaborn create charts. The scikit-learn pipeline imputes
            missing values, scales numeric features, encodes categorical
            features, and evaluates Logistic Regression or Random Forest.

            Dataset sampling, provenance, and label quality may limit
            generalization. Results are educational examples, not clinical
            validation or medical advice.
            """
        )

    with inventory_tab:
        if data is None:
            st.info("Upload a dataset to see its field inventory.")
        else:
            inventory = pd.DataFrame(
                {
                    "Field": data.columns,
                    "Type": [str(data[column].dtype) for column in data.columns],
                    "Missing": [
                        int(data[column].isna().sum())
                        for column in data.columns
                    ],
                    "Unique": [
                        int(data[column].nunique(dropna=True))
                        for column in data.columns
                    ],
                }
            )
            st.dataframe(inventory, use_container_width=True, hide_index=True)

st.divider()
st.caption(
    "CANCERLENS · A student data-science project · "
    "Streamlit · pandas · NumPy · Matplotlib · Seaborn · scikit-learn · OpenAI API"
)