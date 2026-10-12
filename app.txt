# Author: Your Name
# CancerLens: an educational cancer-patient data explorer.
# This app demonstrates exploratory data analysis and classification.
# It is not a diagnostic or clinical decision-support tool.

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
# Streamlit page setup
# ---------------------------------------------------------
st.set_page_config(
    page_title="CancerLens | Patient Data Explorer",
    page_icon="✳",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# App styling
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Manrope', sans-serif;
    }

    .stApp {
        background: #f4f6f8;
        color: #17242a;
    }

    [data-testid="stSidebar"] {
        background: #122d32;
    }

    [data-testid="stSidebar"] * {
        color: #eff8f5;
    }

    .hero {
        background: linear-gradient(120deg, #122d32, #205b59);
        border-radius: 24px;
        padding: 36px 40px;
        color: #f4fbf7;
        margin-bottom: 22px;
    }

    .eyebrow {
        font: 500 11px 'DM Mono', monospace;
        letter-spacing: 2px;
        color: #9dd6bd;
        text-transform: uppercase;
    }

    .hero h1 {
        font-size: 42px;
        letter-spacing: -1.8px;
        margin: 10px 0 8px;
    }

    .hero p {
        color: #ccddd8;
        max-width: 780px;
        font-size: 15px;
        line-height: 1.7;
    }

    .metric-card {
        background: white;
        border: 1px solid #e6ebea;
        border-radius: 16px;
        padding: 18px 20px;
        min-height: 100px;
    }

    .metric-label {
        font: 11px 'DM Mono', monospace;
        color: #718083;
        letter-spacing: 1px;
        text-transform: uppercase;
    }

    .metric-value {
        font-size: 27px;
        font-weight: 800;
        margin-top: 8px;
        color: #193e3c;
    }

    .note {
        border-left: 3px solid #3b9b78;
        background: #eaf4ee;
        padding: 13px 16px;
        border-radius: 0 10px 10px 0;
        color: #385551;
        font-size: 13px;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Reusable functions
# ---------------------------------------------------------
@st.cache_data(show_spinner=False)
def read_uploaded_data(file_name: str, file_bytes: bytes) -> pd.DataFrame:
    """Load a CSV or Excel workbook from uploaded bytes."""
    file_buffer = BytesIO(file_bytes)

    if file_name.lower().endswith(".csv"):
        return pd.read_csv(file_buffer)

    return pd.read_excel(file_buffer)


def normalize_name(value: str) -> str:
    """Lowercase a column name and remove spaces/punctuation."""
    return "".join(character.lower() for character in str(value)
                   if character.isalnum())


def find_target_column(data: pd.DataFrame):
    """Find common target column names."""
    target_names = {
        "level",
        "cancerlevel",
        "target",
        "diagnosis",
        "label",
        "class",
        "outcome",
    }

    for column in data.columns:
        if normalize_name(column) in target_names:
            return column

    return None


def show_metric(label: str, value: str):
    """Render a styled metric card."""
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
    """Apply consistent chart styling."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#dce4e1")
    ax.spines["bottom"].set_color("#dce4e1")
    ax.tick_params(colors="#526564")
    ax.grid(axis="y", alpha=0.15)


def find_local_dataset():
    """Search common locations for an optional bundled dataset."""
    possible_paths = [
        Path("data/cancer patient data sets.xlsx"),
        Path("data/cancer_patient_data.xlsx"),
        Path("cancer patient data sets.xlsx"),
        Path("data/cancer_patient_data.csv"),
    ]

    for file_path in possible_paths:
        if file_path.exists():
            try:
                if file_path.suffix.lower() == ".csv":
                    return pd.read_csv(file_path)

                return pd.read_excel(file_path)
            except Exception:
                continue

    return None


# ---------------------------------------------------------
# Sidebar navigation and data upload
# ---------------------------------------------------------
st.sidebar.markdown("## CANCERLENS")
st.sidebar.caption("Population-level data explorer")

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Explore patterns",
        "Model lab",
        "Data guide",
    ],
    label_visibility="collapsed",
)

uploaded_file = st.sidebar.file_uploader(
    "Upload the cancer-patient dataset",
    type=["xlsx", "xls", "csv"],
    help="Upload your Excel workbook or a CSV version of the dataset.",
)

data = None

if uploaded_file is not None:
    try:
        data = read_uploaded_data(
            uploaded_file.name,
            uploaded_file.getvalue(),
        )
    except Exception as error:
        st.sidebar.error(f"Could not read the uploaded file: {error}")

if data is None:
    data = find_local_dataset()

if data is None:
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">DATA SCIENCE · CANCER PATIENT DATA</div>
            <h1>Turn patterns into<br>better questions.</h1>
            <p>
                Explore a cancer-patient dataset, compare feature distributions,
                and evaluate baseline machine-learning models.
                Upload your Excel or CSV file in the sidebar to begin.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="note">
            Dataset source:
            <a href="https://www.kaggle.com/datasets/rishidamarla/cancer-patients-data">
            Kaggle — Cancer Patients Dataset</a>.
            Download the dataset and upload it using the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.stop()


# Standardize column names and detect target.
data.columns = [str(column).strip() for column in data.columns]
target_column = find_target_column(data)

st.sidebar.divider()
st.sidebar.caption(f"{len(data):,} records · {data.shape[1]} columns")


# Numeric columns for EDA, excluding the target if it is numeric-coded.
numeric_columns = data.select_dtypes(include=np.number).columns.tolist()
numeric_features = [
    column for column in numeric_columns if column != target_column
]


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if page == "Overview":
    st.markdown(
        """
        <div class="hero">
            <div class="eyebrow">
                CANCER PATIENT DATA · EXPLORATORY ANALYSIS
            </div>
            <h1>See the cohort.<br>Question the patterns.</h1>
            <p>
                Explore how recorded features vary across severity labels.
                This application is an educational data-analysis tool and
                should not be used for personal diagnosis or clinical decisions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    missing_count = int(data.isna().sum().sum())
    feature_count = data.shape[1] - (1 if target_column else 0)

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        show_metric("PATIENT RECORDS", f"{len(data):,}")

    with metric2:
        show_metric("FEATURES", f"{feature_count}")

    with metric3:
        show_metric("MISSING CELLS", f"{missing_count:,}")

    with metric4:
        show_metric(
            "TARGET COLUMN",
            str(target_column or "Not detected"),
        )

    st.write("")

    left, right = st.columns([1.2, 1])

    with left:
        st.subheader("About this dataset")
        st.write(
            "Each row represents one record in the supplied dataset. "
            "Columns may describe demographic, environmental, lifestyle, "
            "or symptom-related factors. If a severity label is present, "
            "the app can compare groups and demonstrate classification."
        )

        st.markdown(
            """
            <div class="note">
                Observed associations do not establish causation. The dataset
                may not represent a clinical population and should not be
                used to estimate an individual's cancer risk.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Label distribution")

        if target_column:
            label_counts = (
                data[target_column]
                .fillna("Missing")
                .astype(str)
                .value_counts()
            )

            fig, ax = plt.subplots(figsize=(6, 3.5))
            palette = [
                "#2d7768",
                "#55a88a",
                "#91c7a9",
                "#d8ac68",
                "#bc6c55",
            ]

            ax.bar(
                label_counts.index,
                label_counts.values,
                color=palette[:len(label_counts)],
            )
            ax.set_ylabel("Records")
            ax.tick_params(axis="x", rotation=15)
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info(
                "No common target column was detected. "
                "See Data guide to inspect the available fields."
            )

    st.subheader("Dataset preview")
    st.dataframe(
        data.head(15),
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# PAGE 2: EXPLORE PATTERNS
# =========================================================
elif page == "Explore patterns":
    st.title("Explore patterns")
    st.caption(
        "Filter the records, compare feature distributions, "
        "and inspect missingness."
    )

    categorical_columns = [
        column
        for column in data.columns
        if column not in numeric_columns
    ]

    selected_labels = []
    numeric_filter = "None"
    bounds = None

    with st.expander("Cohort filters", expanded=True):
        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:
            if target_column:
                all_labels = sorted(
                    data[target_column].dropna().astype(str).unique()
                )
                selected_labels = st.multiselect(
                    f"Include {target_column}",
                    options=all_labels,
                    default=all_labels,
                )

        with filter_col2:
            filter_options = ["None"] + numeric_features
            numeric_filter = st.selectbox(
                "Filter by a numeric field",
                filter_options,
            )

            if (
                numeric_filter != "None"
                and data[numeric_filter].notna().any()
            ):
                lower = float(data[numeric_filter].min())
                upper = float(data[numeric_filter].max())

                if lower < upper:
                    bounds = st.slider(
                        f"{numeric_filter} range",
                        min_value=lower,
                        max_value=upper,
                        value=(lower, upper),
                    )
                else:
                    st.caption(
                        f"{numeric_filter} has only one observed value."
                    )

    filtered_data = data.copy()

    if target_column and selected_labels:
        filtered_data = filtered_data[
            filtered_data[target_column].astype(str).isin(selected_labels)
        ]

    if bounds is not None:
        filtered_data = filtered_data[
            filtered_data[numeric_filter].between(bounds[0], bounds[1])
        ]

    st.caption(
        f"Showing **{len(filtered_data):,}** of {len(data):,} records"
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Numeric feature distribution")

        if numeric_features:
            feature = st.selectbox(
                "Choose a numeric feature",
                numeric_features,
            )

            fig, ax = plt.subplots(figsize=(7, 4))
            sns.histplot(
                filtered_data[feature].dropna(),
                bins="auto",
                kde=True,
                color="#32836e",
                ax=ax,
            )
            ax.set_xlabel(feature)
            ax.set_ylabel("Records")
            style_axis(ax)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
        else:
            st.info("No numeric feature columns were found.")

    with right:
        st.subheader("Compare feature by label")

        if target_column and numeric_features:
            compare_feature = st.selectbox(
                "Choose a feature to compare",
                numeric_features,
                key="compare_feature",
            )

            plot_data = filtered_data[
                [target_column, compare_feature]
            ].dropna()

            if not plot_data.empty:
                label_order = (
                    plot_data.groupby(target_column)[compare_feature]
                    .median()
                    .sort_values()
                    .index
                )

                fig, ax = plt.subplots(figsize=(7, 4))
                sns.boxplot(
                    data=plot_data,
                    x=target_column,
                    y=compare_feature,
                    order=label_order,
                    color="#82b99b",
                    ax=ax,
                )
                ax.tick_params(axis="x", rotation=20)
                style_axis(ax)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            else:
                st.info("No records match the current filters.")
        else:
            st.info(
                "A target column and numeric features are needed "
                "for this comparison."
            )

    st.subheader("Missing values by field")

    missing_percent = (
        filtered_data.isna()
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .head(15)
    )

    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(
        missing_percent.index,
        missing_percent.values,
        color="#d3a45b",
    )
    ax.set_ylabel("Missing values (%)")
    ax.tick_params(axis="x", rotation=35)
    style_axis(ax)
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    with st.expander("View filtered records"):
        st.dataframe(
            filtered_data,
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# PAGE 3: MODEL LAB
# =========================================================
elif page == "Model lab":
    st.title("Model lab")
    st.caption(
        "Train and evaluate a baseline model for the recorded target label."
    )

    if not target_column:
        st.warning(
            "No target column was detected. The app searches for a column "
            "named Level, Target, Diagnosis, Label, Class, or Outcome."
        )

    elif data[target_column].nunique(dropna=True) < 2:
        st.warning(
            "The target column needs at least two different categories."
        )

    else:
        st.markdown(
            """
            <div class="note">
                Educational demonstration only. This model is not clinically
                validated and must not be used for diagnosis or treatment.
            </div>
            """,
            unsafe_allow_html=True,
        )

        excluded_id_names = {
            "index",
            "id",
            "patientid",
            "unnamed0",
        }

        id_columns = {
            column
            for column in data.columns
            if normalize_name(column) in excluded_id_names
        }

        candidate_features = [
            column
            for column in data.columns
            if column != target_column and column not in id_columns
        ]

        selected_features = st.multiselect(
            "Features used for prediction",
            options=candidate_features,
            default=candidate_features,
        )

        model_name = st.selectbox(
            "Choose a model",
            [
                "Random Forest",
                "Logistic Regression",
            ],
        )

        test_percent = st.slider(
            "Test set size (%)",
            min_value=15,
            max_value=40,
            value=25,
            step=5,
        )

        if st.button(
            "Train and evaluate",
            type="primary",
            disabled=not selected_features,
        ):
            model_data = data[
                selected_features + [target_column]
            ].dropna(subset=[target_column])

            X = model_data[selected_features]
            original_labels = model_data[target_column].astype(str)

            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(original_labels)

            class_counts = pd.Series(y).value_counts()
            can_stratify = (
                len(class_counts) > 1
                and class_counts.min() >= 2
            )
            stratify_labels = y if can_stratify else None

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X,
                    y,
                    test_size=test_percent / 100,
                    random_state=42,
                    stratify=stratify_labels,
                )

                numeric_model_features = X.select_dtypes(
                    include=np.number
                ).columns.tolist()

                categorical_model_features = [
                    column
                    for column in selected_features
                    if column not in numeric_model_features
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
                            SimpleImputer(
                                strategy="most_frequent"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore"),
                        ),
                    ]
                )

                transformers = []

                if numeric_model_features:
                    transformers.append(
                        (
                            "numeric",
                            numeric_pipeline,
                            numeric_model_features,
                        )
                    )

                if categorical_model_features:
                    transformers.append(
                        (
                            "categorical",
                            categorical_pipeline,
                            categorical_model_features,
                        )
                    )

                preprocessor = ColumnTransformer(
                    transformers=transformers
                )

                if model_name == "Random Forest":
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
                    steps=[
                        ("preprocessor", preprocessor),
                        ("classifier", classifier),
                    ]
                )

                model.fit(X_train, y_train)
                predictions = model.predict(X_test)

                accuracy = accuracy_score(y_test, predictions)
                class_ids = np.arange(len(label_encoder.classes_))

                metric1, metric2, metric3 = st.columns(3)

                with metric1:
                    show_metric("TEST ACCURACY", f"{accuracy:.1%}")

                with metric2:
                    show_metric("TRAIN RECORDS", f"{len(X_train):,}")

                with metric3:
                    show_metric("TEST RECORDS", f"{len(X_test):,}")

                st.subheader("Confusion matrix")

                matrix = confusion_matrix(
                    y_test,
                    predictions,
                    labels=class_ids,
                )

                fig, ax = plt.subplots(figsize=(7, 5))

                ConfusionMatrixDisplay(
                    matrix,
                    display_labels=label_encoder.classes_,
                ).plot(
                    ax=ax,
                    cmap="Greens",
                    colorbar=False,
                    xticks_rotation=25,
                )

                ax.set_title(f"{model_name} · held-out records")
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

                st.subheader("Classification report")

                report = classification_report(
                    y_test,
                    predictions,
                    labels=class_ids,
                    target_names=[
                        str(label)
                        for label in label_encoder.classes_
                    ],
                    zero_division=0,
                    output_dict=True,
                )

                report_frame = pd.DataFrame(report).transpose().round(3)
                st.dataframe(
                    report_frame,
                    use_container_width=True,
                )

                # ROC curve is available for binary classification.
                if len(label_encoder.classes_) == 2:
                    probabilities = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, probabilities)
                    false_positive_rate, true_positive_rate, _ = roc_curve(
                        y_test,
                        probabilities,
                    )

                    st.subheader("ROC curve")
                    st.metric("ROC AUC", f"{auc:.3f}")

                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.plot(
                        false_positive_rate,
                        true_positive_rate,
                        color="#32836e",
                        linewidth=2,
                        label=f"ROC AUC = {auc:.3f}",
                    )
                    ax.plot(
                        [0, 1],
                        [0, 1],
                        linestyle="--",
                        color="#9aa9a5",
                        label="Random baseline",
                    )
                    ax.set_xlabel("False positive rate")
                    ax.set_ylabel("True positive rate")
                    ax.legend()
                    style_axis(ax)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

                # Random Forest supports permutation-free built-in importance.
                if model_name == "Random Forest":
                    fitted_classifier = model.named_steps["classifier"]
                    fitted_preprocessor = model.named_steps["preprocessor"]

                    try:
                        encoded_feature_names = (
                            fitted_preprocessor.get_feature_names_out()
                        )
                        importance = pd.Series(
                            fitted_classifier.feature_importances_,
                            index=encoded_feature_names,
                        ).sort_values(ascending=False).head(15)

                        st.subheader("Most influential model features")
                        fig, ax = plt.subplots(figsize=(9, 4.5))
                        importance.sort_values().plot(
                            kind="barh",
                            color="#55a88a",
                            ax=ax,
                        )
                        ax.set_xlabel("Random Forest feature importance")
                        ax.set_ylabel("")
                        style_axis(ax)
                        fig.tight_layout()
                        st.pyplot(fig, use_container_width=True)
                        plt.close(fig)
                    except Exception:
                        st.info(
                            "Feature importance could not be displayed "
                            "for this dataset."
                        )

                st.caption(
                    "A single train/test split provides a limited estimate. "
                    "Results can vary with the split, class balance, and "
                    "selected features."
                )

            except ValueError as error:
                st.error(
                    "The model could not be trained with this split. "
                    "Try another test-set size or check the target classes. "
                    f"Details: {error}"
                )


# =========================================================
# PAGE 4: DATA GUIDE
# =========================================================
elif page == "Data guide":
    st.title("Data guide")
    st.caption(
        "Dataset source, project questions, methods, and limitations."
    )

    st.markdown(
        """
        ### Dataset source

        This project uses the workbook supplied for the
        [Cancer Patients Dataset on Kaggle](https://www.kaggle.com/datasets/rishidamarla/cancer-patients-data).

        ### Questions explored

        1. How are the recorded target labels distributed?
        2. How do numeric feature distributions vary across label groups?
        3. How well can a baseline classifier predict the recorded label
           on a held-out test set?

        ### Methods and tools

        Pandas and NumPy are used for data inspection and filtering.
        Matplotlib and Seaborn create the visualizations. The scikit-learn
        pipeline imputes missing values, scales numeric fields, one-hot
        encodes categorical fields, and trains a Random Forest or Logistic
        Regression model with a reproducible random seed.

        ### Limitations

        This app does not establish causality, estimate individual cancer
        risk, diagnose cancer, or recommend treatment. Dataset sampling,
        provenance, and label quality may limit how well findings generalize.
        Model results are a classroom demonstration, not clinical validation.
        """
    )

    inventory = pd.DataFrame(
        {
            "Field": data.columns,
            "Data type": [
                str(data[column].dtype) for column in data.columns
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

    st.subheader("Field inventory")
    st.dataframe(
        inventory,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download dataset as CSV",
        data=data.to_csv(index=False).encode("utf-8-sig"),
        file_name="cancer_patient_data.csv",
        mime="text/csv",
    )


# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.divider()
st.caption(
    "CancerLens · Course project · Streamlit, pandas, NumPy, "
    "Matplotlib, Seaborn, scikit-learn"
)