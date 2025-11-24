import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans

import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import plotly.express as px

# --------------------------------------------------
# Streamlit basic config
# --------------------------------------------------
st.set_page_config(
    page_title="Beyond Numbers: Global Salary Lab",
    layout="wide"
)

st.sidebar.title("Navigation")


# --------------------------------------------------
# 1. Data loading & preparation
# --------------------------------------------------

@st.cache_data
def load_data():
    """
    Load processed dataset.

    Expect a CSV exported from your notebook with at least:
    - salary_in_usd
    - cost_of_living_index
    - experience_level_encoded
    - job_group_encoded
    - remote_work_encoded
    - company_size_encoded
    - work_year
    - employee_residence
    - (optional) country_name_United States, region
    """
    # TODO: adjust path to your actual file
    df = pd.read_csv("data/new_df_trim.csv")
    return df


def prepare_features(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Trim outliers, create COL-adjusted salary, and interaction features.

    Returns new_df_inter.
    """
    df = df_raw.copy()

    # Trim salary outliers (1–99%)
    q_low, q_high = df["salary_in_usd"].quantile([0.01, 0.99])
    df = df[(df["salary_in_usd"] >= q_low) &
            (df["salary_in_usd"] <= q_high)].copy()

    # Fill missing COL index with mean
    if "cost_of_living_index" in df.columns:
        df["cost_of_living_index"].fillna(df["cost_of_living_index"].mean(), inplace=True)
    else:
        raise KeyError("Column 'cost_of_living_index' is required in the dataset.")

    # COL-adjusted salary: salary / (COL_index / 100)
    df["salary_adj"] = df["salary_in_usd"] / (df["cost_of_living_index"] / 100.0)

    # ---- Feature engineering ----
    new_df_inter = df.copy()

    # Seniority flag
    if "experience_level_encoded" in new_df_inter.columns:
        new_df_inter["seniority_level"] = (
            new_df_inter["experience_level_encoded"] >= 3
        ).astype(int)

    # Experience × job group
    if {"experience_level_encoded", "job_group_encoded"}.issubset(new_df_inter.columns):
        new_df_inter["exp_x_jobgroup"] = (
            new_df_inter["experience_level_encoded"] * new_df_inter["job_group_encoded"]
        )

    # Remote × company size
    if {"remote_work_encoded", "company_size_encoded"}.issubset(new_df_inter.columns):
        new_df_inter["remote_x_company"] = (
            new_df_inter["remote_work_encoded"] * new_df_inter["company_size_encoded"]
        )

    # Experience × COL index
    if {"experience_level_encoded", "cost_of_living_index"}.issubset(new_df_inter.columns):
        new_df_inter["exp_x_costliving"] = (
            new_df_inter["experience_level_encoded"] * new_df_inter["cost_of_living_index"]
        )

    # Experience × US dummy
    if "country_name_United States" in new_df_inter.columns:
        new_df_inter["exp_x_us"] = (
            new_df_inter["experience_level_encoded"] * new_df_inter["country_name_United States"]
        )
        # -------------------------------------------------
    # Reconstruct employee_residence from country dummies
    # -------------------------------------------------
    country_cols = [c for c in new_df_inter.columns if c.startswith("country_name_")]
    if country_cols:
        # Take the dummy with value 1 and strip the prefix
        new_df_inter["employee_residence"] = (
            new_df_inter[country_cols]
            .idxmax(axis=1)                      # column name with max (i.e., 1)
            .str.replace("country_name_", "", regex=False)
        )

    return new_df_inter


@st.cache_resource
def train_models(new_df_inter: pd.DataFrame):
    """
    Prepare X, y, split, and train Ridge, RF, XGB
    on log1p(COL-adjusted salary).
    Returns:
      - new_df_inter
      - X_train, X_test
      - results dict
      - ridge_model, rf_model, xgb_model
    """
    # Features & targets
    X = new_df_inter.drop(columns=["salary_in_usd", "salary_adj"])
    y_raw = new_df_inter["salary_in_usd"]      # for evaluation
    y_adj = new_df_inter["salary_adj"]         # training target (after log)

    # numeric only
    X = X.select_dtypes(include=[np.number])

    # Split
    X_train, X_test, y_raw_train, y_raw_test, y_adj_train, y_adj_test = train_test_split(
        X, y_raw, y_adj, test_size=0.2, random_state=42
    )

    # Log of adjusted salary
    y_train_log = np.log1p(y_adj_train)

    # Models
    ridge_model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0, random_state=42))
    ])

    rf_model = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42
    )

    xgb_model = xgb.XGBRegressor(
        n_estimators=600,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42
    )

    models = {
        "Ridge": ridge_model,
        "RandomForest": rf_model,
        "XGBoost": xgb_model
    }

    def evaluate_model(name, model, X_train, X_test, y_train_log, y_raw_test):
        model.fit(X_train, y_train_log)

        # Predict log(adjusted)
        y_pred_log = model.predict(X_test)
        y_pred_adj = np.expm1(y_pred_log)

        # Reconstruct USD: adj * (COL_index / 100)
        if "cost_of_living_index" not in X_test.columns:
            raise KeyError("cost_of_living_index must be in X to reconstruct salary_in_usd")

        factor = X_test["cost_of_living_index"] / 100.0
        y_pred_usd = y_pred_adj * factor

        r2 = r2_score(y_raw_test, y_pred_usd)
        mae = mean_absolute_error(y_raw_test, y_pred_usd)
        rmse = np.sqrt(mean_squared_error(y_raw_test, y_pred_usd))

        return y_pred_usd, {"r2": r2, "mae": mae, "rmse": rmse}

    results = {}
    y_raw_test_global = y_raw_test  # for potential future use

    for name, model in models.items():
        y_pred, metrics = evaluate_model(
            name, model, X_train, X_test, y_train_log, y_raw_test_global
        )
        results[name] = metrics

    return new_df_inter, X_train, X_test, results, ridge_model, rf_model, xgb_model


# --------------------------------------------------
# 2. Page functions
# --------------------------------------------------

def show_global_salaries(df):
    st.title("1. Global Salaries")

    country = st.sidebar.selectbox(
        "Filter by country (employee_residence)",
        ["All"] + sorted(df["employee_residence"].dropna().unique().tolist())
    )
    df_f = df.copy()
    if country != "All":
        df_f = df_f[df_f["employee_residence"] == country]

    # Figure A – Salary by experience level & job group
    st.markdown("### Salary by Experience Level and Job Group")

    metric_choice = st.radio(
        "Salary metric",
        ["Raw salary (USD)", "COL-adjusted salary"],
        horizontal=True
    )
    y_col = "salary_in_usd" if metric_choice == "Raw salary (USD)" else "salary_adj"

    # Encoded → label maps
    EXPERIENCE_MAP = {
        1: "Entry",
        2: "Mid",
        3: "Senior / Executive"
    }

    JOB_GROUP_MAP = {
        "Data Engineering": 1,
        "Data Science & Research": 2,
        "Data Analytics & BI": 3,
        "AI & Machine Learning": 4,
        "Data Management & Strategy": 5,
        "Visualization & Modeling": 6,
        "Other": 7
    }
    JOB_GROUP_REV = {v: k for k, v in JOB_GROUP_MAP.items()}
    df_plot = df_f.copy()

    # Add readable labels
    if "experience_level_encoded" in df_plot.columns:
        df_plot["experience_label"] = df_plot["experience_level_encoded"].map(EXPERIENCE_MAP)

    if "job_group_encoded" in df_plot.columns:
        df_plot["job_group_label"] = df_plot["job_group_encoded"].map(JOB_GROUP_REV)

    # Keep only rows where both labels are known
    df_plot = df_plot.dropna(subset=["experience_label", "job_group_label"])



    fig_a = px.box(
        df_plot,
        x="experience_label",
        y=y_col,
        color="job_group_label",
        points="all",
        hover_data={
            "experience_label": True,
            "job_group_label": True,
            y_col: ":,.0f",
            "work_year": True
        },
        category_orders={
            "experience_label": [EXPERIENCE_MAP[k] for k in sorted(EXPERIENCE_MAP.keys())]
        },
        title=f"{y_col} by Experience Level and Job Group"
    )

    fig_a.update_layout(
        xaxis_title="Experience level",
        yaxis_title="Salary (USD)" if y_col == "salary_in_usd" else "COL-adjusted salary (index 100 = US avg)",
        legend_title="Job group",
        boxmode="group"
    )

    st.plotly_chart(fig_a, use_container_width=True)

    st.caption(
        "Each box shows the salary distribution for a combination of experience level and job group. "
        "Dots are individual employees; boxes show median and interquartile range."
    )

    # Figure D – Cluster map
    st.markdown("###  Clusters of Experience vs Adjusted Salary")

        # Maps for readable labels
    experience_labels = {
    1: "Entry",
    2: "Mid",
    3: "Senior / Executive"
    }

    # Clean copy
    df_clust = df_f[["experience_level_encoded", "salary_adj"]].dropna().copy()
    df_clust = df_clust.rename(columns={
        "experience_level_encoded": "ExperienceLevel",
        "salary_adj": "SalaryAdj"
    })

    # Convert experience encodings
    df_clust["ExperienceLabel"] = df_clust["ExperienceLevel"].map(experience_labels)

    # Run K-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    df_clust["Cluster"] = kmeans.fit_predict(df_clust[["ExperienceLevel", "SalaryAdj"]])

    # Cluster centers with readable labels
    centers = pd.DataFrame(kmeans.cluster_centers_, columns=["ExperienceLevel", "SalaryAdj"])
    centers["Cluster"] = centers.index
    centers["ExperienceLabel"] = centers["ExperienceLevel"].round(0).astype(int).map(experience_labels)

    # Sort centers by experience then salary (low → high)
    ordered = centers.sort_values(["ExperienceLevel", "SalaryAdj"]).reset_index(drop=True)
    # Map cluster names for user friendliness
    cluster_names = {
        ordered.loc[0, "Cluster"]: "Cluster A – Early Career, Lower Pay",
        ordered.loc[1, "Cluster"]: "Cluster B – Mid-Level, Stable Pay",
        ordered.loc[2, "Cluster"]: "Cluster C – Senior, Emerging Markets",
        ordered.loc[3, "Cluster"]: "Cluster D – Senior, High Compensation"
    }

    df_clust["ClusterName"] = df_clust["Cluster"].map(cluster_names)
    centers["ClusterName"] = centers["Cluster"].map(cluster_names)

    # Plot
    fig_d = px.scatter(
        df_clust,
        x="ExperienceLabel",
        y="SalaryAdj",
        color="ClusterName",
        hover_data={
            "ExperienceLabel": True,
            "SalaryAdj": ":,.0f",
            "ClusterName": True
        },
        title="Talent Segments Based on Experience and COL-Adjusted Salary",
        opacity=0.6,
    )

    # Add cluster centers as bigger markers
    fig_d.add_scatter(
        x=centers["ExperienceLabel"],
        y=centers["SalaryAdj"],
        mode="markers+text",
        marker=dict(size=16, symbol="x", color="black"),
        text=[name.replace("Cluster ", "") for name in centers["ClusterName"]],
        textposition="top center",
        name="Cluster Centers"
    )

    fig_d.update_layout(
        xaxis_title="Experience Level",
        yaxis_title="COL-Adjusted Salary (USD)",
        legend_title="Talent Segments",
    )

    st.plotly_chart(fig_d, use_container_width=True)

    st.caption("""
    **Interpretation:**
    - Each point represents an individual employee.
    - Colors represent segments of the data with similar experience + salary patterns.
    - Cluster centers (X markers) show the "typical profile" for each group.
    """)


def show_col_impact(df):
    st.title("2. Cost of Living Impact")

    country = st.sidebar.selectbox(
        "Filter by country (employee_residence)",
        ["All"] + sorted(df["employee_residence"].dropna().unique().tolist())
    )
    df_f = df.copy()
    if country != "All":
        df_f = df_f[df_f["employee_residence"] == country]

    st.markdown("### Raw vs COL-adjusted Salary Distributions")

    col1, col2 = st.columns(2)
    with col1:
        fig_raw = px.histogram(
            df_f, x="salary_in_usd", nbins=40, title="Raw Salary (USD)"
        )
        st.plotly_chart(fig_raw, use_container_width=True)
    with col2:
        fig_adj = px.histogram(
            df_f, x="salary_adj", nbins=40, title="COL-adjusted Salary"
        )
        st.plotly_chart(fig_adj, use_container_width=True)

    # Figure B – Salary vs COL index
    st.markdown("### Salary vs Cost of Living Index")

    color_col = "region" if "region" in df_f.columns else "employee_residence"
    fig_b = px.scatter(
        df_f,
        x="cost_of_living_index",
        y="salary_in_usd",
        color=color_col,
        trendline="lowess",
        title="Salary vs Cost of Living (with smoothed trend)",
        opacity=0.6
    )
    fig_b.update_layout(
        xaxis_title="Cost of Living Index",
        yaxis_title="Salary in USD"
    )
    st.plotly_chart(fig_b, use_container_width=True)


def show_model_performance(results):
    st.title("3. Model Performance – Ridge vs RandomForest vs XGBoost")

    metrics_df = pd.DataFrame(results).T  # rows = models
    st.subheader("Metrics in Real Salary (USD)")
    st.dataframe(
        metrics_df.style.format({"r2": "{:.3f}", "mae": "{:,.0f}", "rmse": "{:,.0f}"})
    )

    fig = px.bar(
        metrics_df.reset_index().rename(columns={"index": "Model"}),
        x="Model", y="r2", title="R² by Model"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "- Ridge: linear baseline with regularization\n"
        "- RandomForest: nonlinear tree ensemble\n"
        "- XGBoost: best tradeoff; used as our main explainable model with SHAP"
    )


def show_drivers_shap(xgb_model, rf_model, X_train, X_test):
    st.title("4. Drivers of Salary – Feature Importances & SHAP")

    # Figure E – Feature importances
    st.subheader("Feature Importances")

    rf_importances = pd.Series(
        rf_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)
    xgb_importances = pd.Series(
        xgb_model.feature_importances_, index=X_train.columns
    ).sort_values(ascending=False)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Random Forest – Top 10 Features**")
        fig_rf = px.bar(
            rf_importances.head(10)[::-1],
            x=rf_importances.head(10)[::-1],
            y=rf_importances.head(10)[::-1].index,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"}
        )
        st.plotly_chart(fig_rf, use_container_width=True)

    with col2:
        st.markdown("**XGBoost – Top 10 Features**")
        fig_xgb = px.bar(
            xgb_importances.head(10)[::-1],
            x=xgb_importances.head(10)[::-1],
            y=xgb_importances.head(10)[::-1].index,
            orientation="h",
            labels={"x": "Importance", "y": "Feature"}
        )
        st.plotly_chart(fig_xgb, use_container_width=True)

    # SHAP
    st.markdown("---")
    st.subheader("SHAP – Explaining XGBoost Predictions")

    shap.initjs()
    explainer_xgb = shap.TreeExplainer(xgb_model)
    shap_values_xgb = explainer_xgb.shap_values(X_test)

    tab1, tab2 = st.tabs(["Global Importance", "Detailed Effects"])

    with tab1:
        st.markdown("**SHAP Global Feature Importance (XGBoost)**")
        fig_bar = plt.figure()
        shap.summary_plot(shap_values_xgb, X_test, plot_type="bar", show=False)
        st.pyplot(fig_bar)

    with tab2:
        st.markdown("**SHAP Beeswarm & Dependence**")

        fig_bee = plt.figure()
        shap.summary_plot(shap_values_xgb, X_test, show=False)
        st.pyplot(fig_bee)

        key_features = [
            "experience_level_encoded",
            "job_group_encoded",
            "cost_of_living_index",
            "exp_x_jobgroup",
            "exp_x_costliving",
            "remote_x_company",
            "exp_x_us"
        ]
        available_feats = [f for f in key_features if f in X_test.columns]

        if available_feats:
            feat = st.selectbox("Feature for dependence plot", available_feats)
            fig_dep = plt.figure()
            shap.dependence_plot(feat, shap_values_xgb, X_test, show=False)
            st.pyplot(fig_dep)
        else:
            st.info("No key engineered features found in X_test to plot dependence.")


def show_salary_simulator(xgb_model, X_train):
    st.title("5. Salary Simulator – Try Your Profile")

    # -------------------------------------------
    # Human-readable maps
    # -------------------------------------------
    experience_map = {
    1: "Entry",
    2: "Mid",
    3: "Senior / Executive"
    }

    job_group_map = {
        'Data Engineering': 1,
        'Data Science & Research': 2,
        'Data Analytics & BI': 3,
        'AI & Machine Learning': 4,
        'Data Management & Strategy': 5,
        'Visualization & Modeling': 6,
        'Other': 7
    }

    remote_map = {
        0: "On-site",
        1: "Hybrid",
        2: "Remote"
    }

    company_map = {
        0: "Small (<50)",
        1: "Medium (50–250)",
        2: "Large (250–1000)",
        3: "Enterprise (>1000)"
    }

    # -------------------------------------------
    # User inputs with labels
    # -------------------------------------------
    col1, col2 = st.columns(2)

    with col1:
        exp_label = st.selectbox(
            "Experience Level",
            options=list(experience_map.values()),
            help="Your seniority level in your current or target role"
        )
        exp = list(experience_map.keys())[list(experience_map.values()).index(exp_label)]

        job_group_label = st.selectbox(
            "Job Group",
            options=list(job_group_map.keys()),
            help="Choose the closest category for your job"
        )
        job_group = job_group_map[job_group_label]

        remote_label = st.selectbox(
            "Work Arrangement",
            options=list(remote_map.values()),
            help="On-site, Hybrid, or Fully Remote"
        )
        remote = list(remote_map.keys())[list(remote_map.values()).index(remote_label)]

    with col2:
        company_label = st.selectbox(
            "Company Size",
            options=list(company_map.values()),
            help="Size of the employer you work for or are applying to"
        )
        company_size = list(company_map.keys())[list(company_map.values()).index(company_label)]

        col_index = st.slider(
            "Cost of Living Index",
            min_value=50.0, max_value=140.0, value=90.0,
            help="100 = average COL in the US. Higher = more expensive."
        )

        is_us = st.checkbox(
            "United States role?",
            value=True,
            help="Check if the position is located in the US"
        )

    # -------------------------------------------
    # Build feature row for prediction
    # -------------------------------------------
    input_dict = {c: 0 for c in X_train.columns}

    input_dict["experience_level_encoded"] = exp
    input_dict["job_group_encoded"] = job_group
    input_dict["remote_work_encoded"] = remote
    input_dict["company_size_encoded"] = company_size
    input_dict["cost_of_living_index"] = col_index

    if "country_name_United States" in input_dict:
        input_dict["country_name_United States"] = int(is_us)

    # Interaction features
    if "exp_x_jobgroup" in input_dict:
        input_dict["exp_x_jobgroup"] = exp * job_group
    if "exp_x_costliving" in input_dict:
        input_dict["exp_x_costliving"] = exp * col_index
    if "remote_x_company" in input_dict:
        input_dict["remote_x_company"] = remote * company_size
    if "exp_x_us" in input_dict:
        input_dict["exp_x_us"] = exp * int(is_us)

    X_input = pd.DataFrame([input_dict])

    # -------------------------------------------
    # Predict salary
    # -------------------------------------------
    if st.button("Predict Salary"):
        y_log_pred = xgb_model.predict(X_input)[0]
        y_adj_pred = np.expm1(y_log_pred)
        y_usd_pred = y_adj_pred * (col_index / 100.0)

        st.success(f"Estimated salary: **${y_usd_pred:,.0f} USD**")

    # -------------------------------------------
    # Expandable legend
    # -------------------------------------------
    with st.expander("ℹ️ Understanding the Options"):
        st.markdown("### Experience Levels")
        st.json(experience_map)
        st.markdown("### Job Groups")
        st.json(job_group_map)
        st.markdown("### Work Arrangement")
        st.json(remote_map)
        st.markdown("### Company Size")
        st.json(company_map)
        st.markdown("### Cost of Living Index")
        st.write("100 = average US cost of living, higher = more expensive.")


# --------------------------------------------------
# 3. Main app routing
# --------------------------------------------------
def main():
    df_raw = load_data()
    new_df_inter = prepare_features(df_raw)
    new_df_inter, X_train, X_test, results, ridge_model, rf_model, xgb_model = train_models(new_df_inter)

    page = st.sidebar.radio(
        "Go to",
        [
            "1. Global Salaries",
            "2. Cost of Living Impact",
            "3. Model Performance",
            "4. Drivers of Salary (SHAP)",
            "5. Salary Simulator"
        ]
    )

    if page == "1. Global Salaries":
        show_global_salaries(new_df_inter)
    elif page == "2. Cost of Living Impact":
        show_col_impact(new_df_inter)
    elif page == "3. Model Performance":
        show_model_performance(results)
    elif page == "4. Drivers of Salary (SHAP)":
        show_drivers_shap(xgb_model, rf_model, X_train, X_test)
    elif page == "5. Salary Simulator":
        show_salary_simulator(xgb_model, X_train)




if __name__ == "__main__":
    main()

# ---------------------------------------------------------
# Global footer (appears on every Streamlit page)
# ---------------------------------------------------------
footer = """
<style>
.footer {
    font-size: 14px;
    color: #888888;
    text-align: center;
    margin-top: 50px;
    padding-top: 20px;
    border-top: 1px solid #e5e5e5;
}
</style>

<div class="footer">
    Created for Final Project — <strong>MGSC661</strong>, Fall 2025<br>
    McGill University — Master of Management in Analytics (MMA)
</div>
"""

st.markdown(footer, unsafe_allow_html=True)
