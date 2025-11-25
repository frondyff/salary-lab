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
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# --------------------------------------------------
# Streamlit basic config
# --------------------------------------------------
st.set_page_config(
    page_title="Beyond Numbers: Global Salary Lab",
    layout="wide"
)
# --------------------------------------------------
# McGill MMA Branding
# --------------------------------------------------
MCGILL_RED = "#D6001C"
MCGILL_DARK = "#1F2430"

st.markdown(f"""
<style>
/* Header */
.mcgill-header {{
    padding: 16px 0 12px 0;
    border-bottom: 3px solid {MCGILL_RED};
}}
.mcgill-header h1 {{
    color: {MCGILL_DARK};
    margin-bottom: 0px;
}}
.mcgill-header p {{
    color: #666;
    font-size: 16px;
    margin-top: 5px;
}}

/* Buttons */
.stButton>button {{
    background-color: {MCGILL_RED};
    color: white;
    padding: 8px 22px;
    border-radius: 100px;
    border: none;
    font-weight: 600;
}}
.stButton>button:hover {{
    background-color: #b30018;
}}

/* Card container */
.mcgill-card {{
    background-color: white;
    padding: 18px;
    border-radius: 12px;
    border-top: 4px solid {MCGILL_RED};
    box-shadow: 0 2px 6px rgba(0,0,0,0.06);
    margin-bottom: 20px;
}}

</style>
""", unsafe_allow_html=True)


def mcgill_header(title):
    st.markdown(f"""
    <div class="mcgill-header">
        <h1>{title}</h1>
        <p>McGill MMA — MGSC661 Final Project</p>
    </div>
    """, unsafe_allow_html=True)


def mcgill_card_start(title):
    st.markdown(f"""
    <div class="mcgill-card">
        <h3>{title}</h3>
    """, unsafe_allow_html=True)


def mcgill_card_end():
    st.markdown("</div>", unsafe_allow_html=True)

st.sidebar.markdown(f"<h2 style='color:{MCGILL_RED};'>Navigation</h2>", unsafe_allow_html=True)


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

    # --------- sidebar filter ----------
    country = st.sidebar.selectbox(
        "Filter by country (employee_residence)",
        ["All"] + sorted(df["employee_residence"].dropna().unique().tolist())
    )
    df_f = df.copy()
    if country != "All":
        df_f = df_f[df_f["employee_residence"] == country]

    # ========= TABS =========
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Experience & Job Group",
            "PCA + K-Means Clusters",
            "Company Size & Europe",
            "Remote vs Salary (COVID)",
            "Global Map"
        ]
    )

    # --------------------------------------------------
    # TAB 1 – Salary by Experience Level & Job Group
    # --------------------------------------------------
    with tab1:
        st.markdown("### Salary by Experience Level and Job Group")

        # toggle instead of radio
        salary_mode = st.toggle("Use COL-adjusted salary?")
        y_col = "salary_adj" if salary_mode else "salary_in_usd"

        # Encoded → label maps
        EXPERIENCE_MAP = {1: "Entry", 2: "Mid", 3: "Senior / Executive"}
        JOB_GROUP_MAP = {
            "Data Engineering": 1,
            "Data Science & Research": 2,
            "Data Analytics & BI": 3,
            "AI & Machine Learning": 4,
            "Data Management & Strategy": 5,
            "Visualization & Modeling": 6,
            "Other": 7,
        }
        JOB_GROUP_REV = {v: k for k, v in JOB_GROUP_MAP.items()}

        df_plot = df_f.copy()

        if "experience_level_encoded" in df_plot.columns:
            df_plot["experience_label"] = df_plot["experience_level_encoded"].map(
                EXPERIENCE_MAP
            )

        if "job_group_encoded" in df_plot.columns:
            df_plot["job_group_label"] = df_plot["job_group_encoded"].map(JOB_GROUP_REV)

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
                "work_year": True,
            },
            category_orders={
                "experience_label": [
                    EXPERIENCE_MAP[k] for k in sorted(EXPERIENCE_MAP.keys())
                ]
            },
            title=(
                "COL-adjusted salary by Experience Level and Job Group"
                if salary_mode
                else "Raw salary (USD) by Experience Level and Job Group"
            ),
        )

        fig_a.update_layout(
            xaxis_title="Experience level",
            yaxis_title=(
                "COL-adjusted salary (index 100 = US avg)"
                if salary_mode
                else "Salary (USD)"
            ),
            legend_title="Job group",
            boxmode="group",
        )

        st.plotly_chart(fig_a, use_container_width=True)

        st.caption(
            "Each box shows the salary distribution for a combination of experience level and job group. "
            "Dots are individual employees; boxes show median and interquartile range."
        )

    # --------------------------------------------------
    # TAB 2 – PCA + K-Means Clusters
    # --------------------------------------------------
    with tab2:
        st.markdown("### Talent Segments Using PCA + K-Means")

        cluster_features = [
            "experience_level_encoded",
            "job_group_encoded",
            "remote_work_encoded",
            "company_size_encoded",
            "cost_of_living_index",
            "salary_adj",
        ]
        available_features = [c for c in cluster_features if c in df_f.columns]
        df_clust = df_f[available_features].dropna().copy()

        if df_clust.shape[0] > 0 and len(available_features) >= 2:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_clust)

            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            var_ratio = pca.explained_variance_ratio_
            pc1_var = var_ratio[0] * 100
            pc2_var = var_ratio[1] * 100
            total_var = var_ratio.sum() * 100

            kmeans = KMeans(n_clusters=4, random_state=42)
            clusters = kmeans.fit_predict(X_pca)
            centers_pca = kmeans.cluster_centers_

            plot_df = pd.DataFrame(
                {"PC1": X_pca[:, 0], "PC2": X_pca[:, 1], "Cluster": clusters}
            )

            cluster_name_map = {
                0: "A – Entry / Early Career Talent",
                1: "B – Mid-Level Technical Contributors",
                2: "C – Senior Specialists / High Compensation",
                3: "D – Senior Roles in Emerging Markets",
            }
            plot_df["ClusterLabel"] = plot_df["Cluster"].map(cluster_name_map)

            fig_pca = px.scatter(
                plot_df,
                x="PC1",
                y="PC2",
                color="ClusterLabel",
                opacity=0.7,
                title="K-Means Clusters Visualized with PCA (2 Components)",
                labels={
                    "PC1": f"Principal Component 1 ({pc1_var:.1f}% var)",
                    "PC2": f"Principal Component 2 ({pc2_var:.1f}% var)",
                    "ClusterLabel": "K-Means Cluster",
                },
            )

            fig_pca.add_scatter(
                x=centers_pca[:, 0],
                y=centers_pca[:, 1],
                mode="markers",
                marker=dict(size=16, symbol="x", color="black"),
                name="Cluster centers",
            )

            st.plotly_chart(fig_pca, use_container_width=True)

            st.caption(
                f"Explained variance – PC1: {pc1_var:.1f}%, PC2: {pc2_var:.1f}%, "
                f"total: {total_var:.1f}%. The 2D PCA view is a simplification, "
                "but it still helps us see how K-Means separates different talent profiles."
            )
        else:
            st.info(
                "Not enough numeric features/rows available to build a PCA + K-Means cluster view."
            )

    # --------------------------------------------------
    # TAB 3 – Company Size & European Compensation
    # --------------------------------------------------
    with tab3:
        st.markdown("### Effect of Company Size & European Compensation Structure")

        df_size = df_f.copy()

        company_size_map = {
            0: "Small (<50)",
            1: "Medium (50–250)",
            2: "Large (250–1,000)",
            3: "Enterprise (>1,000)",
        }
        df_size["company_size_label"] = df_size["company_size_encoded"].map(
            company_size_map
        )

        if "region" in df_size.columns:
            df_size["region_group"] = df_size["region"]
        else:
            europe_countries = {
                "United Kingdom",
                "Ireland",
                "France",
                "Germany",
                "Spain",
                "Portugal",
                "Netherlands",
                "Belgium",
                "Sweden",
                "Norway",
                "Denmark",
                "Finland",
                "Switzerland",
                "Austria",
                "Italy",
                "Poland",
                "Czechia",
                "Hungary",
                "Romania",
                "Greece",
            }
            na_countries = {"United States", "Canada"}

            def label_region(country):
                if country in europe_countries:
                    return "Europe"
                elif country in na_countries:
                    return "North America"
                else:
                    return "Other"

            df_size["region_group"] = df_size["employee_residence"].apply(label_region)

        df_size = df_size.dropna(
            subset=["company_size_label", "region_group", "salary_adj"]
        )

        group_df = (
            df_size.groupby(["company_size_label", "region_group"])
            .agg(avg_salary_adj=("salary_adj", "mean"), n_roles=("salary_adj", "size"))
            .reset_index()
        )

        size_order = [
            "Small (<50)",
            "Medium (50–250)",
            "Large (250–1,000)",
            "Enterprise (>1,000)",
        ]
        group_df["company_size_label"] = pd.Categorical(
            group_df["company_size_label"], ordered=True, categories=size_order
        )

        fig_size_region = px.bar(
            group_df.sort_values("company_size_label"),
            x="company_size_label",
            y="avg_salary_adj",
            color="region_group",
            barmode="group",
            text_auto=".0f",
            labels={
                "company_size_label": "Company size",
                "avg_salary_adj": "Avg COL-adjusted salary (USD)",
                "region_group": "Region",
            },
            title="COL-Adjusted Salary by Company Size and Region",
        )
        fig_size_region.update_layout(
            xaxis_title="Company Size",
            yaxis_title="COL-Adjusted Salary (USD)",
        )
        st.plotly_chart(fig_size_region, use_container_width=True)

        europe_df = group_df[group_df["region_group"] == "Europe"]
        if not europe_df.empty:
            fig_europe = px.bar(
                europe_df.sort_values("company_size_label"),
                x="company_size_label",
                y="avg_salary_adj",
                text_auto=".0f",
                labels={
                    "company_size_label": "Company size",
                    "avg_salary_adj": "Avg COL-adjusted salary (USD)",
                },
                title="European Labor & Compensation Structure (COL-adjusted)",
            )
            fig_europe.update_layout(
                xaxis_title="",
                yaxis_title="COL-Adjusted Salary (USD)",
            )
            st.plotly_chart(fig_europe, use_container_width=True)

        st.caption(
            "**Effect of company size:** Larger organizations consistently offer higher COL-adjusted salaries "
            "than small and mid-sized firms. **European structure:** Europe shows lower salary ceilings vs. "
            "North America, but the gap between small and large firms is narrower, consistent with more regulated "
            "labor markets and stronger social benefits."
        )

    # --------------------------------------------------
    # TAB 4 – Remote Work vs Average Salary
    # --------------------------------------------------
    with tab4:
        st.markdown("### Remote Work vs Average Salary by Year")

        if {"work_year", "remote_work_encoded", "salary_in_usd"}.issubset(df_f.columns):
            trend_df = df_f.copy()
            trend_df["is_remote"] = (trend_df["remote_work_encoded"] == 2).astype(int)

            yearly = (
                trend_df.groupby("work_year")
                .agg(
                    remote_share=("is_remote", "mean"),
                    avg_salary_usd=("salary_in_usd", "mean"),
                )
                .reset_index()
            )
            yearly["remote_share_pct"] = yearly["remote_share"] * 100

            fig = make_subplots(
                specs=[[{"secondary_y": True}]],
                subplot_titles=["Remote Roles vs Average Salary Over Time"],
            )

            fig.add_trace(
                go.Scatter(
                    x=yearly["work_year"],
                    y=yearly["remote_share_pct"],
                    mode="lines+markers",
                    name="Remote roles (%)",
                    line=dict(width=3),
                    hovertemplate="Year=%{x}<br>Remote=%{y:.1f}%<extra></extra>",
                ),
                secondary_y=False,
            )

            fig.add_trace(
                go.Scatter(
                    x=yearly["work_year"],
                    y=yearly["avg_salary_usd"],
                    mode="lines+markers",
                    name="Avg salary (USD)",
                    line=dict(width=3, dash="dot"),
                    hovertemplate="Year=%{x}<br>Avg salary=$%{y:,.0f}<extra></extra>",
                ),
                secondary_y=True,
            )

            fig.add_vrect(
                x0=2019.5,
                x1=2021.5,
                fillcolor="LightSalmon",
                opacity=0.18,
                layer="below",
                line_width=0,
                annotation_text="COVID period",
                annotation_position="top left",
            )

            if yearly["work_year"].max() >= 2022:
                fig.add_vrect(
                    x0=2021.5,
                    x1=yearly["work_year"].max() + 0.5,
                    fillcolor="LightGreen",
                    opacity=0.10,
                    layer="below",
                    line_width=0,
                    annotation_text="Post-COVID / RTO",
                    annotation_position="top right",
                )

            fig.update_layout(
                template="plotly_white",
                showlegend=True,
                legend_title_text="Series",
                xaxis_title="Year",
                title_text="Remote Roles vs Average Salary (COVID and Post-COVID)",
                margin=dict(l=40, r=40, t=60, b=40),
            )

            fig.update_yaxes(
                title_text="Remote roles (%)", range=[0, 100], secondary_y=False
            )
            fig.update_yaxes(
                title_text="Average salary (USD)", secondary_y=True
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "The orange band highlights the COVID period (2020–2021), when remote work spiked. "
                "The green band shows the post-COVID / return-to-office phase and how both remote share "
                "and pay evolved afterwards."
            )
        else:
            st.info("Required columns for the time series are missing.")

    # --------------------------------------------------
    # TAB 5 – Global Heatmap
    # --------------------------------------------------
    with tab5:
        st.markdown("### Global Heatmap of Pay vs Cost of Living")

        df_map = df_f.copy()
        if "employee_residence" not in df_map.columns:
            country_cols = [c for c in df_map.columns if c.startswith("country_name_")]
            if country_cols:
                df_map["employee_residence"] = (
                    df_map[country_cols]
                    .idxmax(axis=1)
                    .str.replace("country_name_", "", regex=False)
                )

        required_cols = {
            "employee_residence",
            "salary_adj",
            "salary_in_usd",
            "cost_of_living_index",
        }
        if required_cols.issubset(df_map.columns):
            country_stats = (
                df_map.groupby("employee_residence")
                .agg(
                    avg_salary_adj=("salary_adj", "mean"),
                    avg_salary_raw=("salary_in_usd", "mean"),
                    avg_col_index=("cost_of_living_index", "mean"),
                    n_roles=("salary_in_usd", "size"),
                )
                .reset_index()
            )

            fig_map = px.choropleth(
                country_stats,
                locations="employee_residence",
                locationmode="country names",
                color="avg_salary_adj",
                hover_name="employee_residence",
                hover_data={
                    "avg_salary_adj": ":,.0f",
                    "avg_salary_raw": ":,.0f",
                    "avg_col_index": ":.0f",
                    "n_roles": True,
                },
                color_continuous_scale="Viridis",
                title="COL-Adjusted Average Salary by Country",
            )

            fig_map.update_layout(
                coloraxis_colorbar_title="Avg COL-adjusted salary (USD)",
                margin=dict(l=0, r=0, t=60, b=0),
            )

            st.plotly_chart(fig_map, use_container_width=True)

            st.caption(
                "Countries are shaded by **average salary after adjusting for cost of living** "
                "(darker = better purchasing power for data roles). Hover to see raw salary, "
                "cost-of-living index, and number of roles."
            )
        else:
            st.info(
                "Map not shown: missing employee_residence, salary_adj, salary_in_usd, or cost_of_living_index columns."
            )

def show_col_impact(df):
    mcgill_header("2. Cost of Living Impact")

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
    mcgill_header("3. Model Performance – Ridge vs RandomForest vs XGBoost")

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
    mcgill_header("4. Drivers of Salary – Feature Importances & SHAP")

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
    mcgill_header("5. Salary Simulator – Try Your Profile")

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
footer = f"""
<style>
.footer {{
    font-size: 13px;
    color: white;
    background-color: {MCGILL_DARK};
    text-align: center;
    padding: 14px;
    margin-top: 40px;
    border-top: 3px solid {MCGILL_RED};
}}
</style>

<div class="footer">
    Created for <strong>MGSC661 – Fall 2025</strong> ·
    McGill University · Master of Management in Analytics (MMA)
</div>
"""
st.markdown(footer, unsafe_allow_html=True)
