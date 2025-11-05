import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import io

# Optional mapping libraries for nicer maps/plots
try:
    import geopandas as gpd
    import plotly.express as px
    PLOTLY_GEO_AVAILABLE = True
except Exception:
    PLOTLY_GEO_AVAILABLE = False

st.set_page_config(layout="wide", page_title="China Carbon Emissions — CSV Explorer")

# ---- CONFIG: default CSV path (edit if needed) ----
DEFAULT_CSV = "carbon_emissions_china.csv"  # change to your CSV filename/path

# helpers 
@st.cache_data
def load_csv(path):
    df = pd.read_csv(path)
    # strip whitespace from headers
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    return df

def detect_columns(df):
    date_col = None
    state_col = None
    sector_col = None
    metric_col = None
    for c in df.columns:
        lc = str(c).lower()
        if 'date' in lc and date_col is None:
            date_col = c
        if any(x in lc for x in ['state', 'province', 'region']) and state_col is None:
            state_col = c
        if 'sector' in lc and sector_col is None:
            sector_col = c
        if any(x in lc for x in ['mtco2', 'co2', 'emission', 'mtco2 per day', 'mtco2/day']) and metric_col is None:
            metric_col = c
    if metric_col is None:
        numcols = df.select_dtypes(include='number').columns.tolist()
        # but CSV numeric may be strings; try to detect numeric-like by name
        if not numcols:
            # try coercion test to find numeric-like columns
            for c in df.columns:
                try:
                    pd.to_numeric(df[c].dropna().iloc[:50])
                    numcols.append(c)
                except Exception:
                    pass
        metric_col = numcols[0] if numcols else None
    return date_col, state_col, sector_col, metric_col

def to_period_index(df, date_col, freq):
    if date_col not in df.columns:
        df['__Period'] = pd.NaT
        return df
    if freq == 'D':
        df['__Period'] = df[date_col].dt.date
    elif freq == 'M':
        df['__Period'] = df[date_col].dt.to_period('M').dt.to_timestamp()
    elif freq == 'Y':
        df['__Period'] = df[date_col].dt.to_period('Y').dt.to_timestamp()
    else:
        df['__Period'] = df[date_col]
    return df

def prepare_heatmap_pivot(df, state_col, metric_col, month_period_col='__Month'):
    pivot = (
        df.groupby([state_col, month_period_col])[metric_col]
        .sum()
        .reset_index()
        .pivot(index=state_col, columns=month_period_col, values=metric_col)
        .fillna(0)
    )
    # attempt to sort month columns chronologically if possible
    try:
        pivot.columns = pd.to_datetime(pivot.columns)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)
    except Exception:
        pass
    return pivot

def safe_download_button(df, label, filename):
    csv_bytes = df.to_csv(index=True).encode('utf-8')
    st.download_button(label, csv_bytes, file_name=filename, mime='text/csv')

# data input (csv)
st.sidebar.header("1) Data input (CSV)")
uploaded = st.sidebar.file_uploader("Upload CSV file (or skip to use default CSV)", type=["csv", "txt"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.sidebar.success("Loaded uploaded CSV")
else:
    if Path(DEFAULT_CSV).exists():
        df = load_csv(DEFAULT_CSV)
        st.sidebar.info(f"Loaded default CSV: {DEFAULT_CSV}")
    else:
        st.sidebar.error("Default CSV not found. Please upload your CSV file.")
        st.stop()

# detect columns & parse dates
date_col, state_col, sector_col, metric_col = detect_columns(df)
st.sidebar.write("Detected columns:")
st.sidebar.write({"date": date_col, "state": state_col, "sector": sector_col, "metric": metric_col})

# allow manual overrides
with st.sidebar.expander("Manual column overrides (if detection is wrong)"):
    col_options = [None] + list(df.columns)
    date_col = st.selectbox("Date column", options=col_options, index=(0 if date_col is None else col_options.index(date_col)))
    state_col = st.selectbox("State/Province column", options=col_options, index=(0 if state_col is None else col_options.index(state_col)))
    sector_col = st.selectbox("Sector column", options=col_options, index=(0 if sector_col is None else col_options.index(sector_col)))
    metric_col = st.selectbox("Metric column (emissions)", options=col_options, index=(0 if metric_col is None else col_options.index(metric_col)))

# copy for working modifications
working = df.copy()

# parse / coerce metric column to numeric
if metric_col:
    working[metric_col] = pd.to_numeric(working[metric_col].astype(str).str.replace(',', '').str.strip(), errors='coerce')

# parse dates if available
if date_col:
    working[date_col] = pd.to_datetime(working[date_col], errors='coerce', infer_datetime_format=True)

# fill missing states to keep them in groupby
if state_col and state_col in working.columns:
    working[state_col] = working[state_col].fillna('Unknown')

#sidebar: aggregation/filters (A)
st.sidebar.header("2) Aggregation & filters (A)")
agg_choice = st.sidebar.selectbox("Aggregation level", options=['Daily', 'Monthly', 'Yearly'], index=1)
freq_map = {'Daily':'D', 'Monthly':'M', 'Yearly':'Y'}
freq = freq_map[agg_choice]

# date range filter
if date_col and working[date_col].notnull().any():
    min_date = working[date_col].min().date()
    max_date = working[date_col].max().date()
    date_range = st.sidebar.date_input("Date range", value=(min_date, max_date))
    start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    working = working[(working[date_col] >= start_date) & (working[date_col] <= end_date)]
else:
    start_date = end_date = None

# state filter
if state_col:
    state_list = sorted(working[state_col].dropna().unique().tolist())
    selected_states = st.sidebar.multiselect("Filter states/provinces (empty = all)", options=state_list, default=state_list[:6])
    if selected_states:
        working = working[working[state_col].isin(selected_states)]

# sector filter
if sector_col:
    sector_list = sorted(working[sector_col].dropna().unique().tolist())
    selected_sectors = st.sidebar.multiselect("Filter sectors (empty = all)", options=sector_list, default=[])
    if selected_sectors:
        working = working[working[sector_col].isin(selected_sectors)]

rolling_days = st.sidebar.slider("Rolling window (for smoothing)", min_value=1, max_value=90, value=7)

# ---- Main UI ----
st.title("China Carbon Emissions — CSV Explorer (A + B + C)")

st.markdown(
    """
    - Aggregates emissions by day/month/year and plots a time-series (A).  
    - Produces a state × month heatmap (A).  
    - Optionally draws a choropleth map if you upload a GeoJSON (B).  
    - Lets you download the aggregated data used for each chart (C).
    """
)

# ---- Time series (aggregated) ----
st.header("Time series (aggregated)")

if date_col and date_col in working.columns and metric_col:
    # add __AggPeriod based on aggregation
    working = to_period_index(working, date_col, freq_map[agg_choice])
    # rename to consistent name
    if '__Period' in working.columns:
        working = working.rename(columns={'__Period': '__AggPeriod'})
    ts = (
        working.groupby('__AggPeriod')[metric_col]
        .sum()
        .reset_index()
        .sort_values('__AggPeriod')
    )
    ts['rolling'] = ts[metric_col].rolling(rolling_days, min_periods=1).mean()

    # Use plotly for interactive if available; fallback to line_chart
    if PLOTLY_GEO_AVAILABLE:
        fig = px.line(ts, x='__AggPeriod', y=metric_col, title=f"Total {metric_col} ({agg_choice})")
        fig.add_scatter(x=ts['__AggPeriod'], y=ts['rolling'], mode='lines', name=f'{rolling_days}-period rolling avg')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.line_chart(ts.set_index('__AggPeriod')[[metric_col, 'rolling']])

    st.subheader("Download time series data (C)")
    safe_download_button(ts, "Download time series CSV", f"time_series_{agg_choice.lower()}.csv")
else:
    st.warning("Time series requires a valid Date column and numeric metric column.")

# ---- Heatmap: State x Month ----
st.header("State × Month heatmap (A)")

if date_col and state_col and metric_col and date_col in working.columns:
    w = working.copy()
    w['__Month'] = w[date_col].dt.to_period('M').astype(str)
    pivot = prepare_heatmap_pivot(w, state_col, metric_col, month_period_col='__Month')
    if pivot.empty:
        st.warning("No data available after filters to build heatmap.")
    else:
        if PLOTLY_GEO_AVAILABLE:
            fig_h = px.imshow(
                pivot.values,
                x=pivot.columns.astype(str),
                y=pivot.index.astype(str),
                aspect='auto',
                labels=dict(x="Month", y="State/Province", color=str(metric_col)),
                title=f"{metric_col} by State and Month"
            )
            st.plotly_chart(fig_h, use_container_width=True)
        else:
            st.dataframe(pivot)

        st.subheader("Download heatmap pivot table (C)")
        safe_download_button(pivot, "Download heatmap CSV", "state_month_heatmap.csv")
else:
    st.warning("Heatmap requires Date, State and Metric columns.")

# ---- Choropleth (B) ----
st.header("Choropleth map (B) — optional")

if not PLOTLY_GEO_AVAILABLE:
    st.info("For interactive choropleth, install geopandas and plotly: pip install geopandas plotly fiona shapely")
    st.info("You can still upload a GeoJSON and the app will attempt to show the joined CSV table.")
else:
    st.write("Provide a GeoJSON of China provinces (place at /mnt/data/china_provinces.geojson or upload below). The GeoJSON should contain province names that match your State column.")
    geo = None
    local_geo = Path("/mnt/data/china_provinces.geojson")
    if local_geo.exists():
        try:
            geo = gpd.read_file(local_geo)
            st.success("Loaded local GeoJSON: /mnt/data/china_provinces.geojson")
        except Exception as e:
            st.error(f"Failed to read local geojson: {e}")

    uploaded_geo = st.file_uploader("Upload GeoJSON file (optional)", type=['geojson', 'json'])
    if uploaded_geo:
        try:
            geo = gpd.read_file(uploaded_geo)
            st.success("Uploaded GeoJSON loaded.")
        except Exception as e:
            st.error(f"Failed to read uploaded geojson: {e}")
            geo = None

    if geo is not None and state_col and metric_col:
        geo = geo.copy()
        # find candidate name column
        name_col = None
        for c in geo.columns:
            if str(c).lower() in ('name', 'province', 'province_name', 'adm1_name', 'cn_name'):
                name_col = c
                break
        if name_col is None:
            # fallback to first non-geometry object column
            for c in geo.columns:
                if c != geo.geometry.name and geo[c].dtype == object:
                    name_col = c
                    break
        if name_col is None:
            st.error("Could not find a province name column in GeoJSON. Please ensure it has a name property.")
        else:
            # prepare aggregated metric per state
            agg = working.groupby(state_col)[metric_col].sum().reset_index().rename(columns={state_col:'state_name', metric_col:'metric_sum'})
            agg['key_norm'] = agg['state_name'].astype(str).str.strip().str.lower()
            geo['key_norm'] = geo[name_col].astype(str).str.strip().str.lower()
            merged = geo.merge(agg, on='key_norm', how='left')
            merged['metric_sum'] = merged['metric_sum'].fillna(0)

            # Plot choropleth using plotly
            try:
                center = merged.geometry.centroid.unary_union.centroid
                fig_map = px.choropleth_mapbox(
                    merged,
                    geojson=merged.__geo_interface__,
                    locations=merged.index,
                    color='metric_sum',
                    mapbox_style="carto-positron",
                    zoom=3,
                    center={"lat": center.y, "lon": center.x},
                    opacity=0.7,
                    labels={'metric_sum': f"Total {metric_col}"}
                )
                fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
                st.plotly_chart(fig_map, use_container_width=True)
            except Exception as e:
                st.error(f"Failed to draw interactive choropleth: {e}")
                st.dataframe(merged[[name_col, 'metric_sum']].sort_values('metric_sum', ascending=False))

            st.subheader("Download choropleth data (C)")
            safe_download_button(merged[[name_col, 'metric_sum']].set_index(name_col), "Download choropleth CSV", "choropleth_data.csv")
    else:
        st.info("Upload a GeoJSON or place one at /mnt/data/china_provinces.geojson. Make sure geopandas & plotly are installed to display maps.")

# ---- Footer: quick data checks ----
st.sidebar.header("Quick checks")
if st.sidebar.checkbox("Show sample rows"):
    st.sidebar.write(working.head(50))

st.sidebar.markdown("If you'd like, I can add fuzzy matching to auto-map province names (using python-Levenshtein orfuzzywuzzy) or add monthly/yearly comparison charts.")
