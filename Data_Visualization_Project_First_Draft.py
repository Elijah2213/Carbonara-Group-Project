import streamlit as st
import pandas as pd
import plotly.express as px

# --- Page Configuration ---
# Set page layout to wide
st.set_page_config(
    page_title="China CO₂ Emissions Dashboard",
    layout="wide"
)

# --- Data Loading and Preparation ---
@st.cache_data  # Cache the data loading to improve performance
def load_data(filepath):

    # Loads and prepares the CO2 emissions data.
    # Define the columns we want to load
    relevant_cols = ['State', 'Date', 'Sector', 'MtCO2 per day']
    
    try:
        # Use pd.read_excel to read the file, as it's an Excel file
        # 'usecols' will find these columns regardless of where they are
        df = pd.read_excel(
            filepath, 
            usecols=relevant_cols
        )
        
        # Select only the columns we need (read_excel might load others if not specified)
        df_clean = df[relevant_cols].copy()

    except Exception as e:
        st.error(f"Critical Error loading data: {e}")
        st.info("This error can occur if the 'openpyxl' library is not installed or the file is corrupt.")
        st.info(f"Filepath used: {filepath}")
        return pd.DataFrame()

    # --- Post-Loading Processing ---
    if df_clean.empty:
        st.warning("Data loaded but is empty.")
        return pd.DataFrame()

    # Rename column for easier use
    df_clean.rename(columns={'MtCO2 per day': 'Emissions'}, inplace=True)
    
    # Convert 'Date' column to datetime objects
    # pd.read_excel often auto-converts dates, so we check first
    try:
        if not pd.api.types.is_datetime64_any_dtype(df_clean['Date']):
            # If not already a datetime, convert it
            df_clean['Date'] = pd.to_datetime(df_clean['Date'], format='%d/%m/%Y')
    except Exception as date_e:
        st.error(f"Error converting 'Date' column: {date_e}")
        st.info("Please check if the 'Date' column format is dd/mm/yyyy or if it's already a date.")
        return pd.DataFrame()
    
    # Extract Year and Month for filtering
    df_clean['Year'] = df_clean['Date'].dt.year
    df_clean['Month'] = df_clean['Date'].dt.month_name()
    
    # Drop any rows with missing values
    df_clean.dropna(inplace=True)
    
    return df_clean

# Load the data
df = load_data('carbon_emissions_china.xlsx')

if df.empty:
    st.stop()

# --- Main Application Title ---
st.title("Analyzing Industrial and Regional CO₂ Emissions in China")
st.markdown("Dashboard for visualizing emissions by region, date, and sector (2023-2025)")

# --- Sidebar for Filters ---
st.sidebar.header("Dashboard Filters")

# Get unique values for filters
all_provinces = df['State'].unique()
all_sectors = df['Sector'].unique()
all_years = sorted(df['Year'].unique())

# Year filter
selected_years = st.sidebar.multiselect(
    'Select Year(s)',
    options=all_years,
    default=all_years
)

# Province filter
# Default to top 5 emitting provinces for a cleaner initial view
top_5_provinces = df.groupby('State')['Emissions'].sum().nlargest(5).index.tolist()
selected_provinces = st.sidebar.multiselect(
    'Select Province(s)',
    options=all_provinces,
    default=top_5_provinces
)

# Sector filter
selected_sectors = st.sidebar.multiselect(
    'Select Sector(s)',
    options=all_sectors,
    default=all_sectors
)

# --- Filter Data Based on Selections ---
# Check if filters are empty, if so, use all data to avoid errors
if not selected_years:
    selected_years = all_years
if not selected_provinces:
    selected_provinces = all_provinces
if not selected_sectors:
    selected_sectors = all_sectors

df_filtered = df.query(
    'Year == @selected_years & State == @selected_provinces & Sector == @selected_sectors'
)

if df_filtered.empty:
    st.warning("No data available for the selected filters. Please adjust your selection.")
    st.stop()

# --- Create Tabs ---
tab1, tab2, tab3 = st.tabs([
    "Dashboard & Insights", 
    "Recommendations & Impact", 
    "Project Summary"
])

# --- Tab 1: Dashboard & Insights ---
with tab1:
    st.header("Emission Analysis Dashboard")

    # Key Metrics
    total_emissions = df_filtered['Emissions'].sum()
    avg_daily_emissions = df_filtered['Emissions'].mean()
    
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Total Emissions (Selected Period)", 
        f"{total_emissions:,.2f} MtCO₂"
    )
    col2.metric(
        "Avg. Daily Emissions (Selected Period)", 
        f"{avg_daily_emissions:,.2f} MtCO₂"
    )
    col3.metric(
        "No. of Provinces Selected",
        len(selected_provinces)
    )

    st.markdown("---")

    # Time Series Chart
    st.subheader("Daily CO₂ Emissions Over Time")
    # Group by date and sum emissions for the selected filters
    time_series_data = df_filtered.groupby('Date')['Emissions'].sum().reset_index()
    fig_time = px.line(
        time_series_data, 
        x='Date', 
        y='Emissions', 
        title='Total Daily CO₂ Emissions',
        labels={'Emissions': 'Emissions (MtCO₂)'}
    )
    fig_time.update_layout(hovermode="x unified")
    st.plotly_chart(fig_time, use_container_width=True)

    # Breakdown Charts
    col_bar, col_pie = st.columns([0.6, 0.4]) # Give more space to bar chart

    with col_bar:
        # Bar Chart: Top Emitting Provinces
        st.subheader("Emissions by Province")
        province_data = df_filtered.groupby('State')['Emissions'].sum().reset_index()
        province_data = province_data.sort_values(by='Emissions', ascending=False)
        
        fig_prov = px.bar(
            province_data, 
            x='State', 
            y='Emissions', 
            title='Total Emissions by Province (Top 5)',
            labels={'Emissions': 'Total Emissions (MtCO₂)', 'State': 'Province'},
            height=500
        )
        st.plotly_chart(fig_prov, use_container_width=True)

    with col_pie:
        # Pie Chart: Emissions by Sector
        st.subheader("Emissions by Sector")
        sector_data = df_filtered.groupby('Sector')['Emissions'].sum().reset_index()
        
        fig_sec = px.pie(
            sector_data, 
            names='Sector', 
            values='Emissions', 
            title='Emissions Contribution by Sector',
            hole=0.3 # Donut chart
        )
        fig_sec.update_traces(textposition='inside', textinfo='percent+label')
        fig_sec.update_layout(height=500)
        st.plotly_chart(fig_sec, use_container_width=True)
        
    # Data Insights
    st.subheader("Data Insights")
    with st.expander("Click to see key insights from the data", expanded=True):
        st.markdown("""
        * **Time Trend:** The line chart shows the daily emission trends. You can observe seasonal patterns, such as potential increases during winter (heating) or summer (cooling), or dips during major holidays.
        * **Regional Hotspots:** The bar chart clearly identifies which provinces are the largest contributors to CO₂ emissions within your selection. Provinces with heavy industry or reliance on coal power (like 'Shandong', 'Inner Mongolia') consistently appear at the top.
        * **Sector Contribution:** The pie chart breaks down emissions by economic sector. Typically, the **'Power'** and **'Industry'** sectors are the most significant, highlighting them as key areas for intervention.
        """)

# --- Tab 2: Recommendations & Impact ---
with tab2:
    st.header("Actionable Recommendations & Potential Impact")

    st.subheader("Data-Driven Recommendations")
    st.markdown("""
    Based on the analysis, **Power** and **Industry** sectors in specific provinces are the primary emission sources, and hence, we propose the following targeted recommendations:
    
    1.  **For High-Emission Provinces (Examples: Shandong, Shanxi, Inner Mongolia):**
        * **Policy:** Implement and enforce stricter emission caps for heavy industries (steel, cement, chemicals).
        * **Technology:** Accelerate the deployment of renewable energy sources (solar, wind) to replace coal-fired power plants. Provide incentives for installing Carbon Capture, Utilization, and Storage (CCUS) technologies.
    
    2.  **For the 'Power' Sector:**
        * **Efficiency:** Mandate operational efficiency upgrades for all existing coal-fired power plants to reduce the amount of fuel burned per MWh.
        * **Grid:** Invest in smart grid technology to better integrate variable renewable sources and reduce transmission losses.

    3.  **For the 'Industry' Sector:**
        * **Audits:** Promote mandatory, regular energy audits for large industrial facilities to identify and reduce inefficiencies.
        * **Electrification:** Change the electrification of industrial processes (e.g., heating) using clean electricity instead of fossil fuels.
    """)
    
    st.markdown("---")

    # Outcome/Result Section
    st.subheader("Projected Impact (What-if Scenario)")
    st.markdown("Use the slider below to simulate the potential impact of implementing energy efficiency measures in the **Industry** sector for the selected data.")

    # Calculate total industry emissions for the filtered data
    try:
        industry_emissions = df_filtered[df_filtered['Sector'] == 'Industry']['Emissions'].sum()
    except:
        industry_emissions = 0

    if industry_emissions > 0:
        efficiency_gain = st.slider(
            "Select potential efficiency gain (%) in Industry", 
            min_value=0, 
            max_value=50, 
            value=10
        )
        
        # Calculate potential savings
        savings = industry_emissions * (efficiency_gain / 100)
        
        st.metric(
            label=f"Potential CO₂ Savings at {efficiency_gain}% Efficiency Gain", 
            value=f"{savings:,.2f} MtCO₂"
        )
        st.markdown(f"This represents a saving of **{savings:,.2f} MtCO₂** over the selected period, just from the **Industry** sector in the selected provinces. This demonstrates the significant, measurable impact of targeted efficiency policies.")
    else:
        st.info("Select the 'Industry' sector in the sidebar to use the 'What-if' scenario.")


# --- Tab 3: Project Overview ---
with tab3:
    st.header("Project Summary")

    st.subheader("Problem Statement")
    st.markdown("""
    China is one of the world’s largest contributors to global carbon emissions. While the nation aims for carbon neutrality by 2060, existing public data is often aggregated, making it difficult to identify specific emission hotspots across provinces or sectors. This project addresses this gap by developing an interactive dashboard to visualize emissions by region, date, and sector, enabling targeted reduction strategies.
    """)

    st.subheader("Project Objectives")
    st.markdown("""
    * To analyze China’s carbon emission trends from 2023 to 2025.
    * To identify provinces and sectors with the highest CO₂ emissions.
    * To visualize daily, monthly, and yearly emission trends interactively using Streamlit.
    * To provide actionable recommendations to reduce carbon emissions.
    * To develop a user-friendly dashboard for decision-making and awareness.
    """)
    
    st.subheader("Data Source")
    st.markdown("""
    The data used is from the `carbon_emissions_china.xlsx` file provided for this project. It contains simulated daily CO₂ emissions (in MtCO₂) across 31 Chinese provinces and 5 key sectors (Power, Industry, Ground Transport, Residential, Aviation) from January 1, 2023, to June 30, 2025.
    """)

    st.subheader("Conclusion")
    st.markdown("""
    This Streamlit application successfully transforms a complex dataset into an interactive and accessible tool. It meets the project's objectives by providing clear visualizations of emission hotspots (both regional and sectoral) and temporal trends. Most importantly, it connects data analysis to **actionable solutions** and demonstrates their **potential impact**, providing a valuable tool for anyone involved in energy policy and environmental planning.
    """)