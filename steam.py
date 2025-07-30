import pandas as pd  
import streamlit as st
import os
import plotly.graph_objects as go
import plotly.express as px
import statsmodels.api as sm
import numpy as np
#from clean import clean_and_update_steam_data

st.set_page_config(page_title="Steam Dashboard", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #54565B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold; 
        color: #54565B;
        margin: 1rem 0;
        border-bottom: 2px solid #C5203F;
        padding-bottom: 0.5rem;
    }
    .metric-container {
        background-color: #C5203F;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Steam Usage Analytics Dashboard</div>', unsafe_allow_html=True)

csv_dir = 'csvs'
CSV_FILE = 'steam.csv'
EXCLUDE_COLS = ["Year", "Month", "Week", "Day", "Time", "Date"]

@st.cache_data
def load_data():
    file_path = os.path.join(csv_dir, CSV_FILE)
    df = pd.read_csv(file_path)
    # Robust date parsing for dashboard compatibility
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=True, errors='coerce')
    return df.drop_duplicates()

@st.cache_data
def load_regression_data():
    reg_path = os.path.join(csv_dir, "sum_steam.csv")
    reg_df = pd.read_csv(reg_path)
    # Robust date parsing for dashboard compatibility
    if 'Date' in reg_df.columns:
        reg_df['Date'] = pd.to_datetime(reg_df['Date'], format='mixed', dayfirst=True, errors='coerce')
    reg_df["Year"] = reg_df["Date"].dt.year.astype(str)
    reg_df["Month"] = reg_df["Date"].dt.strftime("%b")
    return reg_df

def get_steam_meter_columns(df):
    additional_excludes = ["HDD 15.5", "HDD15.5", "HDD", "10T Steam", "8T Steam"]
    all_excludes = EXCLUDE_COLS + additional_excludes
    return [col for col in df.columns if col not in all_excludes and pd.api.types.is_numeric_dtype(df[col])]

def calculate_advanced_metrics(df, meter_cols):
    """Calculate advanced performance metrics"""
    metrics = {}
    
    if not meter_cols or df.empty:
        return metrics
    
    # Calculate efficiency metrics
    total_usage = df[meter_cols].sum().sum()
    avg_usage = df[meter_cols].mean().mean()
    peak_usage = df[meter_cols].max().max()
    
    # Load factor (average / peak)
    load_factor = (avg_usage / peak_usage * 100) if peak_usage > 0 else 0
    
    # Variability coefficient
    std_usage = df[meter_cols].std().mean()
    variability = (std_usage / avg_usage * 100) if avg_usage > 0 else 0
    
    metrics.update({
        'load_factor': load_factor,
        'variability': variability,
        'total_usage': total_usage,
        'avg_usage': avg_usage,
        'peak_usage': peak_usage
    })
    
    return metrics

def create_correlation_matrix(df, meter_cols):
    """Create correlation matrix heatmap"""
    if len(meter_cols) < 2:
        return None
    
    corr_matrix = df[meter_cols].corr()
    
    fig = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdYlBu_r",
        title="Steam Meter Correlation Matrix"
    )
    fig.update_layout(
        title_x=0.5,
        height=500,
        template="plotly_white"
    )
    return fig

def create_heatmap_calendar(df, meter_cols):
    """Create calendar heatmap of daily usage"""
    if df.empty or not meter_cols:
        return None
    
    df_daily = df.copy()
    df_daily['Date'] = pd.to_datetime(df_daily['Date'])
    df_daily['DayOfWeek'] = df_daily['Date'].dt.day_name()
    df_daily['Week'] = df_daily['Date'].dt.isocalendar().week
    
    daily_usage = df_daily.groupby(['Date', 'DayOfWeek', 'Week'])[meter_cols].sum().sum(axis=1).reset_index()
    daily_usage.columns = ['Date', 'DayOfWeek', 'Week', 'TotalUsage']
    
    pivot_data = daily_usage.pivot_table(
        values='TotalUsage', 
        index='DayOfWeek', 
        columns='Week', 
        fill_value=0
    )
    
    fig = px.imshow(
        pivot_data,
        labels=dict(x="Week", y="Day of Week", color="Steam Usage"),
        color_continuous_scale="Viridis",
        title="Daily Steam Usage Pattern (Calendar Heatmap)"
    )
    fig.update_layout(
        title_x=0.5,
        height=400,
        template="plotly_white"
    )
    return fig

def create_box_plot_analysis(df, meter_cols):
    """Create box plots for usage distribution analysis"""
    if df.empty or not meter_cols:
        return None
    
    # Melt the data for box plot
    df_melted = df[meter_cols + ['Month']].melt(
        id_vars=['Month'],
        value_vars=meter_cols,
        var_name='Steam_Meter',
        value_name='Usage'
    )
    
    fig = px.box(
        df_melted,
        x='Month',
        y='Usage',
        color='Steam_Meter',
        title="Steam Usage Distribution by Month"
    )
    fig.update_layout(
        title_x=0.5,
        template="plotly_white",
        height=500
    )
    return fig

def apply_filters(df, date_range=None, selected_years=None, selected_months=None, selected_meters=None):
    """Apply consistent filters to any dataframe"""
    if df.empty:
        return df
    
    filtered = df.copy()
    
    # Apply date range filter
    if date_range and len(date_range) == 2:
        try:
            start_date = pd.to_datetime(date_range[0])
            end_date = pd.to_datetime(date_range[1])
            df_dates = pd.to_datetime(filtered["Date"], errors='coerce')
            
            # Filter by date range
            date_mask = (df_dates >= start_date) & (df_dates <= end_date)
            filtered = filtered[date_mask]
        except Exception as e:
            st.warning(f"Date filtering error: {e}")
    
    # Apply year filter
    if selected_years and "All" not in selected_years and "Year" in filtered.columns:
        if not filtered.empty:
            # Convert years to strings for consistent comparison
            filtered_years = filtered['Year'].astype(str)
            selected_years_str = [str(y) for y in selected_years]
            filtered = filtered[filtered_years.isin(selected_years_str)]
    
    # Apply month filter
    if selected_months and "All" not in selected_months and "Month" in filtered.columns:
        if not filtered.empty:
            filtered = filtered[filtered['Month'].isin(selected_months)]
    
    # Apply meter filter (keep essential columns but only analyze selected meters)
    if selected_meters and "All" not in selected_meters:
        available_meters = get_steam_meter_columns(filtered) if not filtered.empty else []
        selected_meter_cols = [col for col in selected_meters if col in available_meters]
        
        # Keep essential columns plus selected meters
        essential_cols = ["Year", "Month", "Week", "Day", "Time", "Date"]
        cols_to_keep = []
        
        # Add essential columns that exist
        for col in essential_cols:
            if col in filtered.columns:
                cols_to_keep.append(col)
        
        # Add selected meter columns
        cols_to_keep.extend(selected_meter_cols)
        
        # Always keep HDD 15.5 for regression analysis (but not for plotting)
        if "HDD 15.5" in filtered.columns:
            cols_to_keep.append("HDD 15.5")
        
        # Only filter columns if we have valid selections
        if selected_meter_cols:
            filtered = filtered[cols_to_keep]
    
    return filtered

def filter_by_sidebar(df):
    st.sidebar.markdown("## Dashboard Filters")
    
    # Initialize filters with error handling
    try:
        # Date range filter
        min_date = pd.to_datetime(df["Date"]).min()
        max_date = pd.to_datetime(df["Date"]).max()
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="sidebar_date_range"
        )
    except Exception:
        date_range = None
        st.sidebar.warning("Date range not available")
    
    # Filter options with validation
    try:
        unique_years = ["All"] + sorted([str(y) for y in df['Year'].unique() if pd.notna(y)])
        unique_months = ["All"] + sorted([m for m in df['Month'].unique() if pd.notna(m)])
        steam_meter_options = ["All"] + get_steam_meter_columns(df)
    except Exception as e:
        st.sidebar.error(f"Error loading filter options: {e}")
        unique_years = ["All"]
        unique_months = ["All"] 
        steam_meter_options = ["All"]

    # Sidebar selections
    selected_meters = st.sidebar.multiselect(
        "Steam Meters", 
        options=steam_meter_options, 
        default=["All"],
        help="Select specific steam meters to analyze"
    )
    selected_years = st.sidebar.multiselect(
        "Years", 
        options=unique_years, 
        default=["All"],
        help="Filter by specific years"
    )
    selected_months = st.sidebar.multiselect(
        "Months", 
        options=unique_months, 
        default=["All"],
        help="Filter by specific months"
    )
    
    # Apply filters using the new function
    filtered = apply_filters(df, date_range, selected_years, selected_months, selected_meters)
    
    # Show filter results
    if not filtered.empty:
        st.sidebar.success(f"{len(filtered):,} records after filtering")
    else:
        st.sidebar.error("⚠️ No data matches current filters")
    
    return filtered, selected_meters, selected_years, selected_months, date_range

# File uploaders and cleaning button
st.sidebar.markdown('### Upload New Data Files')
steam_file = st.sidebar.file_uploader('Upload Steam Meter CSV', type='csv', key='steam_csv')
hdd_file = st.sidebar.file_uploader('Upload HDD CSV', type='csv', key='hdd_csv')

if st.sidebar.button('Clean and Update Data'):
    if steam_file and hdd_file:
        # Save uploaded files to csvs/
        steam_path = os.path.join(csv_dir, 'uploaded_steam.csv')
        hdd_path = os.path.join(csv_dir, 'uploaded_hdd.csv')
        with open(steam_path, 'wb') as f:
            f.write(steam_file.getbuffer())
        with open(hdd_path, 'wb') as f:
            f.write(hdd_file.getbuffer())
        # Rename files to match expected names for cleaning
        os.replace(steam_path, os.path.join(csv_dir, 'Steam_File.csv'))
        os.replace(hdd_path, os.path.join(csv_dir, 'HDD_File.csv'))
        # Run cleaning
        #clean_and_update_steam_data(csv_dir)
        st.sidebar.success('Data cleaned and updated! Refresh dashboard to see new data.')
    else:
        st.sidebar.error('Please upload both Steam and HDD CSV files.')

def main():
    df = load_data()
    filtered, selected_meters, selected_years, selected_months, date_range = filter_by_sidebar(df)
    
    steam_meter_columns = get_steam_meter_columns(filtered)
    
    # If specific meters are selected, use only those; otherwise use all available steam meters
    if selected_meters and "All" not in selected_meters:
        # Only include meters that are actually in the filtered data and were selected
        metric_cols = [col for col in selected_meters if col in steam_meter_columns]
    else:
        # Use all available steam meter columns (excluding HDD and 10T Steam)
        metric_cols = steam_meter_columns

    # ========== SECTION 1: KEY PERFORMANCE INDICATORS ==========
    st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Calculate metrics
    totalSteamFlow = highest_steam = average_steam = highest_steam_type = totalIncomingVsMain = main_steam = "N/A"
    
    if not filtered.empty and metric_cols:
        for col in metric_cols:
            filtered[col] = pd.to_numeric(filtered[col], errors='coerce')
        
        # Fallback logic for 10T Steam: use 8T Steam if 10T is 0 or NaN
        if "10T Steam" in filtered.columns:
            filtered["10T Steam"] = pd.to_numeric(filtered["10T Steam"], errors="coerce")
            if "8T Steam" in filtered.columns:
                filtered["10T_or_8T_Steam"] = filtered["10T Steam"].where(
                    (filtered["10T Steam"].notna()) & (filtered["10T Steam"] != 0),
                    filtered["8T Steam"]
                )
            else:
                filtered["10T_or_8T_Steam"] = filtered["10T Steam"]
        elif "8T Steam" in filtered.columns:
            filtered["10T_or_8T_Steam"] = pd.to_numeric(filtered["8T Steam"], errors="coerce")
        else:
            filtered["10T_or_8T_Steam"] = np.nan

        meter_totals = filtered[metric_cols].sum(numeric_only=True)
        highest_steam = meter_totals.max()
        highest_steam_type = meter_totals.idxmax()
        average_steam = meter_totals.mean().round(1)
        totalSteamFlow = meter_totals.sum()

        # Use 10T_or_8T_Steam for main_steam and loss calculations
        main_steam = filtered["10T_or_8T_Steam"].sum()
        if main_steam > 0:
            loss_percentage = (100 - ((totalSteamFlow / main_steam) * 100))
            totalIncomingVsMain = f"{loss_percentage:.1f}%"
        else:
            totalIncomingVsMain = "N/A"

        # Advanced metrics
        advanced_metrics = calculate_advanced_metrics(filtered, metric_cols)

    with col1:
        st.metric(
            "Highest Steam Flow (Kg)", 
            f"{highest_steam:,.0f}" if isinstance(highest_steam, (int, float)) else highest_steam,
            border=True
        )
    with col2:
        st.metric(
            "Total Steam Flow (Kg)", 
            f"{totalSteamFlow:,.0f}" if isinstance(totalSteamFlow, (int, float, np.number)) else totalSteamFlow,
            border=True
        )
    with col3:
        st.metric(
            "Highest Steam Type", 
            highest_steam_type if isinstance(highest_steam_type, str) else f"{highest_steam_type}",
            border=True
        )
    with col4:
        st.metric(
            "Average Steam Usage (Kg)", 
            f"{average_steam:,.0f}" if isinstance(average_steam, (int, float)) else "N/A",
            border=True
        )
    with col5:
        st.metric(
            "Used/Incoming Loss %", 
            totalIncomingVsMain,
            border=True
        )
    with col6:
        st.metric(
            "10T/8T Steam Total (Kg)",
            f"{main_steam:,.0f}" if isinstance(main_steam, (int, float)) else main_steam,
            border=True
        )

    # ========== SECTION 2: TIME SERIES ANALYSIS ========== 
    st.markdown('<div class="section-header">Time Series Analysis</div>', unsafe_allow_html=True)
    
    col_ts1, col_ts2 = st.columns([3, 1])
    
    with col_ts2:
        resample_interval = st.selectbox(
            "Time Interval",
            options=[("15 Minutes", "15T"), ("1 Hour", "H"), ("1 Day", "D")],
            format_func=lambda x: x[0],
            index=0
        )[1]
        
        chart_type = st.selectbox(
            "Chart Type",
            options=["Line", "Area"],
            index=0
        )
    
    with col_ts1:
        if not filtered.empty and metric_cols:
            # Prepare time series data
            filtered_ts = filtered.copy()
            if "Date" in filtered_ts.columns and "Time" in filtered_ts.columns:
                filtered_ts["DateTime"] = pd.to_datetime(
                    filtered_ts["Date"].astype(str) + " " + filtered_ts["Time"].astype(str), 
                    errors="coerce"
                )
                filtered_ts = filtered_ts.dropna(subset=["DateTime"])
                filtered_ts = filtered_ts.set_index("DateTime").resample(resample_interval).sum(numeric_only=True).reset_index()
                x_col = "DateTime"
            else:
                x_col = "Time" if "Time" in filtered_ts.columns else filtered_ts.columns[0]
            
            fig = go.Figure()
            
            # Use metric_cols instead of steam_meter_columns to respect filtering
            meters_to_plot = metric_cols if metric_cols else steam_meter_columns
            
            for meter in meters_to_plot:
                if meter in filtered_ts.columns:
                    if chart_type == "Area":
                        fig.add_trace(go.Scatter(
                            x=filtered_ts[x_col],
                            y=filtered_ts[meter],
                            mode='lines',
                            fill='tonexty' if meter != meters_to_plot[0] else 'tozeroy',
                            name=meter,
                            stackgroup='one'
                        ))
                    elif chart_type == "Bar":
                        fig.add_trace(go.Bar(
                            x=filtered_ts[x_col],
                            y=filtered_ts[meter],
                            name=meter
                        ))
                    else:  # Line chart
                        fig.add_trace(go.Scattergl(
                            x=filtered_ts[x_col],
                            y=filtered_ts[meter],
                            mode='lines+markers',
                            name=meter,
                            hovertemplate=f"<b>{meter}</b><br>Usage: %{{y}}<extra></extra>"
                        ))
            
            fig.update_layout(
                title="Steam Usage Over Time",
                xaxis_title="Time",
                yaxis_title="Steam Usage",
                template="plotly_white",
                height=500,
                xaxis=dict(
                    showgrid=True,
                    rangeslider=dict(visible=True) if chart_type == "Line" else dict(visible=False)
                ),
                yaxis=dict(showgrid=True)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No data available for the selected filters.")

    col7, col8, col9 = st.columns(3)
    col10, col11, col12, col13 = st.columns(4)
    # Prepare for advanced metrics
    daytime_total = nighttime_total = 0
    day_vs_night_change = "N/A"
    delta_color_day_night = "off"
    spike_intervals_list, inactive_periods_list = [], []
    weekday_total = weekend_total = 0
    weekday_vs_weekend_change = "N/A"
    delta_color_weekday_weekend = "off"
    current_week_change = "N/A"
    delta = None
    delta_color = "off"
    safe_steam_meter_columns = []

    filtered_data = filtered.copy()
    combined_data = filtered.copy()  # For now, use filtered as combined

    if not filtered_data.empty:
        # Use filtered_data for all calculations
        steam_meter_columns_adv = [col for col in filtered_data.columns if col not in ["Year", "Month", "Week", "Day", "Time", "Date", "DateTime"] and pd.api.types.is_numeric_dtype(filtered_data[col])]
        for col in steam_meter_columns_adv:
            filtered_data[col] = pd.to_numeric(filtered_data[col], errors='coerce')

        # --- Day vs Night Calculation ---
        if "DateTime" in filtered_data.columns:
            filtered_data["Hour"] = filtered_data["DateTime"].dt.hour
            daytime = filtered_data[(filtered_data["Hour"] >= 6) & (filtered_data["Hour"] < 18)]
            nighttime = filtered_data[(filtered_data["Hour"] < 6) | (filtered_data["Hour"] >= 18)]
        elif "Time" in filtered_data.columns:
            filtered_data["Hour"] = pd.to_datetime(filtered_data["Time"], errors="coerce").dt.hour
            daytime = filtered_data[(filtered_data["Hour"] >= 6) & (filtered_data["Hour"] < 18)]
            nighttime = filtered_data[(filtered_data["Hour"] < 6) | (filtered_data["Hour"] >= 18)]
        else:
            daytime = nighttime = pd.DataFrame()

        filtered_data.drop(columns=["Hour"], inplace=True, errors='ignore')

        daytime_total = daytime[steam_meter_columns_adv].sum().sum() if not daytime.empty else 0
        nighttime_total = nighttime[steam_meter_columns_adv].sum().sum() if not nighttime.empty else 0

        if (nighttime_total + daytime_total) > 0:
            day_vs_night_change = round(((daytime_total - nighttime_total) / (nighttime_total + daytime_total)) * 100, 1)
            delta_color_day_night = "normal" if day_vs_night_change >= 0 else "inverse"
        else:
            day_vs_night_change = "N/A"
            delta_color_day_night = "off"

        # --- Weekday vs Weekend Change ---
        if "Day" in combined_data.columns:
            weekday_data = combined_data[combined_data["Day"].isin(["Mon", "Tue", "Wed", "Thu", "Fri"])]
            weekend_data = combined_data[combined_data["Day"].isin(["Sat", "Sun"])]
            safe_steam_meter_columns = [col for col in steam_meter_columns_adv if col in weekday_data.columns]
            weekday_total = weekday_data[safe_steam_meter_columns].sum().sum() if not weekday_data.empty else 0
            weekend_total = weekend_data[safe_steam_meter_columns].sum().sum() if not weekend_data.empty else 0
            if weekend_total > 0:
                weekday_vs_weekend_change = round(((weekday_total - weekend_total) / weekend_total) * 100, 1)
                delta_color_weekday_weekend = "normal" if weekday_vs_weekend_change >= 0 else "inverse"
            else:
                weekday_vs_weekend_change = "N/A"
                delta_color_weekday_weekend = "off"

        # --- Week over week ---
        if not filtered_data.empty and "Week" in filtered_data.columns and "Year" in filtered_data.columns:
            # Get current and previous week numbers
            current_weeks = filtered_data["Week"].unique()
            if len(current_weeks) > 0:
                current_week = max(current_weeks)
                current_week_data = filtered_data[filtered_data["Week"] == current_week]
                previous_week = current_week - 1
                previous_week_data = filtered_data[filtered_data["Week"] == previous_week]
                current_week_total = current_week_data[steam_meter_columns_adv].sum().sum() if not current_week_data.empty else 0
                previous_week_total = previous_week_data[steam_meter_columns_adv].sum().sum() if not previous_week_data.empty else 0
                if previous_week_total > 0:
                    current_week_change = round(((current_week_total - previous_week_total) / previous_week_total) * 100, 1)
                    delta_color = "normal" if current_week_change >= 0 else "inverse"
                    delta = current_week_change

    with col7:
        st.metric(
            label="Day vs Night Change (%)",
            value="",
            delta=f"{day_vs_night_change}%" if day_vs_night_change != "N/A" else "None",
            delta_color=delta_color_day_night,
            label_visibility="visible",
            border=True
        )
    with col8:
        st.metric(
            label="Weekday vs Weekend Change (%)",
            value="",
            delta=f"{weekday_vs_weekend_change}%" if weekday_vs_weekend_change != "N/A" else "None",
            delta_color=delta_color_weekday_weekend,
            label_visibility="visible",
            border=True
        )
    with col9:
        st.metric(
            label="Week Over Week Change (%)",
            value="",
            delta=f"{delta}%" if delta is not None else "None",  
            delta_color= delta_color if delta is not None else "off",  
            label_visibility="visible",
            border=True
        )
    with col12:
        with st.expander("Day vs Night Steam Usage"):
            st.write(f"Daytime Total: {daytime_total:,.0f} Kg")
            st.write(f"Nighttime Total: {nighttime_total:,.0f} Kg")
    with col13:
        with st.expander("Weekday vs Weekend Steam Usage"):
            st.write(f"Weekday Total: {weekday_total:,.0f} Kg")
            st.write(f"Weekend Total: {weekend_total:,.0f} Kg")
    with col11:
        with st.expander("Inactive Periods"):
            st.write("No inactive periods detected.")
    with col10:
        with st.expander("Spike Time Intervals"):
            st.write("No spike time intervals detected.")

    # ========== SECTION 3: ADVANCED ANALYTICS ==========
    st.markdown('<div class="section-header">Advanced Analytics</div>', unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4 = st.tabs(["Usage Patterns", "Correlations", "Calendar View", "Distributions"])
    
    with tab1:
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            if not filtered.empty and metric_cols:
                # Usage breakdown pie chart
                usage_breakdown = filtered[metric_cols].sum().reset_index()
                usage_breakdown.columns = ["Steam Meter", "Total Usage"]
                usage_breakdown = usage_breakdown[usage_breakdown["Total Usage"] > 0]
                
                if not usage_breakdown.empty:
                    fig_pie = px.pie(
                        usage_breakdown,
                        names="Steam Meter",
                        values="Total Usage",
                        title="Steam Usage Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    fig_pie.update_layout(height=400)
                    st.plotly_chart(fig_pie, use_container_width=True)
        
        with col_p2:
            if not filtered.empty and metric_cols:
                # Day vs Night comparison
                filtered_copy = filtered.copy()
                if "DateTime" in filtered_copy.columns:
                    filtered_copy["Hour"] = filtered_copy["DateTime"].dt.hour
                elif "Time" in filtered_copy.columns:
                    filtered_copy["Hour"] = pd.to_datetime(filtered_copy["Time"], errors="coerce").dt.hour
                else:
                    filtered_copy["Hour"] = 12  # Default
                
                filtered_copy["Period"] = filtered_copy["Hour"].apply(
                    lambda x: "Day (6AM-6PM)" if 6 <= x < 18 else "Night (6PM-6AM)"
                )
                
                period_usage = filtered_copy.groupby("Period")[metric_cols].sum().sum(axis=1).reset_index()
                period_usage.columns = ["Period", "Total Usage"]
                
                fig_period = px.bar(
                    period_usage,
                    x="Period",
                    y="Total Usage",
                    title="Day vs Night Usage Comparison",
                    color="Period",
                    color_discrete_sequence=["#FFA500", "#4169E1"]
                )
                fig_period.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig_period, use_container_width=True)
    
    with tab2:
        if not filtered.empty and len(metric_cols) > 1:
            corr_fig = create_correlation_matrix(filtered, metric_cols)
            if corr_fig:
                st.plotly_chart(corr_fig, use_container_width=True)
        else:
            st.info("Need at least 2 steam meters for correlation analysis.")
    
    with tab3:
        if not filtered.empty and metric_cols:
            calendar_fig = create_heatmap_calendar(filtered, metric_cols)
            if calendar_fig:
                st.plotly_chart(calendar_fig, use_container_width=True)
        else:
            st.info("No data available for calendar view.")
    
    with tab4:
        if not filtered.empty and metric_cols:
            box_fig = create_box_plot_analysis(filtered, metric_cols)
            if box_fig:
                st.plotly_chart(box_fig, use_container_width=True)
        else:
            st.info("No data available for distribution analysis.")

    # ========== SECTION 4: REGRESSION ANALYSIS ==========
    st.markdown('<div class="section-header">Regression Analysis: Steam vs Temperature</div>', unsafe_allow_html=True)
    
    reg_df = load_regression_data()
    
    # Apply filters to regression data
    filtered_reg_df = reg_df.copy()
    if date_range and len(date_range) == 2:
        filtered_reg_df = filtered_reg_df[
            (filtered_reg_df["Date"] >= pd.to_datetime(date_range[0])) &
            (filtered_reg_df["Date"] <= pd.to_datetime(date_range[1]))
        ]
    if "All" not in selected_years:
        filtered_reg_df = filtered_reg_df[filtered_reg_df["Year"].isin(selected_years)]
    if "All" not in selected_months:
        filtered_reg_df = filtered_reg_df[filtered_reg_df["Month"].isin(selected_months)]
    
    # Add fallback for regression meters as well
    if "10T Steam" in filtered_reg_df.columns:
        filtered_reg_df["10T Steam"] = pd.to_numeric(filtered_reg_df["10T Steam"], errors="coerce")
        if "8T Steam" in filtered_reg_df.columns:
            filtered_reg_df["10T_or_8T_Steam"] = filtered_reg_df["10T Steam"].where(
                (filtered_reg_df["10T Steam"].notna()) & (filtered_reg_df["10T Steam"] != 0),
                filtered_reg_df["8T Steam"]
            )
        else:
            filtered_reg_df["10T_or_8T_Steam"] = filtered_reg_df["10T Steam"]
    elif "8T Steam" in filtered_reg_df.columns:
        filtered_reg_df["10T_or_8T_Steam"] = pd.to_numeric(filtered_reg_df["8T Steam"], errors="coerce")
    else:
        filtered_reg_df["10T_or_8T_Steam"] = np.nan

    regression_meters = [col for col in metric_cols if col in filtered_reg_df.columns]
    if "10T_or_8T_Steam" in filtered_reg_df.columns and "10T Steam" in regression_meters:
        regression_meters = [col if col != "10T Steam" else "10T_or_8T_Steam" for col in regression_meters]

    if regression_meters and "HDD 15.5" in filtered_reg_df.columns:
        col_reg1, col_reg2 = st.columns([3, 1])
        
        with col_reg2:
            selected_reg_meter = st.selectbox(
                "Select Steam Meter",
                options=regression_meters,
                index=0
            )
            
            group_by = st.selectbox(
                "Group by",
                options=["Year", "Month"],
                index=0
            )
        
        with col_reg1:
            if selected_reg_meter in filtered_reg_df.columns:
                color_col = "Year" if group_by == "Year" else "Month"
                
                fig_reg = px.scatter(
                    filtered_reg_df,
                    x="HDD 15.5",
                    y=selected_reg_meter,
                    color=color_col,
                    trendline="ols",
                    hover_data=["Date"],
                    title=f"{selected_reg_meter} vs HDD 15.5",
                    labels={"HDD 15.5": "Heating Degree Days (15.5°C)", selected_reg_meter: "Steam Usage"}
                )
                fig_reg.update_layout(
                    template="plotly_white",
                    height=500,
                    xaxis=dict(showgrid=True),
                    yaxis=dict(showgrid=True)
                )
                st.plotly_chart(fig_reg, use_container_width=True)
    
                 
                valid_data = filtered_reg_df[
                    filtered_reg_df["HDD 15.5"].notna() & 
                    filtered_reg_df[selected_reg_meter].notna()
                ]
                
                if len(valid_data) > 1:
                    X = sm.add_constant(valid_data["HDD 15.5"])
                    y = valid_data[selected_reg_meter]
                    model = sm.OLS(y, X).fit()
                    
                    st.info(f"**R² Score**: {model.rsquared:.3f} | **Correlation Strength**: {'Strong' if model.rsquared > 0.7 else 'Moderate' if model.rsquared > 0.3 else 'Weak'}")
                    
                    # Show monthly R² values when grouped by month
                    if group_by == "Month":
                        st.markdown("### Monthly R² Values")
                        
                        # Calculate R² for each month
                        for month in sorted(valid_data["Month"].unique()):
                            month_data = valid_data[valid_data["Month"] == month]
                            if len(month_data) > 2:  # Need at least 3 points for meaningful regression
                                try:
                                    X_month = sm.add_constant(month_data["HDD 15.5"])
                                    y_month = month_data[selected_reg_meter]
                                    model_month = sm.OLS(y_month, X_month).fit()
                                    r2_month = model_month.rsquared
                                    
                                    
                                    if r2_month > 0.7:
                                        strength = "Strong"
                                        color = "green"
                                    elif r2_month > 0.3:
                                        strength = "Moderate"
                                        color = "orange"
                                    else:
                                        strength = "Weak"
                                        color = "red"
                                    
                                    st.info(f"**{month}**: R² Score: {r2_month:.3f} | Correlation Strength: {strength}")
                                    
                                except Exception as e:
                                    st.warning(f"**{month}**: Could not calculate R² (calculation error)")
                            else:
                                st.warning(f"**{month}**: Insufficient data ({len(month_data)} points - need at least 3)")
    else:
        st.info("No steam meter data available for regression analysis with current filters.")

   

if __name__ == "__main__":
    main()
