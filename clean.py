import pandas as pd
import os
import calendar
import math

def clean_and_update_steam_data(input_dir='csvs'):
    output_file = os.path.join(input_dir, 'steam.csv')
    sum_output_file = os.path.join(input_dir, 'sum_steam.csv')

    # Find HDD and Steam files
    hdd_file = None
    steam_file = None
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.csv'):
            if 'HDD' in file_name:
                hdd_file = os.path.join(input_dir, file_name)
            elif 'Steam' in file_name:
                steam_file = os.path.join(input_dir, file_name)

    if not hdd_file or not steam_file:
        raise FileNotFoundError("Could not find both HDD and Steam CSV files in the 'csvs' folder.")

    # Read and clean HDD data
    hdd = pd.read_csv(hdd_file, skiprows=5)
    hdd['Date'] = pd.to_datetime(hdd['Date'], format='mixed', dayfirst=True, errors='coerce').dt.date
    # Drop rows where date could not be parsed
    hdd = hdd.dropna(subset=['Date'])

    # Read and clean Steam data
    steam = pd.read_csv(steam_file)
    steam.rename(columns={col: col.replace(" (kg)", "") for col in steam.columns}, inplace=True)
    steam.fillna(0, inplace=True)
    steam['Timestamp'] = pd.to_datetime(steam['Timestamp'])
    steam['Date'] = steam['Timestamp'].dt.date
    steam['Time'] = steam['Timestamp'].dt.time

    # Merge cleaned data
    combined = pd.merge(steam, hdd, on='Date', how='inner')
    combined['Year'] = pd.to_datetime(combined['Date']).dt.year
    combined['Month'] = pd.to_datetime(combined['Date']).dt.month.map(lambda x: calendar.month_abbr[x])
    combined['Day'] = pd.to_datetime(combined['Date']).dt.strftime('%a')
    combined['Week'] = pd.to_datetime(combined['Date']).dt.dayofyear.apply(lambda x: math.ceil(x / 7))

    steam_cols = [
        '8T Steam', '10T Steam', 'Packaging', 'TCW-03 & HG080', 'Man. & Admin',
        'VG010/020 Humidifiers', 'VG010/020 PreHeat', 'QC Lab 2 & SBP',
        'Process Line 1', 'Process Line 2', 'PUW MP015', 'PUW MP014'
    ]
    column_order = ['Year', 'Month', 'Day', 'Date', 'Time', 'Week', 'HDD 15.5'] + steam_cols
    combined = combined[column_order]

    # If output_file exists, append only new rows
    if os.path.exists(output_file):
        existing = pd.read_csv(output_file)
        # Convert 'Date' and 'Time' to comparable types
        existing['Date'] = pd.to_datetime(existing['Date'], format='mixed', dayfirst=True, errors='coerce').dt.date
        existing['Time'] = pd.to_datetime(existing['Time']).dt.time
        # Remove duplicates
        combined = pd.concat([existing, combined], ignore_index=True).drop_duplicates(subset=['Date', 'Time'], keep='last')

    combined.to_csv(output_file, index=False)
    print(f"Combined file saved: {output_file}")

    meter_cols = steam_cols
    # Group and sum for sum_steam.csv
    summed = combined.groupby('Date', as_index=False)[meter_cols].sum(numeric_only=True)
    hdd_vals = combined.groupby('Date', as_index=False)['HDD 15.5'].first()
    summed = pd.merge(summed, hdd_vals, on='Date', how='left')
    summed = summed[['Date'] + meter_cols + ['HDD 15.5']]

    # If sum_output_file exists, append only new rows
    if os.path.exists(sum_output_file):
        existing_sum = pd.read_csv(sum_output_file)
        existing_sum['Date'] = pd.to_datetime(existing_sum['Date']).dt.date
        summed['Date'] = pd.to_datetime(summed['Date']).dt.date
        summed = pd.concat([existing_sum, summed], ignore_index=True).drop_duplicates(subset=['Date'], keep='last')

    summed.to_csv(sum_output_file, index=False)
    print(f"Simple format file saved: {sum_output_file}")
    return combined, summed

if __name__ == "__main__":
    clean_and_update_steam_data()
