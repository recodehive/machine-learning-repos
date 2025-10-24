import pandas as pd
import numpy as np
import os

def process_iiot_data(file_path: str, output_path: str = "Sample_Output.csv"):
    """
    Transforms IIoT sensor data from a wide, high-frequency format to a long,
    aggregated time-series.

    Args:
        file_path (str): The path to the input CSV file containing metadata and
                         high-frequency sensor readings.
        output_path (str): The file path to save the final aggregated CSV.
    """
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return

    ## 1. READ METADATA & DATA
    # Dynamically detect the header line to separate metadata from readings
    with open(file_path, 'r') as f:
        header_line = None
        for i, line in enumerate(f):
            if 'Date' in line and 'Time' in line:
                header_line = i
                break

    # Read metadata from the top of the file down to the header line
    meta = pd.read_csv(file_path, nrows=header_line)
    meta = meta.loc[:, ~meta.columns.str.contains('^Unnamed')]

    # Read high-frequency data from the header line onwards
    hf_data = pd.read_csv(file_path, skiprows=header_line)

    ## 2. CREATE MAPPING
    # Use the metadata to create lookup dictionaries for each sensor
    itemid_map = dict(zip(meta["ItemName"], meta["ItemId"]))
    desc_map   = dict(zip(meta["ItemName"], meta["Comment"]))
    unit_map   = dict(zip(meta["ItemName"], meta["Unit"]))

    ## 3. CONVERT TO LONG FORMAT (MELT)
    # Combine Date, Time, Milli Sec into a single timestamp
    hf_data['timestamp'] = pd.to_datetime(hf_data['Date'] + ' ' + hf_data['Time'], dayfirst=True) + pd.to_timedelta(hf_data['Milli Sec'], unit='ms')

    # Get a list of only the sensor columns
    sensor_cols = [col for col in hf_data.columns if col not in ['Date', 'Time', 'Milli Sec', 'timestamp']]

    # Melt the DataFrame from a wide format to a long format
    hf_long = hf_data.melt(
        id_vars=['timestamp'],
        value_vars=sensor_cols,
        var_name='tag_name',
        value_name='tag_value'
    )

    ## 4. MAP METADATA TO THE LONG DATAFRAME
    # Attach the metadata to each sensor reading using the mapping dictionaries
    hf_long['tag__id'] = hf_long['tag_name'].map(itemid_map)
    hf_long['tag__desc'] = hf_long['tag_name'].map(desc_map)
    hf_long['tag__unit'] = 'mps'

    ## 5.Create a 10-second time bucket for grouping
    hf_long['time_bucket'] = hf_long['timestamp'].dt.floor('10s')

    # Group by the time bucket and sensor metadata, then aggregate using MAX
    agg = (
        hf_long
        .groupby(['time_bucket', 'tag__id', 'tag_name', 'tag__desc', 'tag__unit'], as_index=False)
        .agg(tag__value=('tag_value', 'max'))
    )

    ## 6.Format the time_bucket to match the required output timestamp format
    agg['event_timestamp'] = agg['time_bucket'].dt.strftime('%d/%m/%Y %H:%M:%S')

    out = agg[['event_timestamp', 'tag__id', 'tag_name', 'tag__desc', 'tag__value', 'tag__unit']]

    # Export the final DataFrame
    out.to_csv(output_path, index=False)
    print("Transformation complete. Output saved as {}".format(output_path))


if __name__ == "__main__":
    input_file_path = "/content/Sample Input - Sample.csv"

    # Run the data processing pipeline
    process_iiot_data(input_file_path)