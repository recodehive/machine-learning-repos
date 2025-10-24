### IIoT Data Transformation Pipeline

This project demonstrates a core skill in Machine Learning Engineering: transforming raw, high-frequency sensor data into a structured and aggregated format. The pipeline is crucial for handling the immense volume of data produced by industrial systems.

***

### The Metrics That Matter

The need for this pipeline is driven by data volume.

* Industrial systems generate data every **5 milliseconds**.
* In a 10-second window, each sensor produces `10,000 ms / 5 ms = 2000` data points.
* With 50,000 sensors, this amounts to **100 million** data points every 10 seconds, which is impossible to manage directly.

The pipeline's aggregation step compresses this data, reducing the volume by a `2000:1` ratio for each sensor, making it manageable and efficient for analytics and modeling.

***

### Pipeline Overview

<img width="1024" height="1024" alt="Gemini_Generated_Image_eisvv9eisvv9eisv" src="https://github.com/user-attachments/assets/3dd43204-a0bd-4768-874f-e34587719c85" />


The pipeline is implemented in Python using the Pandas library. The process is a series of transformations:

1.  **Ingestion & Unpivoting**: The pipeline reads the raw, "wide" format data and reshapes it into a "long" format, where each row represents a single sensor reading.
2.  **Aggregation**: The data is grouped into 10-second time buckets, and the **MAX** value is calculated for each sensor, drastically reducing data volume.
3.  **Enrichment**: Sensor metadata is mapped to the aggregated data to provide context (ID, description, etc.).



This process results in a clean, structured dataset ready for use in a time-series database or machine learning model.
