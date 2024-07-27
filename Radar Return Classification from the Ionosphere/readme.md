This project involves classifying radar returns from the ionosphere to distinguish between "good" and "bad" radar signals.

Good: Indicate the presence of some structure in the ionosphere, which is detectable through the radar signals.

Bad: Indicate the signals passed through the ionosphere without any observable structure.

The data was collected by a high-frequency radar system deployed in Goose Bay, Labrador, which utilizes a phased array of 16 antennas and transmits at a power level of approximately 6.4 kilowatts.
The primary targets of this radar system are free electrons in the ionosphere.

Each pulse number is represented by two continuous attributes, corresponding to the complex values returned by the autocorrelation function.
With 17 pulse numbers, this results in 34 continuous attributes.

The dataset is imported from UCI through Python.

The classification of radar returns is crucial for understanding and interpreting ionospheric conditions, which can impact communication systems and satellite operations.

Models Used: Random Forest, XGBoost,Decision Tree, Support Vector Machine (SVM), and Neural Network.

Used SMOTE to balance out the classes. 

Using Cross Validation scores, plotted a Mean Cross-Validation Scores with Standard Deviations Error plot.
