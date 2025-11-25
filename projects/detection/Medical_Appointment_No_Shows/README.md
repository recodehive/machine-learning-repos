# Medical Appointment No-Shows Prediction

## Overview
This project implements a machine learning model to predict whether a patient will miss their medical appointment. The model uses various patient features and appointment details to make accurate predictions.

## Dataset
The dataset contains the following information:
- **Total Records**: Approximately 100,000 medical appointments
- **Features**: 14 different features including patient demographics and appointment details
- **Target Variable**: `No-show` (1 = No-show, 0 = Show)

### Dataset Features
1. `PatientId` - Unique patient identifier
2. `AppointmentID` - Unique appointment identifier
3. `Gender` - Patient gender (M/F)
4. `ScheduledDay` - Date when appointment was scheduled
5. `AppointmentDay` - Actual date of appointment
6. `Age` - Patient age in years
7. `Neighbourhood` - Neighbourhood where patient is from
8. `Scholarship` - Indicates if patient is on any scholarship program
9. `Hipertension` - Indicates if patient has hypertension
10. `Diabetes` - Indicates if patient has diabetes
11. `Alcoholism` - Indicates if patient has alcoholism
12. `HandiCap` - Indicates if patient has any handicap
13. `SMS_received` - Indicates if appointment reminder SMS was sent
14. `No-show` - Target variable (0 = Showed up, 1 = No-show)

## Files in This Project
- `model.py` - Main model implementation with preprocessing pipeline and training code
- `requirements.txt` - Python dependencies
- `README.md` - This documentation file

## Data Preprocessing
- Missing values are handled by filling with mean values for numerical columns
- Categorical variables are encoded using LabelEncoder
- Numerical features are scaled using StandardScaler for normalization

## Requirements
- Python 3.7+
- pandas
- numpy
- scikit-learn

## Usage
See `model.py` for implementation details and usage examples.
