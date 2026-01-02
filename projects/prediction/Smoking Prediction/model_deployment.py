
#? STAGE 8: MODEL DEPLOYMENT 

from fastapi import FastAPI, HTTPException, Path, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import os
import sys
import logging
from datetime import datetime
from dotenv import load_dotenv
from fastapi.openapi.utils import get_openapi
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import PolynomialFeatures
from typing import Optional, List, Dict
from contextlib import asynccontextmanager
import socket
import uvicorn
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import HTMLResponse
import json
from .feature_engineering import FeatureEngineer

# Configure logging to both file and console with maximum verbosity
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
API_LOG_DIR = os.path.join(LOG_DIR, 'api')
DEPLOYMENT_LOG_DIR = os.path.join(LOG_DIR, 'deployment')
ERROR_LOG_DIR = os.path.join(LOG_DIR, 'errors')

# Create log directories if they don't exist
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(API_LOG_DIR, exist_ok=True)
os.makedirs(DEPLOYMENT_LOG_DIR, exist_ok=True)
os.makedirs(ERROR_LOG_DIR, exist_ok=True)

# Configure logging with organized file structure
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(API_LOG_DIR, f'api_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.FileHandler(os.path.join(DEPLOYMENT_LOG_DIR, f'deployment_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler(sys.stdout)
    ]
)

# Configure error logging separately
error_handler = logging.FileHandler(os.path.join(ERROR_LOG_DIR, f'error_{datetime.now().strftime("%Y%m%d")}.log'))
error_handler.setLevel(logging.ERROR)
error_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logging.getLogger().addHandler(error_handler)

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Define model path and dictionary to hold loaded models
MODEL_PATH = os.getenv("MODEL_PATH", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models")))
logger.info(f"Using model path: {MODEL_PATH}")
models = {}
model_parameters = {}

# Define best models to be loaded for deployment
BEST_MODELS = {
    "ml_olympiad_improved_final": "ML Olympiad – Improved XGBoost",
    "archive_improved_final": "Archive – Improved Ensemble"
}

# Default model parameters
DEFAULT_MODEL_PARAMETERS = {
    "confidence_threshold": 0.5,
    "class_weights": {"0": 1.0, "1": 1.0},
    "health_indicator_thresholds": {
        "bmi": {"low": 18.5, "high": 25.0},
        "liver_function": {"low": 10.0, "high": 50.0},
        "cardiovascular_risk": {"low": 1.0, "high": 5.0},
        "metabolic_index": {"low": 0.5, "high": 2.5}
    }
}

class ModelParameters(BaseModel):
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    class_weights: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"0": 1.0, "1": 1.0}
    )
    health_indicator_thresholds: Optional[Dict[str, Dict[str, float]]] = Field(
        default_factory=lambda: DEFAULT_MODEL_PARAMETERS["health_indicator_thresholds"]
    )

    class Config:
        json_schema_extra = {
            "example": DEFAULT_MODEL_PARAMETERS
        }

# Define lifespan to load models and handle startup logging
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        logger.info(f"Starting model loading from {MODEL_PATH}")
        if not os.path.exists(MODEL_PATH):
            error_msg = f"Model directory not found at {MODEL_PATH}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        model_files = [f for f in os.listdir(MODEL_PATH) if f.endswith('.pkl')]
        logger.info(f"Found model files: {model_files}")
        
        if not model_files:
            error_msg = f"No .pkl model files found in {MODEL_PATH}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        for model_file in model_files:
            model_name = model_file.replace('.pkl', '')
            if model_name in BEST_MODELS:
                model_path = os.path.join(MODEL_PATH, model_file)
                try:
                    logger.info(f"Loading model {model_name} from {model_path}")
                    model_artifacts = joblib.load(model_path)
                    models[model_name] = model_artifacts['model']
                    logger.info(f"Successfully loaded model: {model_name}")
                except Exception as e:
                    logger.error(f"Error loading model {model_name}: {str(e)}", exc_info=True)
                    raise
                    
        if not models:
            error_msg = f"No best models found for deployment in {MODEL_PATH}. Expected models: {list(BEST_MODELS.keys())}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
        # Startup logging
        logger.info("=== Server Starting ===")
        logger.info(get_ip())
        logger.info("You can access the API at:")
        logger.info("    http://127.0.0.1:8000")
        logger.info("    http://localhost:8000")
        logger.info("API documentation available at:")
        logger.info("    http://127.0.0.1:8000/docs")
        logger.info("    http://localhost:8000/docs")
        logger.info("Try both URLs if one doesn't work")
        
        logger.info("All models loaded successfully. Ready to serve.")
    except Exception as e:
        logger.error(f"Error during startup: {str(e)}", exc_info=True)
        raise e
    yield
    # Cleanup
    logger.info("Cleaning up models")
    models.clear()

# Initialize FastAPI app with lifespan
app = FastAPI(
    lifespan=lifespan,
    title="Smoking Status Prediction API",
    description="API for predicting smoking status using machine learning models",
    version="2.0.0",
    docs_url=None,
    redoc_url=None
)

# Update CORS middleware with more specific origins and headers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Add socket info logging
def get_ip():
    try:
        # Get all network interfaces
        hostname = socket.gethostname()
        local_ip = socket.gethostbyname(hostname)
        return f"Hostname: {hostname}, Local IP: {local_ip}"
    except Exception as e:
        return f"Could not determine IP: {str(e)}"

# Custom OpenAPI schema
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title="Smoking Status Prediction API",
        version="2.0.0",
        description="**API for predicting smoking status using best-performing or ensemble ML models**",
        routes=app.routes,
    )

    # Define tags with descriptions and colors
    openapi_schema["tags"] = [
        {
            "name": "Root",
            "description": "**Root endpoint operations**",
            "x-tag-style": {"background-color": "#FFEB3B"}
        },
        {
            "name": "Models",
            "description": "**Model listing operations**",
            "x-tag-style": {"background-color": "#FF69B4"}
        },
        {
            "name": "Health",
            "description": "**Health check operations**",
            "x-tag-style": {"background-color": "#4CAF50"}
        },
        {
            "name": "Predictions",
            "description": "**Smoking status prediction operations**",
            "x-tag-style": {"background-color": "#2196F3"}
        },
        {
            "name": "Feature Engineering",
            "description": "**Feature engineering rules management**",
            "x-tag-style": {"background-color": "#9C27B0"}
        }
    ]

    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

# Define input schema
class SmokingPredictionInput(BaseModel):
    height_cm: float = Field(..., alias="height(cm)")
    weight_kg: float = Field(..., alias="weight(kg)")
    waist_cm: float = Field(..., alias="waist(cm)")
    age: float
    ALT: float
    Gtp: float
    HDL: float
    LDL: float = Field(0.0)
    Cholesterol: float = Field(0.0)
    systolic: float
    relaxation: float
    hemoglobin: float
    serum_creatinine: float = Field(..., alias="serum creatinine")
    triglyceride: float
    AST: Optional[float] = Field(0.0)
    dental_caries: Optional[int] = Field(0, alias="dental caries")
    eyesight_right: Optional[float] = Field(0.0, alias="eyesight(right)")
    eyesight_left: Optional[float] = Field(0.0, alias="eyesight(left)")
    fasting_blood_sugar: Optional[float] = Field(0.0, alias="fasting blood sugar")

    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "height(cm)": 170.0,
                "weight(kg)": 70.0,
                "waist(cm)": 85.0,
                "eyesight(left)": 1.0,
                "eyesight(right)": 1.0,
                "age": 35.0,
                "ALT": 25.0,
                "AST": 20.0,
                "Gtp": 30.0,
                "HDL": 50.0,
                "LDL": 100.0,
                "Cholesterol": 180.0,
                "dental caries": 0,
                "fasting blood sugar": 90.0,
                "relaxation": 80.0,
                "serum creatinine": 1.0,
                "triglyceride": 150.0,
                "hemoglobin": 15.0,
                "systolic": 120.0
            }
        }

# Root endpoint with enhanced response
@app.get("/", tags=["Root"], response_model=dict)
async def root():
    """Root endpoint with detailed API information and status"""
    try:
        network_info = get_ip()
        logger.info(f"Root endpoint accessed. {network_info}")
        
        response_data = {
            "status": "success",
            "api_info": {
                "name": "Enhanced Smoking Prediction API",
                "version": "2.0.0",
                "description": "Machine Learning API for Smoking Status Prediction"
            },
            "models": {
                "available": list(models.keys()),
                "total_count": len(models),
                "model_path": MODEL_PATH
            },
            "endpoints": {
                "documentation": "/docs",
                "health_check": "/health",
                "models_list": "/models",
                "prediction": "/predict/{model_name}"
            },
            "server_info": {
                "status": "healthy",
                "network": network_info,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return response_data
        
    except Exception as e:
        error_msg = f"Error accessing root endpoint: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail={"error": error_msg})

# Health check endpoint
@app.get("/health", tags=["Health"])
async def health_check():
    logger.info("Health check endpoint accessed")
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "model_path": MODEL_PATH,
        "timestamp": datetime.now().isoformat()
    }

# Endpoint to list models
@app.get("/models", tags=["Models"])
async def list_models():
    logger.info("Models endpoint accessed")
    return {
        "available_models": BEST_MODELS,
        "loaded_models": list(models.keys()),
        "total": len(BEST_MODELS),
        "model_path": MODEL_PATH
    }

# Prediction endpoint
@app.post("/predict/{model_name}", tags=["Predictions"])
async def predict(
    model_name: str = Path(
        ...,
        description="Available models: ml_olympiad_improved_final, archive_improved_final"
    ),
    input_data: SmokingPredictionInput = Body(...)
):
    logger.info(f"Prediction requested for model: {model_name}")
    try:
        # Clean up model name
        model_name = model_name.strip()
        
        if model_name not in models:
            error_msg = f"Model '{model_name}' not found. Available models: {list(models.keys())}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail={"error": error_msg})

        # Get model parameters or use defaults
        model_params = model_parameters.get(model_name, DEFAULT_MODEL_PARAMETERS)
        confidence_threshold = model_params["confidence_threshold"]
        health_thresholds = model_params["health_indicator_thresholds"]

        # Convert input data to DataFrame
        input_dict = input_data.dict(by_alias=True)
        logger.debug(f"Raw input data: {input_dict}")
        data = pd.DataFrame([input_dict])

        try:
            # 1. Initialize all required numeric columns with safe defaults
            default_values = {
                'systolic': data.get('systolic', [0.0])[0],
                'triglyceride': data.get('triglyceride', [0.0])[0],
                'HDL': max(data.get('HDL', [1.0])[0], 1.0),  # Ensure HDL is at least 1
                'LDL': data.get('LDL', [0.0])[0],
                'AST': data.get('AST', [0.0])[0],
                'ALT': data.get('ALT', [0.0])[0],
                'Gtp': data.get('Gtp', [0.0])[0],
                'fasting blood sugar': data.get('fasting blood sugar', [0.0])[0]
            }

            # Update DataFrame with safe values
            for col, value in default_values.items():
                if pd.isna(value):
                    data[col] = 0.0 if col != 'HDL' else 1.0
                else:
                    data[col] = value

            logger.debug("Initialized features with safe values")

            # 2. Calculate basic health indicators
            data['bmi'] = data['weight(kg)'] / ((data['height(cm)']/100) ** 2)
            data['liver_function'] = (data['AST'] + data['ALT'] + data['Gtp']) / 3
            data['cardiovascular_risk'] = (data['systolic'] * data['triglyceride']) / data['HDL']
            data['metabolic_index'] = (data['fasting blood sugar'] * data['bmi']) / data['HDL']

            # 3. Calculate health status indicators
            data['bmi_status'] = ((data['bmi'] >= health_thresholds['bmi']['low']) & 
                               (data['bmi'] <= health_thresholds['bmi']['high'])).astype(int)
            
            data['liver_status'] = ((data['liver_function'] >= health_thresholds['liver_function']['low']) & 
                                 (data['liver_function'] <= health_thresholds['liver_function']['high'])).astype(int)
            
            data['cv_risk_status'] = ((data['cardiovascular_risk'] >= health_thresholds['cardiovascular_risk']['low']) & 
                                   (data['cardiovascular_risk'] <= health_thresholds['cardiovascular_risk']['high'])).astype(int)
            
            data['metabolic_status'] = ((data['metabolic_index'] >= health_thresholds['metabolic_index']['low']) & 
                                    (data['metabolic_index'] <= health_thresholds['metabolic_index']['high'])).astype(int)

            # 4. Calculate additional ratios
            data['hdl_ldl_ratio'] = data['HDL'] / (data['LDL'] + 1)
            data['ast_alt_ratio'] = data['AST'] / (data['ALT'] + 1)
            data['bp_ratio'] = data['systolic'] / (data['relaxation'] + 1)

            # 5. Generate polynomial features based on model type
            if model_name == 'ml_olympiad_improved_final':
                key_features = ['bmi', 'liver_function', 'cardiovascular_risk', 'metabolic_index']
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = poly.fit_transform(data[key_features])
                for i in range(poly_features.shape[1]):
                    data[f'health_poly_{i}'] = poly_features[:, i]
            else:  # archive_improved_final
                # For archive model, we only need specific polynomial features
                key_features = ['bmi', 'liver_function', 'cardiovascular_risk']
                poly = PolynomialFeatures(degree=2, include_bias=False)
                poly_features = poly.fit_transform(data[key_features])
                # Only keep required polynomial features (0, 4, 5)
                data['health_poly_0'] = poly_features[:, 0]  # First feature
                data['health_poly_4'] = poly_features[:, 4]  # Fifth feature
                data['health_poly_5'] = poly_features[:, 5]  # Sixth feature

            logger.debug("All features calculated successfully")
            logger.debug(f"Available features: {list(data.columns)}")

        except Exception as e:
            error_msg = f"Error calculating health indicators: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Data state: {data.to_dict()}")
            raise HTTPException(status_code=400, detail={"error": error_msg})

        # Select features based on model type
        if model_name == 'ml_olympiad_improved_final':
            required_features = [
                "age", "height(cm)", "weight(kg)", "systolic", "relaxation",
                "Cholesterol", "triglyceride", "HDL", "LDL", "hemoglobin",
                "serum creatinine", "AST", "ALT", "Gtp", "dental caries",
                "health_poly_0", "health_poly_1", "health_poly_4", "health_poly_13",
                "bmi", "liver_function", "hdl_ldl_ratio", "ast_alt_ratio"
            ]
        else:  # archive_improved_final
            required_features = [
                "age", "height(cm)", "weight(kg)", "waist(cm)", "systolic",
                "relaxation", "fasting blood sugar", "triglyceride", "HDL",
                "LDL", "hemoglobin", "serum creatinine", "ALT", "Gtp",
                "dental caries", "health_poly_0", "health_poly_4", "health_poly_5",
                "bmi", "liver_function", "hdl_ldl_ratio", "ast_alt_ratio"
            ]

        # Create a new DataFrame with only required features in correct order
        prediction_data = pd.DataFrame()
        for feature in required_features:
            if feature not in data.columns:
                error_msg = f"Missing required feature: {feature}"
                logger.error(error_msg)
                raise HTTPException(status_code=400, detail={"error": error_msg})
            prediction_data[feature] = data[feature]

        logger.debug(f"Final features for prediction: {list(prediction_data.columns)}")
        
        # Make prediction
        model = models[model_name]
        prediction = model.predict(prediction_data)[0]
        probabilities = model.predict_proba(prediction_data)[0]
        confidence = float(max(probabilities))
        
        # Apply confidence threshold
        adjusted_prediction = 1 if confidence >= confidence_threshold and prediction == 1 else 0
        
        result = {
            "model_used": BEST_MODELS[model_name],
            "prediction": int(adjusted_prediction),
            "label": "Smoker" if adjusted_prediction == 1 else "Non-smoker",
            "confidence": f"{confidence:.2%}",
            "confidence_threshold": confidence_threshold,
            "health_indicators": {
                "bmi_status": bool(data['bmi'].iloc[0] >= health_thresholds['bmi']['low'] and 
                                 data['bmi'].iloc[0] <= health_thresholds['bmi']['high']),
                "liver_status": bool(data['liver_function'].iloc[0] >= health_thresholds['liver_function']['low'] and 
                                   data['liver_function'].iloc[0] <= health_thresholds['liver_function']['high']),
                "cardiovascular_status": bool(data['cardiovascular_risk'].iloc[0] >= health_thresholds['cardiovascular_risk']['low'] and 
                                           data['cardiovascular_risk'].iloc[0] <= health_thresholds['cardiovascular_risk']['high']),
                "metabolic_status": bool(data['metabolic_index'].iloc[0] >= health_thresholds['metabolic_index']['low'] and 
                                      data['metabolic_index'].iloc[0] <= health_thresholds['metabolic_index']['high'])
            },
            "calculated_features": {
                "bmi": float(data['bmi'].iloc[0]),
                "liver_function": float(data['liver_function'].iloc[0]),
                "cardiovascular_risk": float(data['cardiovascular_risk'].iloc[0]),
                "metabolic_index": float(data['metabolic_index'].iloc[0])
            },
            "model_type": "XGBoost" if model_name == "ml_olympiad_improved_final" else "Ensemble",
            "features_used": required_features
        }
        
        return result

    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Error making prediction: {str(e)}"
        logger.error(error_msg)
        logger.error("Full traceback: ", exc_info=True)
        raise HTTPException(status_code=500, detail={"error": error_msg})

# Feature engineering rules models
class FeatureEngineeringRule(BaseModel):
    name: str
    formula: str
    enabled: bool = True
    description: Optional[str] = None
    degree: Optional[int] = Field(default=2, ge=1, le=3)

class FeatureEngineeringRules(BaseModel):
    health_indicators: List[FeatureEngineeringRule]
    polynomial_features: List[str]
    feature_ratios: List[FeatureEngineeringRule]
    polynomial_degree: int = Field(default=2, ge=1, le=3)

# Endpoint to update feature engineering rules
@app.put("/feature-engineering/rules", tags=["Feature Engineering"])
async def update_feature_engineering_rules(rules: FeatureEngineeringRules):
    """
    Update feature engineering rules including:
    - Health indicator calculations
    - Polynomial feature generation rules
    - Feature ratio calculations
    """
    try:
        # Save the rules to a configuration file
        rules_dict = rules.dict()
        os.makedirs("config", exist_ok=True)
        with open("config/feature_engineering_rules.json", "w") as f:
            json.dump(rules_dict, f, indent=4)
        
        return {
            "status": "success",
            "message": "Feature engineering rules updated successfully",
            "rules": rules_dict
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update feature engineering rules: {str(e)}"
        )

@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" type="text/css" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
        <link rel="shortcut icon" href="/favicon.ico">
        <title>Smoking Status Prediction API - Swagger UI</title>
        <style>
            /* Hide the filter input */
            .swagger-ui .filter-container {
                display: none !important;
            }
            
            /* API Description and Tag Description Styling */
            .title, .description, .opblock-tag-section h3 span, .opblock-tag-section .markdown p {
                font-weight: bold !important;
                font-size: calc(100% + 2pt) !important;
            }
            .info__title {
                font-weight: bold !important;
                font-size: calc(100% + 4pt) !important;
            }

            /* Method button styling */
            .swagger-ui .opblock-summary-method {
                min-width: 80px !important;
                text-align: center !important;
                border-radius: 3px !important;
                padding: 6px 15px !important;
            }

            /* Root endpoint (Yellow) */
            .swagger-ui #operations-Root-get .opblock-summary-method,
            .swagger-ui #operations-Root-get .btn,
            .swagger-ui #operations-Root-get .execute,
            .swagger-ui #operations-Root-get .try-out__btn {
                background: #FFD700 !important;
                border-color: #FFD700 !important;
                color: #000000 !important;
            }
            .swagger-ui #operations-Root-get.is-open .opblock-summary {
                border-color: #FFD700 !important;
            }

            /* Models endpoint (Purple) */
            .swagger-ui #operations-Models-get .opblock-summary-method,
            .swagger-ui #operations-Models-get .btn,
            .swagger-ui #operations-Models-get .execute,
            .swagger-ui #operations-Models-get .try-out__btn {
                background: #9B59B6 !important;
                border-color: #9B59B6 !important;
                color: #FFFFFF !important;
            }
            .swagger-ui #operations-Models-get.is-open .opblock-summary {
                border-color: #9B59B6 !important;
            }

            /* Health endpoint (Green) */
            .swagger-ui #operations-Health-get .opblock-summary-method,
            .swagger-ui #operations-Health-get .btn,
            .swagger-ui #operations-Health-get .execute,
            .swagger-ui #operations-Health-get .try-out__btn {
                background: #2ECC71 !important;
                border-color: #2ECC71 !important;
                color: #FFFFFF !important;
            }
            .swagger-ui #operations-Health-get.is-open .opblock-summary {
                border-color: #2ECC71 !important;
            }

            /* Predictions endpoint (Orange) */
            .swagger-ui #operations-Predictions-post .opblock-summary-method,
            .swagger-ui #operations-Predictions-post .btn,
            .swagger-ui #operations-Predictions-post .execute,
            .swagger-ui #operations-Predictions-post .try-out__btn {
                background: #E67E22 !important;
                border-color: #E67E22 !important;
                color: #FFFFFF !important;
            }
            .swagger-ui #operations-Predictions-post.is-open .opblock-summary {
                border-color: #E67E22 !important;
            }

            /* Feature Engineering endpoint (Pink) */
            .swagger-ui #operations-FeatureEngineering-put .opblock-summary-method,
            .swagger-ui #operations-FeatureEngineering-put .btn,
            .swagger-ui #operations-FeatureEngineering-put .execute,
            .swagger-ui #operations-FeatureEngineering-put .try-out__btn {
                background: #FF1493 !important;
                border-color: #FF1493 !important;
                color: #FFFFFF !important;
            }
            .swagger-ui #operations-FeatureEngineering-put.is-open .opblock-summary {
                border-color: #FF1493 !important;
            }

            /* Hide operation IDs */
            .swagger-ui .opblock-summary-operation-id {
                display: none !important;
            }

            /* Button hover effects */
            .swagger-ui #operations-Root-get .opblock-summary-method:hover,
            .swagger-ui #operations-Root-get .btn:hover {
                background: #FFE44D !important;
            }
            .swagger-ui #operations-Models-get .opblock-summary-method:hover,
            .swagger-ui #operations-Models-get .btn:hover {
                background: #A569BD !important;
            }
            .swagger-ui #operations-Health-get .opblock-summary-method:hover,
            .swagger-ui #operations-Health-get .btn:hover {
                background: #27AE60 !important;
            }
            .swagger-ui #operations-Predictions-post .opblock-summary-method:hover,
            .swagger-ui #operations-Predictions-post .btn:hover {
                background: #D35400 !important;
            }
            .swagger-ui #operations-FeatureEngineering-put .opblock-summary-method:hover,
            .swagger-ui #operations-FeatureEngineering-put .btn:hover {
                background: #FF69B4 !important;
            }

            /* Active button styles */
            .swagger-ui .try-out__btn:active {
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.2) !important;
            }
        </style>
    </head>
    <body>
        <div id="swagger-ui"></div>
        <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
        <script>
            window.onload = () => {
                const ui = SwaggerUIBundle({
                    url: '/openapi.json',
                    dom_id: '#swagger-ui',
                    deepLinking: true,
                    displayRequestDuration: true,
                    filter: false,
                    operationsSorter: 'alpha',
                    presets: [
                        SwaggerUIBundle.presets.apis,
                        SwaggerUIBundle.SwaggerUIStandalonePreset
                    ],
                    plugins: [
                        SwaggerUIBundle.plugins.DownloadUrl
                    ]
                });

                // Additional styling for buttons after UI loads
                setTimeout(() => {
                    const applyColors = () => {
                        // Root endpoint (Yellow)
                        const rootElements = document.querySelectorAll('#operations-Root-get button');
                        rootElements.forEach(el => {
                            el.style.setProperty('background', '#FFD700', 'important');
                            el.style.setProperty('border-color', '#FFD700', 'important');
                            el.style.setProperty('color', '#000000', 'important');
                        });

                        // Models endpoint (Purple)
                        const modelsElements = document.querySelectorAll('#operations-Models-get button');
                        modelsElements.forEach(el => {
                            el.style.setProperty('background', '#9B59B6', 'important');
                            el.style.setProperty('border-color', '#9B59B6', 'important');
                            el.style.setProperty('color', '#FFFFFF', 'important');
                        });

                        // Health endpoint (Green)
                        const healthElements = document.querySelectorAll('#operations-Health-get button');
                        healthElements.forEach(el => {
                            el.style.setProperty('background', '#2ECC71', 'important');
                            el.style.setProperty('border-color', '#2ECC71', 'important');
                            el.style.setProperty('color', '#FFFFFF', 'important');
                        });

                        // Predictions endpoint (Orange)
                        const predictionElements = document.querySelectorAll('#operations-Predictions-post button');
                        predictionElements.forEach(el => {
                            el.style.setProperty('background', '#E67E22', 'important');
                            el.style.setProperty('border-color', '#E67E22', 'important');
                            el.style.setProperty('color', '#FFFFFF', 'important');
                        });
                        
                        // Feature Engineering endpoint (Pink)
                        const featureElements = document.querySelectorAll('#operations-FeatureEngineering-put button');
                        featureElements.forEach(el => {
                            el.style.setProperty('background', '#FF1493', 'important');
                            el.style.setProperty('border-color', '#FF1493', 'important');
                            el.style.setProperty('color', '#FFFFFF', 'important');
                        });
                    };

                    // Apply colors initially
                    applyColors();

                    // Reapply colors when sections are expanded
                    const observer = new MutationObserver(applyColors);
                    observer.observe(document.getElementById('swagger-ui'), {
                        childList: true,
                        subtree: true
                    });
                }, 100);
            };
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


class ModelDeployment:
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model = models.get('smoking_status', None)  # Assuming models is defined elsewhere
    
    async def predict(self, data: Dict):
        """Make predictions using the deployed model"""
        try:
            # Convert input data to DataFrame
            df = pd.DataFrame([data])
            
            # Apply feature engineering
            df = self.feature_engineer.transform(df)
            
            # Make prediction
            prediction = self.model.predict(df)[0]
            probability = self.model.predict_proba(df)[0][1]
            
            return {
                "prediction": int(prediction),
                "probability": float(probability),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Prediction failed: {str(e)}"
            )


@app.patch("/models/{model_name}/parameters", tags=["Models"])
async def update_model_parameters(
    model_name: str = Path(
        ...,
        description="Model name to update parameters for"
    ),
    parameters: ModelParameters = Body(...)
):
    """
    Update model parameters including:
    - Confidence threshold for predictions
    - Class weights for model predictions
    - Health indicator thresholds
    """
    try:
        if model_name not in models:
            raise HTTPException(
                status_code=404,
                detail=f"Model '{model_name}' not found. Available models: {list(models.keys())}"
            )

        # Initialize parameters for model if not exists
        if model_name not in model_parameters:
            model_parameters[model_name] = DEFAULT_MODEL_PARAMETERS.copy()
        
        # Update only provided parameters
        updated_params = parameters.dict(exclude_unset=True)
        model_parameters[model_name].update(updated_params)
        
        logger.info(f"Updated parameters for model {model_name}: {updated_params}")
        
        return {
            "status": "success",
            "message": f"Parameters updated successfully for model: {model_name}",
            "model": model_name,
            "updated_parameters": updated_params,
            "current_parameters": model_parameters[model_name]
        }
    except HTTPException:
        raise
    except Exception as e:
        error_msg = f"Failed to update model parameters: {str(e)}"
        logger.error(error_msg)
        raise HTTPException(status_code=500, detail={"error": error_msg})

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)