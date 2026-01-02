
#? STAGE 3: FEATURE ENGINEERING

import json
import numpy as np
import pandas as pd
from typing import Dict, List
from sklearn.preprocessing import PolynomialFeatures
from pathlib import Path

class FeatureEngineer:
    def __init__(self):
        self.rules = self._load_rules()
        
    def _load_rules(self) -> Dict:
        """Load feature engineering rules from config file"""
        config_path = Path("config/feature_engineering_rules.json")
        if not config_path.exists():
            return self._get_default_rules()
        
        with open(config_path, "r") as f:
            return json.load(f)
    
    def _get_default_rules(self) -> Dict:
        """Default feature engineering rules if no config exists"""
        return {
            "health_indicators": [
                {
                    "name": "bmi_health_index",
                    "formula": "weight / (height ** 2)",
                    "enabled": True,
                    "description": "BMI-based health indicator"
                }
            ],
            "polynomial_features": ["age", "weight", "height"],
            "feature_ratios": [
                {
                    "name": "age_bmi_ratio",
                    "formula": "age / bmi_health_index",
                    "enabled": True
                }
            ],
            "polynomial_degree": 2
        }
    
    def create_health_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate health indicator features based on configured rules"""
        result = df.copy()
        
        for rule in self.rules["health_indicators"]:
            if rule["enabled"]:
                try:
                    result[rule["name"]] = eval(rule["formula"], 
                                              {"__builtins__": None}, 
                                              {**dict(result), "np": np})
                except Exception as e:
                    print(f"Failed to calculate {rule['name']}: {str(e)}")
        
        return result
    
    def create_polynomial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate polynomial features for specified columns"""
        result = df.copy()
        features_to_transform = [col for col in self.rules["polynomial_features"] 
                               if col in df.columns]
        
        if not features_to_transform:
            return result
            
        poly = PolynomialFeatures(
            degree=self.rules["polynomial_degree"],
            include_bias=False
        )
        
        poly_features = poly.fit_transform(df[features_to_transform])
        feature_names = poly.get_feature_names_out(features_to_transform)
        
        # Add only the interaction terms and higher degree terms
        for i, name in enumerate(feature_names[len(features_to_transform):], 
                               start=len(features_to_transform)):
            result[f"poly_{name}"] = poly_features[:, i]
        
        return result
    
    def create_feature_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate feature ratios based on configured rules"""
        result = df.copy()
        
        for rule in self.rules["feature_ratios"]:
            if rule["enabled"]:
                try:
                    result[rule["name"]] = eval(rule["formula"], 
                                              {"__builtins__": None}, 
                                              {**dict(result), "np": np})
                except Exception as e:
                    print(f"Failed to calculate {rule['name']}: {str(e)}")
        
        return result
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering transformations"""
        result = df.copy()
        result = self.create_health_indicators(result)
        result = self.create_polynomial_features(result)
        result = self.create_feature_ratios(result)
        return result