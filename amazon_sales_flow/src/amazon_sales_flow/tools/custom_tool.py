from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field

import pandas as pd
import json
from ydata_profiling import ProfileReport
import matplotlib.pyplot as 
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pickle


################### Custom Tool ##########################################

## --------------------------------------------------
# 1. Data Cleaning Tool
# --------------------------------------------------
class CleanDataToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the raw CSV file")

class CleanDataTool(BaseTool):
    name: str = "clean_data"
    description: str = "Cleans raw data and outputs clean_data.csv"
    args_schema: Type[BaseModel] = CleanDataToolInput

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            df = df.drop_duplicates()
            for col in df.columns:
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('Unknown')
            for col in df.columns:
                if "date" in col.lower():
                    df[col] = pd.to_datetime(df[col], errors="coerce")
            df.to_csv("clean_data.csv", index=False)
            return "clean_data.csv"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# --------------------------------------------------
# 2. Data Profiling Tool
# --------------------------------------------------
class ProfileDataToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the cleaned CSV file")

class ProfileDataTool(BaseTool):
    name: str = "profile_data"
    description: str = "Generates profiling report and dataset schema"
    args_schema: Type[BaseModel] = ProfileDataToolInput

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            # eda_report.html
            profile = ProfileReport(df, title="EDA Report", explorative=True)
            profile.to_file("eda_report.html")

            # dataset_contract.json
            schema = {
                "columns": {col: str(df[col].dtype) for col in df.columns},
                "shape": df.shape
            }
            with open("dataset_contract.json", "w") as f:
                json.dump(schema, f, indent=4)

            return "eda_report.html, dataset_contract.json"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# --------------------------------------------------
# 3. Business Insights Tool
# --------------------------------------------------
class BusinessInsightsToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the cleaned CSV file")

class BusinessInsightsTool(BaseTool):
    name: str = "business_insights"
    description: str = "Generates visualizations and insights.md"
    args_schema: Type[BaseModel] = BusinessInsightsToolInput

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)

            insights = []
            if "Country" in df.columns and "Sales" in df.columns:
                sales_by_country = df.groupby("Country")["Sales"].sum()
                sales_by_country.plot(kind="bar")
                plt.title("Sales by Country")
                plt.savefig("sales_by_country.png")
                plt.close()
                insights.append("✅ Sales by country calculated.")

            if "Year" in df.columns and "Profit" in df.columns:
                profit_by_year = df.groupby("Year")["Profit"].sum()
                profit_by_year.plot(kind="line")
                plt.title("Profit by Year")
                plt.savefig("profit_by_year.png")
                plt.close()
                insights.append("✅ Profit trends over years calculated.")

            with open("insights.md", "w", encoding="utf-8") as f:
                f.write("# Business Insights\n")
                for line in insights:
                    f.write(f"- {line}\n")

            return "insights.md"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# --------------------------------------------------
# 4. Feature Engineering Tool
# --------------------------------------------------
class FeatureEngineeringToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the cleaned CSV file")

class FeatureEngineeringTool(BaseTool):
    name: str = "feature_engineering"
    description: str = "Performs feature engineering and outputs features.csv"
    args_schema: Type[BaseModel] = FeatureEngineeringToolInput

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)

            if "Sales" in df.columns and "Profit" in df.columns:
                df["Loss"] = df["Sales"] - df["Profit"]

            if "Quantity" in df.columns and "Sales" in df.columns:
                df["PricePerUnit"] = df["Sales"] / df["Quantity"]

            df.to_csv("features.csv", index=False)
            return "features.csv"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# --------------------------------------------------
# 5. Model Training Tool
# --------------------------------------------------
class ModelTrainingToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the features CSV file")

class ModelTrainingTool(BaseTool):
    name: str = "train_model"
    description: str = "Trains a predictive model and outputs trained_model.pkl"
    args_schema: Type[BaseModel] = ModelTrainingToolInput

    def _run(self, file_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            y = df["Profit"] if "Profit" in df.columns else df.iloc[:, -1]
            X = df.drop(columns=["Profit"], errors="ignore")

            X = pd.get_dummies(X)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            model = RandomForestRegressor()
            model.fit(X_train, y_train)

            with open("trained_model.pkl", "wb") as f:
                pickle.dump(model, f)

            return "trained_model.pkl"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# --------------------------------------------------
# 6. Model Evaluation Tool
# --------------------------------------------------
class ModelEvaluationToolInput(BaseModel):
    file_path: str = Field(..., description="Path to the features CSV file")
    model_path: str = Field(..., description="Path to the trained model file")

class ModelEvaluationTool(BaseTool):
    name: str = "evaluate_model"
    description: str = "Evaluates model performance and outputs evaluation_report.md"
    args_schema: Type[BaseModel] = ModelEvaluationToolInput

    def _run(self, file_path: str, model_path: str) -> str:
        try:
            df = pd.read_csv(file_path)
            y = df["Profit"] if "Profit" in df.columns else df.iloc[:, -1]
            X = df.drop(columns=["Profit"], errors="ignore")
            X = pd.get_dummies(X)

            with open(model_path, "rb") as f:
                model = pickle.load(f)

            scores = cross_val_score(model, X, y, cv=5)
            mean_score = scores.mean()

            with open("evaluation_report.md", "w", encoding="utf-8") as f:
                f.write("# Model Evaluation Report\n")
                f.write(f"- Mean CV Score: {mean_score:.4f}\n")

            return "evaluation_report.md"
        except Exception as e:
            return f"❌ Error: {str(e)}"

# --------------------------------------------------
# 7. Model Card Tool
# --------------------------------------------------
class ModelCardToolInput(BaseModel):
    evaluation_file: str = Field(..., description="Path to evaluation_report.md")

class ModelCardTool(BaseTool):
    name: str = "generate_model_card"
    description: str = "Generates model_card.md for documentation"
    args_schema: Type[BaseModel] = ModelCardToolInput

    def _run(self, evaluation_file: str) -> str:
        try:
            with open(evaluation_file, "r", encoding="utf-8") as f:
                evaluation = f.read()

            with open("model_card.md", "w", encoding="utf-8") as f:
                f.write("# Model Card\n")
                f.write("## Purpose\nPredict sales/profits based on historical data.\n")
                f.write("## Data\nDerived from clean_data.csv and features.csv.\n")
                f.write("## Method\nRandomForestRegressor trained with train/test split and CV.\n")
                f.write("## Performance\n")
                f.write(evaluation + "\n")
                f.write("## Limitations\nModel may fail with unseen categories or missing values.\n")
                f.write("## Ethical Considerations\nUse only for business forecasting.\n")

            return "model_card.md"
        except Exception as e:
            return f"❌ Error: {str(e)}"