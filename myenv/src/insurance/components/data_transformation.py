import os
from src.insurance import logger
from sklearn.model_selection import train_test_split
import pandas as pd
from src.insurance.entity.config_entity import DataTransformationConfig
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def perform_eda(self, data: pd.DataFrame):
        """Perform basic EDA."""
        logger.info("Starting EDA process...")
        
        # Check for null values
        logger.info("Null value count:\n{}".format(data.isnull().sum()))
        
        # Correlation heatmap (only on numeric columns)
        numeric_data = data.select_dtypes(include=['number'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_data.corr(), annot=True, cmap="coolwarm")
        plt.title("Feature Correlation")
        plt.savefig(os.path.join(self.config.root_dir, "correlation_heatmap.png"))
        logger.info("Saved correlation heatmap.")
        
        # Visualize distributions
        data.hist(bins=30, figsize=(15, 10))
        plt.savefig(os.path.join(self.config.root_dir, "feature_distributions.png"))
        logger.info("Saved feature distribution plots.")

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by encoding categorical variables."""
        logger.info("Starting preprocessing...")

        # Encode binary categorical variables
        label_encoders = {}
        for col in ['sex', 'smoker']:
            if col in data.columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                label_encoders[col] = le
                logger.info(f"Encoded '{col}' with mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")

        # One-hot encode multi-class categorical variables like 'region'
        if 'region' in data.columns:
            data = pd.get_dummies(data, columns=['region'], drop_first=True)
            logger.info("One-hot encoded 'region' column.")

        # Log final data types after encoding
        logger.info(f"Data types after encoding:\n{data.dtypes}")
        logger.info("Completed preprocessing.")
        return data

    def train_test_spliting(self):
        # Read the dataset
        data = pd.read_csv(self.config.data_path)
        logger.info("Loaded data. First few rows:")
        logger.info(data.head())

        # Perform EDA
        self.perform_eda(data)

        # Preprocess the data
        data = self.preprocess_data(data)

        # Split the data into training and test sets
        train, test = train_test_split(data, test_size=0.25, random_state=42)

        # Save preprocessed datasets
        train.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Saved train and test datasets.")
        logger.info("Split data into training and test sets")
        logger.info(f"Training set shape: {train.shape}")
        logger.info(f"Testing set shape: {test.shape}")

        print(train.shape)
        print(test.shape)
