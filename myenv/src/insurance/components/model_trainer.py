import pandas as pd
import os
from src.insurance import logger
from sklearn.ensemble import RandomForestRegressor
import joblib
from src.insurance.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Load the training and testing data
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)

        # Prepare features and target variable
        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]].values.ravel()
        test_y = test_data[[self.config.target_column]].values.ravel()

        # Initialize Random Forest Regressor with parameters from params.yaml
        rf = RandomForestRegressor(
            n_estimators=self.config.n_estimators, 
            max_depth=self.config.max_depth, 
            random_state=42
        )

        # Fit the model
        rf.fit(train_x, train_y)

        # Log model training information
        logger.info(f"Trained Random Forest model with {self.config.n_estimators} estimators and max depth of {self.config.max_depth}")

        # Save the trained model using joblib
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(rf, model_path)

        # Log that the model has been saved
        logger.info(f"Model saved at {model_path}")
