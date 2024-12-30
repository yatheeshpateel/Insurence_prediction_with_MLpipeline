# Insurence_prediction_with_MLpipeline

## Project Description
This project aims to predict insurance costs using a machine learning pipeline. The goal is to build a model that can accurately estimate the insurance charges based on various features such as age, sex, BMI, children, smoker status, and region.

## Tools Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Workflow of the ML Pipeline
1. **Data Ingestion**: Collect the dataset and load it into the project.
2. **Data Transformation**: Clean the data, handle missing values, encode categorical variables, and normalize numerical features.
3. **Feature Engineering**: Create new features based on domain knowledge and select the most relevant ones.
4. **Model Training**: Train various machine learning models using the processed data.
5. **Model Evaluation**: Evaluate the models using metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), and R-squared.
6. **Prediction**: Use the trained model to make predictions on new data.
7. **Metrics**: Analyze the performance of the model using evaluation metrics and visualize the results.

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/Insurence_prediction_with_MLpipeline.git
    ```
2. Navigate to the project directory:
    ```bash
    cd Insurence_prediction_with_MLpipeline
    ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage
1. Prepare your dataset and place it in the `data` directory.
2. Run the data ingestion script:
    ```bash
    python data_ingestion.py
    ```
3. Run the data transformation script:
    ```bash
    python data_transformation.py
    ```
4. Run the feature engineering script:
    ```bash
    python feature_engineering.py
    ```
5. Train the model:
    ```bash
    python model_trainer.py
    ```
6. Evaluate the model:
    ```bash
    python model_evaluation.py
    ```
7. Make predictions:
    ```bash
    python prediction.py
    ```

## Contributing
1. Fork the repository.
2. Create a new branch:
    ```bash
    git checkout -b feature-branch
    ```
3. Make your changes and commit them:
    ```bash
    git commit -m "Description of changes"
    ```
4. Push to the branch:
    ```bash
    git push origin feature-branch
    ```
5. Open a pull request.

## Output

### Home Page
![Home Page](project%20output%20images/home.jpg)

### Model Train Page
![Train Page](project%20output%20images/train%20page.jpg)

### Prediction Page
![Prediction Page](project%20output%20images/predict%20page.jpg)

### BMI Calculator Page
![BMI Calculator Page](project%20output%20images/bmi%20calculation%20page.jpg)

### Recommendation Page
![Recommendation Page](project%20output%20images/recommendation%20page.jpg)
