# Census Classification Project

This project involves building a machine learning model to classify census data, deploying the model as an API using FastAPI, and setting up CI/CD with GitHub Actions.

## Project Structure

census_classification/
├── .github/
│ └── workflows/
│ └── ci.yml
├── data/
│ ├── census.csv
│ └── clean_census.csv
├── models/
│ ├── model.joblib
│ └── label_encoders.joblib
├── src/
│ ├── init.py
│ ├── data_cleaning.py
│ ├── model.py
│ └── api.py
├── tests/
│ ├── init.py
│ ├── test_data_cleaning.py
│ ├── test_model.py
│ └── test_api.py
├── .gitignore
├── Dockerfile
├── requirements.txt
├── README.md
├── main.py
└── sanitycheck.py


## Instructions

1. Clone the repository.
2. Set up a virtual environment and install the dependencies:
    ```bash
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```
3. Run the data cleaning script:
    ```bash
    python src/data_cleaning.py
    ```
4. Train the model:
    ```bash
    python src/model.py
    ```
5. Run the FastAPI application:
    ```bash
    uvicorn src.api:app --reload
    ```
6. Run the tests:
    ```bash
    pytest
    ```

## Deployment

1. Create a Heroku/Render account.
2. Connect the repository to Heroku/Render and enable automatic deployments.
3. Configure the application to run the FastAPI app.

## CI/CD

This project uses GitHub Actions for CI/CD. The workflow is defined in `.github/workflows/ci.yml`.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

