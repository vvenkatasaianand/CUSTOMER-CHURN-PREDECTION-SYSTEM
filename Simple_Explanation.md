# Simple Explanation for Project Review

## 1. Project Overview (Bird's Eye View)

This project is a **Customer Churn Prediction System**. Its purpose is to help a business identify which customers are likely to leave, based on past customer data. The system does this in a complete flow: first it accepts a dataset, then it prepares the data, trains a machine learning model, and finally allows the user to enter customer details and get a churn prediction with an explanation.

In simple terms, the application answers this question: **"Given customer information, is this customer likely to churn or not, and why?"** It is not just a raw prediction tool. It also gives business-friendly outputs such as risk level, important contributing factors, and suggested actions.

The project combines three ideas in one system:

1. **Data handling** for upload and preprocessing
2. **Machine learning** for churn prediction
3. **Local AI explanation** using Ollama for readable summaries

## 2. What the Application Does End-to-End

The application starts when the user uploads a dataset in CSV or JSON format. After upload, the system studies the columns and shows the dataset structure. The user then selects which column is the target column, meaning the value the model should learn to predict, such as `churn`.

Next, the system preprocesses the dataset. At this stage, the user can remove columns that are not useful for prediction, such as IDs or personal identifiers. The backend then stores the cleaned dataset and keeps track of which columns are features and which column is the target.

After preprocessing, the user clicks **Train Model**. The backend then builds a preprocessing-plus-model pipeline and trains an `XGBoost` classifier. Once training is complete, the system shows evaluation metrics such as accuracy, precision, recall, F1-score, ROC-AUC, confusion matrix, and feature importance.

Once the model is trained, the system automatically creates a prediction form based on the trained model schema. The user fills in values for the customer fields, submits the form, and the system predicts:

- whether the customer is likely to churn or not
- the churn probability
- the risk level as Low, Medium, or High
- key contributing factors
- recommended actions
- a readable explanation generated locally through Ollama

## 3. How a User Interacts with It

From the user's point of view, the system is a guided five-step workflow:

1. **Upload**: The user uploads a dataset file.
2. **Preprocess**: The user selects the target column and excludes unwanted columns.
3. **Train**: The user trains the machine learning model and watches live progress.
4. **Predict**: The user fills a dynamically generated form using the trained model schema.
5. **Results**: The user sees the churn prediction, probability, risk level, explanation, and actions.

This is useful in a review because it shows the system is not just a backend script. It is a complete application with a clear user journey from raw data to decision support.

## 4. High-Level Workflow (Step-by-Step Flow of the System)

The high-level flow of the system is:

1. The frontend accepts the uploaded dataset.
2. The backend saves the file and reads it using `pandas`.
3. The backend extracts metadata such as column names, data types, null counts, and sample values.
4. The user chooses the target column and optionally excludes irrelevant columns.
5. The backend removes rows where the target is missing and saves the processed dataset.
6. The training module splits the data into training and test sets.
7. A preprocessing pipeline is applied, where numeric columns are filled with median values and categorical columns are filled and one-hot encoded.
8. The `XGBoost` model is trained on this processed data.
9. The trained pipeline, schema, metrics, and metadata are stored locally on disk.
10. The schema is used to create a prediction form automatically.
11. When the user enters a new customer record, the backend loads the trained pipeline and makes a prediction.
12. The system calculates probability and maps it to a business-friendly risk level.
13. The system extracts important factors from the model, mainly from per-prediction contribution scores.
14. Ollama generates a clear explanation and summary using the computed facts.
15. The user receives the final result on the results screen.

## 5. Core Working of the System

The core logic of the system is based on a strong separation of responsibilities. The **React frontend** controls the user flow and displays each screen. The **FastAPI backend** handles file upload, preprocessing, training, inference, and metadata management. The **machine learning layer** is responsible for building the training pipeline, fitting the model, evaluating it, and performing prediction. The **Ollama layer** is only used to convert model outputs into readable explanations.

The key modules connect in this way:

- `dataset_service.py` handles dataset upload, validation, preprocessing, and preview generation.
- `training_service.py` controls model training and stores the trained artifact and metrics.
- `pipeline.py` builds the preprocessing pipeline and the `XGBoost` model.
- `trainer.py` performs train-test split, fitting, and metric calculation.
- `schema_service.py` returns the schema used to build the prediction form.
- `prediction_service.py` loads the saved model, prepares the input row, predicts churn, assigns risk level, finds key factors, and creates recommended actions.
- `llm_service.py` connects to Ollama and asks the local LLM to generate JSON-based summaries.

So logically, the system works like this: **data comes in, gets cleaned, is learned by the model, and then future customer inputs are passed through the same pipeline to produce a prediction and explanation.**

## 6. Algorithm and Model Explanation (Main Focus)

The main predictive algorithm used in this project is **XGBoost Classifier**.

XGBoost is a gradient boosting algorithm based on decision trees. Instead of relying on a single tree, it builds many trees in sequence. Each new tree tries to correct the mistakes made by the previous trees. Because of this, XGBoost is often more accurate than a single decision tree and usually more powerful than very basic models on structured tabular data.

This project uses XGBoost because churn prediction is a **binary classification problem** on business data with mixed feature types such as numbers and categories. XGBoost is well-suited for this kind of data because it:

- performs strongly on tabular datasets
- handles non-linear relationships better than simple linear models
- works well with medium-sized structured data
- gives useful feature importance information
- provides prediction probabilities, which are important for risk scoring

### Why XGBoost Was Chosen Over Other Options

**Compared with Logistic Regression:** Logistic Regression is simple and easy to explain, but it may miss more complex relationships between customer features and churn behavior.

**Compared with a single Decision Tree:** A single tree is easy to understand, but it can be unstable and less accurate.

**Compared with Random Forest:** Random Forest is strong, but XGBoost often gives better predictive performance and finer control for tabular classification tasks.

**Compared with Deep Learning:** Neural networks usually need more data, more tuning, and more computing resources. For a student project on structured churn data, that would add complexity without a clear benefit.

### What Problem It Solves Effectively

The model solves the problem of **estimating churn likelihood from customer attributes**. It does not just output yes or no. It also gives a probability, which is more useful in practice because businesses can prioritize customers by risk.

### Trade-Offs and Reasoning

The choice of XGBoost is a practical balance:

- **Accuracy**: usually strong on structured business data
- **Speed**: fast enough for training and prediction on local systems
- **Scalability**: suitable for typical project-sized churn datasets
- **Explainability**: better than black-box deep learning, though still not as simple as Logistic Regression
- **Complexity**: more complex than basic models, but still manageable in a real application

So the main trade-off is this: **we accept a little more model complexity in exchange for better prediction quality and stronger practical usefulness.**

## 7. Why This Approach

The overall design of this project makes sense because it solves the complete problem instead of only one part of it. Many academic projects stop after model training in a notebook. This project goes further by turning the model into an actual application where a user can upload data, train a model, and use it for future predictions through a proper interface.

The architecture is also sensible because it keeps the system modular. The frontend is responsible for user interaction, the backend is responsible for business logic, the ML pipeline is responsible for prediction, and Ollama is responsible only for natural-language explanation. This separation makes the system easier to maintain and explain.

Some alternatives could have been:

- building everything in a Jupyter notebook
- using only a rule-based churn score
- using a cloud LLM API for explanation
- training multiple models and selecting the best one dynamically

These were not the best fit here. A notebook is harder to use as an application. A rule-based system is too rigid. A cloud API raises privacy and cost concerns. A multi-model selection system would increase complexity and reduce clarity for a project review.

## 8. Literature Review (Market Understanding)

In the real world, churn prediction is usually approached in three common ways.

The first is **manual or dashboard-based analysis**, where teams use spreadsheets, BI dashboards, or simple reporting tools. These systems can show trends, but they usually do not provide custom predictive modeling for a specific uploaded dataset.

The second is **commercial customer analytics or CRM platforms**. These can provide churn insights, but they are often expensive, less transparent, and not ideal for academic or local offline use. They may also require data to be moved to external systems.

The third is **custom machine learning notebooks or scripts**. These are good for experimentation, but they are often not user-friendly, not reusable by non-technical users, and usually lack a guided interface.

So existing tools are available, but many of them either focus on business reporting, paid enterprise automation, or technical experimentation rather than a complete local end-to-end academic solution.

## 9. Limitations in Existing Systems

A common limitation in existing approaches is that they solve only part of the problem.

- Some tools predict churn but do not explain it clearly.
- Some tools explain trends but do not allow custom model training from uploaded data.
- Some tools are powerful but depend on cloud APIs, which creates privacy and cost issues.
- Some academic solutions show accuracy scores but stop there, without making the model usable in an application.
- Many systems are not flexible enough to create a prediction form automatically from the trained dataset schema.

This means there is still room for a system that is practical, explainable, local, and easy to demonstrate.

## 10. Gap Analysis and Contribution

The gap this project addresses is the lack of a **simple, end-to-end, locally runnable churn prediction system** that combines prediction with explanation.

The main contributions of this project are:

- it supports the complete workflow from upload to prediction
- it trains on user-provided data rather than a fixed hardcoded input form
- it automatically generates the prediction form from the trained model schema
- it gives business-friendly outputs such as risk level and recommended actions
- it uses a local LLM for explanation, so data does not need to leave the machine

What makes this project different is not only the algorithm, but the integration of **machine learning + application workflow + local explainability** in one system.

## 11. Use of Ollama

Ollama is used in this project to run a large language model locally on the user's machine. In this system, Ollama is **not used for prediction**. The churn prediction itself is produced by the XGBoost model. Ollama is used only after the statistical result is already available.

Its role is to generate:

- dataset summaries
- training summaries
- prediction explanations

This is an important design choice because it keeps the predictive logic deterministic and reliable, while still giving readable text for a human reviewer or business user.

### Benefits of Using Ollama

- **Local inference**: the model runs on the local machine
- **Privacy**: customer data does not need to be sent to a cloud API
- **Low cost**: no paid API is required
- **Offline capability**: the system can still work in local environments
- **Control**: the project can choose its own model and prompt style

### How Ollama Fits into the System

The backend sends structured facts such as probability, risk level, key factors, dataset statistics, and metrics to Ollama. Ollama then returns JSON-formatted natural-language text. If Ollama is unavailable, the backend falls back to deterministic explanations, so the application still works.

## 12. Model Selection

The local language model used through Ollama is **`llama3.1:8b`**.

This model was selected because it gives a practical balance between capability and efficiency. It is large enough to generate clear, useful summaries, but still small enough to run on many student or personal machines compared with much larger models.

The reason for selecting `llama3.1:8b` specifically is:

- it works well for explanation and summarization tasks
- it is suitable for local deployment
- it supports open-source, offline use
- it is a better practical fit than very small weak models or very large heavy models

Compared with a much smaller model, the explanation quality may be more natural and complete. Compared with a much larger model, it is easier to run locally and has lower hardware demand. So again, the choice is based on balance, not on using the biggest possible model.

## 13. Simple Example Walkthrough

Let us take a simple example based on the telecom churn dataset used in this project.

Suppose the user has already trained the model, and now enters one customer's details:

- `gender = Female`
- `SeniorCitizen = No`
- `Partner = Yes`
- `Dependents = No`
- `tenure = 1`
- `InternetService = DSL`
- `TechSupport = No`
- `Contract = Month-to-month`
- `PaperlessBilling = Yes`
- `PaymentMethod = Electronic check`
- `MonthlyCharges = 29.85`
- `TotalCharges = 29.85`

Now the system processes this input step by step:

1. The frontend displays a form created from the trained model schema.
2. The user fills in the values and clicks **Predict**.
3. The backend reads the saved model metadata and checks that all required fields are present.
4. The backend converts the input into a single-row table in the exact feature order expected by the model.
5. The same preprocessing pipeline used during training is applied, so missing values are handled and categorical values are encoded in a consistent way.
6. The XGBoost model calculates the prediction and churn probability.
7. Suppose the probability comes out high. The system then maps it to a **High** risk level using the rule that below 0.33 is Low, 0.33 to 0.66 is Medium, and above 0.66 is High.
8. The backend extracts the most influential factors for that prediction using model contribution scores.
9. Ollama receives only the computed facts, such as probability, risk level, and key factors, and writes a readable explanation.
10. The results page shows the predicted class, probability, risk level, explanation, and suggested retention actions.

This example is useful in a viva because it shows clearly that the system is not guessing in plain language first. It predicts using the ML model first, and only then explains the result using Ollama.

## 14. Conclusion (How to Explain in 1-2 Minutes)

This project is an end-to-end customer churn prediction application. The user uploads a dataset, chooses the target column, preprocesses the data, trains an XGBoost model, and then uses the trained model to predict churn for new customers. The backend stores the model, metrics, and schema, and the frontend automatically creates the prediction form from that schema.

The main algorithm is XGBoost because it works very well on structured churn data and gives strong prediction performance with probability outputs. Ollama with `llama3.1:8b` is used only for local explanations and summaries, not for prediction itself. The main strength of the project is that it combines machine learning, usability, and explainability in one locally runnable system.

## 15. Short Memorization Version

If I have to explain this very quickly, I can say:

"My project is a Customer Churn Prediction System that takes a business dataset, preprocesses it, trains an XGBoost model, and predicts whether a customer is likely to churn. The frontend guides the user through upload, preprocessing, training, and prediction. The backend handles data processing, model training, and inference. I used XGBoost because it is strong for structured tabular data and gives useful probability outputs. I used Ollama with `llama3.1:8b` only for local explanations, so the prediction remains machine-learning based while the explanation stays private, offline, and readable."
