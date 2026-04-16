# Main Backend Files

This file is a quick map of where the **core backend logic** lives.

Use this during review when you want to answer questions like:

- Where does the API call start?
- Where is preprocessing done?
- Where is the XGBoost model created?
- Where is the model trained?
- Where is the saved model loaded for prediction?
- Where does Ollama fit in?

Note: line numbers are based on the current code and may shift if the files change later.

## Fastest Reading Order

If you want the shortest path to understand the backend, open these files in this order:

1. [backend/app/api/routes/models.py](backend/app/api/routes/models.py)
2. [backend/app/services/training_service.py](backend/app/services/training_service.py)
3. [backend/app/ml/pipeline.py](backend/app/ml/pipeline.py)
4. [backend/app/ml/trainer.py](backend/app/ml/trainer.py)
5. [backend/app/services/prediction_service.py](backend/app/services/prediction_service.py)
6. [backend/app/ml/predictor.py](backend/app/ml/predictor.py)

## 1. App Startup and Flow Entry

**File Name:** `main.py`  
**Repo Path:** [backend/app/main.py](backend/app/main.py)  
**Line Number(s):** `27-57`, `60-110`, `113-134`  
**What happens here:** Creates the FastAPI app, sets middleware, startup/shutdown flow, logging, and mounts the main router.

**File Name:** `router.py`  
**Repo Path:** [backend/app/api/router.py](backend/app/api/router.py)  
**Line Number(s):** `9-17`  
**What happens here:** Connects the route groups together. This is where the backend wires `root`, `datasets`, `models`, and `admin` routes.

## 2. Dataset Upload and Preprocessing

**File Name:** `datasets.py`  
**Repo Path:** [backend/app/api/routes/datasets.py](backend/app/api/routes/datasets.py)  
**Line Number(s):** `60-75`, `78-93`, `96-106`  
**What happens here:** API endpoints for dataset upload, dataset preprocessing, and dataset summary. This is where the frontend first triggers backend data flow.

**File Name:** `dataset_service.py`  
**Repo Path:** [backend/app/services/dataset_service.py](backend/app/services/dataset_service.py)  
**Line Number(s):** `27-57`, `59-137`, `147-169`  
**What happens here:** Parses uploaded files, creates dataset metadata, validates the target column, removes rows with missing target values, and stores the processed dataset.

**File Name:** `dataset_store.py`  
**Repo Path:** [backend/app/storage/dataset_store.py](backend/app/storage/dataset_store.py)  
**Line Number(s):** `22-34`, `36-63`, `65-95`  
**What happens here:** Saves raw upload files, reads CSV/JSON into `pandas`, saves processed data as Parquet, and reloads processed datasets for training.

## 3. Training API and Main Training Flow

**File Name:** `models.py`  
**Repo Path:** [backend/app/api/routes/models.py](backend/app/api/routes/models.py)  
**Line Number(s):** `67-80`, `88-137`, `140-164`, `167-229`  
**What happens here:** Main model endpoints live here: train, train with SSE progress, schema lookup, training summary, predict, and predict with SSE progress.

**File Name:** `training_service.py`  
**Repo Path:** [backend/app/services/training_service.py](backend/app/services/training_service.py)  
**Line Number(s):** `33-205`, `95-120`, `139-174`  
**What happens here:** Main training orchestration. It loads preprocessed data, checks columns, calls the training helper, saves the trained pipeline, and writes model metadata.

## 4. XGBoost Pipeline Creation

**File Name:** `pipeline.py`  
**Repo Path:** [backend/app/ml/pipeline.py](backend/app/ml/pipeline.py)  
**Line Number(s):** `47-76`, `79-142`, `123-136`  
**What happens here:** Builds the model input schema and creates the preprocessing + `XGBoost` pipeline. The `XGBClassifier` is created here.

**Why this file matters:**

- `build_schema(...)` creates the dynamic prediction form structure
- `build_xgb_pipeline(...)` builds numeric preprocessing, categorical preprocessing, and the final `XGBClassifier`

## 5. XGBoost Training and Metrics

**File Name:** `trainer.py`  
**Repo Path:** [backend/app/ml/trainer.py](backend/app/ml/trainer.py)  
**Line Number(s):** `81-169`, `171-197`, `199-257`, `259-268`  
**What happens here:** Splits train/test data, fits the pipeline, calculates metrics, builds confusion matrix, and aggregates feature importance back to original columns.

**Where XGBoost is actually used:**

- `125-168`: train/test split and pipeline fitting
- `168`: `pipeline.fit(...)` runs model training
- `172-175`: prediction and probability for evaluation

## 6. Model Save and Load

**File Name:** `model_store.py`  
**Repo Path:** [backend/app/storage/model_store.py](backend/app/storage/model_store.py)  
**Line Number(s):** `21-23`, `25-37`, `39-57`  
**What happens here:** Saves the trained model pipeline to disk with `joblib` and loads it back later during prediction.

## 7. Prediction Flow

**File Name:** `prediction_service.py`  
**Repo Path:** [backend/app/services/prediction_service.py](backend/app/services/prediction_service.py)  
**Line Number(s):** `47-143`, `145-152`, `154-220`, `302-400`  
**What happens here:** Main prediction orchestration. It loads the saved model, builds the input row, runs prediction, maps probability to risk level, generates explanation, and builds recommended actions.

**Important points inside this file:**

- `75-76`: saved model is loaded
- `97-102`: prediction is triggered
- `145-152`: probability is converted into `Low`, `Medium`, or `High`
- `302-400`: key factor extraction is done using XGBoost contribution values

**File Name:** `predictor.py`  
**Repo Path:** [backend/app/ml/predictor.py](backend/app/ml/predictor.py)  
**Line Number(s):** `13-52`  
**What happens here:** This is the actual low-level prediction call. It runs `pipeline.predict(...)` and `pipeline.predict_proba(...)`.

## 8. Dynamic Prediction Form Flow

**File Name:** `schema_service.py`  
**Repo Path:** [backend/app/services/schema_service.py](backend/app/services/schema_service.py)  
**Line Number(s):** `21-70`  
**What happens here:** Reads saved model metadata and returns the schema used by the frontend to build the prediction form automatically.

## 9. Ollama / Explanation Flow

**File Name:** `llm_service.py`  
**Repo Path:** [backend/app/services/llm_service.py](backend/app/services/llm_service.py)  
**Line Number(s):** `21-28`, `28-76`  
**What happens here:** Calls Ollama locally through `/api/generate` and forces JSON output. This is used for summaries and explanations, not for the actual churn prediction.

## 10. Best File-by-File Review Summary

If someone asks you where each major backend action happens, you can answer like this:

- **API starts here:** [backend/app/api/routes/models.py](backend/app/api/routes/models.py) and [backend/app/api/routes/datasets.py](backend/app/api/routes/datasets.py)
- **Data preprocessing happens here:** [backend/app/services/dataset_service.py](backend/app/services/dataset_service.py) and [backend/app/storage/dataset_store.py](backend/app/storage/dataset_store.py)
- **XGBoost model is created here:** [backend/app/ml/pipeline.py](backend/app/ml/pipeline.py)
- **XGBoost training happens here:** [backend/app/ml/trainer.py](backend/app/ml/trainer.py)
- **Training orchestration and model saving happen here:** [backend/app/services/training_service.py](backend/app/services/training_service.py) and [backend/app/storage/model_store.py](backend/app/storage/model_store.py)
- **Saved model loading and prediction happen here:** [backend/app/services/prediction_service.py](backend/app/services/prediction_service.py) and [backend/app/ml/predictor.py](backend/app/ml/predictor.py)
- **Prediction form schema comes from here:** [backend/app/services/schema_service.py](backend/app/services/schema_service.py)
- **Ollama explanation logic is here:** [backend/app/services/llm_service.py](backend/app/services/llm_service.py)

## 11. If You Only Need 3 Files for a Review

If you are under time pressure, these are the three most important files:

1. [backend/app/services/training_service.py](backend/app/services/training_service.py)  
   This shows the main training flow from processed data to saved model.

2. [backend/app/ml/pipeline.py](backend/app/ml/pipeline.py)  
   This shows where preprocessing and `XGBoost` are defined.

3. [backend/app/services/prediction_service.py](backend/app/services/prediction_service.py)  
   This shows how the saved model is loaded, how prediction works, and how explanation is generated.
