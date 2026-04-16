# Customer Churn Prediction System

<p align="center">
  <a href="https://react.dev" target="_blank">
    <img height="40" src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" />
  </a>
  &nbsp;&nbsp;

  <a href="https://fastapi.tiangolo.com" target="_blank">
    <img height="40" src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" />
  </a>
  &nbsp;&nbsp;

  <a href="https://xgboost.readthedocs.io" target="_blank">
    <img height="40" src="https://img.shields.io/badge/XGBoost-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white" />
  </a>
  &nbsp;&nbsp;

  <a href="https://ollama.com/library/llama3.1:8b" target="_blank">
    <img height="40" src="https://img.shields.io/badge/Ollama-000000?style=for-the-badge&logo=ollama&logoColor=white" />
  </a>
  &nbsp;&nbsp;

  <a href="https://www.kaggle.com/datasets/blastchar/telco-customer-churn" target="_blank">
    <img height="40" src="https://img.shields.io/badge/Kaggle%20Dataset-20BEFF?style=for-the-badge&logo=kaggle&logoColor=black" />
  </a>
</p>

End-to-end churn analytics for business datasets: upload raw customer data, preprocess it interactively, train an XGBoost model, and generate predictions with explainability powered by a local open-source LLM (Ollama). The LLM is used only for explanations and summaries; predictions come from the ML model.

## Project Overview

**Problem**: Businesses often have churn datasets but lack a simple, reproducible way to train custom churn models and explain results without paid APIs.

**What this project does**:

- Upload CSV/JSON datasets
- Select target and preprocess features
- Train an XGBoost churn model with live progress
- Auto-generate a prediction form from the model schema
- Produce predictions plus explanations, risks, and recommended actions

**Key features**:

- ML training + prediction workflow
- UI-guided multi-step experience with light/dark mode
- LLM-powered dataset summary, training summary, and prediction explanation (Ollama)
- SSE-based progress streaming for training and prediction
- One-click reset to clear runtime artifacts

## Tech Stack

**Frontend**

- React 18 + TypeScript
- Vite
- Tailwind CSS

**Backend**

- FastAPI (Python)
- Uvicorn

**Machine Learning**

- XGBoost
- scikit-learn
- pandas, numpy

**LLM**

- Ollama (local inference)
- Model: `llama3.1:8b`

**Database**

- None (runtime artifacts stored on local disk under `backend/`)

## Project Architecture

High-level flow:
Frontend (React) -> Backend (FastAPI) -> ML pipeline (XGBoost) -> LLM (Ollama) for explanations only

LLM role (explanations only):

- Dataset summary after preprocessing
- Training summary and key feature insights
- Per-prediction explanation and confidence note

Data flow:

- User uploads dataset -> backend stores file and inspects columns
- User selects target/exclusions -> backend preprocesses and stores processed data
- User trains model -> backend trains XGBoost pipeline and stores model + metadata
- Frontend loads model schema -> renders dynamic prediction form
- User predicts -> backend returns probability, risk level, key factors, and recommendations
- LLM generates summaries/explanations when enabled (fallbacks used if LLM is off)

## Folder Structure

```
/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── routes/
│   │   │   │   ├── admin.py
│   │   │   │   ├── datasets.py
│   │   │   │   ├── models.py
│   │   │   │   └── root.py
│   │   │   ├── router.py
│   │   │   └── __init__.py
│   │   ├── core/
│   │   │   ├── config.py
│   │   │   ├── errors.py
│   │   │   ├── logging.py
│   │   │   └── __init__.py
│   │   ├── ml/
│   │   │   ├── metrics.py
│   │   │   ├── pipeline.py
│   │   │   ├── predictor.py
│   │   │   ├── trainer.py
│   │   │   └── __init__.py
│   │   ├── schemas/
│   │   │   ├── common.py
│   │   │   ├── datasets.py
│   │   │   ├── insights.py
│   │   │   ├── prediction.py
│   │   │   ├── training.py
│   │   │   └── __init__.py
│   │   ├── services/
│   │   │   ├── dataset_service.py
│   │   │   ├── insights_service.py
│   │   │   ├── llm_service.py
│   │   │   ├── prediction_service.py
│   │   │   ├── schema_service.py
│   │   │   ├── training_service.py
│   │   │   └── __init__.py
│   │   ├── storage/
│   │   │   ├── dataset_store.py
│   │   │   ├── metadata_store.py
│   │   │   ├── model_store.py
│   │   │   └── __init__.py
│   │   ├── utils/
│   │   │   ├── files.py
│   │   │   ├── ids.py
│   │   │   └── __init__.py
│   │   ├── main.py
│   │   └── __init__.py
│   ├── metadata/
│   ├── models/
│   ├── processed/
│   ├── uploads/
│   ├── bank.csv
│   ├── telecom.csv
│   └── requirements.txt
├── src/
│   ├── components/
│   │   ├── common/
│   │   │   ├── Alert.tsx
│   │   │   ├── FeatureImportanceList.tsx
│   │   │   ├── MetricCard.tsx
│   │   │   ├── ProgressBar.tsx
│   │   │   ├── SelectMenu.tsx
│   │   │   ├── Spinner.tsx
│   │   │   ├── Stepper.tsx
│   │   │   ├── ThemeToggle.tsx
│   │   │   ├── Toast.tsx
│   │   │   ├── Toaster.tsx
│   │   │   └── TrainedModelDetails.tsx
│   │   ├── DataPreprocessing.tsx
│   │   ├── FileUpload.tsx
│   │   ├── ModelTraining.tsx
│   │   ├── PredictionForm.tsx
│   │   └── PredictionResults.tsx
│   ├── config/
│   │   └── env.ts
│   ├── context/
│   │   └── ThemeProvider.tsx
│   ├── hooks/
│   │   ├── useDatasetSummary.ts
│   │   ├── useModelDetails.ts
│   │   ├── useModelSummary.ts
│   │   ├── useTheme.ts
│   │   ├── useTrainingProgress.ts
│   │   └── useWizard.ts
│   ├── services/
│   │   ├── api.ts
│   │   └── http.ts
│   ├── styles/
│   │   ├── components.css
│   │   ├── motion.css
│   │   └── theme.css
│   ├── theme/
│   │   └── tokens.ts
│   ├── types/
│   │   ├── api.ts
│   │   └── ui.ts
│   ├── utils/
│   │   ├── errors.ts
│   │   └── format.ts
│   ├── App.tsx
│   ├── index.css
│   ├── main.tsx
│   └── vite-env.d.ts
├── UI ScreenShots/
│   ├── DARK/
│   └── LIGHT/
├── dist/
├── node_modules/
├── index.html
├── package.json
├── package-lock.json
├── vite.config.ts
└── README.md
```

## Prerequisites

- Node.js 18+
- Python 3.8+
- pip + virtualenv (recommended)
- Ollama installed locally
- OS: Windows, macOS, or Linux (Ollama-supported platforms)

## Backend Setup and Run

1. Create and activate a virtual environment:

```bash
cd backend
python -m venv .venv
.\.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `backend/.env` (optional):

```env
APP_NAME=CUSTOMER CHURN PREDICTION SYSTEM - Backend
ENVIRONMENT=development
DEBUG=false
HOST=0.0.0.0
PORT=8000
LLM_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_MAX_TOKENS=256
CORS_ALLOW_ORIGINS=http://localhost:5173
```

4. Run the backend:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Frontend Setup and Run

1. Install dependencies:

```bash
npm install
```

2. Configure API base URL (choose one):

**Option A: Dev proxy (recommended for local dev)**  
Set `VITE_BACKEND_URL` so Vite proxies backend routes:

```env
VITE_BACKEND_URL=http://localhost:8000
```

**Option B: Direct API base URL**  
Set `VITE_API_BASE_URL` to a full backend origin:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT_MS=30000
```

3. Run the frontend:

```bash
npm run dev
```

Frontend runs at `http://localhost:5173`.

## LLM (Ollama) Setup and Run

Commands:

```bash
ollama --version
ollama pull llama3.1:8b
ollama run llama3.1:8b
ollama stop llama3.1:8b
ollama list
ollama ps
```

What each command does:

- `ollama --version`: Verify Ollama is installed.
- `ollama pull llama3.1:8b`: Download the model used by this project.
- `ollama run llama3.1:8b`: Start a local session (useful for quick checks).
- `ollama stop llama3.1:8b`: Stop the model process.
- `ollama list`: List all locally available models.
- `ollama ps`: Show currently running models.

When the model must be running:

- Required for dataset summaries, training summaries, and prediction explanations.
- If Ollama is not running, the backend returns deterministic fallbacks (predictions still work).

Why `llama3.1:8b`:

- Small enough for local machines
- Good balance between quality and speed
- Fully open-source and offline-capable

## Application Workflow

1. **Dataset upload**  
   Upload CSV/JSON; backend parses columns, types, and sample values.

2. **Preprocessing**  
   Select target column and exclude features; processed data is stored server-side.

3. **Model training**  
   Train an XGBoost pipeline; view live progress and metrics.

4. **Prediction**  
   Auto-generated form captures feature inputs; backend returns probability and risk level.

5. **LLM-based explanation**

- Dataset summary after preprocessing
- Training summary with top features
- Per-prediction explanation + key factors

## Dataset

This project uses the Telco Customer Churn dataset from Kaggle:  
https://www.kaggle.com/datasets/blastchar/telco-customer-churn

## Environment Variables

There are no committed env files in this repo. Create them as needed:

Backend (optional `.env` in `backend/`, loaded by `backend/app/core/config.py`):

```env
APP_NAME=CUSTOMER CHURN PREDICTION SYSTEM - Backend
APP_VERSION=1.0.0
ENVIRONMENT=development
DEBUG=false
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO
CORS_ALLOW_ORIGINS=http://localhost:5173
MAX_UPLOAD_MB=25
RANDOM_SEED=42
TEST_SIZE=0.2
LLM_ENABLED=true
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b
OLLAMA_MAX_TOKENS=256
```

Frontend (Vite env file at repo root, e.g. `.env` or `.env.local`):

```env
VITE_BACKEND_URL=http://localhost:8000
VITE_API_BASE_URL=http://localhost:8000
VITE_API_TIMEOUT_MS=30000
```

Ollama (set via backend env variables):

- `OLLAMA_BASE_URL` and `OLLAMA_MODEL`

## Screens / UI Flow

Screens in the guided wizard:

- Upload
- Preprocess
- Train (progress + summary)
- Predict (dynamic form)
- Results (probability, risk level, explanation)

The UI supports light and dark mode via the theme toggle. Screenshots are stored in `UI ScreenShots/`.

### Light mode flow

1. Upload screen  
   ![Upload Screen](UI%20ScreenShots/LIGHT/1.%20Upload%20Screen.png)

2. Upload with file loaded  
   ![Upload Screen With File Loaded](UI%20ScreenShots/LIGHT/2.%20Upload%20Screen%20With%20File%20Loaded.png)

3. Preprocessing  
   ![PreProcessing Screen](UI%20ScreenShots/LIGHT/3.%20PreProcessing%20Screen.png)

4. Train preview  
   ![Train Preview Screen](UI%20ScreenShots/LIGHT/4.%20Train%20Preview%20Screen.png)

5. Training in progress  
   ![Training Screen](UI%20ScreenShots/LIGHT/5.%20Training%20Screen.png)

6. Training summary  
   ![Training Complete and Summary Screen](UI%20ScreenShots/LIGHT/6.%20Training%20Complete%20and%20Summary%20Screen.png)

7. Prediction form  
   ![Prediction Screen](UI%20ScreenShots/LIGHT/7.%20Pridection%20Screen.png)

8. Prediction processing  
   ![Prediction Processing Screen](UI%20ScreenShots/LIGHT/8.%20Pridection%20Processing%20Screen.png)

9. Results  
   ![Results Screen](UI%20ScreenShots/LIGHT/9.%20Results%20Screen.png)

### Dark mode flow

1. Upload screen  
   ![Upload Screen Dark](<UI%20ScreenShots/DARK/1.%20Upload%20Screen%20(DARK).png>)

2. Upload with file loaded  
   ![Upload Screen With File Loaded Dark](<UI%20ScreenShots/DARK/2.%20Upload%20Screen%20With%20File%20Loaded%20(DARK).png>)

3. Preprocessing  
   ![PreProcessing Screen Dark](<UI%20ScreenShots/DARK/3.%20PreProcessing%20Screen%20(DARK).png>)

4. Train preview  
   ![Train Preview Screen Dark](<UI%20ScreenShots/DARK/4.%20Train%20Preview%20Screen%20(DARK).png>)

5. Training in progress  
   ![Training Screen Dark](<UI%20ScreenShots/DARK/5.%20Training%20Screen%20(DARK).png>)

6. Training summary  
   ![Training Complete and Summary Screen Dark](<UI%20ScreenShots/DARK/6.%20Training%20Complete%20and%20Summary%20Screen%20(DARK).png>)

7. Prediction form  
   ![Prediction Screen Dark](<UI%20ScreenShots/DARK/7.%20Pridection%20Screen%20(DARK).png>)

8. Prediction processing  
   ![Prediction Processing Screen Dark](<UI%20ScreenShots/DARK/8.%20Pridection%20Processing%20Screen%20(DARK).png>)

9. Results  
   ![Results Screen Dark](<UI%20ScreenShots/DARK/9.%20Results%20Screen%20(DARK).png>)

## Model Plots

To generate evaluation plots (confusion matrix, feature importance, ROC curve), run:

```bash
python backend\scripts\generate_model_plots.py --data "datasets\telecom.csv" --target "churn" --outdir "UI ScreenShots"
```

1. Confusion matrix  
   Shows correct vs. incorrect predictions (TP, TN, FP, FN), so you can see where the model is making mistakes and how balanced the errors are.  
   ![Confusion Matrix](UI%20ScreenShots/confusion_matrix.png)

2. Feature importance  
   Ranks the most influential features used by the model, which helps explain what drives churn predictions and guides feature selection.  
   ![Feature Importance](UI%20ScreenShots/feature_importance.png)

3. ROC curve  
   Plots true positive rate vs. false positive rate across thresholds, which indicates how well the model separates churn vs. non-churn.  
   ![ROC Curve](UI%20ScreenShots/roc_curve.png)

## Error Handling and Troubleshooting

Common issues and fixes:

- **Backend not starting**
  - Check Python version and dependencies
  - Ensure port `8000` is free or change `PORT`

- **Frontend API errors (CORS or 404)**
  - Set `VITE_BACKEND_URL` or `VITE_API_BASE_URL` correctly
  - Ensure backend is running and reachable

- **Ollama not responding**
  - Start the Ollama service and verify with `ollama ps`
  - Confirm `OLLAMA_BASE_URL` is correct (default `http://localhost:11434`)

- **Model not found**
  - Run `ollama pull llama3.1:8b`
  - Verify `OLLAMA_MODEL=llama3.1:8b`

## Future Enhancements

- Add model comparison and hyperparameter tuning
- Provide richer LLM explanations (counterfactuals, scenario analysis)
- Package for production deployment (Docker, cloud storage for artifacts)

## Academic Notes

- This project uses an open-source LLM (Ollama) to avoid paid APIs and external data sharing.
- The LLM is used strictly for explainability and summaries, not for prediction.
- All data remains local to the machine running the system, supporting privacy requirements.
- Reproducibility is supported via deterministic ML settings and stored artifacts.
