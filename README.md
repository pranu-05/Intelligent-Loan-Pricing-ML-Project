# Intelligent Loan Pricing & Risk Management for Australian Banks

A machine learning project for credit risk prediction and pricing optimisation.

## Installation

### Prerequisites
- Python 3.8+
- Git

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/pranu-05/Intelligent-Loan-Pricing-ML-Project.git
cd Intelligent-Loan-Pricing-ML-Project
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

## Quick Start

### Run Notebooks (in order)
```bash
jupyter notebook notebooks/01_eda_and_features.ipynb
jupyter notebook notebooks/02_modeling_and_evaluation.ipynb
jupyter notebook notebooks/03_pricing_and_risk_analysis.ipynb
jupyter notebook notebooks/04_strategic_analysis_and_scenario_testing.ipynb
```

### Launch Dashboard
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

## Project Structure
- `notebooks/` - Jupyter notebooks for data analysis & modeling
- `data/` - Raw and processed datasets
- `models/` - Trained Random Forest model & scaler
- `analysis/` - Strategy comparison results
- `app.py` - Streamlit dashboard

## Key Results
- **Collateral Strategy** increases annual net profit by 159.5%
- From: $1.24M (baseline) → To: $3.21M (with collateral)
- Model Performance: 80.5% accuracy, 84.08% AUC-ROC

## Author
Pranamya Rajbhandari