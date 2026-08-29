# Intelligent Loan Pricing & Risk Management for Australian Banks

A machine learning project for credit risk prediction and pricing optimization.

## Quick Start

### Run Notebooks
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

## Key Results
- **Collateral Strategy** increases annual net profit by 159.5%
- From: $1.24M (baseline) → To: $3.21M (with collateral)
- Model Performance: 80.5% accuracy, 84.08% AUC-ROC

## Project Structure
- `notebooks/` - Jupyter notebooks for data analysis & modeling
- `data/` - Raw and processed datasets
- `models/` - Trained Random Forest model & scaler
- `analysis/` - Strategy comparison results
- `app.py` - Streamlit dashboard

## Australian Regulatory Compliance
- ASIC Credit Management Laws
- Privacy Act 1988 (Cth)
- Equal Opportunity Laws
- RBA Risk Management Standards

## Author
Pranamya Rajbhandari