import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')
import os
import sys

# Set page config FIRST
st.set_page_config(
    page_title="Intelligent Loan Pricing & Risk Management",
    page_icon="🏦",
    layout="wide"
)

st.title("Intelligent Loan Pricing & Risk Management")
st.markdown("**Australian Bank Context | Risk-Based Pricing Strategy**")

    
# ========== LOAD MODEL & DATA ==========
 
@st.cache_resource
def load_model():
    try:
        if os.path.exists('Models\\best_model.pkl'):
            with open('Models\\best_model.pkl', 'rb') as f:
                return pickle.load(f)
        else:
            st.warning("best_model.pkl not found")
            return None
    except Exception as e:
        st.warning(f"Error loading model: {e}")
        return None
 
@st.cache_data
def load_test_data():
    try:
        if os.path.exists('data\\processed\\X_test.csv') and os.path.exists('data\\processed\\y_test.csv'):
            X_test = pd.read_csv('data\\processed\\X_test.csv')
            y_test = pd.read_csv('data\\processed\\y_test.csv').squeeze()
            return X_test, y_test
        else:
            st.warning("X_test.csv or y_test.csv not found")
            return None, None
    except Exception as e:
        st.warning(f"Error loading test data: {e}")
        return None, None
    
@st.cache_data
def load_scenario_data():
    try:
        return pd.read_csv('AnalysisResults\\scenario_comparison.csv')
    except FileNotFoundError:
        st.error("scenario_comparison.csv not found. Run notebook 04 first.")
        return None

@st.cache_data
def load_pricing_data():
    try:
        return pd.read_csv('data\\raw\\pricing_df.csv')
    except FileNotFoundError:
        st.warning("pricing_df.csv not found")
        return None

# Load data
scenario_df = load_scenario_data()
pricing_df = load_pricing_data()
model = load_model()
X_test, y_test = load_test_data()
 
# Create sample pricing data (works even without files)
def create_sample_pricing_data():
    """Create sample pricing data for demonstration"""
    np.random.seed(42)
    
    n_samples = 400
    
    default_probs = np.random.beta(2, 2, n_samples)  # Beta distribution for probabilities
    loan_amounts = np.random.normal(50000, 30000, n_samples)
    loan_amounts = np.clip(loan_amounts, 5000, 300000)
    
    BASE_RATE = 0.05
    LGD = 0.70
    
    expected_loss_pct = default_probs * LGD
    risk_premium = expected_loss_pct * 10000
    suggested_rate = BASE_RATE + (risk_premium / 10000)
    annual_revenue = loan_amounts * suggested_rate
    expected_loss_dollars = loan_amounts * expected_loss_pct
    net_profit = annual_revenue - expected_loss_dollars
    
    def assign_tier(prob):
        if prob <= 0.15:
            return 'Prime'
        elif prob <= 0.30:
            return 'Near-Prime'
        elif prob <= 0.50:
            return 'Subprime'
        else:
            return 'High-Risk'
    
    pricing_df = pd.DataFrame({
        'loan_id': range(n_samples),
        'default_probability': default_probs,
        'loan_amount': loan_amounts,
        'expected_loss_pct': expected_loss_pct,
        'risk_premium': risk_premium,
        'suggested_rate': suggested_rate,
        'annual_revenue': annual_revenue,
        'expected_loss_dollars': expected_loss_dollars,
        'net_profit': net_profit,
        'tier': [assign_tier(p) for p in default_probs]
    })
    
    return pricing_df

# Use real data if available, otherwise use sample
if X_test is not None and model is not None:
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        BASE_RATE = 0.05
        LGD = 0.70
        
        pricing_df = pd.DataFrame({
            'loan_id': range(len(X_test)),
            'default_probability': y_pred_proba,
            'loan_amount': X_test['loan_amount'].values if 'loan_amount' in X_test.columns else np.random.normal(50000, 30000, len(X_test)),
        })
        
        pricing_df['expected_loss_pct'] = pricing_df['default_probability'] * LGD
        pricing_df['risk_premium'] = pricing_df['expected_loss_pct'] * 10000
        pricing_df['suggested_rate'] = BASE_RATE + (pricing_df['risk_premium'] / 10000)
        pricing_df['annual_revenue'] = pricing_df['loan_amount'] * pricing_df['suggested_rate']
        pricing_df['expected_loss_dollars'] = pricing_df['loan_amount'] * pricing_df['expected_loss_pct']
        pricing_df['net_profit'] = pricing_df['annual_revenue'] - pricing_df['expected_loss_dollars']
        
        def assign_tier(prob):
            if prob <= 0.15:
                return 'Prime'
            elif prob <= 0.30:
                return 'Near-Prime'
            elif prob <= 0.50:
                return 'Subprime'
            else:
                return 'High-Risk'
        
        pricing_df['tier'] = pricing_df['default_probability'].apply(assign_tier)
        data_source = "Real Model Predictions"
    except Exception as e:
        st.error(f"Error processing real data: {e}")
        pricing_df = create_sample_pricing_data()
        data_source = "Sample Data (Error Loading Real Data)"
else:
    pricing_df = create_sample_pricing_data()
    data_source = "Sample Data (Files Not Found)"
    
# Add sidebar info
with st.sidebar:
    st.markdown(f"**Data Source:** {data_source}")
    st.markdown("""
    ### About This Project
    
    Built a credit risk model that optimizes profitability while managing default risk.
    
    **Key Achievement:**
    - Collateral strategy increases net profit by 318.9% (\$1.24M → \$5.18M)
    - 100% approval rate with risk-appropriate pricing
    - Expected loss reduction of 42% through collateral requirement
    """)
    
# ========== MAIN TABS ==========
 
tab1, tab2, tab3, tab4 = st.tabs([
    "Portfolio Dashboard",
    "Pricing Calculator",
    "Risk Tier Analysis",
    "About"
])
 
# ========== TAB 1: PORTFOLIO DASHBOARD ==========
 
with tab1:
    st.header("Portfolio Dashboard")
    
    if pricing_df is not None and len(pricing_df) > 0:
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        total_profit = pricing_df['net_profit'].sum()
        total_loss = pricing_df['expected_loss_dollars'].sum()
        total_revenue = pricing_df['annual_revenue'].sum()
        profit_margin = (total_profit / total_revenue * 100) if total_revenue > 0 else 0
        
        with col1:
            st.metric("Total Net Profit", f"${total_profit:,.0f}", "4.2x improvement")
        with col2:
            st.metric("Expected Loss", f"${total_loss:,.0f}", "-42% with collateral")
        with col3:
            st.metric("Total Revenue", f"${total_revenue:,.0f}", "100% approval rate")
        with col4:
            st.metric("Profit Margin", f"{profit_margin:.1f}%", "+318.9% vs baseline")
        
        st.divider()
        
        # Scenario Comparison
        st.subheader("Strategy Comparison")
        
        scenario_df = load_scenario_data()

        if scenario_df is not None:
            col1, col2 = st.columns(2)
        
            with col1:
                fig, ax = plt.subplots(figsize=(10, 6))
                colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
                ax.bar(range(len(scenario_df)), scenario_df['Net Profit']/1e6, color=colors)
                ax.set_xticks(range(len(scenario_df)))
                ax.set_xticklabels(scenario_df['Strategy'], rotation=45, ha='right')
                ax.set_ylabel('Net Profit ($ Millions)')
                ax.set_title('Net Profit by Strategy')
                for i, v in enumerate(scenario_df['Net Profit']/1e6):
                    ax.text(i, v, f'${v:.1f}M', ha='center', va='bottom', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(range(len(scenario_df)), scenario_df['Profit Margin %'], color=colors)
                ax.set_xticks(range(len(scenario_df)))
                ax.set_xticklabels(scenario_df['Strategy'], rotation=45, ha='right')
                ax.set_ylabel('Profit Margin (%)')
                ax.set_title('Profit Margin by Strategy')
                for i, v in enumerate(scenario_df['Profit Margin %']):
                    ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
        
        st.divider()
        
         # Recommended Strategy Explanation
        st.subheader("Why Collateral Strategy is Best")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            **The Collateral Strategy:**
            1. Don't reject high-risk customers outright
            2. Approve them with collateral requirement (car, property, etc.)
            3. If they default, sell collateral to recover loss
            4. Expected loss drops by 42% (\$9.46M → \$5.52M)
            
            **Business Benefits:**
            - 4.2x profit increase (\$1.24M → \$5.18M annually)
            - 100% approval rate (better customer experience)
            - Risk exposure managed through collateral
            - Defensible under ASIC guidelines (risk-based pricing)
            
            **Australian Context:**
            - Aligns with responsible lending standards
            - Reduces regulatory risk
            - Increases financial inclusion (more people approved)
            """)
        
        with col2:
            # Approval rate comparison
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.bar(scenario_df['Strategy'], scenario_df['Approval Rate %'], color=colors)
            ax.set_ylabel('Approval Rate (%)')
            ax.set_title('Approval Rate by Strategy')
            ax.set_ylim(0, 120)
            ax.tick_params(axis='x', rotation=45)
            for i, v in enumerate(scenario_df['Approval Rate %']):
                ax.text(i, v, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.error("Test data not loaded. Please ensure X_test.csv and y_test.csv are in the project directory.")
        
# ========== TAB 2: PRICING CALCULATOR ==========
 
with tab2:
    st.header("Risk-Based Pricing Calculator")
    
    st.markdown("""
    Enter applicant details to calculate:
    - Default probability
    - Suggested interest rate
    - Approval recommendation
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.slider("Age", 18, 80, 35)
        income = st.number_input("Annual Income ($)", min_value=20000, max_value=500000, value=60000, step=5000)
        employment_years = st.slider("Years in Current Role", 0, 40, 5)
        loan_amount = st.number_input("Loan Amount ($)", min_value=5000, max_value=500000, value=50000, step=5000)
    
    with col2:
        total_debt = st.number_input("Total Existing Debt ($)", min_value=0, max_value=300000, value=30000, step=5000)
        marital_status = st.selectbox("Marital Status", ['Single', 'Married', 'Divorced'])
        credit_history = st.slider("Credit History (months)", 0, 600, 60)
        employment_type = st.selectbox("Employment Type", ['Full-time', 'Part-time', 'Casual', 'Self-employed'])
    
    # Simple default probability estimation (using a basic formula for demo)
    debt_to_income = total_debt / income if income > 0 else 0
    income_score = min(1.0, income / 100000)  # Normalize
    employment_score = {'Full-time': 0.9, 'Part-time': 0.7, 'Casual': 0.5, 'Self-employed': 0.6}[employment_type]
    credit_score = min(1.0, credit_history / 600)
    
    # Weighted default probability
    estimated_default_prob = (
        0.3 * debt_to_income +
        0.2 * (1 - income_score) +
        0.2 * (1 - employment_score) +
        0.3 * (1 - credit_score)
    )
    estimated_default_prob = max(0, min(1, estimated_default_prob))
    
    # Calculate pricing
    BASE_RATE = 0.05
    LGD = 0.70
    
    expected_loss_pct = estimated_default_prob * LGD
    risk_premium = expected_loss_pct * 100  # in percentage
    suggested_rate = BASE_RATE + (risk_premium / 100)
    annual_revenue = loan_amount * suggested_rate
    expected_loss = loan_amount * expected_loss_pct
    net_profit = annual_revenue - expected_loss
    
    # Assign tier
    if estimated_default_prob <= 0.15:
        tier = "Prime"
        tier_color = "green"
        recommendation = "APPROVE"
        strategy = "Standard approval at suggested rate"
    elif estimated_default_prob <= 0.30:
        tier = "Near-Prime"
        tier_color = "blue"
        recommendation = "APPROVE"
        strategy = "Approve at elevated rate"
    elif estimated_default_prob <= 0.50:
        tier = "Subprime"
        tier_color = "orange"
        recommendation = "CONDITIONAL"
        strategy = "Require collateral or co-signer"
    else:
        tier = "High-Risk"
        tier_color = "red"
        recommendation = "CONDITIONAL"
        strategy = "Require collateral or alternative lender"
    
    st.divider()
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk Tier", tier)
        st.metric("Default Probability", f"{estimated_default_prob:.1%}")
    
    with col2:
        st.metric("Suggested Interest Rate", f"{suggested_rate:.2%}")
        st.metric("Risk Premium", f"{risk_premium:.2f} bps")
    
    with col3:
        st.metric("Annual Interest Revenue", f"${annual_revenue:,.0f}")
        st.metric("Expected Loss", f"${expected_loss:,.0f}")
    
    st.divider()
    
    # Recommendation
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader(f"Approval Recommendation: {recommendation}")
        st.write(f"**Strategy:** {strategy}")
        st.write(f"**Net Profit (Annual):** ${net_profit:,.0f}")
        
        if recommendation == "APPROVE":
            st.success("✅ This applicant should be approved at the suggested rate")
        elif recommendation == "CONDITIONAL":
            st.warning("Conditional approval - require collateral or co-signer")
        
    with col2:
        # Gauge chart
        fig, ax = plt.subplots(figsize=(8, 6))
        risk_pct = estimated_default_prob * 100
        colors_gauge = ['green' if risk_pct < 15 else 'blue' if risk_pct < 30 else 'orange' if risk_pct < 50 else 'red']
        ax.barh(['Risk'], [risk_pct], color=colors_gauge)
        ax.text(risk_pct/2, 0, f'{risk_pct:.1f}%', ha='center', va='center', 
                fontsize=16, fontweight='bold', color='white')
        ax.set_xlabel('Default Probability (%)')
        ax.set_title('Risk Score')
        plt.tight_layout()
        st.pyplot(fig)