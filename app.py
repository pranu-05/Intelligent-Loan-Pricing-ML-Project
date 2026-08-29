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
        if os.path.exists('models\\best_model.pkl'):
            with open('models\\best_model.pkl', 'rb') as f:
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
        return pd.read_csv('analysis\\scenario_comparison.csv')
    except FileNotFoundError:
        st.error("scenario_comparison.csv not found. Run notebook 04 first.")
        return None

@st.cache_data
def load_pricing_data():
    try:
        # Try multiple locations
        paths = ['data\\raw\\pricing_df.csv', 'data\\processed\\pricing_df.csv', 'pricing_df.csv']
        for path in paths:
            if os.path.exists(path):
                return pd.read_csv(path)
        return None
    except Exception as e:
        st.warning(f"Error loading: {e}")
        return None

# Load data
scenario_df = load_scenario_data()
pricing_df = load_pricing_data()
model = load_model()
X_test, y_test = load_test_data()

if pricing_df is not None:
    column_mapping = {
        'risk_tier': 'tier',
        'suggested_interest_rate': 'suggested_interest_rate',
        'loan_amount': 'loan_amount'
    }
    pricing_df.rename(columns=column_mapping, inplace=True)
 
# Create sample pricing data (works even without files)
def create_sample_pricing_data():
    """Create sample pricing data for demonstration"""
    np.random.seed(42)
    
    n_samples = 400
    
    default_probs = np.random.beta(2, 2, n_samples)  # Beta distribution for probabilities
    loan_amount = np.random.normal(50000, 30000, n_samples)
    loan_amount = np.clip(loan_amount, 5000, 300000)
    
    BASE_RATE = 0.05
    LGD = 0.25
    
    expected_loss_pct = default_probs * LGD
    risk_premium = expected_loss_pct * 10000
    suggested_interest_rate = BASE_RATE + (risk_premium / 10000)
    annual_interest_revenue = loan_amount * suggested_interest_rate
    expected_loss_dollars = loan_amount * expected_loss_pct
    net_profit = annual_interest_revenue - expected_loss_dollars
    
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
        'loan_amount': loan_amount,
        'expected_loss_pct': expected_loss_pct,
        'risk_premium': risk_premium,
        'suggested_interest_rate': suggested_interest_rate,
        'annual_interest_revenue': annual_interest_revenue,
        'expected_loss_dollars': expected_loss_dollars,
        'net_profit': net_profit,
        'risk_tier': [assign_tier(p) for p in default_probs]
    })
    
    return pricing_df

if pricing_df is not None:
    data_source = "Loaded from a synthetic dataset developed for this project"
else:
    data_source = "CSV not found"
    
# Add sidebar info
with st.sidebar:
    st.markdown(f"**Data Source:** {data_source}")
    st.markdown("""
    ### About This Project
    
    Built a personal loan credit risk model that optimises profitability while managing the risk of defaulting.
    
    **Key Achievement:**
    - Collateral strategy increases net profit by 159.4% (\$1.24M → \$3.2M)
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
        # Load scenario data
        scenario_df = load_scenario_data()
        
        if scenario_df is not None:
            # Get current strategy (approve all)
            current_row = scenario_df[scenario_df['Strategy'] == 'Current (Approve All)'].iloc[0]
            
            current_profit = current_row['Net Profit']
            current_loss = current_row['Expected Loss']
            current_revenue = current_row['Total Revenue']
            current_margin = current_row['Profit Margin %']
            
            # Get collateral strategy
            collateral_row = scenario_df[scenario_df['Strategy'] == 'Collateral (Collateral for High-Risk & Subprime)'].iloc[0]
            
            collateral_profit = collateral_row['Net Profit']
            collateral_loss = collateral_row['Expected Loss']
            collateral_revenue = collateral_row['Total Revenue']
            collateral_margin = collateral_row['Profit Margin %']
            
            # Calculate improvements (Collateral vs Current)
            profit_improvement = ((collateral_profit - current_profit) / current_profit * 100)
            loss_reduction = ((current_loss - collateral_loss) / current_loss * 100)
            margin_improvement = (collateral_margin - current_margin)
            
            # Display Collateral Strategy metrics with Current Strategy comparison
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Total Net Profit",
                    f"${collateral_profit:,.0f}",
                    f"+{profit_improvement:.1f}% vs Current"
                )
            with col2:
                st.metric(
                    "Expected Loss",
                    f"${collateral_loss:,.0f}",
                    f"-{loss_reduction:.1f}% vs Current"
                )
            with col3:
                st.metric(
                    "Total Revenue",
                    f"${collateral_revenue:,.0f}",
                    "100% approval rate"
                )
            with col4:
                st.metric(
                    "Profit Margin",
                    f"{collateral_margin:.1f}%",
                    f"+{margin_improvement:.1f}pp vs Current"
                )

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
    st.header("Risk-Based Loan Pricing Calculator")
    
    st.markdown("""
    Enter applicant details to calculate:
    - Default probability (using an Australian-inspired lending data model)
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
    
    # Calculate default probability using Australian lending formula
    # Based on portfolio analysis: 57.74% average default probability
    
    debt_to_income = total_debt / income if income > 0 else 0
    income_normalized = min(1.0, income / 150000)  # Normalize against median income
    
    employment_risk = {
        'Full-time': 0.1,
        'Part-time': 0.3,
        'Casual': 0.5,
        'Self-employed': 0.4
    }[employment_type]
    
    credit_risk = 1.0 - min(1.0, credit_history / 600)  # More history = less risk
    age_risk = 0 if 25 <= age <= 65 else 0.2  # Higher risk if very young/old
    
    # Weighted formula based on portfolio defaults
    estimated_default_prob = (
        0.35 * debt_to_income +           # Debt is highest risk factor
        0.25 * employment_risk +           # Employment stability
        0.20 * credit_risk +               # Credit history
        0.15 * (1 - income_normalized) +   # Income level
        0.05 * age_risk                    # Age effects
    )
    
    # Clamp between 0 and 1
    estimated_default_prob = max(0.01, min(0.95, estimated_default_prob))
    
    # Pricing calculation from your portfolio analysis
    BASE_RATE = 0.05
    LGD = 0.25  # Loss Given Default (25%)
    
    expected_loss_pct = estimated_default_prob * LGD
    risk_premium = expected_loss_pct * 100
    suggested_interest_rate = BASE_RATE + (risk_premium / 100)
    
    # Clamp interest rate to realistic range
    suggested_interest_rate = max(0.05, min(0.70, suggested_interest_rate))
    
    annual_revenue = loan_amount * suggested_interest_rate
    expected_loss = loan_amount * expected_loss_pct
    net_profit = annual_revenue - expected_loss
    
    # Assign tier based on default probability
    if estimated_default_prob <= 0.15:
        risk_tier = "Prime"
        tier_color = "green"
        recommendation = "APPROVE"
        strategy = "Standard approval at suggested rate"
    elif estimated_default_prob <= 0.30:
        risk_tier = "Near-Prime"
        tier_color = "blue"
        recommendation = "APPROVE"
        strategy = "Approve at elevated rate"
    elif estimated_default_prob <= 0.50:
        risk_tier = "Subprime"
        tier_color = "orange"
        recommendation = "CONDITIONAL"
        strategy = "Require collateral or co-signer"
    else:
        risk_tier = "High-Risk"
        tier_color = "red"
        recommendation = "CONDITIONAL"
        strategy = "Require collateral (car, property, etc.)"
    
    st.divider()
    
    # Display results
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk Tier", risk_tier)
        st.metric("Default Probability", f"{estimated_default_prob:.1%}")
    
    with col2:
        st.metric("Suggested Interest Rate", f"{suggested_interest_rate:.2%}")
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
        st.write(f"**Expected Annual Profit:** ${net_profit:,.0f}")
        
        if recommendation == "APPROVE":
            st.success("✅ This applicant should be approved at the suggested rate")
        elif recommendation == "CONDITIONAL":
            st.warning("⚠️ Conditional approval - require collateral or co-signer")
        
    with col2:
        # Risk gauge chart
        fig, ax = plt.subplots(figsize=(5, 4))
        risk_pct = estimated_default_prob * 100
        
        # Color based on tier
        if risk_pct < 15:
            bar_color = 'green'
        elif risk_pct < 30:
            bar_color = 'blue'
        elif risk_pct < 50:
            bar_color = 'orange'
        else:
            bar_color = 'red'
        
        ax.barh(['Risk'], [risk_pct], color=bar_color, edgecolor='black', linewidth=2)
        ax.set_xlim([0, 100])
        ax.text(risk_pct/2, 0, f'{risk_pct:.1f}%', ha='center', va='center', 
                fontsize=13, fontweight='bold', color='white')
        ax.set_xlabel('Default Probability (%)', fontweight='bold')
        ax.set_title('Default Risk Score', fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
    
    # Info box explaining the calculation
    with st.expander("📊 How the calculator works"):
        st.markdown("""
        **Default Probability Factors (weighted):**
        - Debt-to-Income Ratio: 35% weight (highest impact)
        - Employment Stability: 25% weight
        - Credit History: 20% weight
        - Income Level: 15% weight
        - Age: 5% weight
        
        **Pricing Formula:**
        - Base Rate: 5%
        - Loss Given Default: 20%
        - Suggested Rate = Base Rate + (Default Prob × LGD × 100)
        
        **Recommendation Tiers:**
        - Prime (0-15% default): Auto-approve
        - Near-Prime (15-30%): Approve at higher rate
        - Subprime (30-50%): Conditional (requires collateral)
        - High-Risk (50%+): Conditional (requires collateral)
        """)
        
# ========== TAB 3: RISK TIER ANALYSIS  ==========
 
with tab3:
    st.header("Risk Tier Analysis")
    
    if pricing_df is not None and len(pricing_df) > 0:
        st.subheader("Tier Breakdown - Collateral Strategy")
        
        # Calculate tier statistics
        tier_stats = pricing_df.groupby('approval_tier').agg({
            'loan_amount': 'sum',
            'default_probability': 'mean',
            'expected_loss_dollars': 'sum',
            'annual_interest_revenue': 'sum',
            'net_profit': 'sum',
            'suggested_interest_rate': 'mean',
            'loan_id': 'count'
        }).round(2)
        
        tier_stats.columns = [
            'Total Loan Amount',
            'Avg Default Prob',
            'Expected Loss',
            'Total Interest Revenue',
            'Net Profit',
            'Avg Interest Rate',
            'Count'
        ]
        
        tier_stats['Profit Margin %'] = (
            (tier_stats['Net Profit'] / tier_stats['Total Interest Revenue'] * 100)
        ).round(2)
        
        # Reorder columns for display
        tier_stats = tier_stats[['Count', 'Total Loan Amount', 'Avg Default Prob', 
                                  'Expected Loss', 'Total Interest Revenue', 'Avg Interest Rate', 
                                  'Net Profit', 'Profit Margin %']]
        
        st.dataframe(tier_stats.style.format({
            'Total Loan Amount': '${:,.0f}',
            'Avg Default Prob': '{:.2%}',
            'Expected Loss': '${:,.0f}',
            'Total Interest Revenue': '${:,.0f}',
            'Avg Interest Rate': '{:.2%}',
            'Net Profit': '${:,.0f}',
            'Profit Margin %': '{:.2f}%'
        }), use_container_width=True)
        
        st.divider()
        
        # Visualizations
        st.subheader("Visual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Applicants per tier
            fig, ax = plt.subplots(figsize=(10, 6))
            tier_counts = pricing_df['approval_tier'].value_counts()
            tier_order = ['Prime', 'Near-Prime', 'Subprime', 'High-Risk']
            tier_counts = tier_counts.reindex(tier_order)
            colors_tier = {'Prime': 'green', 'Near-Prime': 'blue', 'Subprime': 'orange', 'High-Risk': 'red'}
            tier_colors = [colors_tier.get(t, 'gray') for t in tier_counts.index]
            
            bars = ax.bar(tier_counts.index, tier_counts.values, color=tier_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Number of Applicants', fontweight='bold', fontsize=11)
            ax.set_title('Applicants by Risk Tier', fontweight='bold', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, v in zip(bars, tier_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(v)}', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Net Profit by tier
            fig, ax = plt.subplots(figsize=(10, 6))
            tier_profit = pricing_df.groupby('approval_tier')['net_profit'].sum()
            tier_profit = tier_profit.reindex(tier_order)
            tier_colors = [colors_tier.get(t, 'gray') for t in tier_profit.index]
            
            bars = ax.bar(tier_profit.index, tier_profit.values/1e6, color=tier_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Net Profit ($ Millions)', fontweight='bold', fontsize=11)
            ax.set_title('Net Profit by Risk Tier', fontweight='bold', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, v in zip(bars, tier_profit.values/1e6):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'${v:.1f}M', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Default probability by tier
            fig, ax = plt.subplots(figsize=(10, 6))
            tier_default = pricing_df.groupby('approval_tier')['default_probability'].mean() * 100
            tier_default = tier_default.reindex(tier_order)
            tier_colors = [colors_tier.get(t, 'gray') for t in tier_default.index]
            
            bars = ax.bar(tier_default.index, tier_default.values, color=tier_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Average Default Probability (%)', fontweight='bold', fontsize=11)
            ax.set_title('Default Probability by Risk Tier', fontweight='bold', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, v in zip(bars, tier_default.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            # Add tier thresholds
            ax.axhline(15, color='green', linestyle='--', alpha=0.5, linewidth=1, label='Tier Thresholds')
            ax.axhline(30, color='blue', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(50, color='orange', linestyle='--', alpha=0.5, linewidth=1)
            
            ax.grid(axis='y', alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            # Interest rates by tier
            fig, ax = plt.subplots(figsize=(10, 6))
            tier_rates = pricing_df.groupby('approval_tier')['suggested_interest_rate'].mean() * 100
            tier_rates = tier_rates.reindex(tier_order)
            tier_colors = [colors_tier.get(t, 'gray') for t in tier_rates.index]
            
            bars = ax.bar(tier_rates.index, tier_rates.values, color=tier_colors, alpha=0.7, edgecolor='black', linewidth=1.5)
            ax.set_ylabel('Average Interest Rate (%)', fontweight='bold', fontsize=11)
            ax.set_title('Suggested Interest Rates by Approval Tier', fontweight='bold', fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            for bar, v in zip(bars, tier_rates.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{v:.2f}%', ha='center', va='bottom', fontweight='bold')
            
            ax.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)
        
        st.divider()
        
        # Business Insights
        st.subheader("Key Business Insights")
        
        prime_count = len(pricing_df[pricing_df['approval_tier'] == 'Prime'])
        high_risk_count = len(pricing_df[pricing_df['approval_tier'] == 'High-Risk'])
        prime_profit = pricing_df[pricing_df['approval_tier'] == 'Prime']['net_profit'].sum()
        high_risk_profit = pricing_df[pricing_df['approval_tier'] == 'High-Risk']['net_profit'].sum()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Prime Tier Performance:**
            - Applicants: {prime_count}
            - Total Profit: ${prime_profit:,.0f}
            - Profit per Applicant: ${prime_profit/prime_count if prime_count > 0 else 0:,.0f}
            - These are your best customers - approve all of them
            """)
        
        with col2:
            st.warning(f"""
            **High-Risk Tier Strategy:**
            - Applicants: {high_risk_count}
            - Total Profit: ${high_risk_profit:,.0f}
            - Profit per Applicant: ${high_risk_profit/high_risk_count if high_risk_count > 0 else 0:,.0f}
            - Require collateral to reduce expected loss by 50%
            """)

# ========== TAB 4: ABOUT ==========

with tab4:
    st.header("About This Project")
    
    st.subheader("Project Overview")
    st.markdown("""
    This project implements an **intelligent  loan pricing and risk management system** for Australian banks, with regard to personal loan applications.
    
    **Key Achievement:**
    - Collateral strategy increases annual net profit by 318.9%
    - Maintains 100% approval rate with risk-appropriate collateral requirements
    - Reduces expected loss by 42% through collateral backing
    """)
    
    st.subheader("Model Performance Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", "80.5%")
    with col2:
        st.metric("Precision", "82.89%")
    with col3:
        st.metric("Recall", "86.85%")
    with col4:
        st.metric("F1-Score", "84.82%")
    with col5:
        st.metric("AUC-ROC", "84.08%")
    
    st.subheader("Methodology")
    st.markdown("""
    **1. Default Prediction Model**
    - Random Forest classifier trained on 610+ Australian loan applications
    - High performance: AUC-ROC 84.08%, Accuracy 80.5%
    - Outputs: Probability of default (0-100%) for each applicant
    
    **2. Risk-Based Pricing**
    - Expected Loss = Default Probability × Loss Given Default (35%) × Loan Amount
    - Risk Premium = Expected Loss converted to basis points
    - Suggested Rate = Base Rate (5%) + Risk Premium
    
    **3. Customer Segmentation**
    - **Prime (0-15% risk):** Approve at standard rates - Low risk, high profit
    - **Near-Prime (15-30% risk):** Approve at elevated rates - Moderate risk, good profit
    - **Subprime (30-50% risk):** Conditional - require collateral/co-signer
    - **High-Risk (50%+ risk):** Conditional - require collateral/alternative lender
    
    **4. Recommended Strategy: Collateral**
    - Instead of rejecting high-risk customers, we require collateral to mitigate possible losses
    - Collateral reduces expected loss by approximately 50%
    - Results in 2.67x profit increase with 100% approval rate
    """)
    
    st.subheader("Australian Regulatory Compliance")
    st.markdown("""
    **ASIC Credit Management Laws**
    - Objective financial metrics defends against the usage of risk-based pricing
    - Customers are provided explanations for decisions
    
    **Privacy Act 1988 (Cth)**
    - Model uses only permitted data (employment, income, debt) and creates relevant features for risk assessment
    - No sensitive personal information required as synthetic data is used for modeling
    
    **Equal Opportunity Laws**
    - Risk-based pricing does not discriminate by protected attributes
    - Age/gender/location are not primary decision factors
    - The focus is on financial capacity and risk indicators
    
    """)
    
    st.subheader("Business Impact")
    st.markdown("""
    **Revenue & Profitability (Collateral Strategy):**
    - Annual net profit: $5,179,073
    - Profit margin: 48.40%
    - Expected loss: $5,521,527 (managed with collateral)
    - Revenue improvement: $3.94M annually
    
    **Customer Experience:**
    - 100% approval rate (vs. 34% rejection with alternative strategies)
    - Transparent, risk-appropriate pricing
    - Collateral option provides path to credit access
    - Supports financial inclusion
    
    **Risk Management:**
    - Portfolio diversified across risk tiers
    - Quarterly model retraining ensures accuracy
    - Proactive monitoring of at-risk borrowers
    - Collateral backing reduces loss exposure
    """)
    
    st.subheader("Project Structure")
    st.markdown("""
    **Notebooks:**
    1. `01_eda_and_features.ipynb` - Data exploration & feature engineering
    2. `02_modeling_and_evaluation.ipynb` - Model training & evaluation
    3. `03_pricing_and_risk_analysis.ipynb` - Pricing calculations
    4. `04_strategic_analysis_and_scenario_testing.ipynb` - Scenario comparison
    
    **Models & Data:**
    - `models/best_model.pkl` - Trained Random Forest model
    - `models/scaler.pkl` - Fitted StandardScaler
    - `data/processed/X_test.csv` - Test features
    - `data/processed/y_test.csv` - Test labels
    
    **App:**
    - `app.py` - Streamlit dashboard
    """)
    
    st.divider()
    
    st.markdown("""
    **Built with:** Python | Scikit-learn | Streamlit | Pandas
    
    **For:** Australian Banking Context | CommBank Technology Internship
    
    **Author:** Pranamya Rajbhandari | Sydney, Australia
    """)