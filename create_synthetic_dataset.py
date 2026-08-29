#Create Synthetic Dataset for Loan Approved Predictor
import pandas as pd
import numpy as np
from scipy import stats
from datetime import datetime, timedelta
import os

# Number of samples
num_samples = 2000

# Seed for reproducibility
np.random.seed(42)

def generate_correlated_features(num_samples):
    # Generate base features
    age = np.random.normal(40, 12, num_samples).clip(18, 80).astype(int)
    experience = (age - 18 - np.random.normal(4, 2, num_samples).clip(0)).clip(0).astype(int)
    education_level = np.random.choice(['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'], num_samples, p=[0.3, 0.2, 0.3, 0.15, 0.05])

    # Education affects income and credit score
    edu_impact = {'High School': 0, 'Associate': 0.1, 'Bachelor': 0.2, 'Master': 0.3, 'Doctorate': 0.4}
    edu_factor = np.array([edu_impact[level] for level in education_level])

    # --- FIX #3: generate employment status BEFORE income, so income can depend on it ---
    employment_status_probs = np.column_stack([
        0.9 - edu_factor * 0.3,  # Employed
        0.05 + edu_factor * 0.2,  # Self-Employed
        0.05 + edu_factor * 0.1   # Unemployed
    ])
    employment_status = np.array(['Employed', 'Self-Employed', 'Unemployed'])[np.argmax(np.random.random(num_samples)[:, np.newaxis] < employment_status_probs.cumsum(axis=1), axis=1)]

    # Employment status now directly scales income potential
    employment_income_factor = np.where(
        employment_status == 'Unemployed', 0.15,       # unemployed -> mostly other/passive income, much lower
        np.where(employment_status == 'Self-Employed', 1.05,  # self-employed -> slightly higher variance/upside
                 1.0)                                          # employed -> baseline
    )

    # Generate correlated income, credit score, and employment status
    base_income = np.random.lognormal(10.5, 0.6, num_samples) * (1 + edu_factor) * (1 + experience / 100)
    income_noise = np.random.normal(0, 0.1, num_samples)
    annual_income = (base_income * (1 + income_noise) * employment_income_factor).clip(0, 300000)
    # Unemployed floor much lower than employed floor
    annual_income = np.where(employment_status == 'Unemployed',
                              annual_income.clip(0, 40000),
                              annual_income.clip(15000, 300000)).astype(int)

    credit_score_base = 300 + 300 * stats.beta.rvs(5, 1.5, size=num_samples)
    credit_score = (credit_score_base + edu_factor * 100 + experience * 1.5 + income_noise * 100).clip(300, 850).astype(int)

    return age, experience, education_level, annual_income, credit_score, employment_status

def generate_time_based_features(num_samples, start_date=datetime(2018, 1, 1), end_date=datetime(2023, 12, 31)):
    """
    --- FIX #7: instead of exactly one application per day (perfectly uniform),
    randomly sample dates across the range, with mild seasonality
    (more applications in spring/summer) so a monthly time-series plot
    looks realistically bursty rather than flat.
    """
    total_days = (end_date - start_date).days
    day_offsets = np.arange(total_days + 1)
    dates_all = np.array([start_date + timedelta(days=int(d)) for d in day_offsets])
    months = np.array([d.month for d in dates_all])

    # Seasonal weight: slightly higher probability in Mar-Aug
    weights = np.where((months >= 3) & (months <= 8), 1.3, 1.0)
    weights = weights / weights.sum()

    sampled_dates = np.random.choice(dates_all, size=num_samples, p=weights, replace=True)
    sampled_dates = sorted(sampled_dates)  # keep chronological order like an application log
    return list(sampled_dates)

def generate_loan_purpose_amount_duration(num_samples, annual_income):
    """
    --- FIX #1: LoanPurpose determines both LoanAmount AND LoanDuration together,
    so a $300k Home loan can no longer come with a 12-month term.
    """
    loan_purpose = np.random.choice(
        ['Home', 'Auto', 'Education', 'Debt Consolidation', 'Other'],
        num_samples, p=[0.3, 0.2, 0.15, 0.25, 0.1]
    )

    # (mu, sigma) for lognormal amount per purpose
    purpose_loan_params = {
        'Home':               (12.0, 0.35),  # median ~$160k
        'Auto':                (10.0, 0.30),  # median ~$22k
        'Education':           (10.2, 0.40),  # median ~$27k
        'Debt Consolidation':  (9.5, 0.45),   # median ~$13k
        'Other':               (9.3, 0.55),   # median ~$11k
    }
    purpose_min = {'Home': 60000, 'Auto': 5000, 'Education': 3000, 'Debt Consolidation': 2000, 'Other': 1000}

    # Realistic duration options (months) and weights per purpose
    purpose_duration_options = {
        'Home':               ([120, 180, 240, 300, 360], [0.05, 0.15, 0.25, 0.25, 0.30]),
        'Auto':                ([24, 36, 48, 60, 72],       [0.10, 0.25, 0.30, 0.25, 0.10]),
        'Education':           ([36, 60, 84, 120, 180],      [0.15, 0.30, 0.25, 0.20, 0.10]),
        'Debt Consolidation':  ([12, 24, 36, 48, 60],        [0.15, 0.30, 0.30, 0.15, 0.10]),
        'Other':               ([12, 24, 36, 48],            [0.30, 0.35, 0.25, 0.10]),
    }

    loan_amount = np.empty(num_samples)
    loan_duration = np.empty(num_samples, dtype=int)

    for i, purpose in enumerate(loan_purpose):
        mu, sigma = purpose_loan_params[purpose]
        loan_amount[i] = np.random.lognormal(mu, sigma)

        options, probs = purpose_duration_options[purpose]
        loan_duration[i] = np.random.choice(options, p=probs)

    # Scale amount slightly by income so higher earners tend toward larger loans within their category
    income_scale = 1 + (annual_income - annual_income.mean()) / (annual_income.mean() * 4)
    loan_amount = loan_amount * income_scale.clip(0.7, 1.5)

    floors = np.array([purpose_min[p] for p in loan_purpose])
    loan_amount = np.maximum(loan_amount, floors).astype(int)

    return loan_purpose, loan_amount, loan_duration

age, experience, education_level, annual_income, credit_score, employment_status = generate_correlated_features(num_samples)
application_dates = generate_time_based_features(num_samples)
loan_purpose, loan_amount, loan_duration = generate_loan_purpose_amount_duration(num_samples, annual_income)

# --- FIX #5: tie MonthlyDebtPayments, Savings, and Checking balances to income level ---
# so a minimum-wage earner isn't randomly sitting on a $50k savings balance.
income_ratio = annual_income / annual_income.mean()

monthly_debt_payments = (np.random.lognormal(6, 0.5, num_samples) * income_ratio.clip(0.4, 2.5)).astype(int)
savings_account_balance = (np.random.lognormal(8, 1, num_samples) * income_ratio.clip(0.3, 3.0)).astype(int)
checking_account_balance = (np.random.lognormal(7, 1, num_samples) * income_ratio.clip(0.3, 3.0)).astype(int)

data = {
    'ApplicationDate': application_dates,
    'Age': age,
    'AnnualIncome': annual_income,
    'CreditScore': credit_score,
    'EmploymentStatus': employment_status,
    'EducationLevel': education_level,
    'Experience': experience,
    'LoanAmount': loan_amount,
    'LoanDuration': loan_duration,
    'MaritalStatus': np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], num_samples, p=[0.3, 0.5, 0.15, 0.05]),
    'NumberOfDependents': np.random.choice([0, 1, 2, 3, 4, 5], num_samples, p=[0.3, 0.25, 0.2, 0.15, 0.07, 0.03]),
    'HomeOwnershipStatus': np.random.choice(['Own', 'Rent', 'Mortgage', 'Other'], num_samples, p=[0.2, 0.3, 0.4, 0.1]),
    'MonthlyDebtPayments': monthly_debt_payments,
    'CreditCardUtilisationRate': np.random.beta(2, 5, num_samples),
    'NumberOfOpenCreditLines': np.random.poisson(3, num_samples).clip(0, 15).astype(int),
    'NumberOfCreditInquiries': np.random.poisson(1, num_samples).clip(0, 10).astype(int),
    'BankruptcyHistory': np.random.choice([0, 1], num_samples, p=[0.95, 0.05]),
    'LoanPurpose': loan_purpose,
    'PreviousLoanDefaults': np.random.choice([0, 1], num_samples, p=[0.9, 0.1]),
    'PaymentHistory': np.random.normal(97,3,num_samples).clip(70,100).astype(int),
    'LengthOfCreditHistory': np.random.randint(1, 30, num_samples),
    'SavingsAccountBalance': savings_account_balance,
    'CheckingAccountBalance': checking_account_balance,
    'TotalAssets': np.random.lognormal(11, 1, num_samples).astype(int),
    'TotalLiabilities': np.random.lognormal(10, 1, num_samples).astype(int),
    'MonthlyIncome': annual_income / 12,
    'UtilityBillsPaymentHistory': np.random.beta(8, 2, num_samples),
    'JobTenure': np.random.poisson(5, num_samples).clip(0, 40).astype(int),
}

# Create DataFrame
df = pd.DataFrame(data)

# Ensure TotalAssets is always greater than or equal to the sum of SavingsAccountBalance and CheckingAccountBalance
df['TotalAssets'] = np.maximum(df['TotalAssets'], df['SavingsAccountBalance'] + df['CheckingAccountBalance'])

# Add more complex derived features
min_net_worth = 1000  # Set a minimum net worth
df['NetWorth'] = np.maximum(df['TotalAssets'] - df['TotalLiabilities'], min_net_worth)

# --- FIX #6: interest rate no longer scales directly (and unrealistically) with raw loan amount.
# It's driven mainly by credit score and loan duration/risk, with a much smaller,
# log-scaled nod to loan size instead of a raw linear one.
df['BaseInterestRate'] = (
    0.03
    + (850 - df['CreditScore']) / 2000        # worse credit -> higher rate
    + df['LoanDuration'] / 3000                # longer term -> slightly higher rate
    + np.log1p(df['LoanAmount']) / 4000        # log-scaled loan size effect (much gentler than raw/1e6)
)
df['InterestRate'] = (df['BaseInterestRate'] * (1 + np.random.normal(0, 0.1, num_samples))).clip(0.02, 0.25)

df['MonthlyLoanPayment'] = (df['LoanAmount'] * (df['InterestRate']/12)) / (1 - (1 + df['InterestRate']/12)**(-df['LoanDuration']))

# --- FIX #4: single, consistent Debt-to-Income ratio, computed from real numbers
# instead of having one random beta-distributed column and one calculated column
# that disagree with each other.
df['DebtToIncomeRatio'] = (df['MonthlyDebtPayments'] / (df['MonthlyIncome'] + 1)).clip(0, 2)
df['TotalDebtToIncomeRatio'] = ((df['MonthlyDebtPayments'] + df['MonthlyLoanPayment']) / (df['MonthlyIncome'] + 1)).clip(0, 3)

# Create a more complex loan approval rule
def loan_approval_rule(row):
    score = 0
    score += (row['CreditScore'] - 650) / 350
    score += (80000 - row['AnnualIncome']) / 150000
    score += (row['TotalDebtToIncomeRatio'] - 0.35) * 1.5
    score += (row['LoanAmount'] - 25000) / 150000
    score += (row['InterestRate'] - 0.06) * 5
    score += 0.5 if row['BankruptcyHistory'] == 1 else 0  # Bankruptcy penalty
    score += 0.3 if row['PreviousLoanDefaults'] == 1 else 0  # Previous default penalty
    score += 0.2 if row['EmploymentStatus'] == 'Unemployed' else 0  # Employment status factor
    score -= 0.1 if row['HomeOwnershipStatus'] in ['Own', 'Mortgage'] else 0  # Home ownership factor
    score -= row['PaymentHistory'] / 120  # Payment history factor
    score -= row['LengthOfCreditHistory'] / 60  # Length of credit history factor
    score -= row['NetWorth'] / 500000  # Net worth factor

    # Age factor (slight preference for middle-aged applicants)
    score += abs(row['Age'] - 40) / 100

    # Experience factor
    score -= row['Experience'] / 200

    # Education factor
    edu_score = {'High School': 0.2, 'Associate': 0.1, 'Bachelor': 0, 'Master': -0.1, 'Doctorate': -0.2}
    score += edu_score[row['EducationLevel']]

    # Seasonal factor (higher approval rates in spring/summer)
    month = row['ApplicationDate'].month
    score -= 0.1 if 3 <= month <= 8 else 0

    # Low credit score + high debt is particularly risky
    if row['CreditScore'] < 620 and row['TotalDebtToIncomeRatio'] > 0.45:
        score += 0.8

    # High income offsets larger loans
    if row['AnnualIncome'] > 120000 and row['LoanAmount'] < 80000:
        score -= 0.4

    # Unemployed applicants with low savings are higher risk
    if (
        row['EmploymentStatus'] == 'Unemployed'
        and row['SavingsAccountBalance'] < 5000
    ):
        score += 0.6

    # Long employment and good credit reduce risk
    if (
        row['Experience'] > 10
        and row['CreditScore'] > 720
    ):
        score -= 0.3

    # Random factor to add some unpredictability
    score += np.random.normal(0, 0.5)

    
    # Adjust this threshold to change overall approval rate
    approval_probability = 1 / (1 + np.exp(score - 1))
    return np.random.binomial(1, approval_probability)

df['LoanApproved'] = df.apply(loan_approval_rule, axis=1)

# Add some noise and outliers
noise_mask = np.random.choice([True, False], num_samples, p=[0.01, 0.99])
df.loc[noise_mask, 'AnnualIncome'] = (df.loc[noise_mask, 'AnnualIncome'] * np.random.uniform(1.5, 2.0, noise_mask.sum())).astype(int)

low_net_worth_mask = df['NetWorth'] == min_net_worth
df.loc[low_net_worth_mask, 'NetWorth'] += np.random.randint(0, 10000, size=low_net_worth_mask.sum())

# Print some statistics
print(f"Loan Approval Rate: {df['LoanApproved'].mean():.2%}")
print(f"Average Credit Score: {df['CreditScore'].mean():.0f}")
print(f"Average Annual Income: ${df['AnnualIncome'].mean():.0f}")
print(f"Average Loan Amount: ${df['LoanAmount'].mean():.0f}")
print(f"Average Total Debt-to-Income Ratio: {df['TotalDebtToIncomeRatio'].mean():.2f}")
print(f"Average Interest Rate: {df['InterestRate'].mean():.2%}")

# Sanity checks for the fixes
print("\nLoan Amount by Purpose (median):")
print(df.groupby('LoanPurpose')['LoanAmount'].median().sort_values(ascending=False))

print("\nLoan Duration by Purpose (median, months):")
print(df.groupby('LoanPurpose')['LoanDuration'].median().sort_values(ascending=False))

print("\nAnnual Income by Employment Status (median):")
print(df.groupby('EmploymentStatus')['AnnualIncome'].median().sort_values(ascending=False))

def assign_credit_score_risk(credit_score):
    if credit_score >= 750: return 1
    elif 700 <= credit_score < 750: return 2
    elif 650 <= credit_score < 700: return 3
    elif 600 <= credit_score < 650: return 4
    else: return 5

def assign_dti_risk(dti):
    if dti < 0.20: return 1
    elif 0.20 <= dti < 0.30: return 2
    elif 0.30 <= dti < 0.40: return 3
    elif 0.40 <= dti < 0.50: return 4
    else: return 5

def assign_payment_history_risk(payment_history):
    if payment_history >= 99: return 1
    elif 97 <= payment_history < 99: return 2
    elif 95 <= payment_history < 97: return 3
    elif 90 <= payment_history < 95: return 4
    else: return 5

def assign_bankruptcy_risk(bankruptcy_history):
    return 5 if bankruptcy_history else 1

def assign_previous_defaults_risk(previous_defaults):
    if previous_defaults == 0: return 1
    elif previous_defaults == 1: return 3
    else: return 5

def assign_utilisation_risk(utilisation):
    if utilisation < 0.20: return 1
    elif 0.20 <= utilisation < 0.40: return 2
    elif 0.40 <= utilisation < 0.60: return 3
    elif 0.60 <= utilisation < 0.80: return 4
    else: return 5

def assign_credit_history_risk(length_of_history):
    if length_of_history >= 10: return 1
    elif 7 <= length_of_history < 10: return 2
    elif 5 <= length_of_history < 7: return 3
    elif 3 <= length_of_history < 5: return 4
    else: return 5

def assign_income_risk(annual_income):
    if annual_income >= 120000: return 1
    elif 80000 <= annual_income < 120000: return 2
    elif 50000 <= annual_income < 80000: return 3
    elif 30000 <= annual_income < 50000: return 4
    else: return 5

def assign_employment_risk(employment_status):
    # --- FIX #2: corrected string matching ('Self-Employed' with capital E,
    # matching what's actually generated). Removed the dead 'Part-time'
    # branch since that category is never generated, and it was silently
    # dumping Self-Employed applicants into the worst-risk bucket.
    if employment_status == 'Employed':
        return 1
    elif employment_status == 'Self-Employed':
        return 2
    else:  # Unemployed
        return 4

def assign_net_worth_risk(net_worth):
    if net_worth >= 500000: return 1
    elif 250000 <= net_worth < 500000: return 2
    elif 100000 <= net_worth < 250000: return 3
    elif 50000 <= net_worth < 100000: return 4
    else: return 5

# Refined overall risk calculation
def calculate_overall_risk(row):
    risk = 0
    risk += (850 - row['CreditScore']) / 550 * 3
    risk += row['TotalDebtToIncomeRatio'] * 2
    risk += (60 - row['PaymentHistory']) / 60
    risk += row['BankruptcyHistory'] * 2
    risk += row['PreviousLoanDefaults'] * 1.5
    risk += max(0, (80000-row['AnnualIncome'])/80000)
    risk += np.random.normal(0,0.5)

    return max(0, risk)


# Apply the refined risk calculation
df['RiskScore'] = df.apply(calculate_overall_risk, axis=1)

#create dataset file path
folder_path = r"data//raw"
os.makedirs(folder_path, exist_ok=True)


# Save to CSV
df.to_csv(os.path.join(folder_path, 'focused_synthetic_loan_data.csv'), index=False)
print("\nFocused synthetic data saved to 'focused_synthetic_loan_data.csv'")

# Display final feature count
print(f"\nTotal number of features (including label): {len(df.columns)}")
print("\nFeatures:")
for column in df.columns:
    print(f"- {column}")


