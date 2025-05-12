"""
# B2B2C Health Recommendation System Prototype
# ---------------------------------------------
# A modular health recommendation engine that combines rules-based logic,
# similarity matching, and ML-based recommendations.
"""

# Standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import random
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.impute import SimpleImputer

# Streamlit for UI
import streamlit as st

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_theme(style="whitegrid")

# =============================================================================
# DATA GENERATION
# =============================================================================

def generate_synthetic_data(num_users=1000):
    """
    Generate synthetic health and wellness data for a population of users
    """
    print("Generating synthetic data for", num_users, "users...")

    # Companies and teams
    companies = ['TechCorp', 'HealthNet', 'FinGroup', 'RetailGiant', 'EduSystems']
    teams_by_company = {
        'TechCorp': ['Engineering', 'Design', 'Product', 'Support', 'Sales'],
        'HealthNet': ['Nursing', 'Administration', 'Research', 'Operations', 'IT'],
        'FinGroup': ['Investment', 'Analysis', 'Compliance', 'HR', 'Tech'],
        'RetailGiant': ['Logistics', 'Marketing', 'Store Ops', 'Digital', 'Support'],
        'EduSystems': ['Faculty', 'Administration', 'IT', 'Research', 'Student Services']
    }

    # Generate user demographics
    user_ids = [f"U{i:04d}" for i in range(1, num_users + 1)]
    age = np.random.normal(40, 10, num_users).astype(int)
    age = np.clip(age, 22, 65)  # Clip to reasonable working age range
    gender = np.random.choice(['M', 'F', 'Other'], num_users, p=[0.48, 0.48, 0.04])
    company = np.random.choice(companies, num_users)

    # Assign teams based on company
    team = []
    for c in company:
        team.append(np.random.choice(teams_by_company[c]))

    # Generate HRA (Health Risk Assessment) data
    # Scale 1-10 where 10 is healthiest behavior
    smoking_score = np.random.choice(range(1, 11), num_users, p=[0.15, 0.05, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.2, 0.2])
    sleep_score = np.random.normal(6, 2, num_users).astype(int)
    sleep_score = np.clip(sleep_score, 1, 10)
    diet_score = np.random.normal(5.5, 2, num_users).astype(int)
    diet_score = np.clip(diet_score, 1, 10)
    exercise_score = np.random.gamma(shape=2, scale=1.5, size=num_users).astype(int)
    exercise_score = np.clip(exercise_score, 1, 10)
    stress_score = np.random.normal(5, 2, num_users).astype(int)
    stress_score = np.clip(stress_score, 1, 10)

    # Create correlation between health behaviors (people who exercise tend to eat better, etc.)
    for i in range(num_users):
        if exercise_score[i] > 7:
            diet_score[i] = min(10, diet_score[i] + np.random.randint(0, 3))
            smoking_score[i] = min(10, smoking_score[i] + np.random.randint(0, 2))
        if smoking_score[i] < 4:  # Heavy smokers
            exercise_score[i] = max(1, exercise_score[i] - np.random.randint(0, 3))
            sleep_score[i] = max(1, sleep_score[i] - np.random.randint(0, 2))

    # Generate EMR (Electronic Medical Record) data
    # BMI data that correlates with diet/exercise
    base_bmi = np.random.normal(26, 4, num_users)
    bmi = []
    for i in range(num_users):
        # Adjust BMI based on diet and exercise
        modifier = (10 - diet_score[i])/10 + (10 - exercise_score[i])/10
        bmi.append(max(16, min(40, base_bmi[i] + modifier * 3)))

    # Blood pressure that correlates with BMI, smoking, stress
    systolic_bp = []
    diastolic_bp = []
    for i in range(num_users):
        base_systolic = 110 + (bmi[i] - 20) * 1.5
        base_diastolic = 70 + (bmi[i] - 20) * 0.8

        # Add effects of smoking and stress
        systolic_modifier = (10 - smoking_score[i]) * 1.2 + (10 - stress_score[i]) * 0.8
        diastolic_modifier = (10 - smoking_score[i]) * 0.6 + (10 - stress_score[i]) * 0.4

        # Add random variation
        sys_bp = int(base_systolic + systolic_modifier + np.random.normal(0, 5))
        dia_bp = int(base_diastolic + diastolic_modifier + np.random.normal(0, 3))

        systolic_bp.append(max(90, min(180, sys_bp)))
        diastolic_bp.append(max(60, min(110, dia_bp)))

    # Blood glucose and cholesterol levels (loosely correlated with diet/exercise/BMI)
    glucose_level = []
    cholesterol_level = []
    for i in range(num_users):
        base_glucose = 85 + (bmi[i] - 20) * 1.2
        base_cholesterol = 180 + (bmi[i] - 20) * 2

        # Add effects of diet
        glucose_modifier = (10 - diet_score[i]) * 2 + (10 - exercise_score[i]) * 1
        cholesterol_modifier = (10 - diet_score[i]) * 3 + (10 - exercise_score[i]) * 2

        # Add random variation
        glucose = int(base_glucose + glucose_modifier + np.random.normal(0, 8))
        chol = int(base_cholesterol + cholesterol_modifier + np.random.normal(0, 15))

        glucose_level.append(max(70, min(180, glucose)))
        cholesterol_level.append(max(130, min(300, chol)))

    # Generate diagnoses
    diagnoses = []
    for i in range(num_users):
        user_diagnoses = []

        # Hypertension
        if systolic_bp[i] > 140 or diastolic_bp[i] > 90:
            p_hypertension = 0.7
            if systolic_bp[i] > 160 or diastolic_bp[i] > 100:
                p_hypertension = 0.95
            if random.random() < p_hypertension:
                user_diagnoses.append('Hypertension')

        # Diabetes
        if glucose_level[i] > 125:
            p_diabetes = 0.7
            if glucose_level[i] > 140:
                p_diabetes = 0.95
            if random.random() < p_diabetes:
                user_diagnoses.append('Diabetes')

        # High Cholesterol
        if cholesterol_level[i] > 240:
            p_high_chol = 0.8
            if random.random() < p_high_chol:
                user_diagnoses.append('High Cholesterol')

        # Obesity
        if bmi[i] > 30:
            p_obesity = 0.8
            if bmi[i] > 35:
                p_obesity = 0.95
            if random.random() < p_obesity:
                user_diagnoses.append('Obesity')

        # Anxiety/Depression (related to stress)
        if stress_score[i] < 4:
            p_anxiety_depression = 0.4
            if stress_score[i] < 2:
                p_anxiety_depression = 0.7
            if random.random() < p_anxiety_depression:
                condition = random.choice(['Anxiety', 'Depression'])
                user_diagnoses.append(condition)

        # Sleep Disorders
        if sleep_score[i] < 3:
            p_sleep_disorder = 0.5
            if random.random() < p_sleep_disorder:
                user_diagnoses.append('Sleep Disorder')

        # Heart Disease Risk
        heart_disease_risk = 0
        if age[i] > 50: heart_disease_risk += 1
        if smoking_score[i] < 5: heart_disease_risk += 1
        if cholesterol_level[i] > 240: heart_disease_risk += 1
        if systolic_bp[i] > 140: heart_disease_risk += 1
        if 'Diabetes' in user_diagnoses: heart_disease_risk += 1

        if heart_disease_risk >= 3:
            p_heart_disease = 0.3
            if heart_disease_risk >= 4:
                p_heart_disease = 0.5
            if random.random() < p_heart_disease:
                user_diagnoses.append('Heart Disease')

        # Some people will have no diagnoses
        if not user_diagnoses and random.random() < 0.9:
            user_diagnoses.append('None')

        diagnoses.append(','.join(user_diagnoses))

    # Generate health risk predictions
    # Simple risk scores based on various factors
    diabetes_risk = []
    hypertension_risk = []
    cvd_risk = []

    for i in range(num_users):
        # Diabetes risk factors
        d_risk = 0
        if bmi[i] > 30: d_risk += 0.2
        if age[i] > 45: d_risk += 0.15
        if glucose_level[i] > 100: d_risk += 0.3
        if 'Diabetes' in diagnoses[i]: d_risk += 0.3
        d_risk += random.uniform(-0.05, 0.05)  # Add some noise
        d_risk = max(0.01, min(0.99, d_risk))  # Clamp between 0.01 and 0.99
        diabetes_risk.append(d_risk)

        # Hypertension risk factors
        h_risk = 0
        if systolic_bp[i] > 130: h_risk += 0.2
        if diastolic_bp[i] > 85: h_risk += 0.2
        if age[i] > 50: h_risk += 0.1
        if bmi[i] > 28: h_risk += 0.1
        if smoking_score[i] < 6: h_risk += 0.1
        if stress_score[i] < 4: h_risk += 0.1
        if 'Hypertension' in diagnoses[i]: h_risk += 0.3
        h_risk += random.uniform(-0.05, 0.05)  # Add some noise
        h_risk = max(0.01, min(0.99, h_risk))  # Clamp between 0.01 and 0.99
        hypertension_risk.append(h_risk)

        # Cardiovascular disease risk factors
        c_risk = 0
        if age[i] > 55: c_risk += 0.15
        if smoking_score[i] < 5: c_risk += 0.15
        if cholesterol_level[i] > 220: c_risk += 0.15
        if systolic_bp[i] > 140: c_risk += 0.15
        if 'Diabetes' in diagnoses[i]: c_risk += 0.15
        if 'Hypertension' in diagnoses[i]: c_risk += 0.15
        if 'Heart Disease' in diagnoses[i]: c_risk += 0.3
        c_risk += random.uniform(-0.05, 0.05)  # Add some noise
        c_risk = max(0.01, min(0.99, c_risk))  # Clamp between 0.01 and 0.99
        cvd_risk.append(c_risk)

    # Generate prescription data based on diagnoses
    prescriptions = []
    for i in range(num_users):
        user_prescriptions = []
        user_diag = diagnoses[i].split(',')

        if 'Hypertension' in user_diag:
            bp_med = random.choice(['Lisinopril', 'Amlodipine', 'Losartan', 'Hydrochlorothiazide'])
            user_prescriptions.append(bp_med)

        if 'Diabetes' in user_diag:
            if random.random() < 0.8:
                d_med = random.choice(['Metformin', 'Glipizide', 'Januvia'])
                user_prescriptions.append(d_med)
            if glucose_level[i] > 160 and random.random() < 0.3:
                user_prescriptions.append('Insulin')

        if 'High Cholesterol' in user_diag:
            if random.random() < 0.7:
                chol_med = random.choice(['Atorvastatin', 'Simvastatin', 'Rosuvastatin'])
                user_prescriptions.append(chol_med)

        if 'Anxiety' in user_diag:
            if random.random() < 0.6:
                anx_med = random.choice(['Sertraline', 'Escitalopram', 'Alprazolam'])
                user_prescriptions.append(anx_med)

        if 'Depression' in user_diag:
            if random.random() < 0.7:
                dep_med = random.choice(['Fluoxetine', 'Sertraline', 'Bupropion'])
                user_prescriptions.append(dep_med)

        if 'Sleep Disorder' in user_diag:
            if random.random() < 0.5:
                sleep_med = random.choice(['Zolpidem', 'Eszopiclone', 'Melatonin'])
                user_prescriptions.append(sleep_med)

        if 'Heart Disease' in user_diag:
            hd_treatments = ['Aspirin', 'Clopidogrel', 'Metoprolol', 'Atorvastatin']
            num_hd_meds = random.choices([1, 2, 3], weights=[0.2, 0.5, 0.3])[0]
            selected_hd_meds = random.sample(hd_treatments, num_hd_meds)
            user_prescriptions.extend(selected_hd_meds)

        prescriptions.append(','.join(user_prescriptions) if user_prescriptions else 'None')

    # Generate app usage data
    app_login_frequency = np.random.choice(['Daily', 'Weekly', 'Monthly', 'Rarely'], num_users,
                                           p=[0.3, 0.4, 0.2, 0.1])

    # Health tracking and activity data
    avg_daily_steps = []
    avg_sleep_hours = []
    avg_water_intake = []

    for i in range(num_users):
        # Steps relate to exercise score
        base_steps = 3000 + exercise_score[i] * 800
        avg_daily_steps.append(int(base_steps + np.random.normal(0, 1000)))

        # Sleep hours relate to sleep score
        base_sleep = 5 + sleep_score[i] * 0.3
        avg_sleep_hours.append(round(base_sleep + np.random.normal(0, 0.5), 1))

        # Water intake (cups per day)
        base_water = 2 + diet_score[i] * 0.5
        avg_water_intake.append(round(base_water + np.random.normal(0, 1), 1))

    # Doctor and pharmacy bookings
    doctor_visits_6mo = []
    pharmacy_orders_6mo = []

    for i in range(num_users):
        # Doctor visits based on diagnoses and age
        base_visits = 0
        diagnoses_list = diagnoses[i].split(',')

        if 'None' not in diagnoses_list:
            base_visits += len(diagnoses_list) * 0.5

        if age[i] > 50:
            base_visits += 0.5

        # Add some randomness
        doc_visits = int(base_visits + np.random.poisson(0.5))
        doctor_visits_6mo.append(doc_visits)

        # Pharmacy orders based on prescriptions
        base_orders = 0
        prescriptions_list = prescriptions[i].split(',')

        if 'None' not in prescriptions_list:
            base_orders += len(prescriptions_list) * 0.8

        # Add some randomness
        pharm_orders = int(base_orders + np.random.poisson(0.5))
        pharmacy_orders_6mo.append(pharm_orders)

    # Challenge participation
    challenge_types = ['Steps', 'Nutrition', 'Mindfulness', 'Sleep', 'Hydration']
    challenges_completed = []
    preferred_challenge = []

    for i in range(num_users):
        # Higher engagement with app = more challenges
        if app_login_frequency[i] == 'Daily':
            num_challenges = np.random.choice([0, 1, 2, 3, 4], p=[0.05, 0.15, 0.3, 0.3, 0.2])
        elif app_login_frequency[i] == 'Weekly':
            num_challenges = np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.4, 0.15, 0.05])
        elif app_login_frequency[i] == 'Monthly':
            num_challenges = np.random.choice([0, 1, 2], p=[0.4, 0.4, 0.2])
        else:  # Rarely
            num_challenges = np.random.choice([0, 1], p=[0.8, 0.2])

        challenges_completed.append(num_challenges)

        # Preferred challenge type
        if exercise_score[i] >= 7:
            pref_weights = [0.5, 0.2, 0.1, 0.1, 0.1]  # Favors steps
        elif diet_score[i] >= 7:
            pref_weights = [0.2, 0.5, 0.1, 0.1, 0.1]  # Favors nutrition
        elif stress_score[i] <= 4:
            pref_weights = [0.1, 0.1, 0.6, 0.1, 0.1]  # Favors mindfulness
        elif sleep_score[i] <= 4:
            pref_weights = [0.1, 0.1, 0.1, 0.6, 0.1]  # Favors sleep
        else:
            pref_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Equal preference

        preferred_challenge.append(np.random.choice(challenge_types, p=pref_weights))

    # Generate claims data
    had_claim_12mo = []
    claim_amount = []

    for i in range(num_users):
        # Base probability of having a claim
        p_claim = 0.1  # 10% baseline

        # Factors that increase claim probability
        if age[i] > 55: p_claim += 0.1
        if 'Heart Disease' in diagnoses[i]: p_claim += 0.2
        if 'Diabetes' in diagnoses[i]: p_claim += 0.1
        if 'Hypertension' in diagnoses[i]: p_claim += 0.05
        if doctor_visits_6mo[i] >= 2: p_claim += 0.1

        # Cap probability
        p_claim = min(0.8, p_claim)

        # Determine if had claim
        had_claim = np.random.random() < p_claim
        had_claim_12mo.append(had_claim)

        # Generate claim amount if had claim
        if had_claim:
            # Base amount
            base_amount = 500

            # Add based on conditions
            if 'Heart Disease' in diagnoses[i]: base_amount += 5000
            if 'Diabetes' in diagnoses[i]: base_amount += 2000
            if 'Hypertension' in diagnoses[i]: base_amount += 1000

            # Add based on age
            if age[i] > 60: base_amount *= 1.5
            elif age[i] > 50: base_amount *= 1.3
            elif age[i] > 40: base_amount *= 1.1

            # Add randomness (lognormal to simulate occasional large claims)
            amount = int(base_amount * np.random.lognormal(0, 0.5))
            claim_amount.append(amount)
        else:
            claim_amount.append(0)

    # Create the dataframes

    # Create users DataFrame
    users_df = pd.DataFrame({
        'user_id': user_ids,
        'age': age,
        'gender': gender,
        'company': company,
        'team': team
    })

    # Create HRA DataFrame
    hra_df = pd.DataFrame({
        'user_id': user_ids,
        'smoking_score': smoking_score,
        'sleep_score': sleep_score,
        'diet_score': diet_score,
        'exercise_score': exercise_score,
        'stress_score': stress_score
    })

    # Create EMR DataFrame
    emr_df = pd.DataFrame({
        'user_id': user_ids,
        'bmi': [round(b, 1) for b in bmi],
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'glucose_level': glucose_level,
        'cholesterol_level': cholesterol_level,
        'diagnoses': diagnoses,
        'prescriptions': prescriptions
    })

    # Create health risk predictions DataFrame
    risk_df = pd.DataFrame({
        'user_id': user_ids,
        'diabetes_risk': [round(r, 2) for r in diabetes_risk],
        'hypertension_risk': [round(r, 2) for r in hypertension_risk],
        'cvd_risk': [round(r, 2) for r in cvd_risk]
    })

    # Create app usage DataFrame
    usage_df = pd.DataFrame({
        'user_id': user_ids,
        'app_login_frequency': app_login_frequency,
        'avg_daily_steps': avg_daily_steps,
        'avg_sleep_hours': avg_sleep_hours,
        'avg_water_intake': avg_water_intake,
        'doctor_visits_6mo': doctor_visits_6mo,
        'pharmacy_orders_6mo': pharmacy_orders_6mo,
        'challenges_completed': challenges_completed,
        'preferred_challenge': preferred_challenge
    })

    # Create claims DataFrame
    claims_df = pd.DataFrame({
        'user_id': user_ids,
        'had_claim_12mo': had_claim_12mo,
        'claim_amount': claim_amount
    })

    print("✅ Data generation complete!")

    return users_df, hra_df, emr_df, risk_df, usage_df, claims_df

# =============================================================================
# DATA PREPROCESSING
# =============================================================================

def create_feature_matrix(users_df, hra_df, emr_df, risk_df, usage_df):
    """
    Combine all data sources into a single feature matrix for analysis
    """
    print("Creating feature matrix...")

    # Start with users dataframe
    feature_df = users_df.copy()

    # Add HRA data
    feature_df = pd.merge(feature_df, hra_df, on='user_id')

    # Add EMR numerical data
    emr_numeric = emr_df[['user_id', 'bmi', 'systolic_bp', 'diastolic_bp',
                          'glucose_level', 'cholesterol_level']]
    feature_df = pd.merge(feature_df, emr_numeric, on='user_id')

    # Handle diagnoses - create binary flags for common conditions
    conditions = ['Hypertension', 'Diabetes', 'High Cholesterol',
                  'Obesity', 'Anxiety', 'Depression', 'Sleep Disorder', 'Heart Disease']

    for condition in conditions:
        feature_df[f'has_{condition.lower().replace(" ", "_")}'] = \
            emr_df['diagnoses'].apply(lambda x: 1 if condition in x else 0)

    # Handle medications - create binary flags for medication categories
    feature_df['on_bp_medication'] = emr_df['prescriptions'].apply(
        lambda x: 1 if any(med in x for med in ['Lisinopril', 'Amlodipine', 'Losartan', 'Hydrochlorothiazide']) else 0)

    feature_df['on_diabetes_medication'] = emr_df['prescriptions'].apply(
        lambda x: 1 if any(med in x for med in ['Metformin', 'Glipizide', 'Januvia', 'Insulin']) else 0)

    feature_df['on_cholesterol_medication'] = emr_df['prescriptions'].apply(
        lambda x: 1 if any(med in x for med in ['Atorvastatin', 'Simvastatin', 'Rosuvastatin']) else 0)

    feature_df['on_anxiety_depression_medication'] = emr_df['prescriptions'].apply(
        lambda x: 1 if any(med in x for med in ['Sertraline', 'Escitalopram', 'Alprazolam',
                                              'Fluoxetine', 'Bupropion']) else 0)

    # Add risk scores
    feature_df = pd.merge(feature_df, risk_df, on='user_id')

    # Add app usage data
    usage_numeric = usage_df[['user_id', 'avg_daily_steps', 'avg_sleep_hours',
                              'avg_water_intake', 'doctor_visits_6mo',
                              'pharmacy_orders_6mo', 'challenges_completed']]
    feature_df = pd.merge(feature_df, usage_numeric, on='user_id')

    # Add categorical app data
    app_categorical = usage_df[['user_id', 'app_login_frequency', 'preferred_challenge']]
    feature_df = pd.merge(feature_df, app_categorical, on='user_id')

    print(f"✅ Feature matrix created with {feature_df.shape[0]} rows and {feature_df.shape[1]} columns")
    return feature_df

def preprocess_data(feature_df):
    """
    Prepare data for machine learning models
    """
    print("Preprocessing data...")

    # Create a copy to avoid modifying the original
    df = feature_df.copy()

    # Define categorical and numerical columns
    categorical_cols = ['gender', 'company', 'team', 'app_login_frequency', 'preferred_challenge']

    # For numerical columns, we'll use all except user_id and categorical columns
    numerical_cols = [col for col in df.columns if col not in categorical_cols + ['user_id']]

    # Define the preprocessing for numerical and categorical data
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    X = df.drop('user_id', axis=1)

    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    feature_names = numerical_cols.copy()

    # Add one-hot encoded feature names
    ohe = preprocessor.named_transformers_['cat'].named_steps['onehot']
    for i, col in enumerate(categorical_cols):
        feature_names.extend([f"{col}_{cat}" for cat in ohe.categories_[i]])

    print(f"✅ Preprocessing complete. Transformed data shape: {X_processed.shape}")

    return X_processed, preprocessor, feature_names

# =============================================================================
# RECOMMENDATION ENGINE COMPONENTS
# =============================================================================

class HealthRecommendationEngine:
    """
    A modular health recommendation system that provides tailored recommendations
    based on user health profiles
    """
    def __init__(self, feature_df, preprocessor, feature_names):
        self.feature_df = feature_df
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        self.models = {}

        # Initialize all recommendation components
        self._train_claims_model()
        print("✅ Recommendation engine initialized")

    def _train_claims_model(self):
        """
        Train a model to predict claims probability
        """
        print("Training claims prediction model...")

        # Merge feature data with claims data
        claims_df = pd.read_csv('claims_data.csv')
        model_data = pd.merge(self.feature_df, claims_df, on='user_id')

        # Prepare X and y
        X = model_data.drop(['user_id', 'had_claim_12mo', 'claim_amount'], axis=1)
        y = model_data['had_claim_12mo']

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Preprocess the data
        X_train_processed = self.preprocessor.transform(X_train)
        X_test_processed = self.preprocessor.transform(X_test)

        # Train a Decision Tree model
        model = DecisionTreeClassifier(max_depth=5, random_state=42)
        model.fit(X_train_processed, y_train)

        # Evaluate the model
        y_pred = model.predict(X_test_processed)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"Claims model accuracy: {accuracy:.2f}")
        print("Classification report:")
        print(report)

        # Save the model
        self.models['claims'] = model

    def find_digital_twin(self, user_profile, top_n=5):
        """
        Find similar users based on health profile using cosine similarity
        """
        print("Finding digital twins...")

        # Create a DataFrame with just the input user
        user_df = pd.DataFrame([user_profile])

        # Apply the same preprocessing
        user_processed = self.preprocessor.transform(user_df.drop('user_id', axis=1))

        # Get processed data for all users
        all_users_processed = self.preprocessor.transform(self.feature_df.drop('user_id', axis=1))

        # Calculate cosine similarity
        similarities = cosine_similarity(user_processed, all_users_processed)[0]

        # Get indices of top N similar users
        top_indices = similarities.argsort()[-top_n-1:-1][::-1]  # Exclude the user itself

        # Get user IDs of similar users
        similar_users = self.feature_df.iloc[top_indices]['user_id'].tolist()
        similarity_scores = similarities[top_indices].tolist()

        # Return user IDs and their similarity scores
        results = [{"user_id": user, "similarity": score}
                  for user, score in zip(similar_users, similarity_scores)]

        return results

    def recommend_medical_steps(self, user_profile):
        """
        Recommend medical steps based on rules and health metrics
        """
        print("Generating medical recommendations...")

        recommendations = []
        urgency = "Routine"

        # Extract key health metrics
        bmi = user_profile['bmi']
        systolic = user_profile['systolic_bp']
        diastolic = user_profile['diastolic_bp']
        glucose = user_profile['glucose_level']
        cholesterol = user_profile['cholesterol_level']
        age = user_profile['age']

        # Get diagnosis and medication flags
        has_hypertension = user_profile.get('has_hypertension', 0)
        has_diabetes = user_profile.get('has_diabetes', 0)
        has_high_cholesterol = user_profile.get('has_high_cholesterol', 0)
        has_heart_disease = user_profile.get('has_heart_disease', 0)

        on_bp_meds = user_profile.get('on_bp_medication', 0)
        on_diabetes_meds = user_profile.get('on_diabetes_medication', 0)
        on_cholesterol_meds = user_profile.get('on_cholesterol_medication', 0)

        # BP recommendations
        if systolic >= 180 or diastolic >= 120:
            recommendations.append("Seek immediate medical attention for severe hypertension")
            urgency = "Immediate"
        elif (systolic >= 160 or diastolic >= 100) and not on_bp_meds:
            recommendations.append("Schedule doctor visit within 1 week for high blood pressure")
            urgency = "Urgent"
        elif (systolic >= 140 or diastolic >= 90) and not on_bp_meds:
            recommendations.append("Schedule doctor visit for elevated blood pressure")
            urgency = "Soon"
        elif has_hypertension and (systolic >= 140 or diastolic >= 90) and on_bp_meds:
            recommendations.append("Follow up with doctor to adjust blood pressure medication")
            urgency = "Soon"

        # Blood sugar recommendations
        if glucose >= 200:
            recommendations.append("Seek medical attention for high blood glucose")
            urgency = max(urgency, "Urgent")
        elif glucose >= 126 and not has_diabetes:
            recommendations.append("Schedule doctor visit for diabetes screening")
            urgency = max(urgency, "Soon")
        elif has_diabetes and glucose >= 160 and on_diabetes_meds:
            recommendations.append("Follow up with doctor to review diabetes management")
            urgency = max(urgency, "Soon")

        # Cholesterol recommendations
        if cholesterol >= 240 and not on_cholesterol_meds:
            recommendations.append("Schedule lipid panel and discuss cholesterol management")
            urgency = max(urgency, "Soon")
        elif has_high_cholesterol and cholesterol >= 200 and on_cholesterol_meds:
            recommendations.append("Follow up on cholesterol medication effectiveness")

        # BMI recommendations
        if bmi >= 35:
            recommendations.append("Consider weight management program for obesity")
        elif bmi >= 30:
            recommendations.append("Discuss weight management options with healthcare provider")

        # Age-based screenings
        if age >= 45 and user_profile['gender'] == 'M':
            recommendations.append("Schedule routine prostate exam")
        if age >= 40 and user_profile['gender'] == 'F':
            recommendations.append("Schedule routine mammogram")
        if age >= 45:
            recommendations.append("Schedule routine colonoscopy")

        # Heart disease recommendations
        if has_heart_disease:
            recommendations.append("Continue regular cardiology follow-ups")
            urgency = max(urgency, "Soon")

        # General recommendations
        if not recommendations:
            if age >= 40:
                recommendations.append("Schedule annual physical exam")
            else:
                recommendations.append("Continue routine health maintenance")

        return {
            "recommendations": recommendations,
            "urgency": urgency
        }

    def recommend_insurance_products(self, user_profile):
        """
        Recommend insurance products based on health risks
        """
        print("Recommending insurance products...")

        # Extract key risk factors and demographics
        age = user_profile['age']
        diabetes_risk = user_profile['diabetes_risk']
        hypertension_risk = user_profile['hypertension_risk']
        cvd_risk = user_profile['cvd_risk']

        has_hypertension = user_profile.get('has_hypertension', 0)
        has_diabetes = user_profile.get('has_diabetes', 0)
        has_high_cholesterol = user_profile.get('has_high_cholesterol', 0)
        has_heart_disease = user_profile.get('has_heart_disease', 0)

        # Calculate overall risk level (simple average of the three risks)
        overall_risk = (diabetes_risk + hypertension_risk + cvd_risk) / 3

        # Base recommendations
        core_recommendation = "Standard Health Plan"
        supplemental_recommendations = []

        # Logic for core plan recommendation
        if overall_risk > 0.7 or has_heart_disease:
            core_recommendation = "Premium Health Plan"
        elif overall_risk > 0.5 or has_diabetes or has_hypertension:
            core_recommendation = "Enhanced Health Plan"
        elif overall_risk > 0.3:
            core_recommendation = "Plus Health Plan"

        # Logic for supplemental recommendations
        if diabetes_risk > 0.5 or has_diabetes:
            supplemental_recommendations.append("Diabetes Care Supplement")

        if hypertension_risk > 0.5 or has_hypertension:
            supplemental_recommendations.append("Cardiovascular Care Supplement")

        if cvd_risk > 0.5 or has_heart_disease:
            supplemental_recommendations.append("Critical Illness Coverage")

        if age > 50:
            supplemental_recommendations.append("Senior Care Supplement")

        if len(supplemental_recommendations) == 0:
            supplemental_recommendations.append("Wellness Rewards Program")

        # Return recommendations
        return {
            "core_plan": core_recommendation,
            "supplements": supplemental_recommendations,
            "risk_level": overall_risk
        }

    def recommend_lifestyle_activities(self, user_profile):
        """
        Recommend lifestyle activities and challenges based on health profile
        """
        print("Recommending lifestyle activities...")

        # Extract key metrics
        exercise_score = user_profile['exercise_score']
        diet_score = user_profile['diet_score']
        sleep_score = user_profile['sleep_score']
        stress_score = user_profile['stress_score']
        smoking_score = user_profile['smoking_score']

        bmi = user_profile['bmi']
        avg_steps = user_profile['avg_daily_steps']

        # Determine activity focus areas
        focus_areas = []
        individual_recommendations = []
        team_recommendations = []

        # Exercise recommendations
        if exercise_score <= 4:
            focus_areas.append("exercise")
            individual_recommendations.append("Start a daily walking routine")
            team_recommendations.append("Join a step challenge with your team")
        elif exercise_score <= 7:
            individual_recommendations.append("Increase workout intensity or duration")
            team_recommendations.append("Join a fitness class with colleagues")
        else:
            individual_recommendations.append("Maintain your excellent exercise routine")
            team_recommendations.append("Lead a company fitness challenge")

        # Diet recommendations
        if diet_score <= 4:
            focus_areas.append("nutrition")
            individual_recommendations.append("Try meal prepping healthy lunches")
            team_recommendations.append("Participate in a healthy recipe exchange")
        elif diet_score <= 7:
            individual_recommendations.append("Add more plant-based meals to your diet")
            team_recommendations.append("Join a nutrition workshop with your team")
        else:
            individual_recommendations.append("Continue your healthy eating habits")
            team_recommendations.append("Start a healthy potluck lunch tradition")

        # Sleep recommendations
        if sleep_score <= 4:
            focus_areas.append("sleep")
            individual_recommendations.append("Establish a consistent sleep schedule")
            team_recommendations.append("Join a sleep improvement challenge")
        elif sleep_score <= 7:
            individual_recommendations.append("Try reducing screen time before bed")
            team_recommendations.append("Join a stress management workshop")

        # Stress management
        if stress_score <= 4:
            focus_areas.append("stress")
            individual_recommendations.append("Practice daily mindfulness meditation")
            team_recommendations.append("Join a workplace meditation group")
        elif stress_score <= 7:
            individual_recommendations.append("Schedule regular breaks during work")
            team_recommendations.append("Sign up for a team yoga class")

        # Smoking cessation
        if smoking_score <= 5:
            focus_areas.append("smoking")
            individual_recommendations.append("Enroll in a smoking cessation program")
            team_recommendations.append("Join a quit-smoking support group")

        # Weight management
        if bmi >= 30:
            focus_areas.append("weight")
            individual_recommendations.append("Set achievable weight management goals")
            team_recommendations.append("Join a workplace weight management program")

        # Step count
        if avg_steps < 5000:
            focus_areas.append("activity")
            individual_recommendations.append("Aim to increase daily steps to 7,000")
            team_recommendations.append("Join a step count challenge")

        # Determine primary focus area
        primary_focus = "general wellness"
        if focus_areas:
            # Choose the area with the lowest score
            scores = {
                "exercise": exercise_score,
                "nutrition": diet_score,
                "sleep": sleep_score,
                "stress": stress_score,
                "smoking": smoking_score,
                "weight": (10 - min(10, max(1, (bmi - 18.5) / 3))),
                "activity": min(10, max(1, avg_steps / 1000))
            }

            available_scores = {area: scores[area] for area in focus_areas if area in scores}
            if available_scores:
                primary_focus = min(available_scores, key=available_scores.get)

        # Create final recommendations
        return {
            "primary_focus": primary_focus,
            "individual_recommendations": individual_recommendations[:3],  # Top 3 recommendations
            "team_recommendations": team_recommendations[:2]  # Top 2 recommendations
        }

    def predict_claim_probability(self, user_profile):
        """
        Predict the probability of a user making a medical claim
        """
        print("Predicting claim probability...")

        # Remove user_id for prediction
        features = user_profile.copy()
        if 'user_id' in features:
            user_id = features.pop('user_id')

        # Convert to DataFrame for preprocessing
        user_df = pd.DataFrame([features])

        # Apply preprocessing
        user_processed = self.preprocessor.transform(user_df)

        # Make prediction
        claim_model = self.models['claims']
        claim_prob = claim_model.predict_proba(user_processed)[0, 1]  # Probability of class 1

        # Return prediction
        return {
            "probability": round(claim_prob, 2),
            "risk_level": "High" if claim_prob > 0.6 else "Medium" if claim_prob > 0.3 else "Low"
        }

    def explain_claim_factors(self, user_profile):
        """
        Explain factors contributing to claims prediction
        """
        # Get the decision tree model
        claim_model = self.models['claims']

        # Create a simplified explanation based on decision path
        explanation = []

        # Remove user_id for processing
        features = user_profile.copy()
        if 'user_id' in features:
            user_id = features.pop('user_id')

        # Convert to DataFrame for preprocessing
        user_df = pd.DataFrame([features])

        # Apply preprocessing
        user_processed = self.preprocessor.transform(user_df)

        # Get feature importances
        importances = claim_model.feature_importances_

        # Map importances to feature names
        feature_importance = dict(zip(self.feature_names, importances))

        # Sort by importance
        sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

        # Return top 5 factors
        top_factors = sorted_importance[:5]

        explanation = [{"factor": factor, "importance": float(importance)}
                       for factor, importance in top_factors]

        return explanation

    def generate_recommendations(self, user_profile):
        """
        Generate all recommendations for a user
        """
        print("Generating comprehensive recommendations...")

        # Find digital twins
        digital_twins = self.find_digital_twin(user_profile)

        # Generate other recommendations
        medical_recs = self.recommend_medical_steps(user_profile)
        insurance_recs = self.recommend_insurance_products(user_profile)
        lifestyle_recs = self.recommend_lifestyle_activities(user_profile)
        claim_prediction = self.predict_claim_probability(user_profile)
        claim_factors = self.explain_claim_factors(user_profile)

        # Compile all recommendations
        recommendations = {
            "digital_twins": digital_twins,
            "medical_recommendations": medical_recs,
            "insurance_recommendations": insurance_recs,
            "lifestyle_recommendations": lifestyle_recs,
            "claim_prediction": claim_prediction,
            "claim_factors": claim_factors
        }

        return recommendations

# =============================================================================
# STREAMLIT APP
# =============================================================================

def create_streamlit_app(engine, feature_df):
    """
    Create a Streamlit app for the health recommendation engine
    """
    st.title("Health & Wellness Recommendation Engine")
    st.write("This prototype provides personalized health recommendations based on user profiles")

    with st.expander("ℹ️ About this prototype", expanded=False):
        st.markdown("""
        This prototype demonstrates a modular health recommendation engine that provides:
        * **Digital Twin**: Finding similar users based on health profiles
        * **Medical Recommendations**: Suggested next steps for healthcare
        * **Insurance Recommendations**: Personalized coverage suggestions
        * **Lifestyle Recommendations**: Individual and team wellness activities
        * **Claims Prediction**: Likelihood of future medical claims

        Enter your information in the form below to get personalized recommendations.
        """)

    # Sidebar for user input
    st.sidebar.header("User Profile")

    # Demographics
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", 22, 65, 40)
    gender = st.sidebar.selectbox("Gender", ["M", "F", "Other"])
    company = st.sidebar.selectbox("Company",
                              ["TechCorp", "HealthNet", "FinGroup", "RetailGiant", "EduSystems"])

    # Team selection based on company
    teams_by_company = {
        'TechCorp': ['Engineering', 'Design', 'Product', 'Support', 'Sales'],
        'HealthNet': ['Nursing', 'Administration', 'Research', 'Operations', 'IT'],
        'FinGroup': ['Investment', 'Analysis', 'Compliance', 'HR', 'Tech'],
        'RetailGiant': ['Logistics', 'Marketing', 'Store Ops', 'Digital', 'Support'],
        'EduSystems': ['Faculty', 'Administration', 'IT', 'Research', 'Student Services']
    }
    team = st.sidebar.selectbox("Team", teams_by_company[company])

    # Health metrics
    st.sidebar.subheader("Health Metrics")

    # BMI and BP
    bmi = st.sidebar.slider("BMI", 16.0, 40.0, 25.0, 0.1)
    systolic_bp = st.sidebar.slider("Systolic BP (mmHg)", 90, 180, 120, 1)
    diastolic_bp = st.sidebar.slider("Diastolic BP (mmHg)", 60, 110, 80, 1)

    # Lab values
    glucose_level = st.sidebar.slider("Blood Glucose (mg/dL)", 70, 180, 95, 1)
    cholesterol_level = st.sidebar.slider("Total Cholesterol (mg/dL)", 130, 300, 190, 1)

    # Lifestyle scores (1-10 where higher is better)
    st.sidebar.subheader("Lifestyle Factors (1-10 scale, higher is better)")
    smoking_score = st.sidebar.slider("Smoking Habits", 1, 10, 8,
                                 help="1=Heavy smoker, 10=Non-smoker")
    sleep_score = st.sidebar.slider("Sleep Quality", 1, 10, 6,
                               help="1=Poor sleep, 10=Excellent sleep")
    diet_score = st.sidebar.slider("Diet Quality", 1, 10, 6,
                              help="1=Poor diet, 10=Excellent diet")
    exercise_score = st.sidebar.slider("Exercise Habits", 1, 10, 5,
                                  help="1=Sedentary, 10=Very active")
    stress_score = st.sidebar.slider("Stress Management", 1, 10, 6,
                                help="1=Highly stressed, 10=Well managed")

    # App usage metrics
    st.sidebar.subheader("Activity Metrics")
    avg_daily_steps = st.sidebar.slider("Avg. Daily Steps", 1000, 15000, 7000, 500)
    avg_sleep_hours = st.sidebar.slider("Avg. Sleep (hours)", 4.0, 10.0, 7.0, 0.1)
    avg_water_intake = st.sidebar.slider("Avg. Water Intake (cups)", 1.0, 12.0, 6.0, 0.5)

    # Medical history
    st.sidebar.subheader("Medical History")
    diagnoses = st.sidebar.multiselect("Current Diagnoses",
                                  ["Hypertension", "Diabetes", "High Cholesterol",
                                   "Obesity", "Anxiety", "Depression",
                                   "Sleep Disorder", "Heart Disease"])

    medications = st.sidebar.multiselect("Current Medications",
                                    ["Blood Pressure Medication",
                                     "Diabetes Medication",
                                     "Cholesterol Medication",
                                     "Anxiety/Depression Medication"])

    # App activity
    app_login_frequency = st.sidebar.select_slider("App Usage Frequency",
                                              options=["Rarely", "Monthly", "Weekly", "Daily"])
    challenges_completed = st.sidebar.slider("Wellness Challenges Completed (6 mo)", 0, 10, 2)
    preferred_challenge = st.sidebar.selectbox("Preferred Challenge Type",
                                          ["Steps", "Nutrition", "Mindfulness", "Sleep", "Hydration"])

    doctor_visits_6mo = st.sidebar.slider("Doctor Visits (6 mo)", 0, 10, 1)
    pharmacy_orders_6mo = st.sidebar.slider("Pharmacy Orders (6 mo)", 0, 10, 2)

    # Health risk predictions
    diabetes_risk = round((1 if "Diabetes" in diagnoses else 0) * 0.3 +
                     (glucose_level > 100) * 0.3 +
                     (bmi > 30) * 0.2 +
                     (age > 45) * 0.15 +
                     random.uniform(-0.05, 0.05), 2)
    diabetes_risk = max(0.01, min(0.99, diabetes_risk))

    hypertension_risk = round((1 if "Hypertension" in diagnoses else 0) * 0.3 +
                         (systolic_bp > 130) * 0.2 +
                         (diastolic_bp > 85) * 0.2 +
                         (bmi > 28) * 0.1 +
                         (age > 50) * 0.1 +
                         (smoking_score < 6) * 0.1 +
                         random.uniform(-0.05, 0.05), 2)
    hypertension_risk = max(0.01, min(0.99, hypertension_risk))

    cvd_risk = round((1 if "Heart Disease" in diagnoses else 0) * 0.3 +
                (1 if "Hypertension" in diagnoses else 0) * 0.15 +
                (age > 55) * 0.15 +
                (smoking_score < 5) * 0.15 +
                (cholesterol_level > 220) * 0.15 +
                (systolic_bp > 140) * 0.15 +
                random.uniform(-0.05, 0.05), 2)
    cvd_risk = max(0.01, min(0.99, cvd_risk))

    # Create user profile dictionary
    user_profile = {
        'user_id': 'new_user',
        'age': age,
        'gender': gender,
        'company': company,
        'team': team,
        'bmi': bmi,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'glucose_level': glucose_level,
        'cholesterol_level': cholesterol_level,
        'smoking_score': smoking_score,
        'sleep_score': sleep_score,
        'diet_score': diet_score,
        'exercise_score': exercise_score,
        'stress_score': stress_score,
        'avg_daily_steps': avg_daily_steps,
        'avg_sleep_hours': avg_sleep_hours,
        'avg_water_intake': avg_water_intake,
        'doctor_visits_6mo': doctor_visits_6mo,
        'pharmacy_orders_6mo': pharmacy_orders_6mo,
        'challenges_completed': challenges_completed,
        'app_login_frequency': app_login_frequency,
        'preferred_challenge': preferred_challenge,
        'diabetes_risk': diabetes_risk,
        'hypertension_risk': hypertension_risk,
        'cvd_risk': cvd_risk
    }

    # Add diagnosis flags
    for condition in ["Hypertension", "Diabetes", "High Cholesterol",
                      "Obesity", "Anxiety", "Depression",
                      "Sleep Disorder", "Heart Disease"]:
        condition_key = f'has_{condition.lower().replace(" ", "_")}'
        user_profile[condition_key] = 1 if condition in diagnoses else 0

    # Add medication flags
    user_profile['on_bp_medication'] = 1 if "Blood Pressure Medication" in medications else 0
    user_profile['on_diabetes_medication'] = 1 if "Diabetes Medication" in medications else 0
    user_profile['on_cholesterol_medication'] = 1 if "Cholesterol Medication" in medications else 0
    user_profile['on_anxiety_depression_medication'] = 1 if "Anxiety/Depression Medication" in medications else 0

    # Generate recommendations button
    if st.sidebar.button("Generate Recommendations", type="primary"):
        with st.spinner("Analyzing health profile and generating recommendations..."):
            # Generate all recommendations
            recommendations = engine.generate_recommendations(user_profile)

            # Display recommendations
            st.header("Your Personalized Health Recommendations")

            # Create tabs for different recommendation types
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                "Medical Recommendations",
                "Lifestyle Suggestions",
                "Insurance Options",
                "Digital Twin",
                "Claims Prediction"
            ])

            # Tab 1: Medical Recommendations
            with tab1:
                st.subheader("Medical Recommendations")

                # Display urgency
                urgency = recommendations['medical_recommendations']['urgency']
                if urgency == "Immediate":
                    st.error(f"Urgency: {urgency}")
                elif urgency == "Urgent":
                    st.warning(f"Urgency: {urgency}")
                elif urgency == "Soon":
                    st.info(f"Urgency: {urgency}")
                else:
                    st.success(f"Urgency: {urgency}")

                # Display medical recommendations
                for i, rec in enumerate(recommendations['medical_recommendations']['recommendations']):
                    st.write(f"**{i+1}.** {rec}")

            # Tab 2: Lifestyle Recommendations
            with tab2:
                st.subheader("Lifestyle Recommendations")

                lifestyle_recs = recommendations['lifestyle_recommendations']
                st.write(f"**Primary Focus Area:** {lifestyle_recs['primary_focus'].title()}")

                st.write("**Individual Activities:**")
                for i, rec in enumerate(lifestyle_recs['individual_recommendations']):
                    st.write(f"- {rec}")

                st.write("**Team Activities:**")
                for i, rec in enumerate(lifestyle_recs['team_recommendations']):
                    st.write(f"- {rec}")

            # Tab 3: Insurance Recommendations
            with tab3:
                st.subheader("Insurance Recommendations")

                insurance_recs = recommendations['insurance_recommendations']

                # Display risk level with color
                risk_level = insurance_recs['risk_level']
                if risk_level > 0.7:
                    st.error(f"Risk Level: High ({risk_level:.2f})")
                elif risk_level > 0.4:
                    st.warning(f"Risk Level: Medium ({risk_level:.2f})")
                else:
                    st.success(f"Risk Level: Low ({risk_level:.2f})")

                st.write(f"**Recommended Core Plan:** {insurance_recs['core_plan']}")

                st.write("**Recommended Supplements:**")
                for supp in insurance_recs['supplements']:
                    st.write(f"- {supp}")

            # Tab 4: Digital Twin
            with tab4:
                st.subheader("Your Digital Twin")
                st.write("Users with similar health profiles to yours:")

                # Display digital twins
                for i, twin in enumerate(recommendations['digital_twins'][:3]):
                    st.metric(f"Match {i+1}",
                          f"User {twin['user_id']}",
                          f"{twin['similarity']:.2%} Similar")

                # Visualize similarity
                st.write("**Similarity Visualization:**")

                # Get data for the twins
                twin_ids = [twin['user_id'] for twin in recommendations['digital_twins'][:3]]
                twin_data = feature_df[feature_df['user_id'].isin(twin_ids)]

                # Add current user
                user_df = pd.DataFrame([user_profile])
                viz_data = pd.concat([user_df, twin_data], ignore_index=True)

                                # Normalize data for radar chart
                numeric_cols = ['bmi', 'systolic_bp', 'diastolic_bp', 'glucose_level',
                                'cholesterol_level', 'smoking_score', 'sleep_score',
                                'diet_score', 'exercise_score', 'stress_score']
                viz_data_norm = viz_data.copy()
                for col in numeric_cols:
                    min_val = feature_df[col].min()
                    max_val = feature_df[col].max()
                    viz_data_norm[col] = (viz_data[col] - min_val) / (max_val - min_val)

                # Plot radar chart using Plotly
                import plotly.express as px
                import plotly.graph_objects as go

                categories = numeric_cols
                fig = go.Figure()

                for i, row in viz_data_norm.iterrows():
                    name = "You" if row['user_id'] == 'new_user' else row['user_id']
                    fig.add_trace(go.Scatterpolar(
                        r=row[categories].values,
                        theta=categories,
                        fill='toself',
                        name=name
                    ))

                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)

            # Tab 5: Claims Prediction
            with tab5:
                st.subheader("Predicted Claims Risk")
                claims_risk = recommendations['claims_prediction']['claim_probability']
                st.write(f"**Estimated Claims Probability (next 6 months):** {claims_risk:.2%}")
                if claims_risk > 0.7:
                    st.error("High likelihood of medical claims. Consider preventive actions.")
                elif claims_risk > 0.4:
                    st.warning("Moderate risk. Keep up healthy habits.")
                else:
                    st.success("Low risk of medical claims.")
