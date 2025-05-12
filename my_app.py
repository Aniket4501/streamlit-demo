import pandas as pd
import numpy as np
import streamlit as st
import random
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

#############################################
# 1. DATA GENERATION
#############################################

def generate_dummy_data(num_users=200):
    """Generate synthetic health profile data for demonstration"""
    
    # Demographics data
    user_ids = list(range(1001, 1001 + num_users))
    ages = np.random.randint(22, 65, size=num_users)
    genders = np.random.choice(['Male', 'Female', 'Other'], size=num_users, p=[0.48, 0.48, 0.04])
    companies = np.random.choice(['TechCorp', 'HealthInc', 'FinanceOne', 'RetailPro'], size=num_users)
    teams = np.random.choice(['Engineering', 'Sales', 'HR', 'Marketing', 'Operations'], size=num_users)
    
    demographics = pd.DataFrame({
        'user_id': user_ids,
        'age': ages,
        'gender': genders,
        'company': companies,
        'team': teams
    })
    
    # HRA Data (Health Risk Assessment)
    smoking_status = np.random.choice(['Never', 'Former', 'Occasional', 'Regular'], size=num_users, p=[0.6, 0.2, 0.1, 0.1])
    sleep_hours = np.clip(np.random.normal(7, 1.5, size=num_users), 3, 10).round(1)
    diet_score = np.random.randint(1, 11, size=num_users)  # 1-10 scale
    exercise_days = np.random.randint(0, 8, size=num_users)  # Days per week
    stress_level = np.random.randint(1, 11, size=num_users)  # 1-10 scale
    
    hra_data = pd.DataFrame({
        'user_id': user_ids,
        'smoking_status': smoking_status,
        'sleep_hours': sleep_hours,
        'diet_score': diet_score,
        'exercise_days': exercise_days,
        'stress_level': stress_level
    })
    
    # EMR Data (Electronic Medical Records)
    # Systolic/Diastolic BP
    systolic_bp = np.clip(np.random.normal(120, 15, size=num_users), 90, 180).astype(int)
    diastolic_bp = np.clip(np.random.normal(80, 10, size=num_users), 60, 120).astype(int)
    
    # BMI with slight correlation to exercise
    base_bmi = np.random.normal(26, 4, size=num_users)
    exercise_effect = (7 - exercise_days) * 0.2  # Less exercise = higher BMI adjustment
    bmi = np.clip(base_bmi + exercise_effect, 16, 45).round(1)
    
    # Lab results
    cholesterol = np.clip(np.random.normal(190, 35, size=num_users), 120, 300).astype(int)
    hdl = np.clip(np.random.normal(55, 15, size=num_users), 20, 100).astype(int)
    ldl = np.clip(cholesterol - hdl - np.random.normal(30, 10, size=num_users), 50, 250).astype(int)
    glucose = np.clip(np.random.normal(95, 20, size=num_users), 70, 200).astype(int)
    
    # Existing diagnoses - more likely with age and less exercise
    has_hypertension = (np.random.random(num_users) < (ages/100 + (systolic_bp > 140)*0.3)).astype(int)
    has_diabetes = (np.random.random(num_users) < (ages/150 + (glucose > 125)*0.4)).astype(int)
    has_high_cholesterol = (np.random.random(num_users) < (ages/120 + (cholesterol > 240)*0.4)).astype(int)
    
    # Medications based on diagnoses
    medications = []
    for i in range(num_users):
        user_meds = []
        if has_hypertension[i]:
            user_meds.extend(np.random.choice(['Lisinopril', 'Amlodipine', 'Losartan'], 
                                         size=np.random.randint(0, 3), replace=False).tolist())
        if has_diabetes[i]:
            user_meds.extend(np.random.choice(['Metformin', 'Glipizide', 'Insulin'], 
                                         size=np.random.randint(0, 2), replace=False).tolist())
        if has_high_cholesterol[i]:
            user_meds.extend(np.random.choice(['Atorvastatin', 'Simvastatin'], 
                                         size=np.random.randint(0, 2), replace=False).tolist())
        medications.append(', '.join(user_meds) if user_meds else 'None')
    
    emr_data = pd.DataFrame({
        'user_id': user_ids,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'bmi': bmi,
        'cholesterol': cholesterol,
        'hdl': hdl,
        'ldl': ldl,
        'glucose': glucose,
        'has_hypertension': has_hypertension,
        'has_diabetes': has_diabetes,
        'has_high_cholesterol': has_high_cholesterol,
        'medications': medications
    })
    
    # Health Risk Predictions
    # Simple risk calculation based on existing health metrics
    diabetes_risk = np.clip(
        (glucose - 70) / 130 + (bmi - 18.5) / 30 + ages / 200 - exercise_days / 10, 
        0.05, 0.95
    ).round(2)
    
    hypertension_risk = np.clip(
        (systolic_bp - 90) / 90 + (diastolic_bp - 60) / 60 + ages / 150 - exercise_days / 10,
        0.05, 0.95
    ).round(2)
    
    cvd_risk = np.clip(
        (cholesterol - 150) / 150 + (systolic_bp - 90) / 90 + ages / 100 - hdl / 100 + 
        (smoking_status != 'Never') * 0.2,
        0.05, 0.95
    ).round(2)
    
    health_risks = pd.DataFrame({
        'user_id': user_ids,
        'diabetes_risk': diabetes_risk,
        'hypertension_risk': hypertension_risk,
        'cvd_risk': cvd_risk
    })
    
    # App Usage Data
    # Base usage probability - younger users use the app more
    base_usage_prob = 0.8 - ages / 100
    
    # Number of doctor bookings
    bookings_count = np.random.binomial(10, base_usage_prob * 0.3, size=num_users)
    
    # Number of pharmacy orders
    pharmacy_orders = np.random.binomial(12, base_usage_prob * 0.2 + has_diabetes * 0.2 + 
                                       has_hypertension * 0.2 + has_high_cholesterol * 0.2, 
                                       size=num_users)
    
    # Activity tracker logs (% of days with data)
    activity_tracking_pct = np.clip(base_usage_prob + np.random.normal(0, 0.2, size=num_users), 0, 1).round(2)
    
    # Average daily steps (correlated with exercise and age)
    avg_steps = np.clip(
        8000 - ages * 30 + exercise_days * 500 + np.random.normal(0, 1000, size=num_users),
        1000, 20000
    ).astype(int)
    
    # Sleep tracking (% of days tracked)
    sleep_tracking_pct = np.clip(base_usage_prob * 0.8 + np.random.normal(0, 0.2, size=num_users), 0, 1).round(2)
    
    # Water tracking (% of days tracked)
    water_tracking_pct = np.clip(base_usage_prob * 0.6 + np.random.normal(0, 0.3, size=num_users), 0, 1).round(2)
    
    app_usage = pd.DataFrame({
        'user_id': user_ids,
        'bookings_count': bookings_count,
        'pharmacy_orders': pharmacy_orders,
        'activity_tracking_pct': activity_tracking_pct,
        'avg_steps': avg_steps,
        'sleep_tracking_pct': sleep_tracking_pct, 
        'water_tracking_pct': water_tracking_pct
    })
    
    # Challenge Participation History
    challenges = ['Step Challenge', 'Sleep Improvement', 'Hydration Challenge', 
                 'Nutrition Challenge', 'Meditation Challenge', 'Team Sports', 
                 'Weight Management', 'Smoking Cessation']
    
    challenge_history = []
    
    for user_id in user_ids:
        # Determine number of challenges participated in (0 to 5)
        num_challenges = np.random.binomial(5, base_usage_prob[user_id - 1001])
        
        if num_challenges > 0:
            # Randomly select challenges
            user_challenges = np.random.choice(challenges, size=num_challenges, replace=False)
            
            # For each challenge, generate completion status and satisfaction
            for challenge in user_challenges:
                completion_pct = np.random.choice([0.0, 0.25, 0.5, 0.75, 1.0], p=[0.1, 0.15, 0.2, 0.15, 0.4])
                satisfaction = None if completion_pct == 0 else np.random.randint(1, 6)  # 1-5 scale
                
                challenge_history.append({
                    'user_id': user_id,
                    'challenge_name': challenge,
                    'completion_pct': completion_pct,
                    'satisfaction': satisfaction
                })
    
    challenge_data = pd.DataFrame(challenge_history) if challenge_history else pd.DataFrame({
        'user_id': [], 'challenge_name': [], 'completion_pct': [], 'satisfaction': []
    })
    
    return {
        'demographics': demographics,
        'hra_data': hra_data,
        'emr_data': emr_data,
        'health_risks': health_risks,
        'app_usage': app_usage,
        'challenge_data': challenge_data
    }

#############################################
# 2. RECOMMENDATION ENGINE
#############################################

class HealthRecommendationEngine:
    """Hybrid recommendation engine combining rules-based and ML approaches"""
    
    def __init__(self, data_dict=None):
        """Initialize with data or generate dummy data if none provided"""
        if data_dict is None:
            self.data_dict = generate_dummy_data()
        else:
            self.data_dict = data_dict
            
        # Prepare processed datasets and train models
        self._prepare_data()
        self._train_models()
        
        # Define available challenges and their target health areas
        self.available_challenges = {
            'Step Challenge': ['exercise', 'cardiovascular'],
            'Sleep Improvement': ['sleep', 'stress'],
            'Hydration Challenge': ['nutrition'],
            'Nutrition Challenge': ['nutrition', 'weight'],
            'Meditation Challenge': ['stress', 'mental_health'],
            'Team Sports': ['exercise', 'social', 'stress'],
            'Weight Management': ['weight', 'nutrition'],
            'Smoking Cessation': ['smoking']
        }
        
        # Define medical recommendations based on rules
        self.medical_rule_templates = {
            'high_bp': "Schedule a blood pressure check with your doctor",
            'very_high_bp': "See a doctor soon for your blood pressure",
            'high_glucose': "Schedule a diabetes screening",
            'very_high_glucose': "See a doctor soon for your blood glucose levels",
            'high_cholesterol': "Schedule a cholesterol panel test",
            'diabetes_managed': "Continue diabetes management with your doctor",
            'hypertension_managed': "Continue hypertension management with your healthcare provider",
            'high_bmi': "Consider discussing weight management with your doctor",
            'very_high_bmi': "Schedule an appointment to discuss health risks related to your BMI",
            'smokingces': "Schedule a lung health assessment",
            'general_checkup': "Schedule your annual wellness visit",
            'low_hdl': "Consider a cholesterol panel to check your HDL levels",
            'high_cvd_risk': "Schedule a cardiovascular assessment"
        }
    
    def _prepare_data(self):
        """Join and process datasets for recommendation generation"""
        # Join all data to create feature-rich user profiles
        self.user_profiles = self.data_dict['demographics'].merge(
            self.data_dict['hra_data'], on='user_id'
        ).merge(
            self.data_dict['emr_data'], on='user_id'
        ).merge(
            self.data_dict['health_risks'], on='user_id'
        ).merge(
            self.data_dict['app_usage'], on='user_id'
        )
        
        # Calculate additional features for recommendations
        self.user_profiles['bp_category'] = self.user_profiles.apply(
            lambda x: self._categorize_bp(x['systolic_bp'], x['diastolic_bp']), axis=1
        )
        
        self.user_profiles['bmi_category'] = self.user_profiles['bmi'].apply(self._categorize_bmi)
        
        self.user_profiles['glucose_category'] = self.user_profiles['glucose'].apply(
            lambda x: 'normal' if x < 100 else ('prediabetic' if x < 126 else 'diabetic')
        )
        
        # Create challenge participation matrix
        if not self.data_dict['challenge_data'].empty:
            challenge_matrix = pd.pivot_table(
                self.data_dict['challenge_data'],
                values='completion_pct',
                index='user_id',
                columns='challenge_name',
                fill_value=0
            )
            
            # Merge with user profiles
            self.user_profiles = self.user_profiles.merge(
                challenge_matrix, on='user_id', how='left'
            ).fillna(0)
            
            # Add feature for total challenges completed
            for challenge in self.available_challenges:
                if challenge not in self.user_profiles.columns:
                    self.user_profiles[challenge] = 0
                    
            self.user_profiles['total_challenges'] = self.user_profiles[list(self.available_challenges.keys())].sum(axis=1)
    
    def _train_models(self):
        """Train ML models for challenge recommendations"""
        # Define health need areas for each user based on their data
        self.user_profiles['needs_cardio'] = (
            (self.user_profiles['bp_category'].isin(['elevated', 'hypertension_1', 'hypertension_2', 'hypertensive_crisis'])) | 
            (self.user_profiles['cvd_risk'] > 0.3)
        ).astype(int)
        
        self.user_profiles['needs_weight_management'] = (
            (self.user_profiles['bmi_category'].isin(['overweight', 'obese', 'extremely_obese']))
        ).astype(int)
        
        self.user_profiles['needs_nutrition'] = (
            (self.user_profiles['diet_score'] < 6) | 
            (self.user_profiles['needs_weight_management'] == 1)
        ).astype(int)
        
        self.user_profiles['needs_sleep'] = (
            (self.user_profiles['sleep_hours'] < 6) | 
            (self.user_profiles['sleep_hours'] > 9)
        ).astype(int)
        
        self.user_profiles['needs_exercise'] = (
            (self.user_profiles['exercise_days'] < 3) | 
            (self.user_profiles['avg_steps'] < 6000)
        ).astype(int)
        
        self.user_profiles['needs_stress_management'] = (
            (self.user_profiles['stress_level'] > 7)
        ).astype(int)
        
        self.user_profiles['needs_smoking_cessation'] = (
            (self.user_profiles['smoking_status'].isin(['Occasional', 'Regular']))
        ).astype(int)
        
        # Create feature matrix for ML models
        features = [
            'age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol', 'glucose',
            'sleep_hours', 'diet_score', 'exercise_days', 'stress_level',
            'diabetes_risk', 'hypertension_risk', 'cvd_risk'
        ]
        
        # Add categorical features as one-hot encoding
        cat_features = ['gender', 'smoking_status', 'bp_category', 'bmi_category', 'glucose_category']
        X = pd.get_dummies(self.user_profiles[features + cat_features], columns=cat_features)
        
        # Train a model for each challenge
        self.challenge_models = {}
        self.feature_columns = X.columns  # Store for prediction
        
        # Normalize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train a model for each health need area to predict challenge suitability
        health_needs = [
            'needs_cardio', 'needs_weight_management', 'needs_nutrition',
            'needs_sleep', 'needs_exercise', 'needs_stress_management',
            'needs_smoking_cessation'
        ]
        
        self.need_models = {}
        for need in health_needs:
            y = self.user_profiles[need]
            
            # Use a simple model - Decision Tree for interpretability
            model = DecisionTreeClassifier(max_depth=4, random_state=42)
            model.fit(X_scaled, y)
            self.need_models[need] = model
    
    def _categorize_bp(self, systolic, diastolic):
        """Categorize blood pressure according to guidelines"""
        if systolic < 120 and diastolic < 80:
            return 'normal'
        elif (systolic >= 120 and systolic < 130) and diastolic < 80:
            return 'elevated'
        elif (systolic >= 130 and systolic < 140) or (diastolic >= 80 and diastolic < 90):
            return 'hypertension_1'
        elif (systolic >= 140 and systolic < 180) or (diastolic >= 90 and diastolic < 120):
            return 'hypertension_2'
        else:
            return 'hypertensive_crisis'
    
    def _categorize_bmi(self, bmi):
        """Categorize BMI according to standard guidelines"""
        if bmi < 18.5:
            return 'underweight'
        elif bmi < 25:
            return 'normal'
        elif bmi < 30:
            return 'overweight'
        elif bmi < 35:
            return 'obese'
        else:
            return 'extremely_obese'
    
    def get_medical_recommendation(self, user_data):
        """Generate medical recommendation based on rules"""
        recommendations = []
        urgency = 0  # 0=routine, 1=important, 2=urgent
        
        # BP-based recommendations
        systolic = user_data.get('systolic_bp', 120)
        diastolic = user_data.get('diastolic_bp', 80)
        bp_category = self._categorize_bp(systolic, diastolic)
        
        if bp_category == 'hypertension_2' or bp_category == 'hypertensive_crisis':
            recommendations.append(self.medical_rule_templates['very_high_bp'])
            urgency = max(urgency, 2)
        elif bp_category == 'hypertension_1':
            recommendations.append(self.medical_rule_templates['high_bp'])
            urgency = max(urgency, 1)
        
        # Glucose-based recommendations
        glucose = user_data.get('glucose', 95)
        if glucose >= 200:
            recommendations.append(self.medical_rule_templates['very_high_glucose'])
            urgency = max(urgency, 2)
        elif glucose >= 126:
            recommendations.append(self.medical_rule_templates['high_glucose'])
            urgency = max(urgency, 1)
        elif glucose >= 100:
            recommendations.append(self.medical_rule_templates['high_glucose'])
            urgency = max(urgency, 0)
        
        # Cholesterol-based recommendations
        cholesterol = user_data.get('cholesterol', 180)
        hdl = user_data.get('hdl', 50)
        if cholesterol >= 240:
            recommendations.append(self.medical_rule_templates['high_cholesterol'])
            urgency = max(urgency, 1)
        elif hdl < 40:
            recommendations.append(self.medical_rule_templates['low_hdl'])
            urgency = max(urgency, 0)
        
        # BMI-based recommendations
        bmi = user_data.get('bmi', 24)
        bmi_category = self._categorize_bmi(bmi)
        if bmi_category == 'extremely_obese':
            recommendations.append(self.medical_rule_templates['very_high_bmi'])
            urgency = max(urgency, 1)
        elif bmi_category == 'obese':
            recommendations.append(self.medical_rule_templates['high_bmi'])
            urgency = max(urgency, 0)
        
        # Smoking-based recommendations
        smoking_status = user_data.get('smoking_status', 'Never')
        if smoking_status == 'Regular':
            recommendations.append(self.medical_rule_templates['smokingces'])
            urgency = max(urgency, 0)
        
        # If no specific recommendations, suggest general checkup
        if not recommendations:
            recommendations.append(self.medical_rule_templates['general_checkup'])
        
        # Format the recommendation with urgency level
        primary_rec = recommendations[0]
        if urgency == 2:
            return f"URGENT: {primary_rec} within the next week."
        elif urgency == 1:
            return f"IMPORTANT: {primary_rec} in the next month."
        else:
            return f"ROUTINE: {primary_rec} at your convenience."
    
    def get_challenge_recommendation(self, user_data):
        """Generate lifestyle challenge recommendation based on ML"""
        # Create feature vector from user data
        feature_vector = {}
        
        # Extract all the necessary features
        for feature in ['age', 'bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol', 
                        'glucose', 'sleep_hours', 'diet_score', 'exercise_days', 
                        'stress_level']:
            feature_vector[feature] = user_data.get(feature, 0)
        
        # Calculate risk scores (simplified for the demo)
        diabetes_risk = (
            (feature_vector['glucose'] - 70) / 130 + 
            (feature_vector['bmi'] - 18.5) / 30 + 
            feature_vector['age'] / 200 - 
            feature_vector['exercise_days'] / 10
        )
        feature_vector['diabetes_risk'] = np.clip(diabetes_risk, 0.05, 0.95)
        
        hypertension_risk = (
            (feature_vector['systolic_bp'] - 90) / 90 + 
            (feature_vector['diastolic_bp'] - 60) / 60 + 
            feature_vector['age'] / 150 - 
            feature_vector['exercise_days'] / 10
        )
        feature_vector['hypertension_risk'] = np.clip(hypertension_risk, 0.05, 0.95)
        
        smoking_factor = 0.2 if user_data.get('smoking_status') != 'Never' else 0
        cvd_risk = (
            (feature_vector['cholesterol'] - 150) / 150 + 
            (feature_vector['systolic_bp'] - 90) / 90 + 
            feature_vector['age'] / 100 - 
            user_data.get('hdl', 50) / 100 + 
            smoking_factor
        )
        feature_vector['cvd_risk'] = np.clip(cvd_risk, 0.05, 0.95)
        
        # Add categorical features with one-hot encoding
        for cat_feature in ['gender', 'smoking_status']:
            if cat_feature in user_data:
                feature_vector[f"{cat_feature}_{user_data[cat_feature]}"] = 1
        
        # Add derived categorical features
        bp_category = self._categorize_bp(
            user_data.get('systolic_bp', 120), 
            user_data.get('diastolic_bp', 80)
        )
        feature_vector[f"bp_category_{bp_category}"] = 1
        
        bmi_category = self._categorize_bmi(user_data.get('bmi', 24))
        feature_vector[f"bmi_category_{bmi_category}"] = 1
        
        glucose = user_data.get('glucose', 95)
        glucose_category = 'normal' if glucose < 100 else ('prediabetic' if glucose < 126 else 'diabetic')
        feature_vector[f"glucose_category_{glucose_category}"] = 1
        
        # Create complete feature vector with all expected columns
        X_pred = pd.DataFrame([feature_vector])
        
        # Add missing columns (that were in training data but not in this user data)
        for col in self.feature_columns:
            if col not in X_pred.columns:
                X_pred[col] = 0
        
        # Ensure columns are in the same order as during training
        X_pred = X_pred[self.feature_columns]
        
        # Scale features
        X_pred_scaled = self.scaler.transform(X_pred)
        
        # Predict health needs
        health_needs = {}
        for need, model in self.need_models.items():
            health_needs[need] = model.predict(X_pred_scaled)[0]
        
        # Map needs to challenge areas
        need_to_area = {
            'needs_cardio': 'cardiovascular',
            'needs_weight_management': 'weight',
            'needs_nutrition': 'nutrition',
            'needs_sleep': 'sleep',
            'needs_exercise': 'exercise',
            'needs_stress_management': 'stress',
            'needs_smoking_cessation': 'smoking'
        }
        
        # Determine needed health areas
        needed_areas = []
        for need, value in health_needs.items():
            if value == 1:
                needed_areas.append(need_to_area[need])
        
        # Add mental health area based on stress
        if user_data.get('stress_level', 0) > 7:
            needed_areas.append('mental_health')
        
        # If nothing specific needed, default to exercise and nutrition
        if not needed_areas:
            needed_areas = ['exercise', 'nutrition']
        
        # Find challenges that match needed health areas
        matching_challenges = {}
        for challenge, areas in self.available_challenges.items():
            overlap = set(areas).intersection(set(needed_areas))
            if overlap:
                matching_challenges[challenge] = len(overlap)
        
        # Get team info
        team = user_data.get('team', None)
        company = user_data.get('company', None)
        
        # Rank challenges by relevance to health needs
        if matching_challenges:
            best_challenges = sorted(matching_challenges.items(), key=lambda x: x[1], reverse=True)
            best_challenge = best_challenges[0][0]
            
            # Format the recommendation
            if team and random.random() < 0.7:  # 70% chance for team challenge
                return f"Join your {team} team in the {best_challenge} to improve your health metrics."
            else:
                return f"Start the {best_challenge} to improve your well-being."
        else:
            # Fallback recommendation
            return "Join our Step Challenge to boost your daily activity levels."
    
    def get_recommendations(self, user_data):
        """Generate both medical and lifestyle recommendations"""
        medical_rec = self.get_medical_recommendation(user_data)
        challenge_rec = self.get_challenge_recommendation(user_data)
        
        return {
            'medical_recommendation': medical_rec,
            'challenge_recommendation': challenge_rec
        }

#############################################
# 3. STREAMLIT FRONTEND
#############################################

def run_app():
    """Run the Streamlit application"""
    st.set_page_config(page_title="Health Recommendation Engine", page_icon="❤️", layout="wide")
    
    st.title("❤️ Health & Wellness Recommendation Engine")
    st.write("Enter your health profile to get personalized recommendations.")
    
    # Generate dummy data for the engine
    if 'data_dict' not in st.session_state:
        st.session_state.data_dict = generate_dummy_data()
        st.session_state.engine = HealthRecommendationEngine(st.session_state.data_dict)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        age = st.slider("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        company = st.selectbox("Company", ["TechCorp", "HealthInc", "FinanceOne", "RetailPro"])
        team = st.selectbox("Team", ["Engineering", "Sales", "HR", "Marketing", "Operations"])
        
        st.subheader("Lifestyle")
        smoking_status = st.selectbox("Smoking Status", ["Never", "Former", "Occasional", "Regular"])
        sleep_hours = st.slider("Average Sleep (hours/day)", 3.0, 10.0, 7.0)
        diet_score = st.slider("Diet Quality (1-10)", 1, 10, 6)
        exercise_days = st.slider("Exercise Days per Week", 0, 7, 3)
        stress_level = st.slider("Stress Level (1-10)", 1, 10, 5)
        
    with col2:
        st.subheader("Medical Metrics")
        systolic_bp = st.slider("Systolic BP (mmHg)", 90, 200, 120)
        diastolic_bp = st.slider("Diastolic BP (mmHg)", 60, 120, 80)
        bmi = st.slider("BMI", 15.0, 45.0, 24.5)
        
        st.subheader("Lab Results")
        cholesterol = st.slider("Total Cholesterol (mg/dL)", 120, 300, 180)
        hdl = st.slider("HDL (mg/dL)", 20, 100, 50)
        ldl = st.slider("LDL (mg/dL)", 50, 250, 100)
        glucose = st.slider("Fasting Glucose (mg/dL)", 70, 200, 90)
    
    # Put all user data together
    user_data = {
        'age': age,
        'gender': gender,
        'company': company,
        'team': team,
        'smoking_status': smoking_status,
        'sleep_hours': sleep_hours,
        'diet_score': diet_score,
        'exercise_days': exercise_days,
        'stress_level': stress_level,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'bmi': bmi,
        'cholesterol': cholesterol,
        'hdl': hdl,
        'ldl': ldl,
        'glucose': glucose
    }
    
    # Add additional features for the recommendation engine
    avg_steps = 8000 - age * 30 + exercise_days * 500
    user_data['avg_steps'] = np.clip(avg_steps, 1000, 20000)
    
    st.markdown("---")
    
    # Generate recommendations when button is clicked
    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Generating personalized recommendations..."):
            # Get recommendations from engine
            recommendations = st.session_state.engine.get_recommendations(user_data)
            
            # Display recommendations in a nice format
            st.subheader("Your Personalized Health Recommendations")
            
            # Medical Recommendation
            st.markdown("### 🩺 Medical Recommendation")
            medical_rec = recommendations['medical_recommendation']
            
            # Color-code based on urgency
            if medical_rec.startswith("URGENT"):
                st.error(medical_rec)
            elif medical_rec.startswith("IMPORTANT"):
                st.warning(medical_rec)
            else:
                st.info(medical_rec)
            
            # Challenge Recommendation
            st.markdown("### 🏃‍♀️ Lifestyle Challenge Recommendation")
            st.success(recommendations['challenge_recommendation'])
            
            # Health insights based on key metrics
            st.markdown("### 📊 Health Insights")
            cols = st.columns(3)
            
            with cols[0]:
                bp_category = st.session_state.engine._categorize_bp(systolic_bp, diastolic_bp)
                bp_status = {
                    'normal': '✅ Normal',
                    'elevated': '⚠️ Elevated',
                    'hypertension_1': '🔴 Stage 1 Hypertension',
                    'hypertension_2': '🔴 Stage 2 Hypertension',
                    'hypertensive_crisis': '⛔ Hypertensive Crisis'
                }
                st.metric("Blood Pressure Status", bp_status.get(bp_category, "Unknown"))
            
            with cols[1]:
                bmi_category = st.session_state.engine._categorize_bmi(bmi)
                bmi_status = {
                    'underweight': '⚠️ Underweight',
                    'normal': '✅ Normal',
                    'overweight': '⚠️ Overweight',
                    'obese': '🔴 Obese',
                    'extremely_obese': '🔴 Extremely Obese'
                }
                st.metric("BMI Status", bmi_status.get(bmi_category, "Unknown"))
                
            with cols[2]:
                glucose_category = 'normal' if glucose < 100 else ('prediabetic' if glucose < 126 else 'diabetic')
                glucose_status = {
                    'normal': '✅ Normal',
                    'prediabetic': '⚠️ Prediabetic',
                    'diabetic': '🔴 Diabetic Range'
                }
                st.metric("Glucose Status", glucose_status.get(glucose_category, "Unknown"))

#############################################
# 4. MAIN FUNCTION
#############################################

if __name__ == "__main__":
    run_app()
