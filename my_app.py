import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import json

# Set page configuration
st.set_page_config(
    page_title="Health Recommendation Engine",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== DATA GENERATION ==========

def generate_dummy_data(num_users=1000, seed=42):
    """Generate realistic dummy health data for model training"""
    np.random.seed(seed)
    
    # User demographics
    user_ids = range(1, num_users + 1)
    ages = np.random.normal(40, 10, num_users).astype(int)
    ages = np.clip(ages, 20, 70)  # Clip ages between 20 and 70
    genders = np.random.choice(['Male', 'Female'], size=num_users)
    companies = np.random.choice(['TechCorp', 'HealthInc', 'FinanceGlobal', 'RetailMart', 'EduSystems'], size=num_users)
    teams = np.random.choice(['Engineering', 'HR', 'Finance', 'Marketing', 'Operations', 'Sales', 'Product'], size=num_users)
    
    # Health Risk Assessment (HRA) Data
    smoking_scores = np.random.choice([0, 1, 2, 3, 4, 5], size=num_users, p=[0.6, 0.1, 0.1, 0.1, 0.05, 0.05])  # 0 = non-smoker, 5 = heavy
    sleep_scores = np.random.normal(7, 1.5, num_users)  # Average hours of sleep
    sleep_scores = np.clip(sleep_scores, 3, 10).round(1)
    diet_scores = np.random.normal(6, 2, num_users)  # Diet quality 1-10
    diet_scores = np.clip(diet_scores, 1, 10).round(0).astype(int)
    exercise_scores = np.random.normal(3, 1.5, num_users)  # Days per week
    exercise_scores = np.clip(exercise_scores, 0, 7).round(0).astype(int)
    
    # EMR Data
    # BMI calculation (weight/height^2)
    heights = np.random.normal(1.7, 0.1, num_users).round(2)  # meters
    weights = np.random.normal(75, 15, num_users).round(1)  # kg
    bmis = (weights / (heights ** 2)).round(1)
    
    # Blood pressure - systolic
    systolic_bp = np.random.normal(120, 15, num_users).astype(int)
    systolic_bp = np.clip(systolic_bp, 90, 200)
    
    # Blood pressure - diastolic
    diastolic_bp = np.random.normal(80, 10, num_users).astype(int)
    diastolic_bp = np.clip(diastolic_bp, 50, 120)
    
    # Common diagnoses
    has_hypertension = (systolic_bp > 140) | (diastolic_bp > 90)
    has_diabetes = np.random.choice([True, False], size=num_users, p=[0.12, 0.88])  # 12% prevalence
    has_high_cholesterol = np.random.choice([True, False], size=num_users, p=[0.15, 0.85])  # 15% prevalence
    
    # Lab results
    cholesterol = np.random.normal(190, 35, num_users).round(0).astype(int)
    cholesterol = np.where(has_high_cholesterol, cholesterol + 40, cholesterol)
    cholesterol = np.clip(cholesterol, 120, 300)
    
    glucose = np.random.normal(90, 10, num_users).round(0).astype(int)
    glucose = np.where(has_diabetes, glucose + 40, glucose)
    glucose = np.clip(glucose, 70, 250)
    
    # Health Risk Predictions (would come from a model in real life)
    # Here we'll generate them based on our other variables
    
    # Diabetes risk (influenced by BMI, age, family history, activity)
    diabetes_risk = 0.05 + (0.01 * (ages - 30) / 10) + (0.02 * (bmis - 25) / 5) + (0.1 * has_diabetes.astype(int)) - (0.01 * exercise_scores)
    # Add some randomness
    diabetes_risk = diabetes_risk + np.random.normal(0, 0.05, num_users)
    diabetes_risk = np.clip(diabetes_risk, 0.01, 0.99).round(2)
    
    # Heart disease risk (influenced by BP, cholesterol, smoking, age, gender)
    cvd_risk = 0.05 + (0.01 * (ages - 30) / 10) + (0.01 * (systolic_bp - 120) / 10) + (0.01 * (cholesterol - 180) / 20) + (0.02 * smoking_scores) - (0.01 * exercise_scores)
    # Males have slightly higher base CVD risk
    cvd_risk = np.where(genders == 'Male', cvd_risk + 0.03, cvd_risk)
    # Add some randomness
    cvd_risk = cvd_risk + np.random.normal(0, 0.05, num_users)
    cvd_risk = np.clip(cvd_risk, 0.01, 0.99).round(2)
    
    # Hypertension risk
    hypertension_risk = 0.05 + (0.01 * (ages - 30) / 10) + (0.02 * (systolic_bp - 120) / 10) + (0.005 * (bmis - 25)) + (0.01 * smoking_scores)
    # Add some randomness
    hypertension_risk = hypertension_risk + np.random.normal(0, 0.05, num_users)
    hypertension_risk = np.clip(hypertension_risk, 0.01, 0.99).round(2)
    
    # App Usage Data
    avg_daily_steps = np.random.normal(7000, 3000, num_users).round(-2).astype(int)
    avg_daily_steps = np.clip(avg_daily_steps, 1000, 20000)
    
    sleep_tracking_usage = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=num_users, p=[0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # Days per week
    water_tracking_usage = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7], size=num_users, p=[0.4, 0.1, 0.1, 0.1, 0.1, 0.05, 0.05, 0.1])  # Days per week
    
    doctor_visits_6mo = np.random.choice([0, 1, 2, 3, 4, 5], size=num_users, p=[0.6, 0.2, 0.1, 0.05, 0.03, 0.02])
    
    # Challenge Participation
    challenges_joined = np.random.choice([0, 1, 2, 3, 4, 5], size=num_users, p=[0.4, 0.3, 0.15, 0.1, 0.03, 0.02])  # Last 6 months
    challenges_completed = np.zeros(num_users).astype(int)
    
    for i in range(num_users):
        if challenges_joined[i] > 0:
            # Complete between 0 and all challenges joined
            challenges_completed[i] = np.random.randint(0, challenges_joined[i] + 1)
    
    # Create the user dataframe
    user_df = pd.DataFrame({
        'user_id': user_ids,
        'age': ages,
        'gender': genders,
        'company': companies,
        'team': teams,
        'smoking_score': smoking_scores,
        'sleep_hours': sleep_scores,
        'diet_score': diet_scores,
        'exercise_days_per_week': exercise_scores,
        'height_m': heights,
        'weight_kg': weights,
        'bmi': bmis,
        'systolic_bp': systolic_bp,
        'diastolic_bp': diastolic_bp,
        'diagnosed_hypertension': has_hypertension,
        'diagnosed_diabetes': has_diabetes,
        'diagnosed_high_cholesterol': has_high_cholesterol,
        'cholesterol': cholesterol,
        'glucose': glucose,
        'diabetes_risk': diabetes_risk,
        'cvd_risk': cvd_risk,
        'hypertension_risk': hypertension_risk,
        'avg_daily_steps': avg_daily_steps,
        'sleep_tracking_days_per_week': sleep_tracking_usage,
        'water_tracking_days_per_week': water_tracking_usage,
        'doctor_visits_6mo': doctor_visits_6mo,
        'challenges_joined_6mo': challenges_joined,
        'challenges_completed_6mo': challenges_completed
    })
    
    return user_df

# Generate challenges data
def generate_challenges_data():
    """Generate sample challenges data"""
    challenges_data = [
        {
            'id': 1,
            'name': '10,000 Steps Challenge',
            'description': 'Achieve 10,000 steps daily for 30 days',
            'category': 'Physical Activity',
            'duration_days': 30,
            'team_based': True,
            'difficulty': 'Medium'
        },
        {
            'id': 2,
            'name': 'Meditation Minutes',
            'description': 'Complete 10 minutes of meditation daily for 21 days',
            'category': 'Mental Wellness',
            'duration_days': 21,
            'team_based': False,
            'difficulty': 'Easy'
        },
        {
            'id': 3,
            'name': 'Hydration Hero',
            'description': 'Drink 8 glasses of water daily for 14 days',
            'category': 'Nutrition',
            'duration_days': 14,
            'team_based': False,
            'difficulty': 'Easy'
        },
        {
            'id': 4,
            'name': 'Sleep Improvement',
            'description': 'Get 7+ hours of sleep for 21 consecutive nights',
            'category': 'Rest & Recovery',
            'duration_days': 21,
            'team_based': False,
            'difficulty': 'Medium'
        },
        {
            'id': 5,
            'name': 'Department Step Challenge',
            'description': 'Department with highest average step count wins',
            'category': 'Physical Activity',
            'duration_days': 30,
            'team_based': True,
            'difficulty': 'Medium'
        },
        {
            'id': 6,
            'name': 'Sugar Reduction',
            'description': 'Reduce daily added sugar intake for 14 days',
            'category': 'Nutrition',
            'duration_days': 14,
            'team_based': False,
            'difficulty': 'Hard'
        },
        {
            'id': 7,
            'name': 'Strength Training',
            'description': 'Complete strength training 3x per week for 4 weeks',
            'category': 'Physical Activity',
            'duration_days': 28,
            'team_based': False,
            'difficulty': 'Medium'
        },
        {
            'id': 8,
            'name': 'Team Wellness Bingo',
            'description': 'Teams complete wellness activities to get bingo',
            'category': 'General Wellness',
            'duration_days': 30,
            'team_based': True,
            'difficulty': 'Easy'
        },
        {
            'id': 9,
            'name': 'Mindful Eating',
            'description': 'Practice mindful eating techniques for 14 days',
            'category': 'Nutrition',
            'duration_days': 14,
            'team_based': False,
            'difficulty': 'Medium'
        },
        {
            'id': 10,
            'name': 'Cardio Challenge',
            'description': 'Complete 150 minutes of cardio exercise weekly',
            'category': 'Physical Activity',
            'duration_days': 28,
            'team_based': False,
            'difficulty': 'Hard'
        }
    ]
    
    return pd.DataFrame(challenges_data)

# Generate insurance products data
def generate_insurance_products():
    """Generate sample insurance products"""
    insurance_products = [
        {
            'id': 1,
            'name': 'Basic Health Plan',
            'coverage': 'Essential medical coverage',
            'best_for': 'Young, healthy individuals with low health risks',
            'risk_profile': 'Low risk',
            'monthly_premium_range': '$200-300'
        },
        {
            'id': 2,
            'name': 'Standard Health Plan',
            'coverage': 'Comprehensive medical coverage with moderate copays',
            'best_for': 'Individuals with moderate health risks',
            'risk_profile': 'Moderate risk',
            'monthly_premium_range': '$300-500'
        },
        {
            'id': 3,
            'name': 'Premium Health Plan',
            'coverage': 'Comprehensive coverage with lower deductibles and copays',
            'best_for': 'Families and individuals with higher health risks',
            'risk_profile': 'High risk',
            'monthly_premium_range': '$500-800'
        },
        {
            'id': 4,
            'name': 'Diabetes Care Plan',
            'coverage': 'Enhanced coverage for diabetes management',
            'best_for': 'Individuals with diabetes or elevated diabetes risk',
            'risk_profile': 'Diabetes focus',
            'monthly_premium_range': '$400-600'
        },
        {
            'id': 5,
            'name': 'Heart Health Plan',
            'coverage': 'Enhanced coverage for cardiovascular conditions',
            'best_for': 'Individuals with heart conditions or elevated CVD risk',
            'risk_profile': 'Cardiovascular focus',
            'monthly_premium_range': '$450-650'
        },
        {
            'id': 6,
            'name': 'Family Plus Plan',
            'coverage': 'Comprehensive family coverage with wellness benefits',
            'best_for': 'Families seeking preventive care and wellness benefits',
            'risk_profile': 'Family focus',
            'monthly_premium_range': '$600-900'
        },
        {
            'id': 7,
            'name': 'Senior Care Plan',
            'coverage': 'Tailored coverage for seniors with chronic condition management',
            'best_for': 'Individuals over 55 with multiple health concerns',
            'risk_profile': 'Senior focus',
            'monthly_premium_range': '$500-700'
        }
    ]
    
    return pd.DataFrame(insurance_products)

# Generate medical recommendations data
def generate_medical_recommendations():
    """Generate medical recommendation templates"""
    recommendations = [
        {
            'id': 1,
            'condition': 'high_bp',
            'recommendation': 'Schedule a doctor visit to discuss your blood pressure readings.',
            'urgency': 'Medium',
            'followup_days': 30
        },
        {
            'id': 2,
            'condition': 'very_high_bp',
            'recommendation': 'Urgent: Consult with a healthcare provider about your blood pressure as soon as possible.',
            'urgency': 'High',
            'followup_days': 7
        },
        {
            'id': 3,
            'condition': 'high_glucose',
            'recommendation': 'Schedule a diabetes screening with your healthcare provider.',
            'urgency': 'Medium',
            'followup_days': 30
        },
        {
            'id': 4,
            'condition': 'high_cholesterol',
            'recommendation': 'Consider scheduling a lipid panel test with your doctor.',
            'urgency': 'Medium',
            'followup_days': 60
        },
        {
            'id': 5,
            'condition': 'high_bmi',
            'recommendation': 'Consider consulting with a nutritionist about healthy weight management strategies.',
            'urgency': 'Low',
            'followup_days': 90
        },
        {
            'id': 6, 
            'condition': 'diabetes_risk',
            'recommendation': 'Based on your profile, we recommend scheduling a diabetes screening with your doctor.',
            'urgency': 'Medium',
            'followup_days': 60
        },
        {
            'id': 7,
            'condition': 'cvd_risk',
            'recommendation': 'Consider scheduling a cardiovascular health check with your doctor.',
            'urgency': 'Medium',
            'followup_days': 60
        },
        {
            'id': 8,
            'condition': 'general_checkup',
            'recommendation': 'It\'s time for your annual physical examination.',
            'urgency': 'Low',
            'followup_days': 90
        },
        {
            'id': 9,
            'condition': 'sleep_issues',
            'recommendation': 'Consider consulting with a sleep specialist about improving your sleep quality.',
            'urgency': 'Low',
            'followup_days': 90
        },
        {
            'id': 10,
            'condition': 'preventive_screening',
            'recommendation': 'Schedule recommended preventive screenings based on your age and gender.',
            'urgency': 'Low',
            'followup_days': 180
        }
    ]
    
    return pd.DataFrame(recommendations)

# ========== RECOMMENDATION ENGINE ==========

class HealthRecommendationEngine:
    """Main recommendation engine class that combines rules and ML-based approaches"""
    
    def __init__(self):
        """Initialize the recommendation engine components"""
        # Load or generate necessary data
        self.user_data = generate_dummy_data(1000)
        self.challenges_data = generate_challenges_data()
        self.insurance_products = generate_insurance_products()
        self.medical_recommendations = generate_medical_recommendations()
        
        # Initialize models
        self.lifestyle_challenge_model = None
        self.insurance_recommendation_model = None
        
        # Train models
        self.train_models()
    
    def train_models(self):
        """Train recommendation models on dummy data"""
        # Create training data for challenge recommendation
        # Feature engineering: What features predict challenge completion?
        X_challenge = self.user_data[['age', 'exercise_days_per_week', 'avg_daily_steps', 
                                    'sleep_tracking_days_per_week', 'water_tracking_days_per_week',
                                    'challenges_joined_6mo', 'challenges_completed_6mo']]
        
        # Target: Did they complete challenges? (success rate > 50%)
        y_challenge = (self.user_data['challenges_completed_6mo'] / 
                      (self.user_data['challenges_joined_6mo'] + 0.001) > 0.5).astype(int)
        
        # Simple Decision Tree model
        self.lifestyle_challenge_model = DecisionTreeClassifier(max_depth=5, random_state=42)
        self.lifestyle_challenge_model.fit(X_challenge, y_challenge)
        
        # Create training data for insurance recommendation
        # Feature engineering: What features predict insurance needs?
        X_insurance = self.user_data[['age', 'bmi', 'systolic_bp', 'diabetes_risk', 
                                     'cvd_risk', 'hypertension_risk', 'diagnosed_diabetes',
                                     'diagnosed_hypertension', 'diagnosed_high_cholesterol']]
        
        # Target: High risk (simplified for prototype - would be more sophisticated in production)
        # This creates 3 categories: 0=low risk, 1=moderate risk, 2=high risk
        y_insurance = np.zeros(len(self.user_data)).astype(int)
        
        # Moderate risk: at least one risk factor is elevated
        moderate_risk = ((self.user_data['diabetes_risk'] > 0.15) | 
                         (self.user_data['cvd_risk'] > 0.15) | 
                         (self.user_data['hypertension_risk'] > 0.15) |
                         (self.user_data['bmi'] > 27))
        
        # High risk: diagnosed condition or multiple high risk factors
        high_risk = ((self.user_data['diagnosed_diabetes']) | 
                     (self.user_data['diagnosed_hypertension']) | 
                     (self.user_data['diagnosed_high_cholesterol']) |
                     ((self.user_data['diabetes_risk'] > 0.3) & (self.user_data['cvd_risk'] > 0.3)))
        
        y_insurance[moderate_risk] = 1
        y_insurance[high_risk] = 2
        
        self.insurance_recommendation_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        self.insurance_recommendation_model.fit(X_insurance, y_insurance)
        
    def get_medical_recommendation(self, user_profile):
        """Rules-based medical recommendation logic"""
        recommendations = []
        
        # Check for high blood pressure
        if user_profile['systolic_bp'] >= 140 or user_profile['diastolic_bp'] >= 90:
            if user_profile['systolic_bp'] >= 160 or user_profile['diastolic_bp'] >= 100:
                recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'very_high_bp'].iloc[0])
            else:
                recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'high_bp'].iloc[0])
        
        # Check for high glucose
        if user_profile['glucose'] >= 126:
            recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'high_glucose'].iloc[0])
        
        # Check for high cholesterol
        if user_profile['cholesterol'] >= 240:
            recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'high_cholesterol'].iloc[0])
        
        # Check for high BMI
        if user_profile['bmi'] >= 30:
            recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'high_bmi'].iloc[0])
        
        # Check for diabetes risk
        if user_profile['diabetes_risk'] >= 0.25 and not user_profile['diagnosed_diabetes']:
            recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'diabetes_risk'].iloc[0])
        
        # Check for cardiovascular risk
        if user_profile['cvd_risk'] >= 0.25:
            recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'cvd_risk'].iloc[0])
        
        # Check for sleep issues
        if user_profile['sleep_hours'] < 6:
            recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'sleep_issues'].iloc[0])
        
        # If no specific recommendations, suggest general checkup
        if len(recommendations) == 0:
            recommendations.append(self.medical_recommendations[self.medical_recommendations['condition'] == 'general_checkup'].iloc[0])
        
        # Sort by urgency
        urgency_map = {'High': 0, 'Medium': 1, 'Low': 2}
        recommendations.sort(key=lambda x: urgency_map[x['urgency']])
        
        # Return the top recommendation (most urgent)
        return recommendations[0]
    
    def get_insurance_recommendation(self, user_profile):
        """ML-based insurance recommendation logic"""
        # Extract features for prediction
        features = np.array([[
            user_profile['age'],
            user_profile['bmi'],
            user_profile['systolic_bp'],
            user_profile['diabetes_risk'],
            user_profile['cvd_risk'],
            user_profile['hypertension_risk'],
            user_profile['diagnosed_diabetes'],
            user_profile['diagnosed_hypertension'],
            user_profile['diagnosed_high_cholesterol']
        ]])
        
        # Predict insurance category
        insurance_category = self.insurance_recommendation_model.predict(features)[0]
        
        # Map category to specific product recommendations
        if insurance_category == 0:  # Low risk
            return self.insurance_products[self.insurance_products['id'] == 1].iloc[0]
        elif insurance_category == 1:  # Moderate risk
            # Check specific risk factors for targeted plans
            if user_profile['diabetes_risk'] > 0.2:
                return self.insurance_products[self.insurance_products['id'] == 4].iloc[0]  # Diabetes Care Plan
            elif user_profile['cvd_risk'] > 0.2:
                return self.insurance_products[self.insurance_products['id'] == 5].iloc[0]  # Heart Health Plan
            else:
                return self.insurance_products[self.insurance_products['id'] == 2].iloc[0]  # Standard Health Plan
        else:  # High risk
            if user_profile['age'] > 55:
                return self.insurance_products[self.insurance_products['id'] == 7].iloc[0]  # Senior Care Plan
            else:
                return self.insurance_products[self.insurance_products['id'] == 3].iloc[0]  # Premium Health Plan
    
    def get_lifestyle_recommendation(self, user_profile):
        """ML-based lifestyle/challenge recommendation logic"""
        # Create feature array for prediction
        features = np.array([[
            user_profile['age'],
            user_profile['exercise_days_per_week'],
            user_profile['avg_daily_steps'],
            user_profile['sleep_tracking_days_per_week'],
            user_profile['water_tracking_days_per_week'],
            user_profile['challenges_joined_6mo'],
            user_profile['challenges_completed_6mo']
        ]])
        
        # Predict likelihood of completing challenges
        challenge_completion_likelihood = self.lifestyle_challenge_model.predict_proba(features)[0][1]
        
        # Recommend based on health metrics and activity level
        recommended_challenges = []
        
        # Physical activity recommendations
        if user_profile['exercise_days_per_week'] < 3 or user_profile['avg_daily_steps'] < 7000:
            if challenge_completion_likelihood > 0.6:  # Likely to complete more difficult challenges
                recommended_challenges.append(self.challenges_data[self.challenges_data['id'] == 10].iloc[0])  # Cardio Challenge
            else:
                recommended_challenges.append(self.challenges_data[self.challenges_data['id'] == 1].iloc[0])  # 10,000 Steps
        
        # Mental wellness
        if user_profile['sleep_hours'] < 7:
            recommended_challenges.append(self.challenges_data[self.challenges_data['id'] == 4].iloc[0])  # Sleep Improvement
        
        # Nutrition recommendations
        if user_profile['diet_score'] < 6:
            recommended_challenges.append(self.challenges_data[self.challenges_data['id'] == 9].iloc[0])  # Mindful Eating
        
        # Water tracking
        if user_profile['water_tracking_days_per_week'] < 3:
            recommended_challenges.append(self.challenges_data[self.challenges_data['id'] == 3].iloc[0])  # Hydration Hero
        
        # Team challenges based on previous participation
        if user_profile['challenges_completed_6mo'] > 0 and challenge_completion_likelihood > 0.5:
            if user_profile['team'] is not None:
                recommended_challenges.append(self.challenges_data[self.challenges_data['id'] == 5].iloc[0])  # Department Step Challenge
        
        # If no specific recommendations, suggest general wellness challenge
        if len(recommended_challenges) == 0:
            recommended_challenges.append(self.challenges_data[self.challenges_data['id'] == 8].iloc[0])  # Team Wellness Bingo
        
        # Return the top recommendation
        # Could be enhanced with ranking logic in a full implementation
        return recommended_challenges[0]
    
    def get_all_recommendations(self, user_profile):
        """Get all recommendation types for a given user profile"""
        medical_rec = self.get_medical_recommendation(user_profile)
        insurance_rec = self.get_insurance_recommendation(user_profile)
        lifestyle_rec = self.get_lifestyle_recommendation(user_profile)
        
        return {
            'medical_recommendation': medical_rec,
            'insurance_recommendation': insurance_rec,
            'lifestyle_recommendation': lifestyle_rec
        }

# ========== STREAMLIT UI ==========

def main():
    st.title("🩺 Health & Wellness Recommendation Engine")
    st.write("This prototype demonstrates personalized health recommendations based on user health data.")
    
    # Initialize recommendation engine
    if 'recommendation_engine' not in st.session_state:
        with st.spinner("Initializing recommendation engine..."):
            st.session_state.recommendation_engine = HealthRecommendationEngine()
            st.success("Recommendation engine initialized successfully!")
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["User Input", "Data Explorer", "About"])
    
    with tab1:
        st.header("Enter Your Health Profile")
        
        # Create columns for form layout
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Demographics")
            age = st.slider("Age", 20, 70, 35)
            gender = st.selectbox("Gender", ["Male", "Female"])
            company = st.selectbox("Company", ["TechCorp", "HealthInc", "FinanceGlobal", "RetailMart", "EduSystems"])
            team = st.selectbox("Team", ["Engineering", "HR", "Finance", "Marketing", "Operations", "Sales", "Product"])
            
        with col2:
            st.subheader("Health Metrics")
            height = st.number_input("Height (m)", 1.4, 2.1, 1.7, 0.01)
            weight = st.number_input("Weight (kg)", 40.0, 150.0, 70.0, 0.1)
            bmi = round(weight / (height ** 2), 1)
            st.info(f"Calculated BMI: {bmi}")
            
            systolic_bp = st.slider("Systolic Blood Pressure", 90, 200, 120)
            diastolic_bp = st.slider("Diastolic Blood Pressure", 50, 120, 80)
            
        st.subheader("Lifestyle & Health")
        col3, col4 = st.columns(2)
        
        with col3:
            smoking_score = st.selectbox("Smoking Status", 
                                       [(0, "Non-smoker"), 
                                        (1, "Occasional"), 
                                        (3, "Regular"), 
                                        (5, "Heavy")], format_func=lambda x: x[1])
            smoking_score = smoking_score[0]
            
            sleep_hours = st.slider("Average Sleep (hours/night)", 3.0, 10.0, 7.0, 0.5)
            diet_score = st.slider("Diet Quality (1-10)", 1, 10, 6)
            exercise_days = st.slider("Exercise (days/week)", 0, 7, 3)
            
        with col4:
            diagnosed_conditions = st.multiselect("Diagnosed Conditions", 
                                                ["Hypertension", "Diabetes", "High Cholesterol"],
                                                default=[])
            
            diagnosed_hypertension = "Hypertension" in diagnosed_conditions
            diagnosed_diabetes = "Diabetes" in diagnosed_conditions
            diagnosed_high_cholesterol = "High Cholesterol" in diagnosed_conditions
            
            cholesterol = st.slider("Total Cholesterol", 120, 300, 190)
            glucose = st.slider("Fasting Glucose", 70, 250, 95)
        
        st.subheader("App Usage")
        col5, col6 = st.columns(2)
        
        with col5:
            avg_daily_steps = st.slider("Average Daily Steps", 1000, 20000, 7000, 500)
            sleep_tracking = st.slider("Sleep Tracking (days/week)", 0, 7, 3)
            water_tracking = st.slider("Water Tracking (days/week)", 0, 7, 2)
            
        with col6:
            doctor_visits = st.slider("Doctor Visits (past 6 months)", 0, 5, 1)
            challenges_joined = st.slider("Wellness Challenges Joined (past 6 months)", 0, 5, 1)
            challenges_completed = st.slider("Wellness Challenges Completed (past 6 months)", 
                                          0, max(challenges_joined, 1), min(1, challenges_joined))
        
        # Calculate risk scores (simplified for prototype - would be more sophisticated ML model in production)
        # These are approximations based on the risk factors
        diabetes_risk = 0.05 + (0.01 * (age - 30) / 10) + (0.02 * (bmi - 25) / 5) + (0.1 * diagnosed_diabetes) - (0.01 * exercise_days)
        diabetes_risk = round(max(0.01, min(0.99, diabetes_risk + (glucose - 90) / 500)), 2)
        
        cvd_risk = 0.05 + (0.01 * (age - 30) / 10) + (0.01 * (systolic_bp - 120) / 10) + (0.01 * (cholesterol - 180) / 20) + (0.02 * smoking_score) - (0.01 * exercise_days)
        cvd_risk = round(max(0.01, min(0.99, cvd_risk + (0.03 if gender == "Male" else 0))), 2)
        
        hypertension_risk = 0.05 + (0.01 * (age - 30) / 10) + (0.02 * (systolic_bp - 120) / 10) + (0.005 * (bmi - 25)) + (0.01 * smoking_score)
        hypertension_risk = round(max(0.01, min(0.99, hypertension_risk)), 2)
        
        # Display risk scores
        st.subheader("Calculated Risk Scores")
        risk_col1, risk_col2, risk_col3 = st.columns(3)
        
        risk_col1.metric("Diabetes Risk", f"{diabetes_risk:.0%}")
        risk_col2.metric("Cardiovascular Risk", f"{cvd_risk:.0%}")
        risk_col3.metric("Hypertension Risk", f"{hypertension_risk:.0%}")
        
        # Create user profile dictionary
        user_profile = {
            'age': age,
            'gender': gender,
            'company': company,
            'team': team,
            'height_m': height,
            'weight_kg': weight,
            'bmi': bmi,
            'systolic_bp': systolic_bp,
            'diastolic_bp': diastolic_bp,
            'smoking_score': smoking_score,
            'sleep_hours': sleep_hours,
            'diet_score': diet_score,
            'exercise_days_per_week': exercise_days,
            'diagnosed_hypertension': diagnosed_hypertension,
            'diagnosed_diabetes': diagnosed_diabetes,
            'diagnosed_high_cholesterol': diagnosed_high_cholesterol,
            'cholesterol': cholesterol,
            'glucose': glucose,
            'diabetes_risk': diabetes_risk,
            'cvd_risk': cvd_risk,
            'hypertension_risk': hypertension_risk,
            'avg_daily_steps': avg_daily_steps,
            'sleep_tracking_days_per_week': sleep_tracking,
            'water_tracking_days_per_week': water_tracking,
            'doctor_visits_6mo': doctor_visits,
            'challenges_joined_6mo': challenges_joined,
            'challenges_completed_6mo': challenges_completed
        }
        
        # Get recommendations button
        if st.button("Generate Personalized Recommendations", type="primary"):
            with st.spinner("Analyzing your health profile..."):
                recommendations = st.session_state.recommendation_engine.get_all_recommendations(user_profile)
                
                st.success("Analysis complete! Here are your personalized recommendations:")
                
                # Display recommendations in three columns
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    st.subheader("🏥 Medical Recommendation")
                    med_rec = recommendations['medical_recommendation']
                    st.markdown(f"**{med_rec['recommendation']}**")
                    st.caption(f"Urgency: {med_rec['urgency']}")
                    st.caption(f"Follow-up: {med_rec['followup_days']} days")
                
                with rec_col2:
                    st.subheader("🛡️ Insurance Suggestion")
                    ins_rec = recommendations['insurance_recommendation']
                    st.markdown(f"**{ins_rec['name']}**")
                    st.markdown(f"{ins_rec['coverage']}")
                    st.caption(f"Best for: {ins_rec['best_for']}")
                    st.caption(f"Monthly premium: {ins_rec['monthly_premium_range']}")
                
                with rec_col3:
                    st.subheader("🏆 Wellness Challenge")
                    life_rec = recommendations['lifestyle_recommendation']
                    st.markdown(f"**{life_rec['name']}**")
                    st.markdown(f"{life_rec['description']}")
                    st.caption(f"Category: {life_rec['category']}")
                    st.caption(f"Duration: {life_rec['duration_days']} days")
                    st.caption(f"Difficulty: {life_rec['difficulty']}")
                    st.caption(f"Team-based: {'Yes' if life_rec['team_based'] else 'No'}")
                
                # Detailed explanation
                with st.expander("Why these recommendations?"):
                    st.markdown("""
                    ### Medical Recommendation Logic
                    The medical recommendation is based on your vital signs, lab values, and risk scores.
                    High blood pressure, elevated glucose, high cholesterol, or high BMI can trigger specific recommendations.
                    
                    ### Insurance Product Logic
                    The insurance recommendation uses a machine learning model that considers your age, BMI, 
                    blood pressure, diagnosed conditions, and risk scores to suggest the most appropriate coverage level.
                    
                    ### Wellness Challenge Logic
                    The wellness challenge recommendation analyzes your current activity level, 
                    app usage patterns, and historical challenge participation to suggest activities
                    that you're likely to both benefit from and complete successfully.
                    """)
    
    with tab2:
        st.header("Data Explorer")
        st.write("Explore the dummy data used to train the recommendation models.")
        
        data_type = st.selectbox("Select data to explore", 
                                ["User Health Profiles", 
                                 "Wellness Challenges", 
                                 "Insurance Products", 
                                 "Medical Recommendations"])
        
        if data_type == "User Health Profiles":
            st.dataframe(st.session_state.recommendation_engine.user_data.head(20))
            
            st.subheader("Data Visualizations")
            viz_col1, viz_col2 = st.columns(2)
            
            with viz_col1:
                st.write("BMI Distribution")
                fig, ax = plt.subplots()
                sns.histplot(st.session_state.recommendation_engine.user_data['bmi'], kde=True, ax=ax)
                ax.axvline(x=25, color='orange', linestyle='--', label='Overweight Threshold')
                ax.axvline(x=30, color='red', linestyle='--', label='Obesity Threshold')
                ax.set_xlabel('BMI')
                ax.legend()
                st.pyplot(fig)
            
            with viz_col2:
                st.write("Blood Pressure Distribution")
                fig, ax = plt.subplots()
                sns.scatterplot(x='systolic_bp', y='diastolic_bp', 
                              data=st.session_state.recommendation_engine.user_data, 
                              alpha=0.5, ax=ax)
                ax.axvline(x=140, color='red', linestyle='--', label='High Systolic Threshold')
                ax.axhline(y=90, color='red', linestyle='--', label='High Diastolic Threshold')
                ax.set_xlabel('Systolic BP')
                ax.set_ylabel('Diastolic BP')
                ax.legend()
                st.pyplot(fig)
            
            viz_col3, viz_col4 = st.columns(2)
            
            with viz_col3:
                st.write("Risk Scores by Age")
                fig, ax = plt.subplots()
                data = st.session_state.recommendation_engine.user_data
                ax.scatter(data['age'], data['diabetes_risk'], alpha=0.3, label='Diabetes Risk')
                ax.scatter(data['age'], data['cvd_risk'], alpha=0.3, label='CVD Risk')
                ax.set_xlabel('Age')
                ax.set_ylabel('Risk Score')
                ax.legend()
                st.pyplot(fig)
            
            with viz_col4:
                st.write("Exercise vs. Steps")
                fig, ax = plt.subplots()
                sns.boxplot(x='exercise_days_per_week', y='avg_daily_steps', 
                          data=st.session_state.recommendation_engine.user_data, ax=ax)
                ax.set_xlabel('Exercise Days per Week')
                ax.set_ylabel('Average Daily Steps')
                st.pyplot(fig)
                
        elif data_type == "Wellness Challenges":
            st.dataframe(st.session_state.recommendation_engine.challenges_data)
            
            # Visualize challenge types
            st.subheader("Challenge Categories")
            fig, ax = plt.subplots()
            category_counts = st.session_state.recommendation_engine.challenges_data['category'].value_counts()
            category_counts.plot(kind='bar', ax=ax)
            ax.set_xlabel('Category')
            ax.set_ylabel('Count')
            st.pyplot(fig)
            
        elif data_type == "Insurance Products":
            st.dataframe(st.session_state.recommendation_engine.insurance_products)
            
        elif data_type == "Medical Recommendations":
            st.dataframe(st.session_state.recommendation_engine.medical_recommendations)
    
    with tab3:
        st.header("About This Prototype")
        st.markdown("""
        ### Architecture Overview
        
        This prototype demonstrates a hybrid recommendation engine that combines:
        
        1. **Rules-based logic** for medical recommendations based on clinical guidelines
        2. **Machine learning models** for lifestyle challenges and insurance product recommendations
        
        The system workflow:
        
        ```
        User Profile Input → Risk Assessment → Multi-domain Recommendations
        ```
        
        ### Components
        
        - **Data Generation**: Creates realistic synthetic health profiles for training
        - **Feature Engineering**: Extracts relevant features from user profiles
        - **Recommendation Logic**: Combines rules and ML to generate personalized recommendations
        - **Streamlit UI**: Simple interface for profile input and recommendation display
        
        ### Data Sources (Simulated)
        
        - Demographics and basic health data
        - Health risk assessment scores
        - Medical history and diagnoses
        - Wellness activity and engagement
        
        ### Limitations & Next Steps
        
        This is a prototype with synthetic data. In a production environment, you would:
        
        1. Use real, anonymized user data (with proper consent)
        2. Implement more sophisticated ML models
        3. Add recommendation explanations and evidence
        4. Integrate with actual medical guidelines and insurance products
        5. Implement HIPAA compliance and data security
        6. Add CI/CD pipeline for model updates
        """)
        
        with st.expander("Technical Implementation Notes"):
            st.markdown("""
            ### Model Selection
            
            For this prototype:
            
            - **Decision Tree** for lifestyle recommendations: Easily interpretable and works well with limited data
            - **Random Forest** for insurance recommendations: Better handles complex relationships
            
            In production, consider:
            
            - **Gradient Boosting** for improved accuracy
            - **Collaborative Filtering** for more personalized recommendations
            - **Deep Learning** for complex pattern recognition
            
            ### Scaling Considerations
            
            - Move model training offline
            - Implement model versioning
            - Add A/B testing framework
            - Develop feedback loops to improve recommendations
            """)

# Run the Streamlit app
if __name__ == "__main__":
    main()
