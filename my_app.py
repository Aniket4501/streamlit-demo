import streamlit as st
import numpy as np

# Define the generate_dummy_data function
def generate_dummy_data():
    return {
        'example_data': [1, 2, 3]  # Replace with your actual data
    }

# Define the HealthRecommendationEngine class
class HealthRecommendationEngine:
    def __init__(self, data):
        self.data = data
    
    def get_recommendations(self, user_data):
        return {
            'medical_recommendation': "URGENT: Please consult a doctor.",
            'challenge_recommendation': "Start a regular exercise routine."
        }

    def _categorize_bp(self, systolic, diastolic):
        if systolic < 120 and diastolic < 80:
            return 'normal'
        elif systolic < 130 and diastolic < 80:
            return 'elevated'
        elif systolic < 140 and diastolic < 90:
            return 'hypertension_1'
        elif systolic < 180 and diastolic < 120:
            return 'hypertension_2'
        else:
            return 'hypertensive_crisis'

    def _categorize_bmi(self, bmi):
        if bmi < 18.5:
            return 'underweight'
        elif bmi < 24.9:
            return 'normal'
        elif bmi < 29.9:
            return 'overweight'
        elif bmi < 40:
            return 'obese'
        else:
            return 'extremely_obese'

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

if __name__ == "__main__":
    run_app()
