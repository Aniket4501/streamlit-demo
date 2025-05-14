import streamlit as st
import numpy as np

# Define health category classification functions
def categorize_bp(systolic, diastolic):
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

def categorize_bmi(bmi):
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

def categorize_glucose(glucose):
    if glucose < 100:
        return 'normal'
    elif glucose < 126:
        return 'prediabetic'
    else:
        return 'diabetic'

def categorize_cholesterol(total, hdl, ldl):
    # Total cholesterol
    risk = 'normal'
    
    if total > 240:
        risk = 'high'
    elif total > 200:
        risk = 'borderline'
    
    # HDL (good cholesterol)
    if hdl < 40:
        risk = 'high' if risk != 'very_high' else risk
    
    # LDL (bad cholesterol)
    if ldl > 160:
        risk = 'high' if risk != 'very_high' else risk
    elif ldl > 130:
        risk = 'borderline' if risk == 'normal' else risk
    
    return risk

# Define comprehensive recommendation engine
class HealthRecommendationEngine:
    def __init__(self, data=None):
        self.data = data if data else {}
    
    def get_recommendations(self, user_data):
        recommendations = {
            'medical': {'message': "", 'urgency': "normal"},
            'lifestyle': [],
            'nutrition': [],
            'mental': []
        }
        
        # MEDICAL RECOMMENDATIONS
        
        # Blood pressure rules
        bp_category = self._categorize_bp(user_data['systolic_bp'], user_data['diastolic_bp'])
        if bp_category == 'hypertensive_crisis':
            recommendations['medical']['message'] = "URGENT: Seek immediate medical attention for your blood pressure."
            recommendations['medical']['urgency'] = "critical"
        elif bp_category == 'hypertension_2':
            recommendations['medical']['message'] = "IMPORTANT: Consult your doctor about your Stage 2 Hypertension soon."
            recommendations['medical']['urgency'] = "high"
        elif bp_category == 'hypertension_1':
            recommendations['medical']['message'] = "Schedule a check-up to discuss your Stage 1 Hypertension."
            recommendations['medical']['urgency'] = "medium"
        
        # Glucose rules
        glucose_category = self._categorize_glucose(user_data['glucose'])
        if glucose_category == 'diabetic':
            if recommendations['medical']['urgency'] != "critical":
                recommendations['medical']['message'] = "IMPORTANT: Consult your doctor about your elevated blood glucose levels."
                recommendations['medical']['urgency'] = "high"
            recommendations['nutrition'].append("Reduce intake of simple carbohydrates and sugars.")
            recommendations['nutrition'].append("Consider a low glycemic index diet.")
        elif glucose_category == 'prediabetic':
            if recommendations['medical']['urgency'] == "normal":
                recommendations['medical']['message'] = "Schedule a follow-up for your prediabetic glucose levels."
                recommendations['medical']['urgency'] = "medium"
            recommendations['nutrition'].append("Limit refined carbohydrates and sugary beverages.")
        
        # BMI rules
        bmi_category = self._categorize_bmi(user_data['bmi'])
        if bmi_category == 'extremely_obese':
            if recommendations['medical']['urgency'] not in ["critical", "high"]:
                recommendations['medical']['message'] = "IMPORTANT: Consider speaking with your doctor about weight management options."
                recommendations['medical']['urgency'] = "high"
            recommendations['lifestyle'].append("Start with gentle, low-impact exercises like swimming or walking.")
        elif bmi_category == 'obese':
            if recommendations['medical']['urgency'] == "normal":
                recommendations['medical']['message'] = "Schedule a check-up to discuss weight management strategies."
                recommendations['medical']['urgency'] = "medium"
            recommendations['lifestyle'].append("Aim for 150 minutes of moderate exercise per week.")
        elif bmi_category == 'overweight':
            recommendations['lifestyle'].append("Incorporate more physical activity into your daily routine.")
        elif bmi_category == 'underweight':
            if recommendations['medical']['urgency'] == "normal":
                recommendations['medical']['message'] = "Consider discussing your weight with a healthcare provider."
                recommendations['medical']['urgency'] = "low"
            recommendations['nutrition'].append("Focus on nutrient-dense foods to reach a healthy weight.")
        
        # Cholesterol rules
        cholesterol_category = self._categorize_cholesterol(user_data['cholesterol'], user_data['hdl'], user_data['ldl'])
        if cholesterol_category == 'high':
            if recommendations['medical']['urgency'] not in ["critical", "high"]:
                recommendations['medical']['message'] = "IMPORTANT: Consult with your doctor about your cholesterol levels."
                recommendations['medical']['urgency'] = "high"
            recommendations['nutrition'].append("Reduce intake of saturated and trans fats.")
            recommendations['nutrition'].append("Increase consumption of omega-3 fatty acids and fiber.")
        elif cholesterol_category == 'borderline':
            recommendations['nutrition'].append("Monitor fat intake and consider adding more heart-healthy foods.")
        
        # Smoking rules
        if user_data['smoking_status'] == 'Regular':
            if recommendations['medical']['urgency'] == "normal":
                recommendations['medical']['message'] = "Consider speaking with your doctor about smoking cessation options."
                recommendations['medical']['urgency'] = "medium"
            recommendations['lifestyle'].append("Consider a smoking cessation program or nicotine replacement therapy.")
        elif user_data['smoking_status'] == 'Occasional':
            recommendations['lifestyle'].append("Work towards completely quitting smoking.")
        
        # Exercise rules
        if user_data['exercise_days'] < 2:
            recommendations['lifestyle'].append("Start with at least 2 days of moderate exercise per week.")
        elif user_data['exercise_days'] < 4:
            recommendations['lifestyle'].append("Try to increase exercise frequency to 4-5 days per week.")
        
        # Sleep rules
        if user_data['sleep_hours'] < 6:
            recommendations['lifestyle'].append("Work on improving sleep duration to at least 7 hours per night.")
            recommendations['mental'].append("Consider a bedtime routine to improve sleep quality.")
        elif user_data['sleep_hours'] > 9:
            recommendations['lifestyle'].append("Excessive sleep may indicate other issues - try to maintain 7-9 hours.")
        
        # Diet rules
        if user_data['diet_score'] < 5:
            recommendations['nutrition'].append("Focus on incorporating more whole foods and vegetables.")
        
        # Stress rules
        if user_data['stress_level'] > 7:
            recommendations['mental'].append("Consider stress reduction techniques like meditation or mindfulness.")
            recommendations['mental'].append("Schedule regular breaks during work hours.")
        
        # Age-specific recommendations
        if user_data['age'] > 50:
            if recommendations['medical']['urgency'] == "normal":
                recommendations['medical']['message'] = "Schedule regular preventative check-ups as recommended for your age group."
            recommendations['lifestyle'].append("Include balance and strength training exercises to maintain mobility.")
        
        # If no medical recommendations have been made
        if recommendations['medical']['message'] == "":
            recommendations['medical']['message'] = "Continue routine health check-ups as recommended for your age and risk factors."
        
        # Personalize based on team/company
        if user_data['team'] == "Engineering":
            recommendations['mental'].append("Take regular breaks from screen time to reduce eye strain.")
        elif user_data['team'] == "Sales":
            recommendations['mental'].append("Practice stress management techniques for high-pressure situations.")
        
        if user_data['company'] == "TechCorp":
            recommendations['lifestyle'].append("Take advantage of your company's standing desk options.")
        elif user_data['company'] == "HealthInc":
            recommendations['lifestyle'].append("Utilize your employee wellness program benefits.")
        
        return recommendations
    
    # Use the previously defined functions but make them instance methods
    def _categorize_bp(self, systolic, diastolic):
        return categorize_bp(systolic, diastolic)
    
    def _categorize_bmi(self, bmi):
        return categorize_bmi(bmi)
    
    def _categorize_glucose(self, glucose):
        return categorize_glucose(glucose)
    
    def _categorize_cholesterol(self, total, hdl, ldl):
        return categorize_cholesterol(total, hdl, ldl)

def run_app():
    """Run the Streamlit application"""
    st.set_page_config(page_title="Health Recommendation Engine", page_icon="❤️", layout="wide")
    
    st.title("❤️ Health & Wellness Recommendation Engine")
    st.write("Enter your health profile to get personalized recommendations.")
    
    # Generate engine instance
    if 'engine' not in st.session_state:
        st.session_state.engine = HealthRecommendationEngine()
    
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
            medical_rec = recommendations['medical']['message']
            medical_urgency = recommendations['medical']['urgency']
            
            # Color-code based on urgency
            if medical_urgency == "critical":
                st.error(medical_rec)
            elif medical_urgency == "high":
                st.warning(medical_rec)
            elif medical_urgency == "medium":
                st.warning(medical_rec)
            else:
                st.info(medical_rec)
            
            # Lifestyle Recommendations
            if recommendations['lifestyle']:
                st.markdown("### 🏃‍♀️ Lifestyle Recommendations")
                for rec in recommendations['lifestyle']:
                    st.success(rec)
            
            # Nutrition Recommendations
            if recommendations['nutrition']:
                st.markdown("### 🍎 Nutrition Recommendations")
                for rec in recommendations['nutrition']:
                    st.success(rec)
            
            # Mental Health Recommendations
            if recommendations['mental']:
                st.markdown("### 🧠 Mental Wellbeing Recommendations")
                for rec in recommendations['mental']:
                    st.success(rec)
            
            # Health insights based on key metrics
            st.markdown("### 📊 Health Insights")
            cols = st.columns(4)
            
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
                glucose_category = st.session_state.engine._categorize_glucose(glucose)
                glucose_status = {
                    'normal': '✅ Normal',
                    'prediabetic': '⚠️ Prediabetic',
                    'diabetic': '🔴 Diabetic Range'
                }
                st.metric("Glucose Status", glucose_status.get(glucose_category, "Unknown"))
                
            with cols[3]:
                cholesterol_category = st.session_state.engine._categorize_cholesterol(cholesterol, hdl, ldl)
                cholesterol_status = {
                    'normal': '✅ Normal',
                    'borderline': '⚠️ Borderline High',
                    'high': '🔴 High'
                }
                st.metric("Cholesterol Status", cholesterol_status.get(cholesterol_category, "Unknown"))

if __name__ == "__main__":
    run_app()
