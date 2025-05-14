import streamlit as st
import pandas as pd
import datetime

# Set page config
st.set_page_config(page_title="Health & Wellness Recommendation Engine", layout="wide")

# Custom CSS to make background white
st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
}
</style>
""", unsafe_allow_html=True)

# Define health category classification functions
def categorizeBP(systolic, diastolic):
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

def categorizeBMI(bmi):
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

def categorizeGlucose(glucose):
    if glucose < 100:
        return 'normal'
    elif glucose < 126:
        return 'prediabetic'
    else:
        return 'diabetic'

def categorizeCholesterol(total, hdl, ldl):
    # Total cholesterol
    risk = 'normal'
    
    if total > 240:
        risk = 'high'
    elif total > 200:
        risk = 'borderline'
    
    # HDL (good cholesterol)
    if hdl < 40:
        risk = risk == 'high' and 'very_high' or 'high'
    
    # LDL (bad cholesterol)
    if ldl > 160:
        risk = risk == 'very_high' and 'very_high' or 'high'
    elif ldl > 130:
        risk = risk == 'normal' and 'borderline' or risk
    
    return risk

def categorizeVitaminD(level):
    if level < 20:
        return 'deficient'
    elif level < 30:
        return 'insufficient'
    else:
        return 'sufficient'

def categorizeThyroid(tsh):
    if tsh < 0.4:
        return 'hyperthyroid'
    elif tsh > 4.5:
        return 'hypothyroid'
    else:
        return 'normal'

def categorizeHbA1c(level):
    if level < 5.7:
        return 'normal'
    elif level < 6.5:
        return 'prediabetic'
    else:
        return 'diabetic'

# Define recommendation rules engine
def getRecommendations(userData):
    recommendations = {
        "medical": {"message": "", "urgency": "normal"},
        "lifestyle": [],
        "nutrition": [],
        "mental": [],
        "tests": [],
        "challenges": [],
        "workout_plan": {},
        "diet_plan": {}
    }
    
    # MEDICAL RECOMMENDATIONS
    
    # HbA1c rules
    if userData.get('hba1c', 0) > 6.5:
        recommendations["medical"]["message"] = "IMPORTANT: Your blood sugar is elevated. Consult a diabetes specialist today."
        recommendations["medical"]["urgency"] = "high"
        recommendations["tests"].append("Schedule a follow-up HbA1c test in 3 months")
    
    # Blood pressure rules
    bp_category = categorizeBP(userData.get('systolicBP', 120), userData.get('diastolicBP', 80))
    high_bp_recorded = bp_category in ['hypertension_1', 'hypertension_2', 'hypertensive_crisis']
    
    if bp_category == 'hypertensive_crisis':
        recommendations["medical"]["message"] = "URGENT: Seek immediate medical attention for your blood pressure."
        recommendations["medical"]["urgency"] = "critical"
    elif bp_category == 'hypertension_2':
        recommendations["medical"]["message"] = "IMPORTANT: Consult your doctor about your Stage 2 Hypertension soon."
        recommendations["medical"]["urgency"] = "high"
    elif bp_category == 'hypertension_1':
        recommendations["medical"]["message"] = "Your BP reading needs attention. Book a lifestyle consult now."
        recommendations["medical"]["urgency"] = "medium"
    
    # LDL rules
    if userData.get('ldl', 0) > 130:
        if recommendations["medical"]["urgency"] not in ["critical", "high"]:
            recommendations["medical"]["message"] = "High cholesterol? Take action early. Book a cardiac screening."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["tests"].append("Schedule a lipid profile test")
    
    # Thyroid rules
    tsh_level = userData.get('tsh', 2.0)
    thyroid_status = categorizeThyroid(tsh_level)
    if thyroid_status != 'normal':
        if recommendations["medical"]["urgency"] not in ["critical", "high"]:
            recommendations["medical"]["message"] = "Thyroid levels are off. Book a follow-up consult."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["tests"].append("Schedule a comprehensive thyroid panel")
    
    # Vitamin D rules
    vitamin_d_level = userData.get('vitaminD', 30)
    vitamin_d_status = categorizeVitaminD(vitamin_d_level)
    if vitamin_d_status == 'deficient':
        if recommendations["medical"]["urgency"] == "normal":
            recommendations["medical"]["message"] = "Low Vitamin D detected. Schedule a nutrition consult."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["nutrition"].append("Consider vitamin D supplementation as advised by your doctor")
    
    # Chronic pain conditions
    if userData.get('chronicBackPain', False):
        if recommendations["medical"]["urgency"] == "normal":
            recommendations["medical"]["message"] = "Chronic pain affects daily life. Consult a physiotherapist."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["lifestyle"].append("Consider gentle stretching exercises for back pain relief")
    
    # PCOS diagnosis
    if userData.get('pcosDiagnosis', False) and userData.get('age', 40) < 35:
        if recommendations["medical"]["urgency"] == "normal":
            recommendations["medical"]["message"] = "Track and manage your PCOS with expert help."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["nutrition"].append("Consider a low-glycemic diet to help manage PCOS symptoms")
    
    # Sleep or migraine issues
    if userData.get('sleepDisorder', False) or userData.get('migraineDiagnosis', False):
        if recommendations["medical"]["urgency"] == "normal":
            recommendations["medical"]["message"] = "Struggling with sleep or migraines? Talk to a neurologist."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["mental"].append("Practice relaxation techniques before bedtime")
    
    # Glucose rules
    glucose_category = categorizeGlucose(userData.get('glucose', 90))
    if glucose_category == 'diabetic':
        if recommendations["medical"]["urgency"] != "critical":
            recommendations["medical"]["message"] = "IMPORTANT: Consult your doctor about your elevated blood glucose levels."
            recommendations["medical"]["urgency"] = "high"
        recommendations["nutrition"].append("Reduce intake of simple carbohydrates and sugars.")
        recommendations["nutrition"].append("Consider a low glycemic index diet.")
    elif glucose_category == 'prediabetic':
        if recommendations["medical"]["urgency"] == "normal":
            recommendations["medical"]["message"] = "Schedule a follow-up for your prediabetic glucose levels."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["nutrition"].append("Limit refined carbohydrates and sugary beverages.")
    
    # BMI rules
    bmi_category = categorizeBMI(userData.get('bmi', 24.5))
    if bmi_category == 'extremely_obese':
        if recommendations["medical"]["urgency"] not in ["critical", "high"]:
            recommendations["medical"]["message"] = "IMPORTANT: Consider speaking with your doctor about weight management options."
            recommendations["medical"]["urgency"] = "high"
        recommendations["lifestyle"].append("Start with gentle, low-impact exercises like swimming or walking.")
    elif bmi_category == 'obese':
        if recommendations["medical"]["urgency"] == "normal":
            recommendations["medical"]["message"] = "Schedule a check-up to discuss weight management strategies."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["lifestyle"].append("Aim for 150 minutes of moderate exercise per week.")
    elif bmi_category == 'overweight':
        recommendations["lifestyle"].append("Incorporate more physical activity into your daily routine.")
    elif bmi_category == 'underweight':
        if recommendations["medical"]["urgency"] == "normal":
            recommendations["medical"]["message"] = "Consider discussing your weight with a healthcare provider."
            recommendations["medical"]["urgency"] = "low"
        recommendations["nutrition"].append("Focus on nutrient-dense foods to reach a healthy weight.")
    
    # BMI > 30 challenge
    if userData.get('bmi', 0) > 30:
        recommendations["challenges"].append("Let's take a step toward better weight management. Join our 4-week weight management program.")
    
    # Cholesterol rules
    cholesterol_category = categorizeCholesterol(
        userData.get('cholesterol', 180), 
        userData.get('hdl', 50), 
        userData.get('ldl', 100)
    )
    if cholesterol_category in ['very_high', 'high']:
        if recommendations["medical"]["urgency"] not in ["critical", "high"]:
            recommendations["medical"]["message"] = "IMPORTANT: Consult with your doctor about your cholesterol levels."
            recommendations["medical"]["urgency"] = "high"
        recommendations["nutrition"].append("Reduce intake of saturated and trans fats.")
        recommendations["nutrition"].append("Increase consumption of omega-3 fatty acids and fiber.")
    elif cholesterol_category == 'borderline':
        recommendations["nutrition"].append("Monitor fat intake and consider adding more heart-healthy foods.")
    
    # Smoking rules
    if userData.get('smokingStatus', 'Never') == 'Regular':
        if recommendations["medical"]["urgency"] == "normal":
            recommendations["medical"]["message"] = "Consider speaking with your doctor about smoking cessation options."
            recommendations["medical"]["urgency"] = "medium"
        recommendations["lifestyle"].append("Consider a smoking cessation program or nicotine replacement therapy.")
        recommendations["tests"].append("Smoking impacts your lungs. Book a preventive lung screening.")
    elif userData.get('smokingStatus', 'Never') == 'Occasional':
        recommendations["lifestyle"].append("Work towards completely quitting smoking.")
    
    # Exercise rules
    if userData.get('exerciseDays', 3) < 2:
        recommendations["lifestyle"].append("Start with at least 2 days of moderate exercise per week.")
    elif userData.get('exerciseDays', 3) < 4:
        recommendations["lifestyle"].append("Try to increase exercise frequency to 4-5 days per week.")
    
    # Sleep rules
    if userData.get('sleepHours', 7) < 6:
        recommendations["lifestyle"].append("Work on improving sleep duration to at least 7 hours per night.")
        recommendations["mental"].append("Consider a bedtime routine to improve sleep quality.")
    elif userData.get('sleepHours', 7) > 9:
        recommendations["lifestyle"].append("Excessive sleep may indicate other issues - try to maintain 7-9 hours.")
    
    # Diet rules
    if userData.get('dietScore', 6) < 5:
        recommendations["nutrition"].append("Focus on incorporating more whole foods and vegetables.")
    
    # Stress rules
    if userData.get('stressLevel', 5) > 7:
        recommendations["mental"].append("Consider stress reduction techniques like meditation or mindfulness.")
        recommendations["mental"].append("Schedule regular breaks during work hours.")
    
    # Age-based screening rules
    age = userData.get('age', 35)
    last_checkup_days = userData.get('lastCheckupDays', 0)
    
    if age > 40 and last_checkup_days > 365:
        recommendations["tests"].append("It's time for your annual health check.")
    
    if age > 50 and not userData.get('hadECG', False):
        recommendations["tests"].append("Heart health matters. Book an ECG today.")
    
    if age > 30 and not userData.get('hadLipidProfile', False):
        recommendations["tests"].append("Know your cholesterol levels. Schedule a test today.")
    
    if userData.get('gender', 'Male') == 'Female' and age > 40 and not userData.get('hadMammogram', False):
        recommendations["tests"].append("Book your preventive breast screening today.")
    
    if userData.get('gender', 'Male') == 'Male' and age > 45 and not userData.get('hadPSA', False):
        recommendations["tests"].append("Men's health matters. Book a preventive prostate screen.")
    
    # Activity tracking challenges
    if userData.get('dailySteps', 0) > 5000 and userData.get('stepStreakDays', 0) >= 5:
        recommendations["challenges"].append("You've built a great habit! Join this week's step challenge to reach 8,000 steps.")
    
    if userData.get('inactiveDays', 0) >= 7:
        recommendations["challenges"].append("Let's get back on track. Walk 2,000 steps today.")
    
    if userData.get('hydrationStreakBroken', False):
        recommendations["challenges"].append("Hydration helps everything. Restart your water goal today.")
    
    if userData.get('completedLastChallenge', False):
        recommendations["challenges"].append("Nice job! Ready for a new fitness challenge?")
    
    if userData.get('inactiveDays', 0) > 3 and userData.get('bmi', 0) > 27:
        recommendations["challenges"].append("Time to get active! Try a low-impact step challenge.")
    
    # Personalize based on team/company
    if userData.get('team', '') == "Engineering":
        recommendations["mental"].append("Take regular breaks from screen time to reduce eye strain.")
    elif userData.get('team', '') == "Sales":
        recommendations["mental"].append("Practice stress management techniques for high-pressure situations.")
    
    if userData.get('company', '') == "TechCorp":
        recommendations["lifestyle"].append("Take advantage of your company's standing desk options.")
    elif userData.get('company', '') == "HealthInc":
        recommendations["lifestyle"].append("Utilize your employee wellness program benefits.")
    
    # If no medical recommendations have been made
    if recommendations["medical"]["message"] == "":
        recommendations["medical"]["message"] = "Continue routine health check-ups as recommended for your age and risk factors."
    
    # Add workout plan based on profile
    workout_plan = create_workout_plan(userData)
    recommendations["workout_plan"] = workout_plan
    
    # Add diet plan based on profile
    diet_plan = create_diet_plan(userData)
    recommendations["diet_plan"] = diet_plan
    
    return recommendations

def create_workout_plan(userData):
    """Create a personalized workout plan based on user data"""
    bmi_category = categorizeBMI(userData.get('bmi', 24.5))
    age = userData.get('age', 35)
    health_conditions = []
    
    if categorizeBP(userData.get('systolicBP', 120), userData.get('diastolicBP', 80)) != 'normal':
        health_conditions.append('hypertension')
    
    if categorizeGlucose(userData.get('glucose', 90)) != 'normal':
        health_conditions.append('blood_sugar')
    
    if userData.get('chronicBackPain', False):
        health_conditions.append('back_pain')
    
    # Base workout structure
    workout_plan = {
        "weekly_structure": [],
        "intensity": "",
        "duration_per_session": "",
        "special_considerations": [],
        "warmup": [],
        "cooldown": []
    }
    
    # Assign basic plan based on BMI and age
    if bmi_category in ['obese', 'extremely_obese']:
        workout_plan["intensity"] = "Low to moderate"
        workout_plan["duration_per_session"] = "15-30 minutes, gradually increasing"
        workout_plan["weekly_structure"] = [
            "Day 1: 15-20 minute walk", 
            "Day 2: Rest or gentle stretching",
            "Day 3: 15-20 minute water exercises or recumbent bike",
            "Day 4: Rest or gentle stretching",
            "Day 5: 15-20 minute walk",
            "Days 6-7: Rest or light activity of choice"
        ]
    elif bmi_category == 'overweight':
        workout_plan["intensity"] = "Moderate"
        workout_plan["duration_per_session"] = "30-45 minutes"
        workout_plan["weekly_structure"] = [
            "Day 1: 30 minute brisk walk or light jog",
            "Day 2: 20-30 minutes strength training (major muscle groups)",
            "Day 3: 30 minute cycling or swimming",
            "Day 4: Rest or yoga",
            "Day 5: 30 minute brisk walk or light jog",
            "Day 6: 20-30 minutes strength training (different muscle groups)",
            "Day 7: Rest or light activity"
        ]
    else:  # normal or underweight
        workout_plan["intensity"] = "Moderate to high (as tolerated)"
        workout_plan["duration_per_session"] = "45-60 minutes"
        workout_plan["weekly_structure"] = [
            "Day 1: 30-45 minute cardio (run, bike, swim)",
            "Day 2: 45 minute strength training (upper body)",
            "Day 3: 30 minute high-intensity interval training",
            "Day 4: 45 minute strength training (lower body)",
            "Day 5: 30-45 minute cardio (different than Day 1)",
            "Day 6: Active recovery (yoga, light walking)",
            "Day 7: Rest"
        ]
    
    # Age adjustments
    if age > 60:
        workout_plan["intensity"] = "Low to moderate"
        workout_plan["special_considerations"].append("Focus on balance exercises to prevent falls")
        workout_plan["special_considerations"].append("Include strength training 2-3 times weekly to preserve muscle mass")
    elif age > 40:
        workout_plan["special_considerations"].append("Allow more recovery time between intense workouts")
        workout_plan["special_considerations"].append("Include flexibility work to maintain joint health")
    
    # Health condition adjustments
    if 'hypertension' in health_conditions:
        workout_plan["special_considerations"].append("Monitor blood pressure before and after exercise")
        workout_plan["special_considerations"].append("Avoid breath holding during strength exercises")
    
    if 'blood_sugar' in health_conditions:
        workout_plan["special_considerations"].append("Check blood sugar before and after exercise")
        workout_plan["special_considerations"].append("Have a carbohydrate snack available during longer workouts")
    
    if 'back_pain' in health_conditions:
        workout_plan["special_considerations"].append("Avoid high-impact activities")
        workout_plan["special_considerations"].append("Focus on core strengthening exercises")
        workout_plan["warmup"].append("Extra focus on gentle back mobilization exercises")
    
    # Standard warmup and cooldown
    if not workout_plan["warmup"]:
        workout_plan["warmup"] = [
            "5 minutes light cardio to raise heart rate",
            "Dynamic stretching for major muscle groups",
            "Gradual increase in intensity"
        ]
    
    workout_plan["cooldown"] = [
        "5 minutes of gradually decreasing activity",
        "Static stretches for worked muscle groups",
        "Deep breathing to normalize heart rate"
    ]
    
    return workout_plan

def create_diet_plan(userData):
    """Create a personalized diet plan based on user data"""
    bmi_category = categorizeBMI(userData.get('bmi', 24.5))
    glucose_category = categorizeGlucose(userData.get('glucose', 90))
    cholesterol_category = categorizeCholesterol(userData.get('cholesterol', 180), userData.get('hdl', 50), userData.get('ldl', 100))
    
    # Base diet structure
    diet_plan = {
        "approach": "",
        "calories": "",
        "macronutrient_ratio": "",
        "meal_structure": [],
        "foods_to_emphasize": [],
        "foods_to_limit": [],
        "special_considerations": []
    }
    
    # Assign basic plan based on BMI
    if bmi_category in ['obese', 'extremely_obese']:
        diet_plan["approach"] = "Calorie-controlled with focus on nutrient density"
        diet_plan["calories"] = "Moderate caloric deficit (consult healthcare provider for specific targets)"
        diet_plan["macronutrient_ratio"] = "Protein: 30%, Carbs: 40%, Fat: 30%"
        diet_plan["foods_to_emphasize"] = [
            "Lean proteins (chicken, fish, tofu, legumes)",
            "Non-starchy vegetables (greens, broccoli, cauliflower)",
            "High-fiber foods (beans, whole grains)",
            "Healthy fats in moderation (avocado, nuts, olive oil)"
        ]
        diet_plan["foods_to_limit"] = [
            "Refined carbohydrates (white bread, pastries)",
            "Added sugars and sweetened beverages",
            "Processed meats and fried foods",
            "High-calorie condiments and sauces"
        ]
    elif bmi_category == 'overweight':
        diet_plan["approach"] = "Portion control with balanced nutrition"
        diet_plan["calories"] = "Slight caloric deficit"
        diet_plan["macronutrient_ratio"] = "Protein: 25%, Carbs: 45%, Fat: 30%"
        diet_plan["foods_to_emphasize"] = [
            "Lean proteins",
            "Whole grains and complex carbohydrates",
            "Variety of vegetables and fruits",
            "Healthy fats in moderation"
        ]
        diet_plan["foods_to_limit"] = [
            "Processed foods high in added sugars",
            "Excessive alcohol consumption",
            "Large portion sizes"
        ]
    elif bmi_category == 'underweight':
        diet_plan["approach"] = "Calorie-dense, nutrient-rich foods"
        diet_plan["calories"] = "Caloric surplus to support healthy weight gain"
        diet_plan["macronutrient_ratio"] = "Protein: 20%, Carbs: 50%, Fat: 30%"
        diet_plan["foods_to_emphasize"] = [
            "Nutrient-dense proteins (eggs, dairy, lean meats)",
            "Healthy fats (nut butters, avocados, olive oil)",
            "Whole grain carbohydrates",
            "Smoothies with added nutrients"
        ]
    else:  # normal weight
        diet_plan["approach"] = "Balanced nutrition for maintenance"
        diet_plan["calories"] = "Maintenance calories"
        diet_plan["macronutrient_ratio"] = "Protein: 20-25%, Carbs: 45-50%, Fat: 25-30%"
        diet_plan["foods_to_emphasize"] = [
            "Variety of whole foods",
            "Colorful fruits and vegetables",
            "Lean proteins and plant proteins",
            "Whole grains and healthy fats"
        ]
        diet_plan["foods_to_limit"] = [
            "Ultra-processed foods",
            "Foods with added sugars"
        ]
    
    # Standard meal structure
    diet_plan["meal_structure"] = [
        "Breakfast: Protein + complex carb + fruit/vegetable",
        "Lunch: Protein + vegetables + whole grain/starch",
        "Dinner: Protein + vegetables + small amount of healthy fats",
        "Snacks: 1-2 daily, focusing on protein and fiber"
    ]
    
    # Health condition adjustments
    if glucose_category in ['prediabetic', 'diabetic']:
        diet_plan["special_considerations"].append("Monitor carbohydrate intake and focus on low glycemic index foods")
        diet_plan["special_considerations"].append("Space carbohydrates evenly throughout the day")
        diet_plan["foods_to_emphasize"].append("Non-starchy vegetables and high-fiber foods")
        diet_plan["foods_to_limit"].append("Simple sugars, white bread, and sugary beverages")
    
    if cholesterol_category in ['high', 'very_high', 'borderline']:
        diet_plan["special_considerations"].append("Limit saturated and trans fats")
        diet_plan["special_considerations"].append("Increase soluble fiber intake")
        diet_plan["foods_to_emphasize"].append("Oats, barley, beans, and fatty fish")
        diet_plan["foods_to_limit"].append("Full-fat dairy, fatty cuts of meat, and coconut oil")
    
    if userData.get('pcosDiagnosis', False):
        diet_plan["special_considerations"].append("Consider a low-glycemic, anti-inflammatory diet approach")
        diet_plan["foods_to_emphasize"].append("Anti-inflammatory foods (berries, fatty fish, turmeric)")
    
    if userData.get('vitaminD', 30) < 20:
        diet_plan["special_considerations"].append("Include vitamin D rich foods or discuss supplementation with doctor")
        diet_plan["foods_to_emphasize"].append("Fatty fish, fortified dairy, egg yolks, and safe sun exposure")
    
    return diet_plan

# Status mappings for UI display
bpStatusMap = {
    'normal': '✅ Normal',
    'elevated': '⚠️ Elevated',
    'hypertension_1': '🔴 Stage 1 Hypertension',
    'hypertension_2': '🔴 Stage 2 Hypertension',
    'hypertensive_crisis': '⛔ Hypertensive Crisis'
}

bmiStatusMap = {
    'underweight': '⚠️ Underweight',
    'normal': '✅ Normal',
    'overweight': '⚠️ Overweight',
    'obese': '🔴 Obese',
    'extremely_obese': '🔴 Extremely Obese'
}

glucoseStatusMap = {
    'normal': '✅ Normal',
    'prediabetic': '⚠️ Prediabetic',
    'diabetic': '🔴 Diabetic Range'
}

cholesterolStatusMap = {
    'normal': '✅ Normal',
    'borderline': '⚠️ Borderline High',
    'high': '🔴 High',
    'very_high': '⛔ Very High'
}

hba1cStatusMap = {
    'normal': '✅ Normal',
    'prediabetic': '⚠️ Prediabetic',
    'diabetic': '🔴 Diabetic'
}

thyroidStatusMap = {
    'normal': '✅ Normal',
    'hypothyroid': '🔴 Hypothyroid',
    'hyperthyroid': '🔴 Hyperthyroid'
}

vitaminDStatusMap = {
    'deficient': '🔴 Deficient',
    'insufficient': '⚠️ Insufficient',
    'sufficient': '✅ Sufficient'
}

# App title
st.title("❤️ Health & Wellness Recommendation Engine")
st.subheader("Enter your health profile to get personalized recommendations")

# Create tabs for input sections
tab1, tab2, tab3, tab4 = st.tabs(["Demographics & Lifestyle", "Medical Metrics", "Lab Results", "Health Records"])

# Initialize session state for user data if it doesn't exist
if 'user_data' not in st.session_state:
    st.session_state.user_data = {
        'age': 35,
        'gender': "Male",
        'company': "TechCorp",
        'team': "Engineering",
        'smokingStatus': "Never",
        'sleepHours': 7.0,
        'dietScore': 6,
        'exerciseDays': 3,
        'stressLevel': 5,
        'systolicBP': 120,
        'diastolicBP': 80,
        'bmi': 24.5,
        'cholesterol': 180,
        'hdl': 50,
        'ldl': 100,
        'glucose': 90,
        'hba1c': 5.5,
        'vitaminD': 30,
        'tsh': 2.0,
        'chronicBackPain': False,
        'pcosDiagnosis': False,
        'sleepDisorder': False,
        'migraineDiagnosis': False,
        'lastCheckupDays': 180,
        'hadECG': True,
        'hadLipidProfile': True,
        'hadMammogram': False,
        'hadPSA': False,
        'dailySteps': 4000,
        'stepStreakDays': 3,
        'inactiveDays': 0,
        'hydrationStreakBroken': False,
        'completedLastChallenge': False
    }

# Store recommendations in session state
if 'recommendations' not in st.session_state:
    st.session_state.recommendations = None

# Function to handle changes in user data
def update_user_data(field, value):
    st.session_state.user_data[field] = value

# Tab 1: Demographics & Lifestyle
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Demographics")
        
        st.session_state.user_data['age'] = st.slider("Age", 18, 80, st.session_state.user_data['age'])
        
        st.session_state.user_data['gender'] = st.selectbox(
            "Gender",
            options=["Male", "Female", "Other"],
            index=["Male", "Female", "Other"].index(st.session_state.user_data['gender'])
        )
        
        st.session_state.user_data['company'] = st.selectbox(
            "Company",
            options=["TechCorp", "HealthInc", "FinanceOne", "RetailPro"],
            index=["TechCorp", "HealthInc", "FinanceOne", "RetailPro"].index(st.session_state.user_data['company'])
        )
        
        st.session_state.user_data['team'] = st.selectbox(
            "Team",
            options=["Engineering", "Sales", "HR", "Marketing", "Operations"],
            index=["Engineering", "Sales", "HR", "Marketing", "Operations"].index(st.session_state.user_data['team'])
        )
    
    with col2:
        st.subheader("Lifestyle Factors")
        
        st.session_state.user_data['smokingStatus'] = st.selectbox(
            "Smoking Status",
            options=["Never", "Former", "Occasional", "Regular"],
            index=["Never", "Former", "Occasional", "Regular"].index(st.session_state.user_data['smokingStatus'])
        )
        
        st.session_state.user_data['sleepHours'] = st.slider(
            "Average Sleep (hours)",
            4.0, 12.0, st.session_state.user_data['sleepHours'], 0.5
        )
        
        st.session_state.user_data['dietScore'] = st.slider(
            "Diet Quality (1-10)",
            1, 10, st.session_state.user_data['dietScore']
        )
        
        st.session_state.user_data['exerciseDays'] = st.slider(
            "Exercise Days per Week",
            0, 7, st.session_state.user_data['exerciseDays']
        )
        
        st.session_state.user_data['stressLevel'] = st.slider(
            "Stress Level (1-10)",
            1, 10, st.session_state.user_data['stressLevel']
        )
        
        st.session_state.user_data['dailySteps'] = st.number_input(
            "Average Daily Steps",
            min_value=0, max_value=20000, value=st.session_state.user_data['dailySteps']
        )

# Tab 2: Medical Metrics
with tab2:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Blood Pressure")
        
        col_bp1, col_bp2 = st.columns(2)
        with col_bp1:
            st.session_state.user_data['systolicBP'] = st.number_input(
                "Systolic (mm Hg)",
                90, 200, st.session_state.user_data['systolicBP']
            )
        with col_bp2:
            st.session_state.user_data['diastolicBP'] = st.number_input(
                "Diastolic (mm Hg)",
                60, 130, st.session_state.user_data['diastolicBP']
            )
        
        bp_status = categorizeBP(st.session_state.user_data['systolicBP'], st.session_state.user_data['diastolicBP'])
        st.markdown(f"**Status:** {bpStatusMap[bp_status]}")
        
        st.subheader("BMI")
        st.session_state.user_data['bmi'] = st.number_input(
            "BMI Value",
            15.0, 50.0, st.session_state.user_data['bmi'], 0.1
        )
        
        bmi_status = categorizeBMI(st.session_state.user_data['bmi'])
        st.markdown(f"**Status:** {bmiStatusMap[bmi_status]}")
    
    with col2:
        st.subheader("Cholesterol Profile")
        
        st.session_state.user_data['cholesterol'] = st.number_input(
            "Total Cholesterol (mg/dL)",
            100, 350, st.session_state.user_data['cholesterol']
        )
        
        st.session_state.user_data['hdl'] = st.number_input(
            "HDL (mg/dL)",
            20, 100, st.session_state.user_data['hdl']
        )
        
        st.session_state.user_data['ldl'] = st.number_input(
            "LDL (mg/dL)",
            40, 250, st.session_state.user_data['ldl']
        )
        
        cholesterol_status = categorizeCholesterol(
            st.session_state.user_data['cholesterol'],
            st.session_state.user_data['hdl'],
            st.session_state.user_data['ldl']
        )
        st.markdown(f"**Status:** {cholesterolStatusMap[cholesterol_status]}")
        
        st.subheader("Blood Glucose")
        st.session_state.user_data['glucose'] = st.number_input(
            "Fasting Glucose (mg/dL)",
            70, 250, st.session_state.user_data['glucose']
        )
        
        glucose_status = categorizeGlucose(st.session_state.user_data['glucose'])
        st.markdown(f"**Status:** {glucoseStatusMap[glucose_status]}")

# Tab 3: Lab Results
with tab3:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Blood Tests")
        
        st.session_state.user_data['hba1c'] = st.number_input(
            "HbA1c (%)",
            4.0, 14.0, st.session_state.user_data['hba1c'], 0.1
        )
        
        hba1c_status = categorizeHbA1c(st.session_state.user_data['hba1c'])
        st.markdown(f"**Status:** {hba1cStatusMap[hba1c_status]}")
        
        st.session_state.user_data['vitaminD'] = st.number_input(
            "Vitamin D (ng/mL)",
            5, 80, st.session_state.user_data['vitaminD']
        )
        
        vitamin_d_status = categorizeVitaminD(st.session_state.user_data['vitaminD'])
        st.markdown(f"**Status:** {vitaminDStatusMap[vitamin_d_status]}")
    
    with col2:
        st.subheader("Thyroid Function")
        
        st.session_state.user_data['tsh'] = st.number_input(
            "TSH (mIU/L)",
            0.1, 10.0, st.session_state.user_data['tsh'], 0.1
        )
        
        thyroid_status = categorizeThyroid(st.session_state.user_data['tsh'])
        st.markdown(f"**Status:** {thyroidStatusMap[thyroid_status]}")

# Tab 4: Health Records
with tab4:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Medical Conditions")
        
        st.session_state.user_data['chronicBackPain'] = st.checkbox(
            "Chronic Back Pain",
            st.session_state.user_data['chronicBackPain']
        )
        
        st.session_state.user_data['pcosDiagnosis'] = st.checkbox(
            "PCOS Diagnosis",
            st.session_state.user_data['pcosDiagnosis']
        )
        
        st.session_state.user_data['sleepDisorder'] = st.checkbox(
            "Sleep Disorder",
            st.session_state.user_data['sleepDisorder']
        )
        
        st.session_state.user_data['migraineDiagnosis'] = st.checkbox(
            "Migraine Diagnosis",
            st.session_state.user_data['migraineDiagnosis']
        )
    
    with col2:
        st.subheader("Preventive Screenings")
        
        st.session_state.user_data['lastCheckupDays'] = st.slider(
            "Days Since Last Checkup",
            0, 730, st.session_state.user_data['lastCheckupDays']
        )
        
        st.session_state.user_data['hadECG'] = st.checkbox(
            "Had ECG in past year",
            st.session_state.user_data['hadECG']
        )
        
        st.session_state.user_data['hadLipidProfile'] = st.checkbox(
            "Had Lipid Profile in past year",
            st.session_state.user_data['hadLipidProfile']
        )
        
        st.session_state.user_data['hadMammogram'] = st.checkbox(
            "Had Mammogram in past year",
            st.session_state.user_data['hadMammogram']
        )
        
        st.session_state.user_data['hadPSA'] = st.checkbox(
            "Had PSA Test in past year",
            st.session_state.user_data['hadPSA']
        )

# Hidden data for testing and demonstration
with st.expander("Activity Tracking Data (Demo)", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.user_data['stepStreakDays'] = st.slider(
            "Step Goal Streak (days)",
            0, 30, st.session_state.user_data['stepStreakDays']
        )
        
        st.session_state.user_data['inactiveDays'] = st.slider(
            "Consecutive Inactive Days",
            0, 30, st.session_state.user_data['inactiveDays']
        )
    
    with col2:
        st.session_state.user_data['hydrationStreakBroken'] = st.checkbox(
            "Hydration Streak Broken This Week",
            st.session_state.user_data['hydrationStreakBroken']
        )
        
        st.session_state.user_data['completedLastChallenge'] = st.checkbox(
            "Completed Last Challenge",
            st.session_state.user_data['completedLastChallenge']
        )

# Generate recommendations button
if st.button("Generate Recommendations", type="primary", use_container_width=True):
    with st.spinner("Analyzing your health profile..."):
        # Simulate processing time
        time.sleep(1)
        st.session_state.recommendations = getRecommendations(st.session_state.user_data)
    st.success("Analysis complete!")

# Display recommendations if available
if st.session_state.recommendations:
    st.markdown("---")
    st.header("Your Personalized Health Recommendations")
    
    # Medical recommendation with color coding based on urgency
    medical_rec = st.session_state.recommendations["medical"]
    
    if medical_rec["urgency"] == "critical":
        st.error("🚨 **MEDICAL ALERT:** " + medical_rec["message"])
    elif medical_rec["urgency"] == "high":
        st.warning("⚠️ **IMPORTANT:** " + medical_rec["message"])
    elif medical_rec["urgency"] == "medium":
        st.info("ℹ️ **MEDICAL ADVICE:** " + medical_rec["message"])
    else:
        st.success("✅ **MEDICAL STATUS:** " + medical_rec["message"])
    
    # Create tabs for different recommendation categories
    rec_tab1, rec_tab2, rec_tab3, rec_tab4, rec_tab5 = st.tabs([
        "Lifestyle & Nutrition", 
        "Mental Wellbeing", 
        "Recommended Tests", 
        "Wellness Challenges",
        "Personal Plans"
    ])
    
    # Tab 1: Lifestyle & Nutrition
    with rec_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Lifestyle Recommendations")
            if st.session_state.recommendations["lifestyle"]:
                for item in st.session_state.recommendations["lifestyle"]:
                    st.markdown(f"• {item}")
            else:
                st.markdown("No specific lifestyle recommendations at this time.")
        
        with col2:
            st.subheader("Nutrition Recommendations")
            if st.session_state.recommendations["nutrition"]:
                for item in st.session_state.recommendations["nutrition"]:
                    st.markdown(f"• {item}")
            else:
                st.markdown("No specific nutrition recommendations at this time.")
    
    # Tab 2: Mental Wellbeing
    with rec_tab2:
        st.subheader("Mental Wellbeing Recommendations")
        if st.session_state.recommendations["mental"]:
            for item in st.session_state.recommendations["mental"]:
                st.markdown(f"• {item}")
        else:
            st.markdown("No specific mental wellbeing recommendations at this time.")
    
    # Tab 3: Recommended Tests
    with rec_tab3:
        st.subheader("Recommended Health Tests")
        if st.session_state.recommendations["tests"]:
            for item in st.session_state.recommendations["tests"]:
                st.markdown(f"• {item}")
        else:
            st.markdown("No specific test recommendations at this time.")
    
    # Tab 4: Wellness Challenges
    with rec_tab4:
        st.subheader("Wellness Challenges For You")
        if st.session_state.recommendations["challenges"]:
            for i, challenge in enumerate(st.session_state.recommendations["challenges"]):
                st.info(f"**Challenge {i+1}:** {challenge}")
        else:
            st.markdown("No specific wellness challenges at this time.")
    
    # Tab 5: Personal Plans
    with rec_tab5:
        workout_col, diet_col = st.columns(2)
        
        with workout_col:
            st.subheader("Your Personalized Workout Plan")
            workout_plan = st.session_state.recommendations["workout_plan"]
            
            st.markdown(f"**Intensity:** {workout_plan['intensity']}")
            st.markdown(f"**Duration:** {workout_plan['duration_per_session']}")
            
            st.markdown("**Weekly Structure:**")
            for day in workout_plan["weekly_structure"]:
                st.markdown(f"• {day}")
            
            with st.expander("Warmup Routine"):
                for item in workout_plan["warmup"]:
                    st.markdown(f"• {item}")
            
            with st.expander("Cooldown Routine"):
                for item in workout_plan["cooldown"]:
                    st.markdown(f"• {item}")
            
            if workout_plan["special_considerations"]:
                with st.expander("Special Considerations"):
                    for item in workout_plan["special_considerations"]:
                        st.markdown(f"• {item}")
        
        with diet_col:
            st.subheader("Your Personalized Diet Plan")
            diet_plan = st.session_state.recommendations["diet_plan"]
            
            st.markdown(f"**Approach:** {diet_plan['approach']}")
            st.markdown(f"**Calories:** {diet_plan['calories']}")
            st.markdown(f"**Macro Ratio:** {diet_plan['macronutrient_ratio']}")
            
            st.markdown("**Meal Structure:**")
            for meal in diet_plan["meal_structure"]:
                st.markdown(f"• {meal}")
            
            with st.expander("Foods to Emphasize"):
                for item in diet_plan["foods_to_emphasize"]:
                    st.markdown(f"• {item}")
            
            with st.expander("Foods to Limit"):
                for item in diet_plan["foods_to_limit"]:
                    st.markdown(f"• {item}")
            
            if diet_plan["special_considerations"]:
                with st.expander("Special Dietary Considerations"):
                    for item in diet_plan["special_considerations"]:
                        st.markdown(f"• {item}")

# Add footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.8em;">
    Health & Wellness Recommendation Engine | Disclaimer: This tool provides general wellness recommendations 
    and should not replace professional medical advice.
    </div>
    """, 
    unsafe_allow_html=True
)
