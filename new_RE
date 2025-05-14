import { useState } from "react";

// Define health category classification functions
const categorizeBP = (systolic, diastolic) => {
  if (systolic < 120 && diastolic < 80) return 'normal';
  else if (systolic < 130 && diastolic < 80) return 'elevated';
  else if (systolic < 140 && diastolic < 90) return 'hypertension_1';
  else if (systolic < 180 && diastolic < 120) return 'hypertension_2';
  else return 'hypertensive_crisis';
};

const categorizeBMI = (bmi) => {
  if (bmi < 18.5) return 'underweight';
  else if (bmi < 24.9) return 'normal';
  else if (bmi < 29.9) return 'overweight';
  else if (bmi < 40) return 'obese';
  else return 'extremely_obese';
};

const categorizeGlucose = (glucose) => {
  if (glucose < 100) return 'normal';
  else if (glucose < 126) return 'prediabetic';
  else return 'diabetic';
};

const categorizeCholesterol = (total, hdl, ldl) => {
  // Total cholesterol
  let risk = 'normal';
  
  if (total > 240) risk = 'high';
  else if (total > 200) risk = 'borderline';
  
  // HDL (good cholesterol)
  if (hdl < 40) risk = risk === 'high' ? 'very_high' : 'high';
  
  // LDL (bad cholesterol)
  if (ldl > 160) risk = risk === 'very_high' ? 'very_high' : 'high';
  else if (ldl > 130) risk = risk === 'normal' ? 'borderline' : risk;
  
  return risk;
};

// Define recommendation rules engine
const getRecommendations = (userData) => {
  const recommendations = {
    medical: { message: "", urgency: "normal" },
    lifestyle: [],
    nutrition: [],
    mental: []
  };
  
  // MEDICAL RECOMMENDATIONS
  
  // Blood pressure rules
  const bpCategory = categorizeBP(userData.systolicBP, userData.diastolicBP);
  if (bpCategory === 'hypertensive_crisis') {
    recommendations.medical.message = "URGENT: Seek immediate medical attention for your blood pressure.";
    recommendations.medical.urgency = "critical";
  } else if (bpCategory === 'hypertension_2') {
    recommendations.medical.message = "IMPORTANT: Consult your doctor about your Stage 2 Hypertension soon.";
    recommendations.medical.urgency = "high";
  } else if (bpCategory === 'hypertension_1') {
    recommendations.medical.message = "Schedule a check-up to discuss your Stage 1 Hypertension.";
    recommendations.medical.urgency = "medium";
  }
  
  // Glucose rules
  const glucoseCategory = categorizeGlucose(userData.glucose);
  if (glucoseCategory === 'diabetic') {
    if (recommendations.medical.urgency !== "critical") {
      recommendations.medical.message = "IMPORTANT: Consult your doctor about your elevated blood glucose levels.";
      recommendations.medical.urgency = "high";
    }
    recommendations.nutrition.push("Reduce intake of simple carbohydrates and sugars.");
    recommendations.nutrition.push("Consider a low glycemic index diet.");
  } else if (glucoseCategory === 'prediabetic') {
    if (recommendations.medical.urgency === "normal") {
      recommendations.medical.message = "Schedule a follow-up for your prediabetic glucose levels.";
      recommendations.medical.urgency = "medium";
    }
    recommendations.nutrition.push("Limit refined carbohydrates and sugary beverages.");
  }
  
  // BMI rules
  const bmiCategory = categorizeBMI(userData.bmi);
  if (bmiCategory === 'extremely_obese') {
    if (recommendations.medical.urgency !== "critical" && recommendations.medical.urgency !== "high") {
      recommendations.medical.message = "IMPORTANT: Consider speaking with your doctor about weight management options.";
      recommendations.medical.urgency = "high";
    }
    recommendations.lifestyle.push("Start with gentle, low-impact exercises like swimming or walking.");
  } else if (bmiCategory === 'obese') {
    if (recommendations.medical.urgency === "normal") {
      recommendations.medical.message = "Schedule a check-up to discuss weight management strategies.";
      recommendations.medical.urgency = "medium";
    }
    recommendations.lifestyle.push("Aim for 150 minutes of moderate exercise per week.");
  } else if (bmiCategory === 'overweight') {
    recommendations.lifestyle.push("Incorporate more physical activity into your daily routine.");
  } else if (bmiCategory === 'underweight') {
    if (recommendations.medical.urgency === "normal") {
      recommendations.medical.message = "Consider discussing your weight with a healthcare provider.";
      recommendations.medical.urgency = "low";
    }
    recommendations.nutrition.push("Focus on nutrient-dense foods to reach a healthy weight.");
  }
  
  // Cholesterol rules
  const cholesterolCategory = categorizeCholesterol(userData.cholesterol, userData.hdl, userData.ldl);
  if (cholesterolCategory === 'very_high' || cholesterolCategory === 'high') {
    if (recommendations.medical.urgency !== "critical" && recommendations.medical.urgency !== "high") {
      recommendations.medical.message = "IMPORTANT: Consult with your doctor about your cholesterol levels.";
      recommendations.medical.urgency = "high";
    }
    recommendations.nutrition.push("Reduce intake of saturated and trans fats.");
    recommendations.nutrition.push("Increase consumption of omega-3 fatty acids and fiber.");
  } else if (cholesterolCategory === 'borderline') {
    recommendations.nutrition.push("Monitor fat intake and consider adding more heart-healthy foods.");
  }
  
  // Smoking rules
  if (userData.smokingStatus === 'Regular') {
    if (recommendations.medical.urgency === "normal") {
      recommendations.medical.message = "Consider speaking with your doctor about smoking cessation options.";
      recommendations.medical.urgency = "medium";
    }
    recommendations.lifestyle.push("Consider a smoking cessation program or nicotine replacement therapy.");
  } else if (userData.smokingStatus === 'Occasional') {
    recommendations.lifestyle.push("Work towards completely quitting smoking.");
  }
  
  // Exercise rules
  if (userData.exerciseDays < 2) {
    recommendations.lifestyle.push("Start with at least 2 days of moderate exercise per week.");
  } else if (userData.exerciseDays < 4) {
    recommendations.lifestyle.push("Try to increase exercise frequency to 4-5 days per week.");
  }
  
  // Sleep rules
  if (userData.sleepHours < 6) {
    recommendations.lifestyle.push("Work on improving sleep duration to at least 7 hours per night.");
    recommendations.mental.push("Consider a bedtime routine to improve sleep quality.");
  } else if (userData.sleepHours > 9) {
    recommendations.lifestyle.push("Excessive sleep may indicate other issues - try to maintain 7-9 hours.");
  }
  
  // Diet rules
  if (userData.dietScore < 5) {
    recommendations.nutrition.push("Focus on incorporating more whole foods and vegetables.");
  }
  
  // Stress rules
  if (userData.stressLevel > 7) {
    recommendations.mental.push("Consider stress reduction techniques like meditation or mindfulness.");
    recommendations.mental.push("Schedule regular breaks during work hours.");
  }
  
  // Age-specific recommendations
  if (userData.age > 50) {
    if (recommendations.medical.urgency === "normal") {
      recommendations.medical.message = "Schedule regular preventative check-ups as recommended for your age group.";
    }
    recommendations.lifestyle.push("Include balance and strength training exercises to maintain mobility.");
  }
  
  // If no medical recommendations have been made
  if (recommendations.medical.message === "") {
    recommendations.medical.message = "Continue routine health check-ups as recommended for your age and risk factors.";
  }
  
  // Personalize based on team/company
  if (userData.team === "Engineering") {
    recommendations.mental.push("Take regular breaks from screen time to reduce eye strain.");
  } else if (userData.team === "Sales") {
    recommendations.mental.push("Practice stress management techniques for high-pressure situations.");
  }
  
  if (userData.company === "TechCorp") {
    recommendations.lifestyle.push("Take advantage of your company's standing desk options.");
  } else if (userData.company === "HealthInc") {
    recommendations.lifestyle.push("Utilize your employee wellness program benefits.");
  }
  
  return recommendations;
};

// Status mappings for UI display
const bpStatusMap = {
  'normal': '✅ Normal',
  'elevated': '⚠️ Elevated',
  'hypertension_1': '🔴 Stage 1 Hypertension',
  'hypertension_2': '🔴 Stage 2 Hypertension',
  'hypertensive_crisis': '⛔ Hypertensive Crisis'
};

const bmiStatusMap = {
  'underweight': '⚠️ Underweight',
  'normal': '✅ Normal',
  'overweight': '⚠️ Overweight',
  'obese': '🔴 Obese',
  'extremely_obese': '🔴 Extremely Obese'
};

const glucoseStatusMap = {
  'normal': '✅ Normal',
  'prediabetic': '⚠️ Prediabetic',
  'diabetic': '🔴 Diabetic Range'
};

const cholesterolStatusMap = {
  'normal': '✅ Normal',
  'borderline': '⚠️ Borderline High',
  'high': '🔴 High',
  'very_high': '⛔ Very High'
};

export default function HealthRecommendationEngine() {
  // State for form inputs
  const [userData, setUserData] = useState({
    age: 35,
    gender: "Male",
    company: "TechCorp",
    team: "Engineering",
    smokingStatus: "Never",
    sleepHours: 7,
    dietScore: 6,
    exerciseDays: 3,
    stressLevel: 5,
    systolicBP: 120,
    diastolicBP: 80,
    bmi: 24.5,
    cholesterol: 180,
    hdl: 50,
    ldl: 100,
    glucose: 90
  });
  
  // State for recommendations
  const [recommendations, setRecommendations] = useState(null);
  
  // Handle form changes
  const handleChange = (field, value) => {
    setUserData({
      ...userData,
      [field]: value
    });
  };
  
  // Generate recommendations
  const generateRecommendations = () => {
    const recs = getRecommendations(userData);
    setRecommendations(recs);
  };
  
  return (
    <div className="max-w-6xl mx-auto p-6">
      <div className="text-center mb-8">
        <h1 className="text-3xl font-bold mb-2">❤️ Health & Wellness Recommendation Engine</h1>
        <p className="text-lg">Enter your health profile to get personalized recommendations.</p>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
        {/* Left Column - Demographics & Lifestyle */}
        <div>
          <div className="bg-white p-6 rounded-lg shadow-md mb-6">
            <h2 className="text-xl font-semibold mb-4">Demographics</h2>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Age</label>
              <input 
                type="range" 
                min="18" 
                max="80" 
                value={userData.age} 
                onChange={(e) => handleChange('age', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>18</span>
                <span>Current: {userData.age}</span>
                <span>80</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Gender</label>
              <select 
                value={userData.gender}
                onChange={(e) => handleChange('gender', e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option>Male</option>
                <option>Female</option>
                <option>Other</option>
              </select>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Company</label>
              <select 
                value={userData.company}
                onChange={(e) => handleChange('company', e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option>TechCorp</option>
                <option>HealthInc</option>
                <option>FinanceOne</option>
                <option>RetailPro</option>
              </select>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Team</label>
              <select 
                value={userData.team}
                onChange={(e) => handleChange('team', e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option>Engineering</option>
                <option>Sales</option>
                <option>HR</option>
                <option>Marketing</option>
                <option>Operations</option>
              </select>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Lifestyle</h2>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Smoking Status</label>
              <select 
                value={userData.smokingStatus}
                onChange={(e) => handleChange('smokingStatus', e.target.value)}
                className="w-full p-2 border rounded"
              >
                <option>Never</option>
                <option>Former</option>
                <option>Occasional</option>
                <option>Regular</option>
              </select>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Average Sleep (hours/day)</label>
              <input 
                type="range" 
                min="3" 
                max="10" 
                step="0.5"
                value={userData.sleepHours} 
                onChange={(e) => handleChange('sleepHours', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>3</span>
                <span>Current: {userData.sleepHours}</span>
                <span>10</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Diet Quality (1-10)</label>
              <input 
                type="range" 
                min="1" 
                max="10" 
                value={userData.dietScore} 
                onChange={(e) => handleChange('dietScore', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>1</span>
                <span>Current: {userData.dietScore}</span>
                <span>10</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Exercise Days per Week</label>
              <input 
                type="range" 
                min="0" 
                max="7" 
                value={userData.exerciseDays} 
                onChange={(e) => handleChange('exerciseDays', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>0</span>
                <span>Current: {userData.exerciseDays}</span>
                <span>7</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Stress Level (1-10)</label>
              <input 
                type="range" 
                min="1" 
                max="10" 
                value={userData.stressLevel} 
                onChange={(e) => handleChange('stressLevel', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>1</span>
                <span>Current: {userData.stressLevel}</span>
                <span>10</span>
              </div>
            </div>
          </div>
        </div>
        
        {/* Right Column - Medical Metrics & Lab Results */}
        <div>
          <div className="bg-white p-6 rounded-lg shadow-md mb-6">
            <h2 className="text-xl font-semibold mb-4">Medical Metrics</h2>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Systolic BP (mmHg)</label>
              <input 
                type="range" 
                min="90" 
                max="200" 
                value={userData.systolicBP} 
                onChange={(e) => handleChange('systolicBP', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>90</span>
                <span>Current: {userData.systolicBP}</span>
                <span>200</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Diastolic BP (mmHg)</label>
              <input 
                type="range" 
                min="60" 
                max="120" 
                value={userData.diastolicBP} 
                onChange={(e) => handleChange('diastolicBP', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>60</span>
                <span>Current: {userData.diastolicBP}</span>
                <span>120</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">BMI</label>
              <input 
                type="range" 
                min="15" 
                max="45" 
                step="0.5"
                value={userData.bmi} 
                onChange={(e) => handleChange('bmi', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>15</span>
                <span>Current: {userData.bmi}</span>
                <span>45</span>
              </div>
            </div>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-md">
            <h2 className="text-xl font-semibold mb-4">Lab Results</h2>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Total Cholesterol (mg/dL)</label>
              <input 
                type="range" 
                min="120" 
                max="300" 
                value={userData.cholesterol} 
                onChange={(e) => handleChange('cholesterol', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>120</span>
                <span>Current: {userData.cholesterol}</span>
                <span>300</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">HDL (mg/dL)</label>
              <input 
                type="range" 
                min="20" 
                max="100" 
                value={userData.hdl} 
                onChange={(e) => handleChange('hdl', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>20</span>
                <span>Current: {userData.hdl}</span>
                <span>100</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">LDL (mg/dL)</label>
              <input 
                type="range" 
                min="50" 
                max="250" 
                value={userData.ldl} 
                onChange={(e) => handleChange('ldl', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>50</span>
                <span>Current: {userData.ldl}</span>
                <span>250</span>
              </div>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium mb-1">Fasting Glucose (mg/dL)</label>
              <input 
                type="range" 
                min="70" 
                max="200" 
                value={userData.glucose} 
                onChange={(e) => handleChange('glucose', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="flex justify-between text-sm">
                <span>70</span>
                <span>Current: {userData.glucose}</span>
                <span>200</span>
              </div>
            </div>
          </div>
        </div>
      </div>
      
      <div className="mt-8">
        <button 
          onClick={generateRecommendations}
          className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-4 rounded-md transition"
        >
          Get Recommendations
        </button>
      </div>
      
      {recommendations && (
        <div className="mt-8 bg-white p-6 rounded-lg shadow-md">
          <h2 className="text-2xl font-bold mb-6 text-center">Your Personalized Health Recommendations</h2>
          
          {/* Medical Recommendation */}
          <div className="mb-6">
            <h3 className="text-xl font-semibold mb-2">🩺 Medical Recommendation</h3>
            
            <div className={`p-4 rounded-lg ${
              recommendations.medical.urgency === "critical" ? "bg-red-100 text-red-800 border-l-4 border-red-600" :
              recommendations.medical.urgency === "high" ? "bg-orange-100 text-orange-800 border-l-4 border-orange-600" :
              recommendations.medical.urgency === "medium" ? "bg-yellow-100 text-yellow-800 border-l-4 border-yellow-600" :
              "bg-blue-100 text-blue-800 border-l-4 border-blue-600"
            }`}>
              {recommendations.medical.message}
            </div>
          </div>
          
          {/* Lifestyle Recommendations */}
          {recommendations.lifestyle.length > 0 && (
            <div className="mb-6">
              <h3 className="text-xl font-semibold mb-2">🏃‍♀️ Lifestyle Recommendations</h3>
              <div className="bg-green-100 p-4 rounded-lg text-green-800">
                <ul className="list-disc pl-5 space-y-1">
                  {recommendations.lifestyle.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          
          {/* Nutrition Recommendations */}
          {recommendations.nutrition.length > 0 && (
            <div className="mb-6">
              <h3 className="text-xl font-semibold mb-2">🍎 Nutrition Recommendations</h3>
              <div className="bg-emerald-100 p-4 rounded-lg text-emerald-800">
                <ul className="list-disc pl-5 space-y-1">
                  {recommendations.nutrition.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          
          {/* Mental Health Recommendations */}
          {recommendations.mental.length > 0 && (
            <div className="mb-6">
              <h3 className="text-xl font-semibold mb-2">🧠 Mental Wellbeing Recommendations</h3>
              <div className="bg-purple-100 p-4 rounded-lg text-purple-800">
                <ul className="list-disc pl-5 space-y-1">
                  {recommendations.mental.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
          
          {/* Health Insights */}
          <div className="mt-8">
            <h3 className="text-xl font-semibold mb-4">📊 Health Insights</h3>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="bg-gray-100 p-4 rounded-lg">
                <h4 className="font-medium text-gray-700 mb-1">Blood Pressure Status</h4>
                <p className="text-lg font-semibold">
                  {bpStatusMap[categorizeBP(userData.systolicBP, userData.diastolicBP)]}
                </p>
              </div>
              
              <div className="bg-gray-100 p-4 rounded-lg">
                <h4 className="font-medium text-gray-700 mb-1">BMI Status</h4>
                <p className="text-lg font-semibold">
                  {bmiStatusMap[categorizeBMI(userData.bmi)]}
                </p>
              </div>
              
              <div className="bg-gray-100 p-4 rounded-lg">
                <h4 className="font-medium text-gray-700 mb-1">Glucose Status</h4>
                <p className="text-lg font-semibold">
                  {glucoseStatusMap[categorizeGlucose(userData.glucose)]}
                </p>
              </div>
              
              <div className="bg-gray-100 p-4 rounded-lg">
                <h4 className="font-medium text-gray-700 mb-1">Cholesterol Status</h4>
                <p className="text-lg font-semibold">
                  {cholesterolStatusMap[categorizeCholesterol(userData.cholesterol, userData.hdl, userData.ldl)]}
                </p>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
