#import libraries
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Apply custom theme colors using markdown and CSS
st.markdown(
    """
    <style>
    .title {
        color: #900C3F;
        font-size: 36px;
        font-weight: bold;
    }
    .subheader {
        color: #900C3F;
        font-size: 24px;
        font-weight: bold;
    }
    .text {
        color: #0000FF; /* OBlue */
        font-size: 18px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title with theme color
st.markdown('<div class="title">INX COMPANY-EMPLOYEE PERFORMANCE PREDICTION APPLICATION</div>', unsafe_allow_html=True)

# Description with theme color
st.markdown(
    '<div class="text">This Web application is used to predict employee performance based on the data provided. A user is supposed to fill in a form provided where the application will predict the employee performance rating.</div>',
    unsafe_allow_html=True
)

# Load dataset
url = "https://raw.githubusercontent.com/smbindyo/Datasets/refs/heads/main/INX_Future_Inc_Employee_Performance_CDS_Project2_Data_V1.8.xls"
df = pd.read_excel(url)

# Function to remove outliers
def remove_outliers(df, features):
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df = df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]
    return df

# Remove outliers
features_to_clean = [
    'TotalWorkExperienceInYears',
    'ExperienceYearsInCurrentRole',
    'ExperienceYearsAtThisCompany',
    'YearsSinceLastPromotion'
]
df = remove_outliers(df, features_to_clean)

# Function to encode categorical features
def encode_categorical_features(df):
    label_encoders = {}
    categorical_features = df.select_dtypes(include=['object']).columns.tolist()
    for feature in categorical_features:
        le = LabelEncoder()
        df[feature] = le.fit_transform(df[feature].astype(str))
        label_encoders[feature] = le
    return df, label_encoders

# Encode categorical features
df, label_encoders = encode_categorical_features(df)

# Function to preprocess data
def preprocess_data(df, target_column):
    scaler = StandardScaler()
    normalizer = MinMaxScaler()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    scaler.fit(X)
    normalizer.fit(scaler.transform(X))
    X_scaled = scaler.transform(X)
    X_normalized = normalizer.transform(X_scaled)
    X_normalized = pd.DataFrame(X_normalized, columns=X.columns)
    return X_normalized, y, scaler, normalizer

# Preprocess the dataset
target_column = 'PerformanceRating'
X_normalized, y, scaler, normalizer = preprocess_data(df, target_column)

# Train RandomForest model
def train_random_forest(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return model, accuracy

model, accuracy = train_random_forest(X_normalized, y)

# Display model accuracy
#st.write(f"Random Forest Model Accuracy: {accuracy:.2f}")

# Display the key mappings for categorical variables
st.markdown('<div class="subheader">Kindly refer to the following value-Rating translation</div>', unsafe_allow_html=True)

# Create a dictionary for the mappings
key_mappings = {
    "EmpEducationLevel": {
        1: "Below College",
        2: "College",
        3: "Bachelor",
        4: "Master",
        5: "Doctor"
    },
    "EmpEnvironmentSatisfaction": {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    },
    "EmpJobInvolvement": {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    },
    "EmpJobSatisfaction": {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    },
    "PerformanceRating": {
        1: "Low",
        2: "Good",
        3: "Excellent",
        4: "Outstanding"
    },
    "RelationshipSatisfaction": {
        1: "Low",
        2: "Medium",
        3: "High",
        4: "Very High"
    },
    "EmpWorkLifeBalance": {
        1: "Bad",
        2: "Good",
        3: "Better",
        4: "Best"
    }
}

# Display the mappings in columns
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.write("###### Education Level")
    for key, value in key_mappings["EmpEducationLevel"].items():
        st.write(f"{key}: {value}")
with col2:
    st.write("######  Environment Satisfaction")
    for key, value in key_mappings["EmpEnvironmentSatisfaction"].items():
        st.write(f"{key}: {value}")
with col3:

    st.write("###### Job Involvement")
    for key, value in key_mappings["EmpJobInvolvement"].items():
        st.write(f"{key}: {value}")
with col4:
    st.write("######  Job Satisfaction")
    for key, value in key_mappings["EmpJobSatisfaction"].items():
        st.write(f"{key}: {value}")

with col5:

    st.write("######  Relationship Satisfaction")
    for key, value in key_mappings["RelationshipSatisfaction"].items():
        st.write(f"{key}: {value}")
with col6:
    st.write("######  Work Life Balance")
    for key, value in key_mappings["EmpWorkLifeBalance"].items():
        st.write(f"{key}: {value}")



# Create a user interface form for input
st.markdown('<div class="subheader">Employee Performance Prediction Form</div>', unsafe_allow_html=True)

with st.form("employee_input_form"):
    st.write("Please fill in the details below:")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        emp_number = st.number_input("Employee Number", min_value=1, step=1)
        age = st.number_input("Age", min_value=18, max_value=100, step=1)
        gender = st.selectbox("Gender", ["Male", "Female"])
        education_background = st.selectbox("Education Background", ["Life Sciences", "Medical", "Marketing", "Technical Degree", "Other"])
        marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
        emp_department = st.selectbox("Department", ["Sales", "Human Resources", "Development", "Data Science", "Research & Development", "Finance"])
        emp_job_role = st.selectbox("Job Role", ["Sales Executive", "Manager", "Developer", "Sales Representative", "Human Resources", "Senior Developer", "Data Scientist", "Senior Manager R&D", "Laboratory Technician", "Manufacturing Director", "Research Scientist", "Healthcare Representative", "Research Director", "Manager R&D", "Finance Manager", "Technical Architect", "Business Analyst", "Technical Lead", "Delivery Manager"])
    with col2:
        business_travel_frequency = st.selectbox("Business Travel Frequency", ["Non-Travel", "Travel_Rarely", "Travel_Frequently"])
        distance_from_home = st.number_input("Distance From Home (in km)", min_value=0, max_value=100, step=1)
        emp_education_level = st.selectbox("Education Level", [1, 2, 3, 4, 5])
        emp_environment_satisfaction = st.selectbox("Environment Satisfaction", [1, 2, 3, 4])
        emp_hourly_rate = st.number_input("Hourly Rate", min_value=0, max_value=100, step=1)
        emp_job_involvement = st.selectbox("Job Involvement", [1, 2, 3, 4])
        emp_job_level = st.number_input("Job Level", min_value=1, max_value=5, step=1)

    with col3:
        emp_job_satisfaction = st.selectbox("Job Satisfaction", [1, 2, 3, 4])
        num_companies_worked = st.number_input("Number of Companies Worked", min_value=0, max_value=20, step=1)
        over_time = st.selectbox("Over Time", ["Yes", "No"])
        emp_last_salary_hike_percent = st.number_input("Last Salary Hike Percent", min_value=0, max_value=100, step=1)
        emp_relationship_satisfaction = st.selectbox("Relationship Satisfaction", [1, 2, 3, 4])
        total_work_experience_in_years = st.number_input("Total Work Experience (in years)", min_value=0, max_value=50, step=1)
        training_times_last_year = st.number_input("Training Times Last Year", min_value=0, max_value=10, step=1)

    with col4:
        emp_work_life_balance = st.selectbox("Work Life Balance", [1, 2, 3, 4])
        experience_years_at_this_company = st.number_input("Experience Years At This Company", min_value=0, max_value=50, step=1)
        experience_years_in_current_role = st.number_input("Experience Years In Current Role", min_value=0, max_value=50, step=1)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", min_value=0, max_value=50, step=1)
        years_with_curr_manager = st.number_input("Years With Current Manager", min_value=0, max_value=50, step=1)
        attrition = st.selectbox("Attrition", ["Yes", "No"])

    submitted = st.form_submit_button("Submit")

if submitted:
    user_data = {
        "EmpNumber": emp_number,
        "Age": age,
        "Gender": gender,
        "EducationBackground": education_background,
        "MaritalStatus": marital_status,
        "EmpDepartment": emp_department,
        "EmpJobRole": emp_job_role,
        "BusinessTravelFrequency": business_travel_frequency,
        "DistanceFromHome": distance_from_home,
        "EmpEducationLevel": emp_education_level,
        "EmpEnvironmentSatisfaction": emp_environment_satisfaction,
        "EmpHourlyRate": emp_hourly_rate,
        "EmpJobInvolvement": emp_job_involvement,
        "EmpJobLevel": emp_job_level,
        "EmpJobSatisfaction": emp_job_satisfaction,
        "NumCompaniesWorked": num_companies_worked,
        "OverTime": over_time,
        "EmpLastSalaryHikePercent": emp_last_salary_hike_percent,
        "EmpRelationshipSatisfaction": emp_relationship_satisfaction,
        "TotalWorkExperienceInYears": total_work_experience_in_years,
        "TrainingTimesLastYear": training_times_last_year,
        "EmpWorkLifeBalance": emp_work_life_balance,
        "ExperienceYearsAtThisCompany": experience_years_at_this_company,
        "ExperienceYearsInCurrentRole": experience_years_in_current_role,
        "YearsSinceLastPromotion": years_since_last_promotion,
        "YearsWithCurrManager": years_with_curr_manager,
        "Attrition": attrition
    }

    user_input = pd.DataFrame([user_data])

    def preprocess_input(input_data, scaler, normalizer):
        for feature in ["Gender", "EducationBackground", "MaritalStatus", "EmpDepartment", "EmpJobRole", "BusinessTravelFrequency", "OverTime", "Attrition"]:
            input_data[feature] = label_encoders[feature].transform(input_data[feature])
        input_scaled = scaler.transform(input_data)
        input_normalized = normalizer.transform(input_scaled)
        return pd.DataFrame(input_normalized, columns=input_data.columns)

    processed_input = preprocess_input(user_input, scaler, normalizer)

    # Debugging: Display processed input
    #st.write("Processed Input:", processed_input)

    # Make prediction using the trained model
    prediction = model.predict(processed_input)

    # Debugging: Display prediction
    st.write("Prediction:", prediction)

    # Map prediction to performance rating
    performance_mapping = {1: "Low", 2: "Good", 3: "Excellent", 4: "Outstanding"}
    predicted_rating = performance_mapping[prediction[0]]

    # Display the prediction
    st.markdown('<div class="subheader">Performance Rating Prediction</div>', unsafe_allow_html=True)
    if predicted_rating in ["Low", "Good"]:
        st.error(f"The predicted performance rating is: **{predicted_rating}**")
    else:
        st.success(f"The predicted performance rating is: **{predicted_rating}**")