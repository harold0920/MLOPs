import requests

url = "http://localhost:8000/predict"

# ✅ Example employee input (raw categorical data)
data = {
    "Age": 29,
    "Gender": "Male",
    "BusinessTravel": "Travel_Rarely",
    "MaritalStatus": "Single",
    "EducationField": "Life Sciences",
    "Department": "Research & Development",
    "OverTime": "No",
    "DistanceFromHome": 10,
    "MonthlyIncome": 5000,
    "NumCompaniesWorked": 2,
    "TotalWorkingYears": 6,
    "YearsAtCompany": 3,
    "JobSatisfaction": 3,
    "EnvironmentSatisfaction": 4,
    "WorkLifeBalance": 2
}

print("🚀 Sending request to FastAPI...")
response = requests.post(url, json=data)

print("✅ Response Status Code:", response.status_code)
print("Predictions:", response.json())
