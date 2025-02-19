import requests

url = "http://localhost:8000/predict"

# Example employee input (raw categorical data)
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

print("Sending request to FastAPI...")

try:
    response = requests.post(url, json=data)
    
    # Handle HTTP errors
    if response.status_code == 200:
        prediction = response.json().get("attrition", "Unknown")
        print(f"Prediction: Employee Attrition Risk → {prediction}")
    else:
        print(f"Error {response.status_code}: {response.text}")

except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
