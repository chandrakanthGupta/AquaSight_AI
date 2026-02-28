import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,classification_report
import pickle 
import matplotlib.pyplot as plt
def calculate_wqi(ph, solids, turbidity, conductivity, organic_carbon):

    ideal_ph_low, ideal_ph_high = 6.5, 8.5
    max_solids = 500
    max_turbidity = 5
    max_conductivity = 400
    max_organic = 10

    score = 0

    if ideal_ph_low <= ph <= ideal_ph_high:
        score += 20
    else:
        score += 5

    score += 20 if solids <= max_solids else 5
    score += 20 if turbidity <= max_turbidity else 5
    score += 20 if conductivity <= max_conductivity else 5
    score += 20 if organic_carbon <= max_organic else 5

    return score


def get_wqi_status(wqi):
    if wqi >= 80:
        return "Excellent"
    elif wqi >= 60:
        return "Good"
    elif wqi >= 40:
        return "Poor"
    else:
        return "Hazardous"

       

df=pd.read_csv('water_potability.csv')

df.fillna(df.mean(), inplace=True)

x=df.drop('Potability',axis=1)
y=df['Potability']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)

model=RandomForestClassifier(n_estimators=300,class_weight="balanced",random_state=42)  # pyright: ignore[reportUndefinedVariable]
model.fit(x_train,y_train)


probability=model.predict_proba(x_test)
threshold=0.4
y_pred=(probability[:,1]>threshold).astype(int)

importances = model.feature_importances_
features = x.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.show()
print("\nSample Risk Scores:")

for i in range(5):
    unsafe_risk = probability[i][0] * 100  # pyright: ignore[reportUndefinedVariable]
    print(f"Sample {i+1} - Risk of Unsafe Water: {unsafe_risk:.2f}%")
    
print("Model Accuracy: ",accuracy_score(y_test,y_pred))
print("Model Classification Report: ",classification_report(y_test,y_pred))

print("\nWQI Results for First 3 Samples:\n")

print("\n===== FINAL HYBRID RESULTS =====\n")

for i in range(3):

    row = x_test.iloc[i]

    # ML Unsafe Risk %
    ml_unsafe_risk = probability[i][0] * 100

    # WQI Score
    wqi = calculate_wqi(
        row["ph"],
        row["Solids"],
        row["Turbidity"],
        row["Conductivity"],
        row["Organic_carbon"]
    )

    # Hybrid Risk Score
    final_risk_score = (ml_unsafe_risk * 0.6) + ((100 - wqi) * 0.4)

    # Final Status
    if final_risk_score < 30:
        final_status = "Safe"
    elif final_risk_score < 60:
        final_status = "Moderate Risk"
    else:
        final_status = "High Risk"

    print(f"\nSample {i+1}")
    print(f"ML Unsafe Risk: {ml_unsafe_risk:.2f}%")
    print(f"WQI Score: {wqi}")
    print(f"Final Hybrid Risk: {final_risk_score:.2f}%")
    print(f"Final Status: {final_status}")

pickle.dump(model,open('water_potability.pkl','wb'))    



