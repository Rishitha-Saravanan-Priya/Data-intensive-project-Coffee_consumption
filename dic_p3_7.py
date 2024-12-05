import numpy as np
from sklearn.neighbors import KNeighborsRegressor
import streamlit as st
import pandas as pd
import requests
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit.components.v1 as components

# Firebase Database URL
FIREBASE_URL = "https://impacts-of-coffee-consumption-default-rtdb.firebaseio.com/"

# Fetch data from Firebase
def fetch_data():
    response = requests.get(FIREBASE_URL + ".json")
    if response.status_code == 200:
        data = response.json()
        
        # Check if the data is a dictionary or list
        if isinstance(data, dict):
            return pd.DataFrame(data.values())
        elif isinstance(data, list):
            return pd.DataFrame(data)
        else:
            st.warning("Unexpected data format received from Firebase.")
            return pd.DataFrame()
    else:
        st.error("Failed to fetch data from Firebase.")
        return pd.DataFrame()

# Add data to Firebase
def add_data(data):
    response = requests.post(FIREBASE_URL + ".json", json=data)
    if response.status_code == 200:
        st.success("Data added successfully!")
    else:
        st.error("Failed to add data to Firebase.")

# Main App
st.title("Impact of Coffee Consumption on Sleep Quality and Health☕")
st.sidebar.title("App Navigation")
page = st.sidebar.selectbox(
    "Go to", ["Home", "Data Entry", "View Data", "Visualizations"]
)

# 1. Home Page
if page == "Home":
    st.header("Welcome!")

    # Typewriting Animation using HTML + CSS + JavaScript
    typewriting_html = """
    <div style="font-family: 'Arial', sans-serif; font-size: 18px; line-height: 1.6; color: #333;">
        <h3>🌙 Welcome to the Sleep & Health Analyzer! 🛌</h3>
        <p id="typewriter"></p>
    </div>
    <script>
        const text = [
            "This app helps you analyze the impact of lifestyle habits on your sleep quality and overall health.",
            "🗂️ Add, View, and Manage your sleep schedule effortlessly.",
            "📊 Explore Visualizations to uncover insights.",
            "🔍 Gain Actionable Insights to improve your lifestyle.",
            "🌟 Start your journey towards better sleep and health right away!"
        ];
        let index = 0;
        let charIndex = 0;
        const speed = 50; // Typing speed
        const delay = 1500; // Delay between messages

        function typeEffect() {
            const typewriter = document.getElementById("typewriter");
            if (index < text.length) {
                if (charIndex < text[index].length) {
                    typewriter.innerHTML += text[index].charAt(charIndex);
                    charIndex++;
                    setTimeout(typeEffect, speed);
                } else {
                    charIndex = 0;
                    index++;
                    typewriter.innerHTML += "<br><br>"; // Line break after each sentence
                    setTimeout(typeEffect, delay);
                }
            }
        }
        document.addEventListener("DOMContentLoaded", typeEffect);
    </script>
    <style>
        #typewriter {
            font-size: 1.1em;
            color: #444;
            font-weight: 400;
            font-family: 'Courier New', monospace;
            margin-top: 10px;
            white-space: pre-wrap;
        }
    </style>
    """

    # Embed the Typewriting Effect in Streamlit
    components.html(typewriting_html, height=300)


# 2. Data Entry Page
# Inside the Data Entry Page

elif page == "Data Entry":
    st.header("Enter Your Data")
    with st.form("data_entry_form"):
        id = st.number_input("ID:", min_value=1, value=1, step=1)
        age = st.number_input("Age:", min_value=1, max_value=120, value=25)
        city = st.number_input("City (Code):", min_value=0, value=0)
        state = st.number_input("State (Code):", min_value=0, value=0)
        
        occupation_dict = {
            "Manager": 0,
            "Artist": 1,
            "Lawyer": 2,
            "Teacher": 3,
            "Engineer": 4,
            "Doctor": 5,
            "Scientist": 6,
            "Nurse": 7
        }
        occupation_names = list(occupation_dict.keys())

        selected_occupation = st.selectbox("Select Occupation:", occupation_names)
        occupation = occupation_dict[selected_occupation]

        gender = st.selectbox("Gender (Code):", [0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
        bedtime_input = st.text_input("Bedtime (YYYY-MM-DD HH:MM:SS):", "2021-03-06 01:00:00")
        try:
            bedtime = pd.to_datetime(bedtime_input)
        except ValueError:
            st.error("Please enter a valid timestamp in the format 'YYYY-MM-DD HH:MM:SS'.")

        wakeup_input = st.text_input("Wakeup Time (YYYY-MM-DD HH:MM:SS):", "2021-03-06 07:00:00")
        try:
            wakeup_time = pd.to_datetime(wakeup_input)
        except ValueError:
            st.error("Please enter a valid timestamp in the format 'YYYY-MM-DD HH:MM:SS'.")

        sleep_duration = st.number_input("Sleep Duration (hours):", min_value=0.0, max_value=24.0, value=6.0)
        sleep_efficiency = st.slider("Sleep Efficiency:", min_value=0.0, max_value=1.0, value=0.88)
        rem_sleep = st.number_input("REM Sleep Percentage:", min_value=0, max_value=100, value=18)
        deep_sleep = st.number_input("Deep Sleep Percentage:", min_value=0, max_value=100, value=70)
        light_sleep = st.number_input("Light Sleep Percentage:", min_value=0, max_value=100, value=12)
        awakenings = st.number_input("Number of Awakenings:", min_value=0, value=0)
        caffeine_consumption = st.number_input("Caffeine Consumption (mg):", min_value=0, value=50)
        alcohol_consumption = st.number_input("Alcohol Consumption (units):", min_value=0, value=0)
        smoking_status = st.selectbox("Smoking Status:", [0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
        exercise_frequency = st.number_input("Exercise Frequency (days/week):", min_value=0, max_value=7, value=3)

        # Add submit button inside the form
        submit = st.form_submit_button("Submit")

        # Handle form submission
        if submit:
            new_data = {
                "ID": id,
                "Age": age,
                "City": city,
                "State": state,
                "Occupation": occupation,
                "Gender": gender,
                "Bedtime": str(bedtime),
                "Wakeup time": str(wakeup_time),
                "Sleep duration": sleep_duration,
                "Sleep efficiency": sleep_efficiency,
                "REM sleep percentage": rem_sleep,
                "Deep sleep percentage": deep_sleep,
                "Light sleep percentage": light_sleep,
                "Awakenings": awakenings,
                "Caffeine consumption": caffeine_consumption,
                "Alcohol consumption": alcohol_consumption,
                "Smoking status": smoking_status,
                "Exercise frequency": exercise_frequency,
            }
            add_data(new_data)

# 3. View Data Page
elif page == "View Data":
    st.header("View and Manage Data")

    # Fetch data from Firebase
    df = fetch_data()

    if df.empty:
        st.warning("No data available in the database.")
    else:
        st.write("### All Records")
        st.write(df)

        # Filter by Age
        age_filter = st.slider("Select Age Range:", min_value=0, max_value=100, value=(0, 100))

        # Apply Age Filter
        filtered_df = df[(df["Age"] >= age_filter[0]) & (df["Age"] <= age_filter[1])]

        # Display filtered results
        if filtered_df.empty:
            st.warning("No records found matching the filter criteria.")
        else:
            st.write("### Filtered Records")
            st.write(filtered_df)

# 4. Visualizations Page
elif page == "Visualizations":
    st.header("Visualizations")
    
    # Fetch data from Firebase
    df = fetch_data()

    if not df.empty:
        # LightGBM Model for Sleep Duration Prediction
        features_used = ['Light sleep percentage', 'REM sleep percentage', 'Alcohol consumption', 
                         'Caffeine consumption', 'Sleep efficiency', 'Deep sleep percentage', 'Age', 'Exercise frequency']
        X = df[features_used]
        y = df['Sleep duration']

        X = X.dropna()
        y = y.loc[X.index]

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Initialize LightGBM Model
        lgb_model = lgb.LGBMRegressor(random_state=42)
        lgb_model.fit(X_train, y_train)

        y_pred_lgb = lgb_model.predict(X_test)

        mse_lgb = mean_squared_error(y_test, y_pred_lgb)
        r2_lgb = r2_score(y_test, y_pred_lgb)

        # Model Evaluation (LightGBM)
        st.write(f"### Model Evaluation (LightGBM)")
        st.write(f"Mean Squared Error (MSE): {mse_lgb}")
        st.write(f"R-squared (R2): {r2_lgb}")

        # Plotting Prediction vs Actual
        st.write("### Prediction vs Actual Values (LightGBM)")
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred_lgb, alpha=0.5, color='orange')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        plt.xlabel("Actual Sleep Duration")
        plt.ylabel("Predicted Sleep Duration")
        plt.title("Prediction using LightGBM Regression Model")
        st.pyplot(plt)

        # Logistic Regression for Sleep Efficiency Prediction
        df['Sleep efficiency'] = pd.to_numeric(df['Sleep efficiency'], errors='coerce')
        df['High_Sleep_Efficiency'] = (df['Sleep efficiency'] > df['Sleep efficiency'].median()).astype(int)

        features = ['Age','Caffeine consumption', 'Alcohol consumption', 'REM sleep percentage', 'Deep sleep percentage', 'Smoking status']
        X = df[features].dropna()
        y = df['High_Sleep_Efficiency'][X.index]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

        # Polynomial Features & Scaling
        polynomialFeat = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
        X_train_poly = polynomialFeat.fit_transform(X_train)
        X_test_poly = polynomialFeat.transform(X_test)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_poly)
        X_test_scaled = scaler.transform(X_test_poly)

        parameterGrid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']}
        cv = StratifiedKFold(n_splits=5)

        logReg = GridSearchCV(LogisticRegression(max_iter=1000, class_weight='balanced'), parameterGrid, cv=cv, scoring='accuracy')
        logReg.fit(X_train_scaled, y_train)
        best_logReg = logReg.best_estimator_

        y_pred_log = best_logReg.predict(X_test_scaled)

        # Accuracy and Classification Report
        st.write("### Logistic Regression Results")
        st.write(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")
        st.write(f"Classification Report: \n{classification_report(y_test, y_pred_log)}")

        # Plotting ROC Curve
        y_pred_prob_log = best_logReg.predict_proba(X_test_scaled)[:, 1]
        fpr_log, tpr_log, _ = roc_curve(y_test, y_pred_prob_log)
        roc_auc_log = auc(fpr_log, tpr_log)

        plt.figure(figsize=(12, 6))
        plt.plot(fpr_log, tpr_log, color='blue', label=f'Logistic Regression (AUC = {roc_auc_log:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Logistic Regression')
        plt.legend(loc='best')
        st.pyplot(plt)

        # SVM Model for Caffeine Level Prediction
        df['Caffeine consumption'] = (df['Caffeine consumption'] > df['Caffeine consumption'].median()).astype(int)
        X_caffeine = df[['Sleep duration', 'Age', 'Awakenings', 'Exercise frequency']]
        y_caffeine = df['Caffeine consumption']

        X_train_caffeine, X_test_caffeine, y_train_caffeine, y_test_caffeine = train_test_split(X_caffeine, y_caffeine, test_size=0.2, random_state=42)

        X_train_caffeine = X_train_caffeine.replace('Unknown', np.nan).fillna(X_train_caffeine.mode().iloc[0])
        X_test_caffeine = X_test_caffeine.replace('Unknown', np.nan).fillna(X_test_caffeine.mode().iloc[0])

        X_train_caffeine = pd.get_dummies(X_train_caffeine, drop_first=True)
        X_test_caffeine = pd.get_dummies(X_test_caffeine, drop_first=True)

        X_test_caffeine = X_test_caffeine.reindex(columns=X_train_caffeine.columns, fill_value=0)

        svm_caffeine_model = SVC(kernel='linear', class_weight={0: 1, 1: 6}, random_state=42)
        svm_caffeine_model.fit(X_train_caffeine, y_train_caffeine)

        y_pred_caffeine_svm = svm_caffeine_model.predict(X_test_caffeine)

        # Accuracy and Confusion Matrix for SVM
        st.write(f"### SVM Model for Caffeine Level Prediction")
        st.write(f"Accuracy: {accuracy_score(y_test_caffeine, y_pred_caffeine_svm) * 100}%")

        conf_matrix_caffeine_svm = confusion_matrix(y_test_caffeine, y_pred_caffeine_svm)
        sns.heatmap(conf_matrix_caffeine_svm, annot=True, fmt="d", cmap="Reds")
        plt.title("SVM Confusion Matrix for Caffeine Level Prediction")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        st.pyplot(plt)

    
        # K-Nearest Neighbors Regressor for Sleep Efficiency Prediction
        knn_regressor = KNeighborsRegressor(n_neighbors=3)
        knn_regressor.fit(X_train, y_train)

        # Predict Sleep Efficiency
        y_pred_efficiency = knn_regressor.predict(X_test)

        # Performance Metrics for KNN Regressor
        mse_efficiency = mean_squared_error(y_test, y_pred_efficiency)
        mae_efficiency = mean_absolute_error(y_test, y_pred_efficiency)
        r2_efficiency = r2_score(y_test, y_pred_efficiency)

        # Display Results
        st.write("### K-Nearest Neighbors Regressor Results for Sleep Efficiency Prediction")
        st.write(f"Mean Squared Error (MSE): {mse_efficiency:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae_efficiency:.2f}")
        st.write(f"R² Score: {r2_efficiency:.2f}")

        # Visualization of True vs Predicted Sleep Efficiency (Scatter Plot)
        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, y_pred_efficiency, alpha=0.6, color='green', label="Predicted vs Actual")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label="Ideal Fit")
        plt.xlabel("Actual Sleep Efficiency")
        plt.ylabel("Predicted Sleep Efficiency")
        plt.title("True vs. Predicted Sleep Efficiency (KNN Regressor)")
        plt.legend()
        st.pyplot(plt)

    else:
        st.warning("No data available for visualizations.")