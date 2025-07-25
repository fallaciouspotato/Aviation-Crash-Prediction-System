import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import joblib

print("Starting the one-time training and data preparation process...")

# --- Load Full Dataset ---
data = pd.read_csv('AviationData.csv', encoding='latin-1')
data.columns = data.columns.str.replace('.', '_', regex=False)

# --- Data Cleaning ---
data['Event_Date'] = pd.to_datetime(data['Event_Date'], errors='coerce')
data = data.dropna(subset=['Aircraft_damage', 'Event_Date'])
data = data[data['Aircraft_damage'] != 'Unknown']
num_cols_to_impute = ['Number_of_Engines', 'Total_Fatal_Injuries', 'Total_Serious_Injuries', 'Total_Minor_Injuries', 'Total_Uninjured']
for col in num_cols_to_impute:
    data[col].fillna(data[col].median(), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col] = data[col].fillna('Unknown')
data['Year'] = data['Event_Date'].dt.year.astype(int)
data['Month'] = data['Event_Date'].dt.month.astype(int)

print("Step 1/3: Data has been cleaned.")

# --- Prepare Data for Charts Tab ---
chart_data = {
    "full_data_head": data.head(),
    "full_data_describe": data.describe(),
    "damage_distribution": data['Aircraft_damage'].value_counts(),
    "incidents_by_year": data['Year'].value_counts().sort_index(),
    "damage_by_weather": data.groupby(['Weather_Condition', 'Aircraft_damage']).size().reset_index(name='count'),
    "damage_by_phase": data.groupby(['Broad_phase_of_flight', 'Aircraft_damage']).size().reset_index(name='count'),
    "damage_by_engine": data.groupby(['Engine_Type', 'Aircraft_damage']).size().reset_index(name='count'),
    "country_counts": data['Country'].value_counts().reset_index(),
    "dropdown_options": {col: sorted(list(data[col].unique())) for col in ['Make', 'Model', 'Engine_Type', 'Country', 'Weather_Condition', 'Broad_phase_of_flight', 'Purpose_of_flight']}
}
joblib.dump(chart_data, 'chart_data.joblib')
print("Step 2/3: All chart data has been prepared and saved.")

# --- Train Models and Calculate Performance ---
features = ['Make', 'Model', 'Engine_Type', 'Number_of_Engines', 'Weather_Condition', 'Broad_phase_of_flight', 'Purpose_of_flight', 'Country', 'Year', 'Month', 'Total_Fatal_Injuries', 'Total_Serious_Injuries']
target = 'Aircraft_damage'
df = data[features + [target]].copy()

encoders = {col: LabelEncoder() for col in df.select_dtypes(include=['object']).columns if col != target}
for col, encoder in encoders.items():
    known_classes = list(encoder.fit(df[col]).classes_)
    if 'Unknown' not in known_classes:
        encoder.classes_ = sorted(known_classes + ['Unknown'])
    df[col] = encoder.transform(df[col])

X = df[features]
y_encoder = LabelEncoder()
y_encoded = y_encoder.fit_transform(df[target])
encoders['target'] = y_encoder

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
xgb_tuned_params = {'objective': 'multi:softmax', 'n_estimators': 300, 'max_depth': 8, 'learning_rate': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8, 'use_label_encoder': False, 'eval_metric': 'mlogloss', 'random_state': 42}

models_to_train = {
    "Random Forest (Baseline)": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost (Optimized)": XGBClassifier(**xgb_tuned_params)
}

trained_models_info = {}
for name, model in models_to_train.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    performance = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred, average='weighted'),
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classes": y_encoder.classes_
    }
    if hasattr(model, 'feature_importances_'):
        performance['Feature Importances'] = model.feature_importances_
        performance['Feature Names'] = X.columns
    trained_models_info[name] = {"model": model, "performance": performance, "encoders": encoders, "y_encoder": y_encoder}

joblib.dump(trained_models_info, 'models_and_performance.joblib')
print("Step 3/3: Models trained and performance data saved.")
print("\nPROCESS COMPLETE! You are ready to deploy the final app.py.")