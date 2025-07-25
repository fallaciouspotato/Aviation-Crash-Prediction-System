import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_option_menu import option_menu
import joblib

# -----------------
# Page Configuration
# -----------------
st.set_page_config(
    page_title="Aviation Damage Predictor",
    page_icon="✈️",
    layout="wide",
)

st.markdown("<h3 style='text-align: center; color: grey;'>A Team 2LPA Project</h3>", unsafe_allow_html=True)

# -----------------
# Load All Pre-Built Assets
# -----------------
@st.cache_resource
def load_assets():
    """Loads all pre-trained models, encoders, and chart data from files."""
    models_info = joblib.load('models_and_performance.joblib')
    chart_data = joblib.load('chart_data.joblib')
    return models_info, chart_data

models_info, chart_data = load_assets()
best_model_name = "XGBoost (Optimized)"
best_model_pack = models_info[best_model_name]


# --- Main Title Added Here ---
st.markdown("<h1 style='text-align: center; font-weight: bold;'>AVIATION DAMAGE PREDICTION SYSTEM</h1>", unsafe_allow_html=True)


# -----------------
# Horizontal Navigation Bar
# -----------------
selected = option_menu(
    menu_title=None,
    options=["Home", "Live Prediction", "Crash Case Studies", "Model Performance", "Data Analysis"],
    icons=["house", "rocket-launch", "journal-text", "bar-chart-line", "search"],
    menu_icon="cast", default_index=0, orientation="horizontal",
)

# -----------------
# Page Content
# -----------------

# --- Home Page ---
if selected == "Home":
    st.title("✈️ Aircraft Damage Severity Predictor")
    st.markdown("### A Machine Learning project to enhance aviation safety through data.")
    st.write("""
    Welcome! This application uses a pre-trained machine learning model to predict aircraft damage severity. 
    By analyzing key factors from over 80,000 historical incidents, our system identifies risk patterns to help prevent future accidents.
    """)
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    col1.metric("Incidents Analyzed", "80,000+")
    col2.metric("Prediction Accuracy", f"{best_model_pack['performance']['Accuracy']:.2%}")
    col3.metric("Key Risk Factor", "Fatal Injuries")
    st.info("Navigate through the tabs to predict outcomes, review case studies, and explore historical data trends.")

# --- Live Prediction Page ---
elif selected == "Live Prediction":
    st.title("🚀 Live Aircraft Damage Prediction")
    st.markdown(f"Enter incident details below. Our **{best_model_name}** model will predict the outcome.")
    st.markdown("---")
    input_features = {}
    col1, col2, col3 = st.columns(3)
    options = chart_data['dropdown_options']
    with col1:
        input_features['Make'] = st.selectbox("Aircraft Make", options=[''] + options['Make'])
        input_features['Model'] = st.selectbox("Aircraft Model", options=[''] + options['Model'])
        input_features['Engine_Type'] = st.selectbox("Engine Type", options=[''] + options['Engine_Type'])
        input_features['Country'] = st.selectbox("Country", options=[''] + options['Country'])
    with col2:
        input_features['Weather_Condition'] = st.selectbox("Weather Condition", options=[''] + options['Weather_Condition'])
        input_features['Broad_phase_of_flight'] = st.selectbox("Phase of Flight", options=[''] + options['Broad_phase_of_flight'])
        input_features['Purpose_of_flight'] = st.selectbox("Purpose of Flight", options=[''] + options['Purpose_of_flight'])
        input_features['Number_of_Engines'] = st.number_input('Number of Engines', min_value=0, max_value=8, value=2)
    with col3:
        input_features['Year'] = st.number_input('Year of Incident', min_value=1940, max_value=2025, value=2024)
        input_features['Month'] = st.number_input('Month of Incident', min_value=1, max_value=12, value=7)
        input_features['Total_Fatal_Injuries'] = st.number_input('Total Fatal Injuries', min_value=0, value=0)
        input_features['Total_Serious_Injuries'] = st.number_input('Total Serious Injuries', min_value=0, value=0)
    st.markdown("---")
    if st.button("Predict Severity", type="primary"):
        cat_features = ['Make', 'Model', 'Engine_Type', 'Country', 'Weather_Condition', 'Broad_phase_of_flight', 'Purpose_of_flight']
        if any(input_features[key] == '' for key in cat_features):
            st.error("Please fill in all dropdown details before predicting.")
        else:
            input_df = pd.DataFrame([input_features])
            model_to_predict = best_model_pack['model']
            encoders = best_model_pack['encoders']
            
            for col, encoder in encoders.items():
                if col in input_df.columns:
                    val = input_df.iloc[0][col]
                    if val not in encoder.classes_: input_df.loc[0, col] = 'Unknown'
                    input_df[col] = encoder.transform(input_df[col])
            
            input_df = input_df[model_to_predict.get_booster().feature_names]
            
            # --- Prediction Logic with Probabilities ---
            prediction_encoded = model_to_predict.predict(input_df)[0]
            prediction_proba = model_to_predict.predict_proba(input_df)
            prediction_decoded = encoders['target'].inverse_transform([prediction_encoded])[0]
            
            # --- Display Results ---
            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.subheader("Prediction Result")
                if prediction_decoded == "Destroyed":
                    st.error(f"Predicted Damage Severity: **{prediction_decoded}**")
                elif prediction_decoded == "Substantial":
                    st.warning(f"Predicted Damage Severity: **{prediction_decoded}**")
                else:
                    st.success(f"Predicted Damage Severity: **{prediction_decoded}**")

            with res_col2:
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame(prediction_proba, columns=encoders['target'].classes_)
                st.dataframe(prob_df)

# --- Crash Case Studies Page ---
elif selected == "Crash Case Studies":
    st.title("📖 Crash Case Studies & Model Analysis")
    st.markdown("A review of notable aviation incidents. Click the button on any case to see how our model would have predicted the outcome based on the known data.")

    crashes = {
        "Tenerife Airport Disaster (1977)": {
            "summary": "**Date:** Mar 27, 1977 | **Aircraft:** Boeing 747 | **Fatalities:** 583\n\nThe deadliest accident in aviation history. Two Boeing 747s collided on the runway in heavy fog due to miscommunications.",
            "data": {'Make': 'Boeing', 'Model': '747-206B', 'Engine_Type': 'Turbofan', 'Number_of_Engines': 4, 'Weather_Condition': 'IMC', 'Broad_phase_of_flight': 'Takeoff', 'Purpose_of_flight': 'Scheduled Passenger', 'Country': 'Spain', 'Year': 1977, 'Month': 3, 'Total_Fatal_Injuries': 583, 'Total_Serious_Injuries': 61}
        },
        "Japan Airlines Flight 123 (1985)": {
            "summary": "**Date:** Aug 12, 1985 | **Aircraft:** Boeing 747SR | **Fatalities:** 520\n\nThe deadliest single-aircraft accident in history. A faulty repair led to an explosive decompression which destroyed all hydraulic controls.",
            "data": {'Make': 'Boeing', 'Model': '747SR-46', 'Engine_Type': 'Turbofan', 'Number_of_Engines': 4, 'Weather_Condition': 'VMC', 'Broad_phase_of_flight': 'Climb', 'Purpose_of_flight': 'Scheduled Passenger', 'Country': 'Japan', 'Year': 1985, 'Month': 8, 'Total_Fatal_Injuries': 520, 'Total_Serious_Injuries': 4}
        },
        "Indian Airlines Flight 113 (1988)": {
            "summary": "**Date:** Oct 19, 1988 | **Aircraft:** Boeing 737-200 | **Fatalities:** 133\n\nThe flight crashed on final approach to Ahmedabad in poor visibility, striking trees and a pylon. The investigation cited crew error.",
            "data": {'Make': 'Boeing', 'Model': '737-2A8', 'Engine_Type': 'Turbojet', 'Number_of_Engines': 2, 'Weather_Condition': 'IMC', 'Broad_phase_of_flight': 'Approach', 'Purpose_of_flight': 'Scheduled Passenger', 'Country': 'India', 'Year': 1988, 'Month': 10, 'Total_Fatal_Injuries': 133, 'Total_Serious_Injuries': 0}
        },
        "US Airways Flight 1549 (2009)": {
            "summary": "**Date:** Jan 15, 2009 | **Aircraft:** Airbus A320-214 | **Fatalities:** 0\n\nKnown as the 'Miracle on the Hudson,' the aircraft lost all engine power after a bird strike. The crew successfully ditched the plane on the Hudson River with no fatalities.",
            "data": {'Make': 'Airbus', 'Model': 'A320-214', 'Engine_Type': 'Turbofan', 'Number_of_Engines': 2, 'Weather_Condition': 'VMC', 'Broad_phase_of_flight': 'Climb', 'Purpose_of_flight': 'Scheduled Passenger', 'Country': 'United States', 'Year': 2009, 'Month': 1, 'Total_Fatal_Injuries': 0, 'Total_Serious_Injuries': 5}
        },
        "Air France Flight 447 (2009)": {
            "summary": "**Date:** June 1, 2009 | **Aircraft:** Airbus A330-203 | **Fatalities:** 228\n\nThe aircraft stalled at high altitude after its airspeed sensors (pitot tubes) froze over. The crew's incorrect response led to the crash in the Atlantic Ocean.",
            "data": {'Make': 'Airbus', 'Model': 'A330-203', 'Engine_Type': 'Turbofan', 'Number_of_Engines': 2, 'Weather_Condition': 'IMC', 'Broad_phase_of_flight': 'Cruise', 'Purpose_of_flight': 'Scheduled Passenger', 'Country': 'Unknown', 'Year': 2009, 'Month': 6, 'Total_Fatal_Injuries': 228, 'Total_Serious_Injuries': 0}
        },
        "Lion Air Flight 610 (2018)": {
            "summary": "**Date:** Oct 29, 2018 | **Aircraft:** Boeing 737 MAX 8 | **Fatalities:** 189\n\nOne of two fatal crashes involving the Boeing 737 MAX. A faulty angle of attack sensor activated the MCAS flight control system, repeatedly pushing the aircraft's nose down until the pilots lost control.",
            "data": {'Make': 'Boeing', 'Model': '737-8 (MAX)', 'Engine_Type': 'Turbofan', 'Number_of_Engines': 2, 'Weather_Condition': 'VMC', 'Broad_phase_of_flight': 'Climb', 'Purpose_of_flight': 'Scheduled Passenger', 'Country': 'Indonesia', 'Year': 2018, 'Month': 10, 'Total_Fatal_Injuries': 189, 'Total_Serious_Injuries': 0}
        },
        "Ethiopian Airlines Flight 302 (2019)": {
            "summary": "**Date:** Mar 10, 2019 | **Aircraft:** Boeing 737 MAX 8 | **Fatalities:** 157\n\nSimilar to the Lion Air crash, a faulty sensor activated the MCAS system, leading to the worldwide grounding of the 737 MAX fleet.",
            "data": {'Make': 'Boeing', 'Model': '737-8 (MAX)', 'Engine_Type': 'Turbofan', 'Number_of_Engines': 2, 'Weather_Condition': 'VMC', 'Broad_phase_of_flight': 'Climb', 'Purpose_of_flight': 'Scheduled Passenger', 'Country': 'Ethiopia', 'Year': 2019, 'Month': 3, 'Total_Fatal_Injuries': 157, 'Total_Serious_Injuries': 0}
        },
    }

    for crash_name, crash_info in crashes.items():
        with st.expander(f"✈️ **{crash_name}**"):
            st.markdown(crash_info['summary'])
            if st.button(f"Run Prediction for {crash_name}", key=crash_name):
                input_df = pd.DataFrame([crash_info['data']])
                model_to_predict = best_model_pack['model']
                encoders = best_model_pack['encoders']
                for col, encoder in encoders.items():
                    if col in input_df.columns:
                        val = input_df.iloc[0][col]
                        if val not in encoder.classes_: input_df.loc[0, col] = 'Unknown'
                        input_df[col] = encoder.transform(input_df[col])
                
                input_df = input_df[model_to_predict.get_booster().feature_names]
                
                prediction_encoded = model_to_predict.predict(input_df)[0]
                prediction_proba = model_to_predict.predict_proba(input_df)
                prediction_decoded = encoders['target'].inverse_transform([prediction_encoded])[0]
                
                res_col1, res_col2 = st.columns(2)
                with res_col1:
                    st.subheader("Model's Analysis:")
                    if prediction_decoded == "Destroyed":
                        st.error(f"Predicted Damage: **{prediction_decoded}**")
                    else:
                        st.warning(f"Predicted Damage: **{prediction_decoded}**")
                with res_col2:
                    st.subheader("Probabilities:")
                    prob_df = pd.DataFrame(prediction_proba, columns=encoders['target'].classes_)
                    st.dataframe(prob_df)

# --- Model Performance Page ---
elif selected == "Model Performance":
    st.title("📊 Model Performance & In-Depth Analysis")
    st.header("📈 Model Comparison")
    perf_list = []
    for name, info in models_info.items():
        p = info['performance']
        perf_list.append({'Model': name, 'Metric': 'Accuracy', 'Score': p['Accuracy']})
        perf_list.append({'Model': name, 'Metric': 'F1 Score', 'Score': p['F1 Score']})
    perf_df = pd.DataFrame(perf_list)
    fig_comp = px.bar(perf_df, x="Metric", y="Score", color="Model", barmode="group", title="Side-by-Side Model Performance Metrics")
    st.plotly_chart(fig_comp, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.header("Feature Importance")
        p = best_model_pack['performance']
        feature_imp_df = pd.DataFrame({'feature': p['Feature Names'], 'importance': p['Feature Importances']}).sort_values('importance', ascending=False).head(10)
        fig_imp = px.bar(feature_imp_df, x='importance', y='feature', orientation='h', title=f"Top 10 Features for {best_model_name}")
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)
    with col2:
        st.header("Confusion Matrix")
        cm = p['Confusion Matrix']
        classes = p['Classes']
        fig_cm, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, ax=ax)
        ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
        st.pyplot(fig_cm)

# --- Data Analysis Page ---
elif selected == "Data Analysis":
    st.title("🔍 Exploratory Data Analysis (EDA)")
    st.header("Dataset Overview")
    st.dataframe(chart_data['full_data_head'])
    with st.expander("See Full Dataset Statistics"):
        st.dataframe(chart_data['full_data_describe'])
    st.markdown("---")
    st.header("Visual Analysis")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Damage Severity Distribution")
        df_pie = chart_data['damage_distribution']
        fig_pie = px.pie(df_pie, values=df_pie.values, names=df_pie.index, title='Proportion of Aircraft Damage Severity', hole=0.3)
        st.plotly_chart(fig_pie, use_container_width=True)
    with col2:
        st.subheader("Incidents Over Time")
        df_line = chart_data['incidents_by_year']
        fig_line = px.line(x=df_line.index, y=df_line.values, labels={'x': 'Year', 'y': 'Number of Incidents'}, title='Total Aviation Incidents per Year')
        st.plotly_chart(fig_line, use_container_width=True)
    st.subheader("Damage Severity by Key Factors")
    tab1, tab2, tab3 = st.tabs(["Weather", "Phase of Flight", "Engine Type"])
    with tab1:
        df_weather = chart_data['damage_by_weather']
        fig_weather = px.bar(df_weather, x='Weather_Condition', y='count', color='Aircraft_damage', title='Damage Severity by Weather Condition', barmode='group')
        st.plotly_chart(fig_weather, use_container_width=True)
    with tab2:
        df_phase = chart_data['damage_by_phase']
        fig_phase = px.bar(df_phase, x='Broad_phase_of_flight', y='count', color='Aircraft_damage', title='Damage Severity by Phase of Flight', barmode='group')
        st.plotly_chart(fig_phase, use_container_width=True)
    with tab3:
        df_engine = chart_data['damage_by_engine']
        fig_engine = px.bar(df_engine, x='Engine_Type', y='count', color='Aircraft_damage', title='Damage Severity by Engine Type', barmode='group')
        st.plotly_chart(fig_engine, use_container_width=True)
    st.subheader("Geographical Distribution of Incidents")
    df_map = chart_data['country_counts']
    df_map.columns = ['Country', 'Incidents']
    fig_map = px.choropleth(df_map, locations='Country', locationmode='country names', color='Incidents', hover_name='Country', color_continuous_scale=px.colors.sequential.Plasma, title='Total Incidents by Country')
    st.plotly_chart(fig_map, use_container_width=True)
