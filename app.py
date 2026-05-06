import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────
# Page Config
# ─────────────────────────────────────
st.set_page_config(
    page_title="Alzheimer's Detection System",
    page_icon="UoH",
    layout="wide"
)

# ─────────────────────────────────────
# Paths
# ─────────────────────────────────────
PREPROCESSED = r'C:\Users\nbush\OneDrive - University of Huddersfield\Desktop\FYP\preprocessed'

# ─────────────────────────────────────
# Load ML Models
# ─────────────────────────────────────
@st.cache_resource
def load_ml_models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from xgboost import XGBClassifier
    from sklearn.preprocessing import StandardScaler

    # Scaled data — model training ke liye
    X_train_scaled = pd.read_csv(
        f'{PREPROCESSED}\\X_train_tabular.csv'
    )
    y_train = pd.read_csv(
        f'{PREPROCESSED}\\y_train_tabular.csv'
    ).squeeze()

    # Original unscaled data
    original_path = (
        r'C:\Users\nbush\OneDrive - University of '
        r'Huddersfield\Desktop\FYP\data_tabular'
        r'\alzheimers_disease_data.csv'
    )
    df_original = pd.read_csv(original_path)

    # Scaler — original data par fit karo
    feature_cols = X_train_scaled.columns.tolist()

    # Drop target column
    df_features = df_original.drop(
        columns=['Diagnosis', 'PatientID',
                 'DoctorInCharge'],
        errors='ignore'
    )[feature_cols]

    scaler = StandardScaler()
    scaler.fit(df_features)

    models = {
        'Logistic Regression': LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            class_weight='balanced',
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            scale_pos_weight=2,
            random_state=42,
            eval_metric='logloss'
        ),
        'SVM': SVC(
            kernel='rbf',
            class_weight='balanced',
            probability=True,
            random_state=42
        )
    }

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)

    return models, feature_cols, scaler
# ─────────────────────────────────────
# Load VGG16 Binary Model
# ─────────────────────────────────────
@st.cache_resource
def load_vgg16_binary():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            f'{PREPROCESSED}\\best_binary_vgg16.keras'
        )
        return model
    except:
        return None

# ─────────────────────────────────────
# Load VGG16 4-Class Model
# ─────────────────────────────────────
@st.cache_resource
def load_vgg16_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            f'{PREPROCESSED}\\best_vgg16_model.keras'
        )
        return model
    except:
        return None

# ─────────────────────────────────────
# Load CNN Model
# ─────────────────────────────────────
@st.cache_resource
def load_cnn_model():
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(
            f'{PREPROCESSED}\\best_cnn_model.keras'
        )
        return model
    except:
        return None

# ─────────────────────────────────────
# Header
# ─────────────────────────────────────
st.title("Alzheimer's Disease Detection System")
st.markdown(
    "**University of Huddersfield"
    " — Final Year Project 2026**"
)
st.markdown("---")

# ─────────────────────────────────────
# Tabs
# ─────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "Patient Data — ML Prediction",
    "MRI Image — DL Prediction",
    "Model Comparison"
])

# ═══════════════════════════════════════
# TAB 1 — Patient Data
# ═══════════════════════════════════════
with tab1:
    st.header("Patient Clinical Data — ML Prediction")
    st.markdown(
        "Enter patient information to "
        "predict Alzheimer's diagnosis."
    )

    with st.spinner("Loading ML models..."):
        ml_models, feature_names, scaler = load_ml_models()

    selected_model = st.selectbox(
        "Select ML Model:",
        list(ml_models.keys())
    )

    st.markdown("### Patient Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        age            = st.slider(
            "Age", 60, 90, 75,
            key="age")
        mmse           = st.slider(
            "MMSE Score", 0, 30, 24,
            key="mmse")
        functional     = st.slider(
            "Functional Assessment", 0, 10, 5,
            key="functional")
        adl            = st.slider(
            "ADL Score", 0, 10, 5,
            key="adl")
        memory         = st.selectbox(
            "Memory Complaints", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="memory")
        confusion      = st.selectbox(
            "Confusion", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="confusion")
        disorientation = st.selectbox(
            "Disorientation", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="disorientation")
        forgetfulness  = st.selectbox(
            "Forgetfulness", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="forgetfulness")

    with col2:
        gender      = st.selectbox(
            "Gender", [0, 1],
            format_func=lambda x:
            "Male" if x == 1 else "Female",
            key="gender")
        bmi         = st.slider(
            "BMI", 15.0, 40.0, 27.0,
            key="bmi")
        behavioral  = st.selectbox(
            "Behavioral Problems", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="behavioral")
        depression  = st.selectbox(
            "Depression", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="depression")
        smoking     = st.selectbox(
            "Smoking", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="smoking")
        personality = st.selectbox(
            "Personality Changes", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="personality")
        difficulty  = st.selectbox(
            "Difficulty Completing Tasks",
            [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="difficulty")
        family      = st.selectbox(
            "Family History Alzheimers",
            [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="family")

    with col3:
        systolic_bp  = st.slider(
            "Systolic BP", 90, 180, 130,
            key="systolic")
        diastolic_bp = st.slider(
            "Diastolic BP", 60, 120, 80,
            key="diastolic")
        cholesterol  = st.slider(
            "Cholesterol Total", 150, 300, 200,
            key="cholesterol")
        sleep        = st.slider(
            "Sleep Quality", 0, 10, 5,
            key="sleep")
        diet         = st.slider(
            "Diet Quality", 0, 10, 5,
            key="diet")
        alcohol      = st.slider(
            "Alcohol Consumption", 0, 10, 3,
            key="alcohol")
        physical     = st.slider(
            "Physical Activity", 0, 10, 5,
            key="physical")
        cardio       = st.selectbox(
            "Cardiovascular Disease", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="cardio")
        diabetes     = st.selectbox(
            "Diabetes", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="diabetes")
        head         = st.selectbox(
            "Head Injury", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="head")
        hypertension = st.selectbox(
            "Hypertension", [0, 1],
            format_func=lambda x:
            "Yes" if x == 1 else "No",
            key="hypertension")

    # Known values dictionary
    known = {
        'Age'                      : age,
        'Gender'                   : gender,
        'BMI'                      : bmi,
        'Smoking'                  : smoking,
        'AlcoholConsumption'       : alcohol,
        'PhysicalActivity'         : physical,
        'DietQuality'              : diet,
        'SleepQuality'             : sleep,
        'FamilyHistoryAlzheimers'  : family,
        'CardiovascularDisease'    : cardio,
        'Diabetes'                 : diabetes,
        'Depression'               : depression,
        'HeadInjury'               : head,
        'Hypertension'             : hypertension,
        'SystolicBP'               : systolic_bp,
        'DiastolicBP'              : diastolic_bp,
        'CholesterolTotal'         : cholesterol,
        'MMSE'                     : mmse,
        'FunctionalAssessment'     : functional,
        'MemoryComplaints'         : memory,
        'BehavioralProblems'       : behavioral,
        'ADL'                      : adl,
        'Confusion'                : confusion,
        'Disorientation'           : disorientation,
        'PersonalityChanges'       : personality,
        'DifficultyCompletingTasks': difficulty,
        'Forgetfulness'            : forgetfulness,
    }

    # Predict Button
    if st.button(
            "Predict Alzheimer's",
            type="primary",
            use_container_width=True,
            key="predict_btn"
    ):
        # Original data se mean values
        original_path = (
            r'C:\Users\nbush\OneDrive - University of '
            r'Huddersfield\Desktop\FYP\data_tabular'
            r'\alzheimers_disease_data.csv'
        )
        df_orig = pd.read_csv(original_path)
        df_orig = df_orig.drop(
            columns=['Diagnosis', 'PatientID',
                     'DoctorInCharge'],
            errors='ignore'
        )

        # Mean se fill karo
        input_data = pd.DataFrame(
            [df_orig[feature_names].mean()],
            columns=feature_names
        )

        # User values override karo
        for col, val in known.items():
            if col in input_data.columns:
                input_data[col] = val

        # Scale karo
        input_scaled = scaler.transform(input_data)
        input_scaled = pd.DataFrame(
            input_scaled,
            columns=feature_names
        )

        # Predict
        model = ml_models[selected_model]
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(
            input_scaled
        )[0]

        # Result
        st.markdown("---")
        st.markdown("### Prediction Result")

        col_r1, col_r2 = st.columns(2)

        with col_r1:
            if prediction == 1:
                st.error(
                    "## DEMENTED\n"
                    "Alzheimer's indicators detected"
                )
            else:
                st.success(
                    "## NON-DEMENTED\n"
                    "No significant indicators"
                )

        with col_r2:
            st.metric(
                "Confidence",
                f"{max(probability)*100:.1f}%"
            )
            st.metric(
                "Model Used",
                selected_model
            )

        # Probability chart
        st.markdown("### Probability Distribution")
        prob_df = pd.DataFrame({
            'Class'      : ['Non-Demented',
                            'Demented'],
            'Probability': [probability[0]*100,
                            probability[1]*100]
        })

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.barh(
            prob_df['Class'],
            prob_df['Probability'],
            color=['#4ECDC4', '#FF6B6B']
        )
        ax.set_xlabel('Probability (%)')
        ax.set_xlim(0, 100)
        for i, v in enumerate(
                prob_df['Probability']):
            ax.text(
                v + 1, i,
                f'{v:.1f}%',
                va='center',
                fontweight='bold'
            )
        ax.set_title('Prediction Probabilities')
        plt.tight_layout()
        st.pyplot(fig)

        st.warning(
            "Disclaimer: This tool is for "
            "educational purposes only. "
            "Always consult a medical professional."
        )

# ═══════════════════════════════════════
# TAB 2 — MRI Image
# ═══════════════════════════════════════
with tab2:
    st.header("MRI Brain Scan — Deep Learning")
    st.markdown(
        "Upload an MRI brain scan image "
        "for Alzheimer's classification."
    )

    classes = [
        'MildDemented', 'ModerateDemented',
        'NonDemented', 'VeryMildDemented'
    ]

    dl_model_choice = st.radio(
        "Select Deep Learning Model:",
        ["CNN 4-Class (56.67%)",
         "VGG16 4-Class (75.94%)",
         "VGG16 2-Class (91.46%)"],
        horizontal=True
    )

    uploaded_image = st.file_uploader(
        "Upload MRI Image",
        type=['jpg', 'jpeg', 'png']
    )

    if uploaded_image is not None:
        from PIL import Image
        import tensorflow as tf

        col_img, col_res = st.columns(2)

        with col_img:
            st.image(
                uploaded_image,
                caption="Uploaded MRI Scan",
                use_column_width=True
            )

        # Load model
        with st.spinner("Loading model..."):
            if "VGG16 2-Class" in dl_model_choice:
                model      = load_vgg16_binary()
                model_name = "VGG16 2-Class"
                is_binary  = True
            elif "VGG16 4-Class" in dl_model_choice:
                model      = load_vgg16_model()
                model_name = "VGG16 4-Class"
                is_binary  = False
            else:
                model      = load_cnn_model()
                model_name = "CNN 4-Class"
                is_binary  = False

        if model is not None:
            # Preprocess image
            img       = Image.open(
                uploaded_image
            ).convert('RGB')
            img       = img.resize((128, 128))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(
                img_array, axis=0
            )

            # Predict
            with st.spinner("Analyzing..."):
                predictions = model.predict(img_array)

                if is_binary:
                    prob       = float(
                        predictions[0][0]
                    )
                    pred_class = int(prob > 0.5)
                    confidence = prob * 100 \
                        if pred_class == 1 \
                        else (1 - prob) * 100
                    pred_label = 'Demented' \
                        if pred_class == 1 \
                        else 'NonDemented'
                else:
                    pred_class = np.argmax(predictions)
                    confidence = predictions[0][
                        pred_class
                    ] * 100
                    pred_label = classes[pred_class]

            with col_res:
                st.markdown("### Result")

                if is_binary:
                    if pred_label == 'NonDemented':
                        st.success(
                            "## NonDemented\n"
                            "No Alzheimer's indicators"
                        )
                    else:
                        st.error(
                            "## Demented\n"
                            "Alzheimer's indicators detected"
                        )
                else:
                    if pred_label == 'NonDemented':
                        st.success(
                            f"## {pred_label}"
                        )
                    elif pred_label == \
                            'VeryMildDemented':
                        st.info(
                            f"## {pred_label}"
                        )
                    elif pred_label == 'MildDemented':
                        st.warning(
                            f"## {pred_label}"
                        )
                    else:
                        st.error(
                            f"## {pred_label}"
                        )

                st.metric(
                    "Confidence",
                    f"{confidence:.1f}%"
                )
                st.metric("Model", model_name)

            # Probability chart
            st.markdown("### All Class Probabilities")

            if is_binary:
                prob_val  = float(predictions[0][0])
                prob_data = pd.DataFrame({
                    'Class': [
                        'NonDemented',
                        'Demented'
                    ],
                    'Probability (%)': [
                        (1 - prob_val) * 100,
                        prob_val * 100
                    ]
                })
            else:
                prob_data = pd.DataFrame({
                    'Class': classes,
                    'Probability (%)': [
                        p * 100
                        for p in predictions[0]
                    ]
                }).sort_values(
                    'Probability (%)',
                    ascending=False
                )

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.bar(
                prob_data['Class'],
                prob_data['Probability (%)'],
                color=['#96CEB4', '#FF6B6B',
                       '#4ECDC4', '#45B7D1'],
                edgecolor='black'
            )
            ax2.set_ylabel('Probability (%)')
            ax2.set_ylim(0, 100)
            ax2.tick_params(axis='x', rotation=15)
            ax2.set_title(
                f'{model_name} — Class Probabilities'
            )
            plt.tight_layout()
            st.pyplot(fig2)

        else:
            st.error(
                "Model load nahi hua! "
                "Check preprocessed folder."
            )

        st.warning(
            "Disclaimer: Educational purposes only."
        )

# ═══════════════════════════════════════
# TAB 3 — Comparison
# ═══════════════════════════════════════
with tab3:
    st.header("Model Comparison & Results")

    st.markdown("### All Models Performance")

    results_data = {
        'Model': [
            'XGBoost', 'Random Forest',
            'SVM', 'Logistic Regression',
            'VGG16 4-Class', 'CNN 4-Class',
            'VGG16 2-Class', 'CNN 2-Class'
        ],
        'Type': [
            'Classical ML', 'Classical ML',
            'Classical ML', 'Classical ML',
            'Deep Learning', 'Deep Learning',
            'Deep Learning', 'Deep Learning'
        ],
        'Dataset': [
            'Tabular', 'Tabular',
            'Tabular', 'Tabular',
            'MRI 4-Class', 'MRI 4-Class',
            'MRI 2-Class', 'MRI 2-Class'
        ],
        'Accuracy (%)': [
            94.88, 92.79, 84.19, 81.63,
            75.94, 56.67, 91.46, 67.92
        ],
        'F1-Score (%)': [
            94.87, 92.70, 84.38, 81.96,
            76.11, 56.51, 91.45, 67.92
        ],
        'ROC-AUC (%)': [
            94.39, 94.46, 89.74, 88.31,
            90.83, 72.50, 97.47, 73.54
        ]
    }

    results_df = pd.DataFrame(results_data)

    st.dataframe(
        results_df.style.background_gradient(
            subset=['Accuracy (%)',
                    'F1-Score (%)',
                    'ROC-AUC (%)'],
            cmap='YlOrRd'
        ),
        use_container_width=True
    )

    # Key findings
    st.markdown("### Key Findings")

    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        st.success(
            "**Best Overall Model**\n\n"
            "XGBoost — 94.88%\n\n"
            "Classical ML on Tabular Data"
        )

    with col_f2:
        st.info(
            "**Best Image Model**\n\n"
            "VGG16 2-Class — 91.46%\n\n"
            "Transfer Learning — Binary"
        )

    with col_f3:
        st.warning(
            "**Key Finding**\n\n"
            "Tabular ML > Image DL\n\n"
            "Clinical data more informative"
        )

    # Graphs
    st.markdown("### Performance Charts")

    graph_path = r'C:\Users\nbush\OneDrive - University of Huddersfield\Desktop\FYP\notebooks'

    col_g1, col_g2 = st.columns(2)

    with col_g1:
        heatmap_path = os.path.join(
            graph_path, 'performance_heatmap.png'
        )
        if os.path.exists(heatmap_path):
            st.image(
                heatmap_path,
                caption="Performance Heatmap",
                use_column_width=True
            )

    with col_g2:
        comparison_path = os.path.join(
            graph_path, 'tabular_vs_image.png'
        )
        if os.path.exists(comparison_path):
            st.image(
                comparison_path,
                caption="Tabular vs Image DL",
                use_column_width=True
            )

    # SHAP
    st.markdown("### SHAP Feature Importance")

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        shap_bar_path = os.path.join(
            graph_path, 'shap_bar.png'
        )
        if os.path.exists(shap_bar_path):
            st.image(
                shap_bar_path,
                caption="SHAP Bar Plot",
                use_column_width=True
            )

    with col_s2:
        shap_dot_path = os.path.join(
            graph_path, 'shap_dot.png'
        )
        if os.path.exists(shap_dot_path):
            st.image(
                shap_dot_path,
                caption="SHAP Summary",
                use_column_width=True
            )

    st.markdown("---")
    st.markdown(
        "**University of Huddersfield "
        "— FYP 2026** | "
        "Alzheimer's Detection using ML"
    )