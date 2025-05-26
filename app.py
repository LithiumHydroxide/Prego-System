import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import random
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Pregnancy Health AI System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-medium {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
    }
    .risk-low {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'patients_data' not in st.session_state:
    st.session_state.patients_data = []
if 'current_user_role' not in st.session_state:
    st.session_state.current_user_role = 'Admin'
if 'ai_model_trained' not in st.session_state:
    st.session_state.ai_model_trained = False

# Generate sample data
@st.cache_data
def generate_sample_data():
    np.random.seed(42)
    n_patients = 500
    
    data = []
    for i in range(n_patients):
        # Patient demographics
        age = np.random.normal(28, 6)
        age = max(16, min(45, age))
        
        # Risk factors
        previous_complications = np.random.choice([0, 1], p=[0.8, 0.2])
        hypertension = np.random.choice([0, 1], p=[0.85, 0.15])
        diabetes = np.random.choice([0, 1], p=[0.9, 0.1])
        bmi = np.random.normal(24, 4)
        bmi = max(16, min(40, bmi))
        
        # Clinical measurements
        systolic_bp = np.random.normal(120, 15)
        diastolic_bp = np.random.normal(80, 10)
        hemoglobin = np.random.normal(12, 1.5)
        
        # Geographic and social factors
        urban = np.random.choice([0, 1], p=[0.6, 0.4])
        education_level = np.random.choice(['Primary', 'Secondary', 'Tertiary'], p=[0.4, 0.4, 0.2])
        anc_visits = np.random.poisson(6)
        
        # Calculate risk score (simplified)
        risk_score = 0
        if age < 18 or age > 35: risk_score += 2
        if previous_complications: risk_score += 3
        if hypertension: risk_score += 2
        if diabetes: risk_score += 2
        if bmi > 30: risk_score += 1
        if systolic_bp > 140: risk_score += 2
        if hemoglobin < 11: risk_score += 1
        if anc_visits < 4: risk_score += 1
        
        # Determine outcomes
        high_risk = 1 if risk_score >= 5 else 0
        complications = np.random.choice([0, 1], p=[0.9, 0.1] if risk_score < 5 else [0.6, 0.4])
        
        patient = {
            'patient_id': f'P{i+1:04d}',
            'age': round(age),
            'gestational_age': np.random.randint(12, 40),
            'previous_complications': previous_complications,
            'hypertension': hypertension,
            'diabetes': diabetes,
            'bmi': round(bmi, 1),
            'systolic_bp': round(systolic_bp),
            'diastolic_bp': round(diastolic_bp),
            'hemoglobin': round(hemoglobin, 1),
            'urban': urban,
            'education_level': education_level,
            'anc_visits': anc_visits,
            'risk_score': risk_score,
            'high_risk': high_risk,
            'complications': complications,
            'hospital': f'Hospital_{np.random.choice(["A", "B", "C", "D", "E"])}',
            'county': np.random.choice(['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret']),
            'date_registered': datetime.now() - timedelta(days=np.random.randint(1, 365))
        }
        data.append(patient)
    
    return pd.DataFrame(data)

# Load sample data
df = generate_sample_data()

# Sidebar - User Role Selection
st.sidebar.markdown("### 👤 User Role")
user_roles = ['Admin', 'Hospital Data Officer', 'Health Worker', 'Researcher/Data Analyst', 'System Developer']
selected_role = st.sidebar.selectbox("Select your role:", user_roles)
st.session_state.current_user_role = selected_role

# Main header
st.markdown("""
<div class="main-header">
    <h1>🏥 Pregnancy Health AI System</h1>
    <p>AI-Powered Risk Prediction & Data Collection Platform</p>
</div>
""", unsafe_allow_html=True)

# Role-based interface
if selected_role == 'Admin':
    st.markdown("## 👑 Admin Dashboard")
    
    tab1, tab2, tab3, tab4 = st.tabs(["System Overview", "User Management", "AI Model Status", "System Health"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Patients", len(df), "↑ 23 this week")
        with col2:
            high_risk_count = df['high_risk'].sum()
            st.metric("High Risk Cases", high_risk_count, f"{high_risk_count/len(df)*100:.1f}%")
        with col3:
            complications_count = df['complications'].sum()
            st.metric("Complications", complications_count, f"↓ {random.randint(1,5)}")
        with col4:
            st.metric("Active Hospitals", df['hospital'].nunique(), "5 connected")
        
        # Geographic distribution
        st.markdown("### 📍 Geographic Distribution")
        county_data = df.groupby('county').agg({
            'patient_id': 'count',
            'high_risk': 'sum',
            'complications': 'sum'
        }).reset_index()
        county_data.columns = ['County', 'Total Patients', 'High Risk', 'Complications']
        
        fig_map = px.bar(county_data, x='County', y='Total Patients', 
                        color='High Risk', title="Patient Distribution by County")
        st.plotly_chart(fig_map, use_container_width=True)
    
    with tab2:
        st.markdown("### 👥 User Management")
        
        # User roles breakdown
        user_stats = pd.DataFrame({
            'Role': ['Hospital Data Officers', 'Health Workers', 'Researchers', 'System Developers'],
            'Active Users': [12, 45, 8, 3],
            'Permissions': ['Data Entry, View Reports', 'View Profiles, Update Observations', 
                          'Export Data, Analytics', 'System Maintenance']
        })
        st.dataframe(user_stats, use_container_width=True)
        
        # Add new user form
        with st.expander("Add New User"):
            col1, col2 = st.columns(2)
            with col1:
                new_user_name = st.text_input("Full Name")
                new_user_role = st.selectbox("Role", user_roles[1:])  # Exclude Admin
            with col2:
                new_user_email = st.text_input("Email")
                new_user_hospital = st.selectbox("Hospital", df['hospital'].unique())
            if st.button("Add User"):
                st.success(f"User {new_user_name} added successfully!")
    
    with tab3:
        st.markdown("### 🤖 AI Model Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model Accuracy", "87.3%", "↑ 2.1%")
            st.metric("Predictions Made", "1,247", "↑ 156 today")
        with col2:
            st.metric("Model Version", "v2.1.3", "Updated 3 days ago")
            st.metric("Training Data", f"{len(df)} records", "Updated daily")
        
        # Model performance visualization
        performance_data = pd.DataFrame({
            'Metric': ['Sensitivity', 'Specificity', 'Precision', 'F1-Score'],
            'Score': [0.85, 0.89, 0.82, 0.84]
        })
        fig_perf = px.bar(performance_data, x='Metric', y='Score', 
                         title="AI Model Performance Metrics")
        st.plotly_chart(fig_perf, use_container_width=True)
        
        if st.button("Retrain AI Model"):
            with st.spinner("Retraining model..."):
                import time
                time.sleep(3)
                st.success("Model retrained successfully! New accuracy: 88.1%")
    
    with tab4:
        st.markdown("### 💻 System Health")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Server Uptime", "99.8%", "↑ 0.1%")
        with col2:
            st.metric("API Response Time", "145ms", "↓ 12ms")
        with col3:
            st.metric("Database Size", "2.3GB", "↑ 45MB")
        
        # System logs
        st.markdown("### 📝 Recent System Logs")
        logs = [
            {"Timestamp": "2024-05-23 14:30", "Level": "INFO", "Message": "Daily backup completed successfully"},
            {"Timestamp": "2024-05-23 13:15", "Level": "INFO", "Message": "AI model predictions generated for 23 new patients"},
            {"Timestamp": "2024-05-23 12:45", "Level": "WARNING", "Message": "High memory usage detected on server"},
            {"Timestamp": "2024-05-23 11:20", "Level": "INFO", "Message": "New user registered: Dr. Jane Smith"},
        ]
        st.dataframe(pd.DataFrame(logs), use_container_width=True)

elif selected_role == 'Hospital Data Officer':
    st.markdown("## 📊 Hospital Data Officer Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Data Entry", "Patient Monitoring", "Verification"])
    
    with tab1:
        st.markdown("### ➕ New Patient Registration")
        
        with st.form("patient_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                patient_name = st.text_input("Patient Name")
                patient_age = st.number_input("Age", min_value=15, max_value=50, value=25)
                gestational_age = st.number_input("Gestational Age (weeks)", min_value=4, max_value=42, value=20)
                previous_complications = st.checkbox("Previous Pregnancy Complications")
                hypertension = st.checkbox("Hypertension")
            
            with col2:
                diabetes = st.checkbox("Diabetes")
                bmi = st.number_input("BMI", min_value=15.0, max_value=50.0, value=23.0)
                systolic_bp = st.number_input("Systolic BP", min_value=80, max_value=200, value=120)
                diastolic_bp = st.number_input("Diastolic BP", min_value=50, max_value=120, value=80)
                hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=6.0, max_value=18.0, value=12.0)
            
            submitted = st.form_submit_button("Register Patient & Get AI Prediction")
            
            if submitted:
                # Simple risk calculation for demo
                risk_factors = sum([
                    patient_age < 18 or patient_age > 35,
                    previous_complications,
                    hypertension,
                    diabetes,
                    bmi > 30,
                    systolic_bp > 140,
                    hemoglobin < 11
                ])
                
                if risk_factors >= 3:
                    risk_level = "HIGH"
                    risk_class = "risk-high"
                elif risk_factors >= 1:
                    risk_level = "MEDIUM"
                    risk_class = "risk-medium"
                else:
                    risk_level = "LOW"
                    risk_class = "risk-low"
                
                st.markdown(f"""
                <div class="{risk_class}">
                    <h4>🤖 AI Prediction Result</h4>
                    <p><strong>Risk Level:</strong> {risk_level}</p>
                    <p><strong>Risk Factors Detected:</strong> {risk_factors}</p>
                    <p><strong>Recommended Actions:</strong> {'Immediate specialist referral' if risk_level == 'HIGH' else 'Regular monitoring' if risk_level == 'MEDIUM' else 'Standard care protocol'}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.success("Patient registered successfully!")
    
    with tab2:
        st.markdown("### 📈 Patient Monitoring")
        
        # Filter patients by hospital
        hospital_filter = st.selectbox("Select Hospital", df['hospital'].unique())
        hospital_patients = df[df['hospital'] == hospital_filter]
        
        # High risk patients alert
        high_risk_patients = hospital_patients[hospital_patients['high_risk'] == 1]
        if len(high_risk_patients) > 0:
            st.warning(f"⚠️ {len(high_risk_patients)} high-risk patients require immediate attention!")
            
            # Show high risk patients
            st.markdown("#### High Risk Patients")
            risk_display = high_risk_patients[['patient_id', 'age', 'gestational_age', 'risk_score']].head(10)
            st.dataframe(risk_display, use_container_width=True)
        
        # Trends visualization
        st.markdown("#### Monthly Trends")
        monthly_trends = hospital_patients.groupby(hospital_patients['date_registered'].dt.month).agg({
            'patient_id': 'count',
            'high_risk': 'sum',
            'complications': 'sum'
        }).reset_index()
        monthly_trends['month_name'] = monthly_trends['date_registered'].apply(
            lambda x: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][x-1]
        )
        
        fig_trends = px.line(monthly_trends, x='month_name', y=['patient_id', 'high_risk'], 
                           title="Patient Registration and Risk Trends")
        st.plotly_chart(fig_trends, use_container_width=True)
    
    with tab3:
        st.markdown("### ✅ AI Prediction Verification")
        
        # Sample predictions to verify
        verification_data = df.sample(10)[['patient_id', 'high_risk', 'complications', 'risk_score']]
        verification_data['ai_prediction'] = verification_data['high_risk'].apply(
            lambda x: 'High Risk' if x else 'Low Risk'
        )
        verification_data['actual_outcome'] = verification_data['complications'].apply(
            lambda x: 'Complications' if x else 'Normal'
        )
        
        st.markdown("#### Recent AI Predictions to Verify")
        for idx, row in verification_data.iterrows():
            with st.expander(f"Patient ID: {row['patient_id']} - AI Prediction: {row['ai_prediction']}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Risk Score:** {row['risk_score']}")
                    st.write(f"**AI Prediction:** {row['ai_prediction']}")
                with col2:
                    verification_result = st.selectbox(
                        "Verify Prediction:", 
                        ["Correct", "Incorrect", "Partially Correct"], 
                        key=f"verify_{row['patient_id']}"
                    )
                    if st.button(f"Submit Verification", key=f"submit_{row['patient_id']}"):
                        st.success("Verification recorded. This will help improve the AI model.")

elif selected_role == 'Health Worker':
    st.markdown("## 👩‍⚕️ Health Worker Dashboard")
    
    tab1, tab2 = st.tabs(["Patient Profiles", "Clinical Updates"])
    
    with tab1:
        st.markdown("### 📋 Patient Risk Profiles")
        
        # Search functionality
        search_term = st.text_input("🔍 Search Patient ID or filter by risk level")
        risk_filter = st.selectbox("Filter by Risk Level", ["All", "High Risk", "Medium Risk", "Low Risk"])
        
        # Filter data
        filtered_df = df.copy()
        if search_term:
            filtered_df = filtered_df[filtered_df['patient_id'].str.contains(search_term, case=False)]
        if risk_filter != "All":
            if risk_filter == "High Risk":
                filtered_df = filtered_df[filtered_df['high_risk'] == 1]
            elif risk_filter == "Low Risk":
                filtered_df = filtered_df[filtered_df['high_risk'] == 0]
        
        # Display patient cards
        for idx, patient in filtered_df.head(10).iterrows():
            risk_level = "HIGH" if patient['high_risk'] else "LOW"
            risk_class = "risk-high" if patient['high_risk'] else "risk-low"
            
            st.markdown(f"""
            <div class="{risk_class}">
                <h4>Patient ID: {patient['patient_id']}</h4>
                <div style="display: flex; gap: 20px;">
                    <div>
                        <p><strong>Age:</strong> {patient['age']} years</p>
                        <p><strong>Gestational Age:</strong> {patient['gestational_age']} weeks</p>
                        <p><strong>Risk Level:</strong> {risk_level}</p>
                    </div>
                    <div>
                        <p><strong>BP:</strong> {patient['systolic_bp']}/{patient['diastolic_bp']} mmHg</p>
                        <p><strong>BMI:</strong> {patient['bmi']}</p>
                        <p><strong>Hemoglobin:</strong> {patient['hemoglobin']} g/dL</p>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Action buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button(f"📝 Update Observations", key=f"update_{patient['patient_id']}"):
                    st.info("Clinical observation update form would open here")
            with col2:
                if st.button(f"📞 Schedule Follow-up", key=f"schedule_{patient['patient_id']}"):
                    st.info("Follow-up scheduling interface would open here")
            with col3:
                if st.button(f"📊 View History", key=f"history_{patient['patient_id']}"):
                    st.info("Patient history timeline would display here")
            
            st.markdown("---")
    
    with tab2:
        st.markdown("### 📝 Clinical Observations Update")
        
        patient_id_update = st.selectbox("Select Patient for Update", df['patient_id'].head(20))
        selected_patient = df[df['patient_id'] == patient_id_update].iloc[0]
        
        st.markdown(f"#### Current Profile: {patient_id_update}")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Age:** {selected_patient['age']} years")
            st.write(f"**Gestational Age:** {selected_patient['gestational_age']} weeks")
            st.write(f"**Current Risk Level:** {'HIGH' if selected_patient['high_risk'] else 'LOW'}")
        with col2:
            st.write(f"**Hospital:** {selected_patient['hospital']}")
            st.write(f"**County:** {selected_patient['county']}")
            st.write(f"**ANC Visits:** {selected_patient['anc_visits']}")
        
        # Update form
        st.markdown("#### Add New Observations")
        with st.form("observation_form"):
            observation_date = st.date_input("Observation Date", datetime.now())
            vital_signs = st.text_area("Vital Signs & Measurements")
            clinical_notes = st.text_area("Clinical Notes")
            risk_changes = st.selectbox("Risk Level Changes", 
                                      ["No Change", "Increased Risk", "Decreased Risk"])
            follow_up_needed = st.checkbox("Requires Follow-up")
            
            if st.form_submit_button("Save Observations"):
                st.success("Clinical observations saved successfully!")
                st.info("AI model will be updated with new data for improved predictions.")

elif selected_role == 'Researcher/Data Analyst':
    st.markdown("## 📊 Research & Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Data Export", "Trend Analysis", "Model Performance"])
    
    with tab1:
        st.markdown("### 📥 Data Export & Download")
        
        # Export filters
        col1, col2, col3 = st.columns(3)
        with col1:
            export_hospital = st.multiselect("Select Hospitals", df['hospital'].unique(), default=df['hospital'].unique())
        with col2:
            export_county = st.multiselect("Select Counties", df['county'].unique(), default=df['county'].unique())
        with col3:
            date_range = st.date_input("Date Range", [datetime.now() - timedelta(days=365), datetime.now()])
        
        # Apply filters
        export_data = df[
            (df['hospital'].isin(export_hospital)) & 
            (df['county'].isin(export_county))
        ]
        
        st.markdown(f"**Filtered Dataset:** {len(export_data)} records")
        
        # Preview data
        st.markdown("#### Data Preview")
        st.dataframe(export_data.head(10), use_container_width=True)
        
        # Export options
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("📊 Export to CSV"):
                csv = export_data.to_csv(index=False)
                st.download_button("Download CSV", csv, "pregnancy_data.csv", "text/csv")
        with col2:
            if st.button("📈 Export to Excel"):
                st.info("Excel export functionality would be implemented here")
        with col3:
            if st.button("🔗 Generate API Key"):
                api_key = f"pk_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=32))}"
                st.code(f"API Key: {api_key}")
    
    with tab2:
        st.markdown("### 📈 Trend Analysis")
        
        # Risk factors analysis
        risk_factors = ['age', 'hypertension', 'diabetes', 'previous_complications', 'bmi']
        correlation_data = df[risk_factors + ['high_risk']].corr()['high_risk'].drop('high_risk')
        
        fig_corr = px.bar(x=correlation_data.index, y=correlation_data.values,
                         title="Risk Factor Correlation with High-Risk Pregnancies")
        fig_corr.update_xaxes(title="Risk Factors")
        fig_corr.update_yaxes(title="Correlation Coefficient")
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Geographic analysis
        st.markdown("#### Geographic Risk Analysis")
        geo_analysis = df.groupby('county').agg({
            'high_risk': ['count', 'sum', 'mean'],
            'complications': ['sum', 'mean']
        }).round(3)
        geo_analysis.columns = ['Total_Patients', 'High_Risk_Count', 'High_Risk_Rate', 
                               'Complications_Count', 'Complications_Rate']
        st.dataframe(geo_analysis, use_container_width=True)
        
        # Age distribution analysis
        fig_age = px.histogram(df, x='age', color='high_risk', 
                              title="Age Distribution by Risk Level", nbins=20)
        st.plotly_chart(fig_age, use_container_width=True)
    
    with tab3:
        st.markdown("### 🤖 AI Model Performance Analysis")
        
        # Confusion matrix simulation
        from sklearn.metrics import confusion_matrix, classification_report
        
        # Generate some predictions for demo
        y_true = df['high_risk'].values
        y_pred = np.random.choice([0, 1], size=len(y_true), p=[0.8, 0.2])
        
        # Make predictions more realistic
        for i in range(len(y_true)):
            if y_true[i] == 1:  # If actually high risk
                y_pred[i] = np.random.choice([0, 1], p=[0.2, 0.8])  # 80% chance of correct prediction
            else:  # If actually low risk
                y_pred[i] = np.random.choice([0, 1], p=[0.9, 0.1])  # 90% chance of correct prediction
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, aspect="auto",
                          title="Confusion Matrix - AI Model Performance",
                          labels=dict(x="Predicted", y="Actual"))
        st.plotly_chart(fig_cm, use_container_width=True)
        
        # Performance metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{report['accuracy']:.3f}")
        with col2:
            st.metric("Precision", f"{report['1']['precision']:.3f}")
        with col3:
            st.metric("Recall", f"{report['1']['recall']:.3f}")
        with col4:
            st.metric("F1-Score", f"{report['1']['f1-score']:.3f}")
        
        # Feature importance (simulated)
        features = ['Age', 'BMI', 'Blood Pressure', 'Hemoglobin', 'Previous Complications', 
                   'Gestational Age', 'ANC Visits', 'Education Level']
        importance = np.random.dirichlet(np.ones(len(features)))
        
        fig_importance = px.bar(x=features, y=importance, 
                               title="Feature Importance in AI Model")
        fig_importance.update_xaxes(title="Features")
        fig_importance.update_yaxes(title="Importance Score")
        st.plotly_chart(fig_importance, use_container_width=True)

elif selected_role == 'System Developer':
    st.markdown("## 💻 System Developer Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["System Status", "API Management", "Database"])
    
    with tab1:
        st.markdown("### ⚙️ System Status & Monitoring")
        
        # System metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("CPU Usage", "45%", "↓ 5%")
        with col2:
            st.metric("Memory Usage", "67%", "↑ 3%")
        with col3:
            st.metric("Disk Usage", "34%", "↑ 1%")
        with col4:
            st.metric("Active Users", "23", "↑ 2")
        
        # System performance over time
        performance_data = pd.DataFrame({
            'Time': pd.date_range(start='2024-05-20', periods=72, freq='H'),
            'CPU': np.random.normal(45, 10, 72),
            'Memory': np.random.normal(67, 15, 72),
            'Response_Time': np.random.normal(145, 30, 72)
        })
        
        fig_perf = px.line(performance_data, x='Time', y=['CPU', 'Memory'], 
                          title="System Performance Over Time")
        st.plotly_chart(fig_perf, use_container_width=True)
        
        # Service status
        services = [
            {"Service": "Web Application", "Status": "🟢 Running", "Uptime": "99.8%"},
            {"Service": "AI Model API", "Status": "🟢 Running", "Uptime": "99.5%"},
            {"Service": "Database", "Status": "🟢 Running", "Uptime": "99.9%"},
            {"Service": "File Storage", "Status": "🟡 Warning", "Uptime": "98.2%"},
            {"Service": "Notification Service", "Status": "🟢 Running", "Uptime": "99.6%"}
        ]
        st.dataframe(pd.DataFrame(services), use_container_width=True)
        
        # Recent system events
        st.markdown("#### Recent System Events")
        events = [
            {"Timestamp": "2024-05-23 15:45", "Event": "Database backup completed", "Severity": "INFO"},
            {"Timestamp": "2024-05-23 14:20", "Event": "High memory usage alert resolved", "Severity": "WARNING"},
            {"Timestamp": "2024-05-23 13:10", "Event": "AI model retrained successfully", "Severity": "INFO"},
            {"Timestamp": "2024-05-23 12:30", "Event": "New hospital connection established", "Severity": "INFO"}
        ]
        st.dataframe(pd.DataFrame(events), use_container_width=True)
    
    with tab2:
        st.markdown("### 🔌 API Management")
        
        # API endpoints
        st.markdown("#### Active API Endpoints")
        endpoints = [
            {"Endpoint": "/api/v1/patients", "Method": "GET, POST", "Status": "Active", "Rate Limit": "1000/hour"},
            {"Endpoint": "/api/v1/predictions", "Method": "POST", "Status": "Active", "Rate Limit": "500/hour"},
            {"Endpoint": "/api/v1/hospitals", "Method": "GET", "Status": "Active", "Rate Limit": "2000/hour"},
            {"Endpoint": "/api/v1/export", "Method": "GET", "Status": "Active", "Rate Limit": "100/hour"},
            {"Endpoint": "/api/v1/analytics", "Method": "GET", "Status": "Active", "Rate Limit": "200/hour"}
        ]
        st.dataframe(pd.DataFrame(endpoints), use_container_width=True)
        
        # API usage statistics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total API Calls (Today)", "15,847", "↑ 8%")
            st.metric("Failed Requests", "23", "↓ 12")
        with col2:
            st.metric("Average Response Time", "142ms", "↓ 8ms")
            st.metric("Active API Keys", "34", "↑ 2")
        
        # API usage chart
        api_usage = pd.DataFrame({
            'Hour': range(24),
            'Requests': np.random.poisson(600, 24),
            'Errors': np.random.poisson(5, 24)
        })
        fig_api = px.line(api_usage, x='Hour', y=['Requests', 'Errors'], 
                         title="API Usage Over Last 24 Hours")
        st.plotly_chart(fig_api, use_container_width=True)
        
        # Generate new API key
        st.markdown("#### Generate New API Key")
        with st.form("api_key_form"):
            key_name = st.text_input("API Key Name")
            key_permissions = st.multiselect("Permissions", 
                                           ["Read Patients", "Write Patients", "AI Predictions", "Export Data", "Analytics"])
            rate_limit = st.selectbox("Rate Limit", ["100/hour", "500/hour", "1000/hour", "2000/hour"])
            
            if st.form_submit_button("Generate API Key"):
                new_key = f"pk_{''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=32))}"
                st.success("API Key generated successfully!")
                st.code(f"API Key: {new_key}")
                st.warning("⚠️ Save this key securely. It won't be shown again.")
    
    with tab3:
        st.markdown("### 🗄️ Database Management")
        
        # Database statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}", "↑ 156")
        with col2:
            st.metric("Database Size", "2.34 GB", "↑ 45MB")
        with col3:
            st.metric("Daily Backups", "✅ 7/7", "All successful")
        with col4:
            st.metric("Query Performance", "23ms avg", "↓ 5ms")
        
        # Table statistics
        st.markdown("#### Table Statistics")
        table_stats = pd.DataFrame({
            'Table Name': ['patients', 'predictions', 'hospitals', 'users', 'audit_logs'],
            'Record Count': [len(df), len(df)*2, 5, 68, 15847],
            'Size (MB)': [245, 189, 1, 12, 89],
            'Last Updated': ['2 min ago', '5 min ago', '1 day ago', '3 hours ago', '1 min ago']
        })
        st.dataframe(table_stats, use_container_width=True)
        
        # Database operations
        st.markdown("#### Database Operations")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Manual Backup"):
                with st.spinner("Creating backup..."):
                    import time
                    time.sleep(2)
                    st.success("Backup created successfully!")
        
        with col2:
            if st.button("🧹 Cleanup Old Records"):
                st.info("This would remove records older than retention policy")
        
        with col3:
            if st.button("📊 Analyze Performance"):
                st.info("Database performance analysis would run here")
        
        # Query monitor
        st.markdown("#### Recent Database Queries")
        queries = [
            {"Time": "15:42:33", "Query": "SELECT * FROM patients WHERE high_risk = 1", "Duration": "12ms", "Status": "✅"},
            {"Time": "15:41:28", "Query": "INSERT INTO predictions (patient_id, risk_score)", "Duration": "8ms", "Status": "✅"},
            {"Time": "15:40:15", "Query": "UPDATE patients SET last_visit = NOW()", "Duration": "45ms", "Status": "✅"},
            {"Time": "15:39:02", "Query": "SELECT COUNT(*) FROM hospitals", "Duration": "3ms", "Status": "✅"}
        ]
        st.dataframe(pd.DataFrame(queries), use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>Improving maternal health outcomes through AI-powered risk prediction</p>
</div>
""", unsafe_allow_html=True)

# Sidebar additional info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 System Info")
st.sidebar.info(f"""
**Current Role:** {selected_role}
**Total Patients:** {len(df):,}
**High Risk Cases:** {df['high_risk'].sum()}
**Active Hospitals:** {df['hospital'].nunique()}
**System Status:** 🟢 Operational
""")

st.sidebar.markdown("### 🔧 Quick Actions")
if st.sidebar.button("🚨 Emergency Alert"):
    st.sidebar.error("Emergency alert sent to all health workers!")

if st.sidebar.button("📊 Generate Report"):
    st.sidebar.success("Monthly report generation started!")

if st.sidebar.button("🔄 Refresh Data"):
    st.sidebar.success("Data refreshed successfully!")

# Model training simulation
if not st.session_state.ai_model_trained:
    if st.sidebar.button("🤖 Train AI Model"):
        with st.sidebar:
            with st.spinner("Training AI model..."):
                import time
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.02)
                    progress_bar.progress(i + 1)
                st.success("AI model trained successfully!")
                st.session_state.ai_model_trained = True