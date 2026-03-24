import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Exoplanet Classification Analysis",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">🌌 Exoplanet Classification Analysis</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Data Upload & Cleaning", "Data Exploration", "Model Training", "Results"])

# Initialize session state
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}

def load_and_clean_data(uploaded_files):
    """Load and clean the uploaded data files"""
    dataframes = []
    
    for uploaded_file in uploaded_files:
        # Try to read with different skip rows based on file name
        if 'cumulative' in uploaded_file.name.lower():
            df = pd.read_csv(uploaded_file, skiprows=53)
        elif 'k2' in uploaded_file.name.lower():
            df = pd.read_csv(uploaded_file, skiprows=98)
        elif 'toi' in uploaded_file.name.lower():
            df = pd.read_csv(uploaded_file, skiprows=69)
        else:
            df = pd.read_csv(uploaded_file)
        dataframes.append(df)
    
    # Merge datasets
    if len(dataframes) > 1:
        merged_df = pd.concat(dataframes, ignore_index=True)
    else:
        merged_df = dataframes[0]
    
    return merged_df

def clean_data(df):
    """Clean and preprocess the data"""
    # Keep important columns (only if they exist)
    keep_cols = ['pl_name', 'hostname', 'default_flag', 'disposition', 'disp_refname']
    keep_cols = [col for col in keep_cols if col in df.columns]  # Filter existing columns
    
    # Keep columns with at least 40% data
    threshold = len(df) * 0.4
    cols_to_keep = [col for col in df.columns if df[col].count() >= threshold]
    usable_cols = list(set(keep_cols).union(set(cols_to_keep)))
    
    # Filter dataframe
    df_filtered = df[usable_cols].copy()
    
    # Handle numerical columns
    num_cols = df_filtered.select_dtypes(include=['float64', 'int64']).columns
    num_cols = [col for col in num_cols if col not in keep_cols]  # Exclude non-numeric keep_cols
    
    for col in num_cols:
        median_val = df_filtered[col].median()
        missing_flag_col = col + '_missing'
        df_filtered[missing_flag_col] = df_filtered[col].isnull().astype(int)
        df_filtered[col] = df_filtered[col].fillna(median_val)
    
    # Handle categorical columns
    cat_cols = df_filtered.select_dtypes(include=['object']).columns
    exclude_cols = ['disposition', 'pl_name', 'hostname', 'disp_refname']
    cat_cols = [col for col in cat_cols if col not in exclude_cols]
    
    for col in cat_cols:
        if not df_filtered[col].mode().empty:
            mode_val = df_filtered[col].mode()[0]
            df_filtered[col] = df_filtered[col].fillna(mode_val)
    
    return df_filtered

# Page 1: Data Upload & Cleaning
if page == "Data Upload & Cleaning":
    st.markdown('<h2 class="sub-header">📁 Data Upload & Cleaning</h2>', unsafe_allow_html=True)
    
    # File upload
    uploaded_files = st.file_uploader(
        "Upload Exoplanet CSV files",
        type=['csv'],
        accept_multiple_files=True,
        help="Upload cumulative, K2, and/or TOI CSV files"
    )
    
    if uploaded_files:
        with st.spinner("Loading and processing data..."):
            # Load data
            raw_df = load_and_clean_data(uploaded_files)
            
            # Display raw data info
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Raw Data Shape:**", raw_df.shape)
            with col2:
                st.write("**Missing Values:**", raw_df.isnull().sum().sum())
            
            # Clean data
            cleaned_df = clean_data(raw_df)
            st.session_state.df_cleaned = cleaned_df
            
            # Display cleaned data info
            st.success("Data cleaned successfully!")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Cleaned Data Shape:**", cleaned_df.shape)
            with col2:
                st.write("**Missing Values:**", cleaned_df.isnull().sum().sum())
            
            # Show sample data
            st.write("**Sample of Cleaned Data:**")
            st.dataframe(cleaned_df.head())
            
            # Data types summary
            st.write("**Data Types Summary:**")
            dtype_summary = pd.DataFrame({
                'Column': cleaned_df.columns,
                'Data Type': cleaned_df.dtypes,
                'Non-Null Count': cleaned_df.count(),
                'Null Count': cleaned_df.isnull().sum()
            })
            st.dataframe(dtype_summary)

# Page 2: Data Exploration
elif page == "Data Exploration":
    st.markdown('<h2 class="sub-header">📊 Data Exploration</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned
        
        # Basic statistics
        st.write("**Dataset Overview:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", df.shape[0])
        with col2:
            st.metric("Total Features", df.shape[1])
        with col3:
            numerical_cols = df.select_dtypes(include=['number']).columns
            st.metric("Numerical Features", len(numerical_cols))
        
        # Distribution plots
        st.write("**Feature Distributions:**")
        numerical_cols = df.select_dtypes(include=['number']).columns[:12]  # Limit to first 12
        
        if len(numerical_cols) > 0:
            fig = make_subplots(
                rows=3, cols=4,
                subplot_titles=numerical_cols[:12]
            )
            
            for i, col in enumerate(numerical_cols[:12]):
                row = i // 4 + 1
                col_pos = i % 4 + 1
                
                fig.add_trace(
                    go.Histogram(x=df[col].dropna(), name=col, showlegend=False),
                    row=row, col=col_pos
                )
            
            fig.update_layout(height=800, title_text="Distribution of Numerical Features")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation heatmap
        if len(numerical_cols) > 1:
            st.write("**Correlation Analysis:**")
            corr_matrix = df[numerical_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        
        # Target variable analysis (if exists)
        target_options = ['koi_disposition', 'disposition'] + [col for col in df.columns if 'disp' in col.lower()]
        available_targets = [col for col in target_options if col in df.columns]
        
        if available_targets:
            st.write("**Target Variable Analysis:**")
            target_col = st.selectbox("Select target variable:", available_targets)
            
            if target_col:
                # Target distribution
                target_counts = df[target_col].value_counts()
                fig = px.pie(
                    values=target_counts.values,
                    names=target_counts.index,
                    title=f"Distribution of {target_col}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Please upload and clean data first in the 'Data Upload & Cleaning' page.")

# Page 3: Model Training
elif page == "Model Training":
    st.markdown('<h2 class="sub-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    if st.session_state.df_cleaned is not None:
        df = st.session_state.df_cleaned.copy()
        
        # Target selection
        target_options = ['koi_disposition', 'disposition'] + [col for col in df.columns if 'disp' in col.lower()]
        available_targets = [col for col in target_options if col in df.columns]
        
        if available_targets:
            target_col = st.selectbox("Select target variable:", available_targets)
            
            if target_col and st.button("Train Models"):
                with st.spinner("Training models..."):
                    # Prepare data
                    # Encode categorical variables
                    label_encoders = {}
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            le = LabelEncoder()
                            df[col] = le.fit_transform(df[col].astype(str))
                            label_encoders[col] = le
                    
                    # Split features and target
                    y = df[target_col]
                    X = df.drop(columns=[target_col])
                    
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Preprocessing
                    imputer = SimpleImputer(strategy='mean')
                    X_train_imputed = imputer.fit_transform(X_train)
                    X_test_imputed = imputer.transform(X_test)
                    
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train_imputed)
                    X_test_scaled = scaler.transform(X_test_imputed)
                    
                    # Models
                    models = {
                        "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
                        'Random Forest': RandomForestClassifier(random_state=42),
                        'Decision Tree': DecisionTreeClassifier(random_state=42),
                        'Support Vector Classifier': SVC(random_state=42),
                        'K-Nearest Neighbors': KNeighborsClassifier(),
                        'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                        'Naive Bayes': GaussianNB()
                    }
                    
                    # Train and evaluate models
                    results = {}
                    progress_bar = st.progress(0)
                    
                    for i, (name, model) in enumerate(models.items()):
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        results[name] = {
                            'model': model,
                            'accuracy': accuracy,
                            'predictions': y_pred,
                            'classification_report': classification_report(y_test, y_pred)
                        }
                        
                        progress_bar.progress((i + 1) / len(models))
                    
                    st.session_state.model_results = results
                    st.session_state.y_test = y_test
                    st.session_state.models_trained = True
                    
                    st.success("Models trained successfully!")
        else:
            st.error("No suitable target variable found in the dataset.")
    
    else:
        st.warning("Please upload and clean data first in the 'Data Upload & Cleaning' page.")

# Page 4: Results
elif page == "Results":
    st.markdown('<h2 class="sub-header">📈 Model Results</h2>', unsafe_allow_html=True)
    
    if st.session_state.models_trained and st.session_state.model_results:
        results = st.session_state.model_results
        
        # Model comparison
        st.write("**Model Performance Comparison:**")
        
        # Create comparison dataframe
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[model]['accuracy'] for model in results.keys()]
        }).sort_values('Accuracy', ascending=False)
        
        # Display results table
        st.dataframe(comparison_df.style.highlight_max(subset=['Accuracy']))
        
        # Accuracy comparison chart
        fig = px.bar(
            comparison_df,
            x='Accuracy',
            y='Model',
            orientation='h',
            title="Model Accuracy Comparison",
            color='Accuracy',
            color_continuous_scale='viridis'
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Best model details
        best_model_name = comparison_df.iloc[0]['Model']
        best_accuracy = comparison_df.iloc[0]['Accuracy']
        
        st.write(f"**Best Model: {best_model_name}**")
        st.write(f"**Accuracy: {best_accuracy:.4f}**")
        
        # Detailed results for selected model
        selected_model = st.selectbox("Select model for detailed analysis:", list(results.keys()))
        
        if selected_model:
            st.write(f"**Detailed Results for {selected_model}:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Classification Report:**")
                st.text(results[selected_model]['classification_report'])
            
            with col2:
                # Confusion matrix
                y_test = st.session_state.y_test
                y_pred = results[selected_model]['predictions']
                cm = confusion_matrix(y_test, y_pred)
                
                fig = px.imshow(
                    cm,
                    title=f"Confusion Matrix - {selected_model}",
                    color_continuous_scale="Blues",
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Download results
        if st.button("Download Results"):
            results_df = pd.DataFrame({
                'Model': list(results.keys()),
                'Accuracy': [results[model]['accuracy'] for model in results.keys()]
            })
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="model_results.csv",
                mime="text/csv"
            )
    
    else:
        st.warning("Please train models first in the 'Model Training' page.")

# Footer
st.markdown("---")
st.markdown("**Exoplanet Classification Analysis** - Built with Streamlit 🚀")
