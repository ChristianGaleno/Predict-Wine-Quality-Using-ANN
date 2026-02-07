import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import tensorflow as tf
import joblib

# Konfigurasi halaman
st.set_page_config(
    page_title="Wine Quality Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #722F37, #B05F6D);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding: 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .prediction-box {
        text-align: center; 
        padding: 2rem; 
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
        border-radius: 15px; 
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .prediction-score {
        font-size: 4rem; 
        margin: 0;
        font-weight: bold;
    }
    .prediction-label {
        font-size: 1.5rem; 
        margin: 0;
    }
    .feature-input {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<p class="main-header">🍷 Wine Quality Predictor</p>', unsafe_allow_html=True)
st.markdown("---")


@st.cache_resource
def load_model_and_tools():
    model = tf.keras.models.load_model("model.h5")
    scaler = joblib.load("scaler.pkl")
    le_type = joblib.load("le_type.pkl")
    le_quality = joblib.load("le_quality.pkl")
    return model, scaler, le_type, le_quality

model, scaler, le_type, le_quality = load_model_and_tools()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('winequality.csv')
    return df

# Sidebar
st.sidebar.title("🎯 Navigation")
page = st.sidebar.radio(
    "Pilih Halaman:",
    ["🏠 Home", "📊 Eksplorasi Data", "🔮 Prediksi"]
)

# Load data
df = load_data()

# ==================== HOME PAGE ====================
if page == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Data", f"{len(df):,}")
    with col2:
        st.metric("Features", "12")
    with col3:
        red_count = len(df[df['type'] == 'red'])
        st.metric("Red Wine", f"{red_count:,}")
    with col4:
        white_count = len(df[df['type'] == 'white'])
        st.metric("White Wine", f"{white_count:,}")
    
    st.markdown("### 📖 Tentang Aplikasi")
    st.info("""
    Aplikasi prediksi kualitas wine menggunakan **Deep Learning (TensorFlow/Keras)**.
    
    **Fitur Utama:**
    - 📊 Eksplorasi data interaktif dengan visualisasi
    - 🔮 Prediksi kualitas wine menggunakan model TensorFlow
    - 📈 Input manual dengan feature engineering yang dapat dikustomisasi
    
    **Model:**
    - Framework: TensorFlow/Keras
    - Input: 12 fitur kimia wine
    - Output: Quality score (3-9)
    """)
    
    st.markdown("### 📋 Preview Data")
    st.dataframe(df.head(20), use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Statistik Deskriptif")
        st.dataframe(df.describe(), use_container_width=True)
    
    with col2:
        st.markdown("### 📈 Distribusi Quality")
        quality_counts = df['quality'].value_counts().sort_index()
        fig = px.bar(x=quality_counts.index, y=quality_counts.values,
                    labels={'x': 'Quality', 'y': 'Count'},
                    title='Distribusi Quality Score',
                    color=quality_counts.values,
                    color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)

# ==================== EKSPLORASI DATA ====================
elif page == "📊 Eksplorasi Data":
    st.header("📊 Eksplorasi Data Wine Quality")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Distribusi", "🔗 Korelasi", "📦 Comparison", "🎨 Relationships"])
    
    with tab1:
        st.subheader("Distribusi Fitur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Quality by type
            quality_type = df.groupby(['type', 'quality']).size().reset_index(name='count')
            fig = px.bar(quality_type, x='quality', y='count', color='type',
                        title='Quality Distribution by Wine Type',
                        barmode='group',
                        color_discrete_map={'red': '#722F37', 'white': '#F4E4C1'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Type distribution
            type_counts = df['type'].value_counts()
            fig = px.pie(values=type_counts.values, 
                        names=type_counts.index,
                        title='Wine Type Distribution',
                        color=type_counts.index,
                        color_discrete_map={'red': '#722F37', 'white': '#F4E4C1'})
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature distribution
        st.markdown("#### 📊 Pilih Fitur untuk Analisis")
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols.remove('quality')
        
        feature = st.selectbox("Pilih Fitur:", numeric_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(df, x=feature, 
                              color='type',
                              marginal='box',
                              title=f'Distribusi {feature}',
                              color_discrete_map={'red': '#722F37', 'white': '#F4E4C1'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.violin(df, y=feature, x='type', color='type',
                           title=f'Violin Plot: {feature} by Type',
                           color_discrete_map={'red': '#722F37', 'white': '#F4E4C1'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Analisis Korelasi")
        
        # Select numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Compute correlation
        corr = numeric_df.corr()
        
        # Create heatmap
        fig = px.imshow(corr,
                       labels=dict(color="Correlation"),
                       x=corr.columns,
                       y=corr.columns,
                       color_continuous_scale='RdBu_r',
                       aspect="auto",
                       title="Correlation Heatmap",
                       zmin=-1, zmax=1)
        fig.update_layout(height=700)
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation with quality
        st.markdown("#### 🎯 Korelasi dengan Quality")
        quality_corr = corr['quality'].drop('quality').sort_values(ascending=False)
        
        fig = px.bar(x=quality_corr.values, 
                    y=quality_corr.index,
                    orientation='h',
                    title='Feature Correlation with Quality',
                    labels={'x': 'Correlation', 'y': 'Feature'},
                    color=quality_corr.values,
                    color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📦 Perbandingan Fitur")
        
        col1, col2 = st.columns(2)
        
        with col1:
            feature_x = st.selectbox("Pilih Feature X:", numeric_cols, index=0)
        
        with col2:
            feature_y = st.selectbox("Pilih Feature Y:", numeric_cols, index=1)
        
        # Box plot comparison
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.box(df, x='quality', y=feature_x, color='type',
                        title=f'{feature_x} by Quality',
                        color_discrete_map={'red': '#722F37', 'white': '#F4E4C1'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.box(df, x='quality', y=feature_y, color='type',
                        title=f'{feature_y} by Quality',
                        color_discrete_map={'red': '#722F37', 'white': '#F4E4C1'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("🎨 Relationship Analysis")
        
        # Select features for scatter
        col1, col2 = st.columns(2)
        
        with col1:
            x_feature = st.selectbox("X-axis:", numeric_cols, key='scatter_x')
        with col2:
            y_feature = st.selectbox("Y-axis:", numeric_cols, key='scatter_y', index=1)
        
        fig = px.scatter(df, x=x_feature, y=y_feature, 
                        color='quality', 
                        size='alcohol',
                        hover_data=['type'],
                        title=f'{x_feature} vs {y_feature}',
                        color_continuous_scale='Viridis')
        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
        
        # Pair plot option
        st.markdown("#### 🔍 Multi-feature Analysis")
        selected_features = st.multiselect(
            "Pilih 2-4 fitur untuk Scatter Matrix:",
            numeric_cols,
            default=numeric_cols[:3]
        )
        
        if len(selected_features) >= 2:
            fig = px.scatter_matrix(df,
                                   dimensions=selected_features,
                                   color='quality',
                                   title='Scatter Matrix',
                                   height=800,
                                   color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)

# ==================== PREDIKSI ====================
else:  # Prediksi page
    st.header("🔮 Prediksi Kualitas Wine")
    
    st.info("""
    💡 **Cara Menggunakan:**
    1. Masukkan nilai untuk 12 fitur di bawah ini
    2. Sistem akan menampilkan data yang Anda input dalam format DataFrame
    3. **Anda dapat melakukan preprocessing dan feature engineering sendiri**
    4. Load model TensorFlow Anda dan lakukan prediksi
    
    📝 **Note:** Aplikasi ini hanya menyediakan interface input. Preprocessing, feature engineering, 
    dan prediksi dilakukan oleh Anda menggunakan model TensorFlow yang sudah Anda buat.
    """)
    
    st.markdown("### 📝 Input Data Wine")
    
    # Create two columns for inputs
    col1, col2 = st.columns(2)
    
    # Column 1 - Acidity & Sugar
    with col1:
        st.markdown("#### 🧪 Acidity & Sweetness")
        fixed_acidity = st.number_input(
            "Fixed Acidity",
            min_value=0.0,
            max_value=20.0,
            value=7.4,
            step=0.1,
            help="Tartaric acid content (g/dm³)"
        )
        
        volatile_acidity = st.number_input(
            "Volatile Acidity",
            min_value=0.0,
            max_value=2.0,
            value=0.7,
            step=0.01,
            help="Acetic acid content (g/dm³)"
        )
        
        citric_acid = st.number_input(
            "Citric Acid",
            min_value=0.0,
            max_value=2.0,
            value=0.0,
            step=0.01,
            help="Citric acid content (g/dm³)"
        )
        
        residual_sugar = st.number_input(
            "Residual Sugar",
            min_value=0.0,
            max_value=20.0,
            value=1.9,
            step=0.1,
            help="Remaining sugar after fermentation (g/dm³)"
        )
        
        chlorides = st.number_input(
            "Chlorides",
            min_value=0.0,
            max_value=1.0,
            value=0.076,
            step=0.001,
            format="%.3f",
            help="Salt content (g/dm³)"
        )
    
    # Column 2 - Sulfur & Physical Properties
    with col2:
        st.markdown("#### 💨 Sulfur & Physical Properties")
        free_sulfur = st.number_input(
            "Free Sulfur Dioxide",
            min_value=0.0,
            max_value=100.0,
            value=11.0,
            step=1.0,
            help="Free SO₂ (mg/dm³)"
        )
        
        total_sulfur = st.number_input(
            "Total Sulfur Dioxide",
            min_value=0.0,
            max_value=300.0,
            value=34.0,
            step=1.0,
            help="Total SO₂ (mg/dm³)"
        )
        
        density = st.number_input(
            "Density",
            min_value=0.98,
            max_value=1.01,
            value=0.9978,
            step=0.0001,
            format="%.4f",
            help="Wine density (g/cm³)"
        )
        
        pH = st.number_input(
            "pH",
            min_value=2.0,
            max_value=5.0,
            value=3.51,
            step=0.01,
            help="pH level (0-14 scale)"
        )
        
        sulphates = st.number_input(
            "Sulphates",
            min_value=0.0,
            max_value=2.5,
            value=0.56,
            step=0.01,
            help="Potassium sulphate content (g/dm³)"
        )
    
    # Alcohol and Type in full width
    st.markdown("#### 🍾 Alcohol & Wine Type")
    col1, col2 = st.columns(2)
    
    with col1:
        alcohol = st.number_input(
            "Alcohol",
            min_value=0.0,
            max_value=20.0,
            value=9.4,
            step=0.1,
            help="Alcohol percentage (%)"
        )
    
    with col2:
        wine_type = st.selectbox(
            "Wine Type",
            options=['red', 'white'],
            help="Type of wine"
        )
    
    st.markdown("---")
    
    # Display input data
    st.markdown("### 📊 Data Input Anda")
    
    # Create DataFrame from input
    input_data = pd.DataFrame([{
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_sulfur,
        'total sulfur dioxide': total_sulfur,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol,
        'type': wine_type
    }])
    
    # Display as styled dataframe
    st.dataframe(input_data.T.rename(columns={0: 'Value'}), use_container_width=True)
    
    # Download button for input data
    csv = input_data.to_csv(index=False)
    st.download_button(
        label="📥 Download Input Data (CSV)",
        data=csv,
        file_name="wine_input.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Placeholder for user's model
    st.markdown("### 🤖 Model & Prediksi")
    def feat_engineer(df):
        df['total_acidity'] = df['fixed acidity'] + df['volatile acidity']
        df['sulfur_ratio'] = df['free sulfur dioxide'] / (df['total sulfur dioxide'] + 1e-6)
        df['chloride_density_ratio'] = df['chlorides'] / df['density']
        df['alcohol_sugar_ratio'] = df['alcohol'] / (df['residual sugar'] + 1e-6)
        df['alcohol_x_sulphates'] = df['alcohol'] * df['sulphates']
        df['acidity_x_pH'] = df['total_acidity'] * df['pH']
        df['acid_strength'] = df['total_acidity'] / (df['pH'] + 1e-6)
        df['so2_excess'] = df['total sulfur dioxide'] - df['free sulfur dioxide']
        return df
    
    def preprocess(df):
        df['type'] = le_type.transform(df['type'])
        return df
    
    predict_btn = st.button("🔮 Predict Quality", use_container_width=True)
    
    if predict_btn:
        input_data_fe = feat_engineer(input_data.copy())
        input_data_pp = preprocess(input_data_fe)

        input_scaled = pd.DataFrame(
            scaler.transform(input_data_pp),
            columns=input_data_pp.columns
        )

        prediction = model.predict(input_scaled)
        quality_pred = np.argmax(prediction, axis=1)[0]
        y_pred_label = le_quality.inverse_transform([quality_pred])[0]

        st.markdown("### 🧠 Hasil Prediksi")
        st.success(f"{y_pred_label}")



    st.info("""
    **📌 Catatan Penting:**
    - Data input sudah tersedia dalam variabel `input_data` (pandas DataFrame)
    - Anda perlu melakukan preprocessing sesuai dengan training model Anda
    - Feature engineering (jika ada) dilakukan sesuai kebutuhan model
    - Load model TensorFlow dengan `tf.keras.models.load_model()`
    - Lakukan prediksi dan tampilkan hasilnya
    """)
    
    # Statistics comparison
    st.markdown("### 📈 Perbandingan dengan Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    numeric_features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 
                       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 
                       'density', 'pH', 'sulphates', 'alcohol']
    
    comparison_data = []
    for feature in numeric_features:
        comparison_data.append({
            'Feature': feature,
            'Your Value': input_data[feature].values[0],
            'Dataset Mean': df[feature].mean(),
            'Dataset Min': df[feature].min(),
            'Dataset Max': df[feature].max()
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    st.dataframe(comparison_df.style.format({
        'Your Value': '{:.4f}',
        'Dataset Mean': '{:.4f}',
        'Dataset Min': '{:.4f}',
        'Dataset Max': '{:.4f}'
    }), use_container_width=True)
    
    # Visualization of input vs dataset
    st.markdown("### 📊 Visualisasi Input vs Dataset")
    
    selected_viz = st.selectbox(
        "Pilih fitur untuk visualisasi:",
        numeric_features
    )
    
    fig = go.Figure()
    
    # Add histogram of dataset
    fig.add_trace(go.Histogram(
        x=df[selected_viz],
        name='Dataset Distribution',
        opacity=0.7,
        marker_color='#722F37'
    ))
    
    # Add vertical line for user input
    fig.add_vline(
        x=input_data[selected_viz].values[0],
        line_dash="dash",
        line_color="red",
        annotation_text=f"Your Value: {input_data[selected_viz].values[0]:.2f}",
        annotation_position="top"
    )
    
    fig.update_layout(
        title=f'Your {selected_viz} vs Dataset Distribution',
        xaxis_title=selected_viz,
        yaxis_title='Count',
        height=400
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 1rem;'>
    <p>🍷 Wine Quality Predictor with TensorFlow | Built with Streamlit</p>
    <p style='font-size: 0.9rem;'>Preprocessing & Feature Engineering: User-Defined</p>
</div>
""", unsafe_allow_html=True)
