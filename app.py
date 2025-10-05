import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sdv.single_table import CTGANSynthesizer, GaussianCopulaSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from scipy.stats import ks_2samp
import warnings
import logging
from faker import Faker
from tqdm import tqdm
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def clean_cogs(value):
    try:
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (int, float)):
            return float(value)
        value = str(value).strip().replace('$', '').replace(',', '')
        if '$' in value[1:]:
            value = value.split('$')[1]
        if 'B' in value.upper():
            return float(value.upper().replace('B', '')) * 1e9
        elif 'M' in value.upper():
            return float(value.upper().replace('M', '')) * 1e6
        elif 'K' in value.upper():
            return float(value.upper().replace('K', '')) * 1e3
        else:
            return float(value)
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid COGS value: {value}. Returning NaN.")
        return pd.NA

def preprocess_extreme_values(df, columns=['Supplier Count', 'Cost of Goods Sold (COGS)', 'Total Implementation Cost']):
    df_transformed = df.copy()
    transforms = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if df[col].min() < 0:
                logger.warning(f"Cannot log-transform {col} due to negative values.")
                continue
            cap = df[col].quantile(0.99)
            df_transformed[col] = df[col].clip(upper=cap)
            df_transformed[col] = np.log1p(df_transformed[col])
            transforms[col] = 'log1p'
            logger.info(f"Applied log1p transformation and capped {col} at 99th percentile")
    return df_transformed, transforms

def reverse_transform(df, transforms):
    df_transformed = df.copy()
    for col, transform in transforms.items():
        if transform == 'log1p':
            df_transformed[col] = np.expm1(df_transformed[col])
    return df_transformed

def clean_data(df, missing_threshold=0.2):
    df = df.copy()
    cols_to_drop_by_name = [col for col in df.columns if any(keyword in col.lower() for keyword in ['date', 'text', 'full text'])]
    df = df.drop(columns=cols_to_drop_by_name, errors='ignore')
    
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_datetime64_any_dtype(df[col]):
            if col not in ['Company Name', 'SCM Practices', 'Technology Utilized', 'Supply Chain Agility', 'Supply Chain Integration Level', 'Industry', 'Supplier Collaboration Level', 'Transportation Cost Efficiency', 'Supply Chain Complexity Index', 'Supply Chain Resilience Score']:
                df = df.drop(columns=[col])

    missing_percentage = df.isnull().sum() / len(df)
    cols_to_drop_by_na = missing_percentage[missing_percentage > missing_threshold].index.tolist()
    df = df.drop(columns=cols_to_drop_by_na, errors='ignore')

    for col in df.columns:
        if df[col].isna().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].mean())
            else:
                df[col] = df[col].fillna(df[col].mode()[0])

    original_columns = df.columns.tolist()
    invariant_cols = [col for col in df.columns if df[col].nunique() <= 1 and col != 'Company Name']
    dropped_data = df[invariant_cols].copy() if invariant_cols else pd.DataFrame()
    df_cleaned = df.drop(columns=invariant_cols)
    df_cleaned, transforms = preprocess_extreme_values(df_cleaned)
    return df_cleaned, dropped_data, transforms, original_columns

def infer_industry(df):
    def infer(company):
        company = company.lower()
        mappings = {
            'Auto Services': ['tire', 'auto', 'pep boys', 'jiffy lube', 'mavis', 'les schwab'],
            'Agriculture/Fertilizer': ['fertilizer', 'agro', 'bayer', 'novozymes', 'crop'],
            'Steel/Metals': ['steel', 'iron', 'metals', 'stainless'],
            'Automotive': ['ford', 'volvo', 'rivian', 'bugatti', 'motorcycles'],
            'Tech/Software': ['microsoft', 'nvidia', 'zoom', 'openai', 'snowflake'],
            'Pharma/Healthcare': ['novartis', 'medtronic', 'illumina', 'health'],
            'Tire Manufacturing': ['michelin', 'pirelli', 'maxxis', 'yokohama']
        }
        for industry, keywords in mappings.items():
            if any(keyword in company for keyword in keywords):
                return industry
        return 'Other'
    df['Industry'] = df['Company Name'].apply(infer)
    return df

def synthesize_data(df, n_synthetic_rows, ctgan_params, invariant_cols, data_type, model_type):
    existing_invariant_cols = [col for col in invariant_cols if col in df.columns]
    df_for_synthesis = df.drop(columns=['Company Name'] + existing_invariant_cols, errors='ignore')
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_for_synthesis)
    
    categorical_columns = ['SCM Practices', 'Technology Utilized', 'Supply Chain Agility',
                          'Supply Chain Integration Level', 'Industry', 'Supplier Collaboration Level', 
                          'Transportation Cost Efficiency', 'Supply Chain Complexity Index', 
                          'Supply Chain Resilience Score']
    for col in categorical_columns:
        if col in df_for_synthesis.columns:
            metadata.update_column(column_name=col, sdtype='categorical')
    
    if data_type == 'Numerical':
        for col in df_for_synthesis.columns:
            if pd.api.types.is_numeric_dtype(df_for_synthesis[col]):
                metadata.update_column(column_name=col, sdtype='numerical')
            else:
                metadata.update_column(column_name=col, sdtype='unknown')
    elif data_type == 'Categorical':
        for col in df_for_synthesis.columns:
            metadata.update_column(column_name=col, sdtype='categorical')
    
    try:
        if model_type == "CTGAN":
            synthesizer = CTGANSynthesizer(metadata, **ctgan_params)
        else:  # GaussianCopula
            synthesizer = GaussianCopulaSynthesizer(metadata)
        
        with st.empty():
            with tqdm(total=ctgan_params.get('epochs', 100), desc=f"Training {model_type}") as pbar:
                def callback(epoch, loss):
                    pbar.update(1)
                if model_type == "CTGAN":
                    synthesizer._fit_progress_callback = callback
                synthesizer.fit(df_for_synthesis)
                
        synthetic_data = synthesizer.sample(num_rows=n_synthetic_rows)
    except Exception as e:
        logger.error(f"{model_type} synthesis failed: {e}")
        st.error(f"{model_type} synthesis failed: {e}")
        raise
    
    fake = Faker()
    synthetic_data['Company Name'] = [fake.company() for _ in range(n_synthetic_rows)]
    return synthetic_data, metadata

def combine_data(synthetic_data, dropped_data, original_columns, transforms):
    synthetic_data = synthetic_data.copy()
    synthetic_data = reverse_transform(synthetic_data, transforms)
    if not dropped_data.empty:
        constants = dropped_data.iloc[0]
        for col in dropped_data.columns:
            synthetic_data[col] = constants[col]
    
    # Ensure all original columns are present
    for col in original_columns:
        if col not in synthetic_data.columns:
            synthetic_data[col] = pd.NA
    combined_df = synthetic_data.reindex(columns=original_columns)
    return combined_df

def fix_categorical_contradictions(df):
    st.info("No categorical contradictions detected or fixed in this version.")
    return df

def handle_missing_data(df, option):
    if df is None:
        raise ValueError("Input DataFrame is None.")
    df = df.copy()
    if option == "Interpolate missing data":
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].interpolate(method='linear', limit_direction='both')
            else:
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
    elif option == "Keep missing":
        pass
    elif option == "Manual edit":
        st.warning("Manual edit not implemented in this version. Keeping missing values.")
    return df

def plot_distributions(df1, df2, title1, title2, columns):
    for col in columns:
        if col not in df1.columns or col not in df2.columns:
            st.warning(f"Skipping plot for '{col}' as it does not exist in both dataframes.")
            continue
            
        if pd.api.types.is_numeric_dtype(df1[col]):
            fig, ax = plt.subplots(figsize=(10, 7))
            sns.kdeplot(df1[col], label=title1, fill=True, ax=ax)
            sns.kdeplot(df2[col], label=title2, fill=True, ax=ax)
            ax.set_title(f"KDE Plot for {col}")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            
            sns.countplot(x=col, data=df1, ax=ax[0])
            ax[0].set_title(f"{title1} {col}")
            ax[0].tick_params(axis='x', rotation=45, labelsize=10)
            
            sns.countplot(x=col, data=df2, ax=ax[1])
            ax[1].set_title(f"{title2} {col}")
            ax[1].tick_params(axis='x', rotation=45, labelsize=10)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
def plot_correlation_heatmap(df1, df2, title1, title2):
    numerical_cols1 = df1.select_dtypes(include=np.number).columns
    numerical_cols2 = df2.select_dtypes(include=np.number).columns
    common_cols = list(set(numerical_cols1) & set(numerical_cols2))
    
    if len(common_cols) > 1:
        corr1 = df1[common_cols].corr()
        corr2 = df2[common_cols].corr()
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 10))
        
        sns.heatmap(corr1, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[0])
        axes[0].set_title(f"Correlation Heatmap: {title1}")
        axes[0].tick_params(axis='x', rotation=45)
        
        sns.heatmap(corr2, annot=True, fmt=".2f", cmap="coolwarm", ax=axes[1])
        axes[1].set_title(f"Correlation Heatmap: {title2}")
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.warning("Not enough numerical columns to plot a correlation heatmap.")

def evaluate_statistical_model(original_df, synthetic_df, metadata):
    try:
        # Exclude 'Company Name' from evaluation
        common_cols = [col for col in (set(original_df.columns) & set(synthetic_df.columns)) if col != 'Company Name']
        if not common_cols:
            raise ValueError("No common columns (excluding 'Company Name') between original and synthetic data.")
        original_df = original_df[common_cols].copy()
        synthetic_df = synthetic_df[common_cols].copy()
        
        # Update metadata to match common columns
        metadata_eval = SingleTableMetadata()
        metadata_eval.detect_from_dataframe(original_df)
        for col in metadata.columns:
            if col in common_cols:
                metadata_eval.update_column(column_name=col, sdtype=metadata.columns[col]['sdtype'])
        
        quality_report = evaluate_quality(
            real_data=original_df,
            synthetic_data=synthetic_df,
            metadata=metadata_eval
        )
        overall_score = quality_report.get_score()
        details = quality_report.get_details(property_name='Column Shapes')
        pair_trends = quality_report.get_details(property_name='Column Pair Trends')
        return overall_score, details, pair_trends
    except Exception as e:
        logger.error(f"Statistical evaluation failed: {str(e)}")
        st.error(f"Statistical evaluation failed with error: {str(e)}")
        return None, None, None

st.set_page_config(page_title="Synthetic Data Generation App", layout="wide", page_icon="📊")

st.markdown("""
    <div style="text-align: center;">
        <h1>📊 Synthetic Data Generation (SDG) App</h1>
        <p style="font-style: italic;">Powered by CTGAN and GaussianCopula for Realistic Data Synthesis</p>
    </div>
""", unsafe_allow_html=True)

page = st.sidebar.selectbox("Navigation", ["Home", "About", "Generate", "Post-Processing"], help="Select a page to navigate.")

if 'original_df' not in st.session_state:
    st.session_state.original_df = None
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None
if 'dropped_data' not in st.session_state:
    st.session_state.dropped_data = None
if 'transforms' not in st.session_state:
    st.session_state.transforms = None
if 'original_columns' not in st.session_state:
    st.session_state.original_columns = None
if 'synthetic_data' not in st.session_state:
    st.session_state.synthetic_data = None
if 'combined_synthetic_data' not in st.session_state:
    st.session_state.combined_synthetic_data = None
if 'post_processed_data' not in st.session_state:
    st.session_state.post_processed_data = None

if page == "Home":
    st.header("Welcome to the SDG App")
    st.markdown("""
    This app allows you to upload your dataset, generate synthetic data using CTGAN or GaussianCopula models, and perform post-processing.
    
    **Key Features:**
    - Data Cleaning & Preprocessing
    - Synthetic Data Generation with CTGAN or Statistical (GaussianCopula) Models
    - Statistical Evaluation and Visual Comparisons
    - Post-Processing Options for Data Quality
    
    Upload your file below to get started!
    """)
    
    uploaded_file = st.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "xls", "csv"], help="Upload your dataset here.")
    
    if uploaded_file:
        try:
            file_ext = uploaded_file.name.split('.')[-1]
            if file_ext in ['xlsx', 'xls']:
                st.session_state.original_df = pd.read_excel(uploaded_file)
            elif file_ext == 'csv':
                st.session_state.original_df = pd.read_csv(uploaded_file)
            
            st.success("File uploaded successfully!")
            
            if 'Cost of Goods Sold (COGS)' in st.session_state.original_df.columns:
                st.session_state.original_df['Cost of Goods Sold (COGS)'] = st.session_state.original_df['Cost of Goods Sold (COGS)'].apply(clean_cogs)
                st.session_state.original_df['Cost of Goods Sold (COGS)'] = pd.to_numeric(st.session_state.original_df['Cost of Goods Sold (COGS)'], errors='coerce')
            
            if 'Company Name' in st.session_state.original_df.columns:
                st.session_state.original_df = infer_industry(st.session_state.original_df)
            
            st.session_state.df_cleaned, st.session_state.dropped_data, st.session_state.transforms, st.session_state.original_columns = clean_data(st.session_state.original_df)
            st.success("Data cleaned and prepared!")
            
            st.subheader("Dataset Information")
            st.write(f"**Rows:** {st.session_state.original_df.shape[0]}")
            st.write(f"**Columns:** {st.session_state.original_df.shape[1]}")
            st.write(f"**Missing Values:** {st.session_state.original_df.isnull().sum().sum()}")
            
            st.subheader("Data Preview (First 10 Rows)")
            st.dataframe(st.session_state.original_df.head(10))
            
            st.subheader("Data Summary")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Numerical Summary**")
                st.dataframe(st.session_state.original_df.describe())
            with col2:
                st.write("**Categorical Summary**")
                cat_cols = st.session_state.original_df.select_dtypes(include='object').columns
                for col in cat_cols:
                    st.write(f"**{col} Value Counts:**")
                    st.dataframe(st.session_state.original_df[col].value_counts())
        
        except Exception as e:
            st.error(f"An error occurred while loading the file: {e}")

elif page == "About":
    st.header("About the App")
    st.markdown("""
    This Streamlit app is designed for **Synthetic Data Generation (SDG)** using CTGAN or GaussianCopula models from the SDV library.
    
    **Purpose:**
    - Generate realistic synthetic datasets that mimic the statistical properties of your original data.
    - Useful for data augmentation, privacy preservation, and testing.
    
    **Model Options:**
    - **CTGAN**: A deep learning-based model for generating synthetic data, suitable for complex datasets.
    - **GaussianCopula**: A statistical model that captures data distributions using copulas, ideal for simpler datasets.
    
    **Technologies Used:**
    - Streamlit for the web interface
    - CTGAN and GaussianCopula for data synthesis
    - Pandas, NumPy, Matplotlib, Seaborn for data handling and visualization
    - SDV for statistical evaluation
    
    **How it Works:**
    1. Upload your data on the Home page.
    2. Select a model (CTGAN or GaussianCopula) and configure parameters on the Generate page.
    3. Evaluate statistical quality and visualize comparisons.
    4. Apply post-processing on the Post-Processing page.
    5. Download the final dataset.
    
    For more details, check the documentation or contact the developers.
    """)

elif page == "Generate":
    st.header("Generate Synthetic Data")
    if st.session_state.original_df is None:
        st.warning("Please upload a dataset on the Home page first.")
    else:
        st.info("Configure parameters, select a model, and generate synthetic data.")
        
        n_synthetic_rows = st.number_input("Number of synthetic rows", min_value=1, value=len(st.session_state.original_df), help="Number of rows to generate.")
        model_type = st.selectbox("Model Type", ["CTGAN", "GaussianCopula"], help="Choose the model for synthetic data generation.")
        
        # CTGAN-specific parameters
        ctgan_params = {}
        if model_type == "CTGAN":
            epochs = st.number_input("CTGAN Epochs", min_value=1, value=100, help="Number of training epochs for CTGAN.")
            batch_size = st.number_input("Batch Size", min_value=1, value=100, help="Batch size for CTGAN training.")
            generator_decay = st.number_input("Generator Decay", min_value=0.0, max_value=1.0, value=1e-6, step=1e-7, format="%.7f", help="Decay rate for CTGAN generator.")
            ctgan_params = {
                'epochs': epochs,
                'batch_size': batch_size,
                'generator_decay': generator_decay,
                'verbose': False
            }
        
        accuracy_threshold = st.number_input("Accuracy Threshold", min_value=0.0, max_value=1.0, value=0.8, help="Target accuracy for synthesis (not directly used).")
        data_type = st.selectbox("Data Type", ["Mixed", "Numerical", "Categorical"], help="Type of data for metadata adjustment.")
        
        if st.button("Generate Synthetic Data", help="Click to start the synthesis process."):
            try:
                with st.spinner(f'Training {model_type} and generating synthetic data...'):
                    st.session_state.synthetic_data, metadata = synthesize_data(
                        st.session_state.df_cleaned, 
                        n_synthetic_rows, 
                        ctgan_params, 
                        st.session_state.dropped_data.columns, 
                        data_type,
                        model_type
                    )
                    st.session_state.combined_synthetic_data = combine_data(
                        st.session_state.synthetic_data, 
                        st.session_state.dropped_data, 
                        st.session_state.original_columns, 
                        st.session_state.transforms
                    )
                
                st.success("Synthetic data generated successfully!")
                
                st.subheader("Original vs Synthetic Data (Preview)")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Data**")
                    st.dataframe(st.session_state.original_df.head())
                with col2:
                    st.write("**Synthetic Data**")
                    st.dataframe(st.session_state.combined_synthetic_data.head())
                
                st.subheader("Statistical Analysis & Visualizations")
                
                st.write("### 1. Statistical Quality Evaluation")
                overall_score, column_shapes, pair_trends = evaluate_statistical_model(
                    st.session_state.df_cleaned,
                    st.session_state.synthetic_data,
                    metadata
                )
                if overall_score is not None:
                    st.write(f"**Overall Quality Score:** {overall_score:.3f} (Higher is better, max 1.0)")
                    st.markdown("""
                    **Interpretation:**
                    - **Column Shapes**: Measures how well the synthetic data matches the distribution of each column (using KS Distance for numerical, Chi-Squared for categorical).
                    - **Column Pair Trends**: Measures how well pairwise relationships (correlations) are preserved.
                    - A score close to 1 indicates high similarity between original and synthetic data.
                    """)
                    st.write("**Column Shapes Details**")
                    st.dataframe(column_shapes)
                    st.write("**Column Pair Trends Details**")
                    st.dataframe(pair_trends)
                else:
                    st.error("Statistical evaluation failed. Check the error message above for details.")
                
                st.write("### 2. Statistical Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Data**")
                    st.dataframe(st.session_state.original_df.describe())
                with col2:
                    st.write("**Synthetic Data**")
                    st.dataframe(st.session_state.combined_synthetic_data.describe())

                st.write("### 3. Correlation Heatmap")
                plot_correlation_heatmap(st.session_state.df_cleaned, st.session_state.synthetic_data, "Original (Cleaned)", "Synthetic")

                st.write("### 4. Distribution Plots (KDE & Count)")
                plot_distributions(st.session_state.df_cleaned, st.session_state.synthetic_data, "Original (Cleaned)", "Synthetic", st.session_state.df_cleaned.columns)
                
                st.success(f"Data synthesis with {model_type}, visualization, and statistical evaluation complete! 🎉")
                st.balloons()
                
            except Exception as e:
                st.error(f"An error occurred during synthesis: {e}")

elif page == "Post-Processing":
    st.header("Post-Processing & Validation")
    if st.session_state.combined_synthetic_data is None:
        st.warning("Please generate synthetic data on the Generate page first.")
    else:
        st.info("Apply post-processing to refine the synthetic data.")
        
        if st.button("Fix Categorical Contradictions", help="Detect and resolve inconsistencies in categorical fields."):
            st.session_state.post_processed_data = fix_categorical_contradictions(st.session_state.combined_synthetic_data.copy())
            st.success("Categorical contradictions fixed!")
        
        missing_option = st.selectbox("Handle Missing Data", ["Interpolate missing data", "Keep missing", "Manual edit"], help="Choose how to handle missing values.")
        if st.button("Apply Missing Data Handling", help="Apply the selected option to missing values."):
            try:
                st.session_state.post_processed_data = handle_missing_data(
                    st.session_state.post_processed_data if st.session_state.post_processed_data is not None else st.session_state.combined_synthetic_data.copy(),
                    missing_option
                )
                st.success("Missing data handled!")
            except Exception as e:
                st.error(f"Error handling missing data: {e}")
        
        if st.session_state.post_processed_data is not None:
            st.subheader("Missing Values Comparison")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Before Post-Processing**")
                st.dataframe(st.session_state.combined_synthetic_data.isnull().sum().to_frame(name="Missing Count"))
            with col2:
                st.write("**After Post-Processing**")
                st.dataframe(st.session_state.post_processed_data.isnull().sum().to_frame(name="Missing Count"))
        
        if st.session_state.post_processed_data is not None:
            csv_buffer = io.StringIO()
            st.session_state.post_processed_data.to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Final Dataset",
                data=csv_buffer.getvalue(),
                file_name="post_processed_synthetic_data.csv",
                mime="text/csv",
                help="Download the post-processed synthetic dataset as CSV."
            )