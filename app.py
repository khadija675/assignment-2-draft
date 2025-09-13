import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sdv.evaluation.single_table import evaluate_quality
from scipy.stats import ks_2samp
import warnings
import logging

# Configure logging and suppress warnings
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# --- Data Cleaning and Preprocessing Functions ---

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
    return df_cleaned, dropped_data, transforms

# --- Synthesis and Comparison Functions ---

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

def synthesize_data(df, n_synthetic_rows, ctgan_params, invariant_cols):
    from faker import Faker
    from tqdm import tqdm
    
    existing_invariant_cols = [col for col in invariant_cols if col in df.columns]
    df_for_synthesis = df.drop(columns=['Company Name'] + existing_invariant_cols, errors='ignore')
    
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_for_synthesis)
    
    categorical_columns = ['SCM Practices', 'Technology Utilized', 'Supply Chain Agility',
                          'Supply Chain Integration Level', 'Industry', 'Supplier Collaboration Level', 'Transportation Cost Efficiency', 'Supply Chain Complexity Index', 'Supply Chain Resilience Score']
    for col in categorical_columns:
        if col in df_for_synthesis.columns:
            metadata.update_column(column_name=col, sdtype='categorical')
    try:
        synthesizer = CTGANSynthesizer(metadata, **ctgan_params)
        with st.empty():
            with tqdm(total=ctgan_params['epochs'], desc="Training CTGAN") as pbar:
                def callback(epoch, loss):
                    pbar.update(1)
                synthesizer._fit_progress_callback = callback
                synthesizer.fit(df_for_synthesis)
                
        synthetic_data = synthesizer.sample(num_rows=n_synthetic_rows)
    except Exception as e:
        logger.error(f"CTGAN synthesis failed: {e}")
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
    
    combined_df = synthetic_data.reindex(columns=original_columns)
    return combined_df

# --- Visualization Functions (Updated) ---

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

# --- Streamlit App Layout ---

st.set_page_config(layout="wide")
st.title("Data Synthesis and Analysis App")

st.markdown("""
Upload your Excel or CSV file to perform data synthesis using CTGAN.
The app will clean the data, apply CTGAN to generate synthetic data,
and provide a comparative analysis and visualizations.
""")

uploaded_file = st.file_uploader("Choose a file (Excel or CSV)", type=["xlsx", "xls", "csv"])

if uploaded_file:
    try:
        file_ext = uploaded_file.name.split('.')[-1]
        if file_ext in ['xlsx', 'xls']:
            original_df = pd.read_excel(uploaded_file)
        elif file_ext == 'csv':
            original_df = pd.read_csv(uploaded_file)
        
        st.success("File uploaded successfully!")
        st.subheader("Original Data")
        st.dataframe(original_df.head())

        st.subheader("Data Preprocessing and Synthesis")
        st.info("Cleaning and transforming data...")

        if 'Cost of Goods Sold (COGS)' in original_df.columns:
            original_df['Cost of Goods Sold (COGS)'] = original_df['Cost of Goods Sold (COGS)'].apply(clean_cogs)
            original_df['Cost of Goods Sold (COGS)'] = pd.to_numeric(original_df['Cost of Goods Sold (COGS)'], errors='coerce')
        
        if 'Company Name' in original_df.columns:
            original_df = infer_industry(original_df)
        original_columns = original_df.columns.tolist()

        df_cleaned, dropped_data, transforms = clean_data(original_df)
        st.success("Data cleaned and prepared for synthesis!")

        n_synthetic_rows = st.sidebar.number_input("Number of synthetic rows", min_value=1, value=len(original_df))
        epochs = st.sidebar.number_input("CTGAN Epochs", min_value=1, value=500)

        ctgan_params = {'epochs': epochs, 'verbose': False}
        
        if st.sidebar.button("Generate Synthetic Data"):
            try:
                with st.spinner('Training CTGAN and generating synthetic data...'):
                    synthetic_data, metadata = synthesize_data(df_cleaned, n_synthetic_rows, ctgan_params, dropped_data.columns)
                    combined_synthetic_data = combine_data(synthetic_data, dropped_data, original_columns, transforms)
                
                st.success("Synthetic data generated successfully!")

                st.subheader("Generated Synthetic Data")
                st.dataframe(combined_synthetic_data.head())

                st.subheader("Statistical Analysis & Visualizations")
                
                st.write("### 1. Statistical Summary")
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Original Data**")
                    st.dataframe(original_df.describe())
                with col2:
                    st.write("**Synthetic Data**")
                    st.dataframe(combined_synthetic_data.describe())

                st.write("### 2. Correlation Heatmap")
                # Plot heatmap only for numerical columns in the cleaned dataframe
                plot_correlation_heatmap(df_cleaned, synthetic_data, "Original (Synthesized)", "Synthetic")

                st.write("### 3. Distribution Plots (KDE & Count)")
                # Plot distributions only for columns used in synthesis
                plot_distributions(df_cleaned, synthetic_data, "Original (Synthesized)", "Synthetic", df_cleaned.columns)
                
                st.success("Data synthesis and visualization complete! 🎉")
                st.balloons()
                
            except Exception as e:
                st.error(f"An error occurred during synthesis: {e}")

    except Exception as e:
        st.error(f"An error occurred while loading the file: {e}")