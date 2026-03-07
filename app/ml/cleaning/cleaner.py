import pandas as pd
import numpy as np

def clean_and_restore_data(df):
    # 1. Drop the garbage columns we injected
    cols_to_drop = ['raw_import_id', 'data_status', 'unwanted_garbage_col', 'garbage_unwanted_column']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # 2. Filter out corrupted "System Error" rows
    error_keywords = ['SYSTEM_ERROR', 'CORRUPT', '!!!', '###']
    for word in error_keywords:
        df = df[~df.apply(lambda row: row.astype(str).str.contains(word).any(), axis=1)]

    # 3. Handle Garbage Strings
    garbage_strings = ['?', 'N/A', 'none', 'unknown_val', 'null']
    df = df.replace(garbage_strings, np.nan)

    # 4. Remove Duplicates
    df = df.drop_duplicates()

    # 5. Fill Missing Values
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if not df[col].mode().empty:
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna('Unknown')

    # 6. Final Formatting (Keep original names to avoid capital-gains errors)
    df.columns = [c.strip() for c in df.columns]

    return df