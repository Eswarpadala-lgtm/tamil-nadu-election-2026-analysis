# =============================================================================
# FILE: src/data_cleaning.py
# PROJECT: Tamil Nadu Election 2026 - Winner Prediction
# PURPOSE: Load raw data, clean it, engineer features, save processed data
# =============================================================================

import pandas as pd
import numpy as np
import os

# =============================================================================
# STEP 2.1 — LOAD THE RAW DATA
# =============================================================================

def load_data(filepath):
    """
    Loads the CSV dataset from the given path.
    """
    print("=" * 60)
    print("STEP 2.1: Loading Raw Data")
    print("=" * 60)

    df = pd.read_csv(filepath)

    print(f"✅ Data loaded successfully!")
    print(f"   Rows    : {df.shape[0]}")
    print(f"   Columns : {df.shape[1]}")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nFirst 3 rows:\n{df.head(3).to_string()}")

    return df


# =============================================================================
# STEP 2.2 — BASIC DATA INSPECTION
# =============================================================================

def inspect_data(df):
    """
    Prints a full inspection report of the dataset.
    WHY: Before cleaning, you must understand what you're working with.
    """
    print("\n" + "=" * 60)
    print("STEP 2.2: Data Inspection Report")
    print("=" * 60)

    print(f"\n📌 Shape: {df.shape[0]} rows x {df.shape[1]} columns")

    print(f"\n📌 Data Types:")
    print(df.dtypes)

    print(f"\n📌 Missing Values:")
    missing = df.isnull().sum()
    print(missing)
    print(f"   Total missing values: {missing.sum()}")

    print(f"\n📌 Duplicate Rows: {df.duplicated().sum()}")

    print(f"\n📌 Unique Parties: {df['Party'].nunique()}")
    print(f"   Top 5 parties by candidate count:")
    print(df['Party'].value_counts().head(5).to_string())

    print(f"\n📌 Unique Constituencies: {df['Constituency'].nunique()}")

    print(f"\n📌 Vote Statistics:")
    print(df[['EVM Votes', 'Postal Votes', 'Total Votes', '% Votes']].describe().round(2).to_string())


# =============================================================================
# STEP 2.3 — DATA CLEANING
# =============================================================================

def clean_data(df):
    """
    Cleans the raw dataset.
    WHY: Raw data always has issues. We fix them before analysis or ML.
    """
    print("\n" + "=" * 60)
    print("STEP 2.3: Cleaning Data")
    print("=" * 60)

    # Work on a copy — NEVER modify the original dataframe directly
    df_clean = df.copy()

    # --- Remove duplicate rows ---
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    after = len(df_clean)
    print(f"✅ Removed {before - after} duplicate rows")

    # --- Strip whitespace from text columns ---
    # WHY: "  DMK  " and "DMK" would be treated as different parties!
    text_cols = ['Constituency', 'Candidate', 'Party', 'Code']
    for col in text_cols:
        df_clean[col] = df_clean[col].str.strip()
    print(f"✅ Stripped whitespace from text columns")

    # --- Rename columns to snake_case ---
    # WHY: '% Votes' with spaces is annoying to type. Industry uses snake_case.
    df_clean = df_clean.rename(columns={
        'Code'              : 'code',
        'Constituency'      : 'constituency',
        'Candidate'         : 'candidate',
        'Party'             : 'party',
        'EVM Votes'         : 'evm_votes',
        'Postal Votes'      : 'postal_votes',
        'Total Votes'       : 'total_votes',
        '% Votes'           : 'pct_votes',
        'Round'             : 'round',
        'Last Updated Time' : 'last_updated_time',
        'Last Updated Date' : 'last_updated_date'
    })
    print(f"✅ Renamed columns to snake_case")

    # --- Extract round numbers from "26/26" format ---
    # WHY: "26/26" is a string. We want numbers we can actually use.
    df_clean['total_rounds']     = df_clean['round'].str.split('/').str[1].astype(int)
    df_clean['completed_rounds'] = df_clean['round'].str.split('/').str[0].astype(int)
    print(f"✅ Extracted total_rounds and completed_rounds")

    # --- Drop columns not useful for ML ---
    df_clean = df_clean.drop(columns=['last_updated_time', 'last_updated_date', 'round'])
    print(f"✅ Dropped non-useful columns")

    print(f"\n✅ Cleaned data shape: {df_clean.shape}")
    return df_clean


# =============================================================================
# STEP 2.4 — FEATURE ENGINEERING
# =============================================================================

def engineer_features(df):
    """
    Creates new columns (features) that help the ML model learn better.
    WHY: Raw columns alone aren't enough. We create smarter signals.
    """
    print("\n" + "=" * 60)
    print("STEP 2.4: Feature Engineering")
    print("=" * 60)

    df_feat = df.copy()

    # --- TARGET: is_winner ---
    # The candidate with highest votes in each constituency = winner = 1
    df_feat['is_winner'] = (
        df_feat.groupby('constituency')['total_votes']
        .transform(lambda x: x == x.max())
        .astype(int)
    )
    print(f"✅ Created 'is_winner' — {df_feat['is_winner'].sum()} winners total")

    # --- winning_margin ---
    # Difference between 1st and 2nd place votes in each constituency
    def calc_margin(group):
        sorted_votes = group['total_votes'].sort_values(ascending=False).values
        if len(sorted_votes) >= 2:
            return sorted_votes[0] - sorted_votes[1]
        return sorted_votes[0]

    margins = df_feat.groupby('constituency').apply(calc_margin).reset_index()
    margins.columns = ['constituency', 'winning_margin']
    df_feat = df_feat.merge(margins, on='constituency', how='left')
    print(f"✅ Created 'winning_margin' — avg: {df_feat['winning_margin'].mean():.0f} votes")

    # --- candidates_in_constituency ---
    # How many candidates contested in the same constituency
    candidate_count = df_feat.groupby('constituency')['candidate'].count().reset_index()
    candidate_count.columns = ['constituency', 'candidates_in_constituency']
    df_feat = df_feat.merge(candidate_count, on='constituency', how='left')
    print(f"✅ Created 'candidates_in_constituency'")

    # --- postal_vote_ratio ---
    # What fraction of votes came via postal ballot
    df_feat['postal_vote_ratio'] = (
        df_feat['postal_votes'] / (df_feat['total_votes'] + 1)
    ).round(4)
    print(f"✅ Created 'postal_vote_ratio'")

    # --- is_major_party ---
    # 1 if the candidate belongs to a major party, 0 otherwise
    major_parties = [
        'Tamilaga Vettri Kazhagam',
        'Dravida Munnetra Kazhagam',
        'All India Anna Dravida Munnetra Kazhagam',
        'Indian National Congress',
        'Bharatiya Janata Party'
    ]
    df_feat['is_major_party'] = df_feat['party'].isin(major_parties).astype(int)
    print(f"✅ Created 'is_major_party'")

    # --- is_independent ---
    # 1 if Independent candidate — they almost never win
    df_feat['is_independent'] = (df_feat['party'] == 'Independent').astype(int)
    print(f"✅ Created 'is_independent'")

    # --- party_win_rate ---
    # What % of seats did this party win overall
    party_wins = df_feat.groupby('party')['is_winner'].mean().reset_index()
    party_wins.columns = ['party', 'party_win_rate']
    df_feat = df_feat.merge(party_wins, on='party', how='left')
    df_feat['party_win_rate'] = df_feat['party_win_rate'].round(4)
    print(f"✅ Created 'party_win_rate'")

    print(f"\n✅ Feature engineering complete. Final shape: {df_feat.shape}")
    return df_feat


# =============================================================================
# STEP 2.5 — SAVE PROCESSED DATA
# =============================================================================

def save_processed_data(df, output_path):
    """
    Saves cleaned + engineered data to data/processed/
    WHY: Save once, reuse many times. Never redo cleaning.
    """
    print("\n" + "=" * 60)
    print("STEP 2.5: Saving Processed Data")
    print("=" * 60)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"✅ Saved to: {output_path}")
    print(f"   Rows: {df.shape[0]} | Columns: {df.shape[1]}")

    print(f"\n📌 Final Column List:")
    for i, col in enumerate(df.columns, 1):
        print(f"   {i:2}. {col}")


# =============================================================================
# MAIN — Runs all steps in order when you execute this file
# =============================================================================

if __name__ == "__main__":

    RAW_DATA_PATH       = "data/raw/eci_results_tamilnadu_2026.csv"
    PROCESSED_DATA_PATH = "data/processed/tn_election_cleaned.csv"

    print("\n🚀 TAMIL NADU ELECTION 2026 — DATA CLEANING PIPELINE")
    print("=" * 60)

    df_raw   = load_data(RAW_DATA_PATH)
    inspect_data(df_raw)
    df_clean = clean_data(df_raw)
    df_final = engineer_features(df_clean)
    save_processed_data(df_final, PROCESSED_DATA_PATH)

    print("\n" + "=" * 60)
    print("✅ ALL STEPS COMPLETE!")
    print("   Processed file ready at: data/processed/tn_election_cleaned.csv")
    print("=" * 60)