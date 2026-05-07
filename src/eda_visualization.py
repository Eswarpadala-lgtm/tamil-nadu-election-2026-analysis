# =============================================================================
# FILE: src/eda_visualization.py
# PROJECT: Tamil Nadu Election 2026 - Winner Prediction
# PURPOSE: Exploratory Data Analysis + Professional Visualizations
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import os
import warnings

warnings.filterwarnings('ignore')

# --- Chart style settings ---
# WHY: Consistent styling makes your portfolio look professional
plt.rcParams['figure.dpi']       = 150
plt.rcParams['font.family']      = 'DejaVu Sans'
plt.rcParams['axes.spines.top']  = False
plt.rcParams['axes.spines.right']= False

# Color palette for parties
PARTY_COLORS = {
    'Tamilaga Vettri Kazhagam'                   : '#1a6b3c',
    'Dravida Munnetra Kazhagam'                  : '#e63946',
    'All India Anna Dravida Munnetra Kazhagam'   : '#2196F3',
    'Indian National Congress'                   : '#00bcd4',
    'Independent'                                : '#9e9e9e',
    'Others'                                     : '#ff9800'
}

OUTPUT_DIR = 'outputs/figures'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# HELPER: Save chart
# =============================================================================

def save_chart(filename):
    path = os.path.join(OUTPUT_DIR, filename)
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   💾 Saved: {path}")


# =============================================================================
# CHART 1 — Seats Won by Party (Bar Chart)
# WHY: The #1 question in any election — who won how many seats?
# =============================================================================

def chart_seats_won(df):
    print("\n📊 Chart 1: Seats Won by Party")

    winners = df[df['is_winner'] == 1].copy()
    party_seats = winners['party'].value_counts().reset_index()
    party_seats.columns = ['party', 'seats']

    # Shorten long party names for display
    name_map = {
        'Tamilaga Vettri Kazhagam'                   : 'TVK',
        'Dravida Munnetra Kazhagam'                  : 'DMK',
        'All India Anna Dravida Munnetra Kazhagam'   : 'AIADMK',
        'Indian National Congress'                   : 'INC',
        'Pattali Makkal Katchi'                      : 'PMK',
        'Viduthalai Chiruthaigal Katchi'             : 'VCK',
        'Communist Party of India (Marxist)'         : 'CPI(M)',
        'Communist Party of India'                   : 'CPI',
        'Indian Union Muslim League'                 : 'IUML',
        'Amma Makkal Munnettra Kazagam'              : 'AMMK'
    }
    party_seats['short_name'] = party_seats['party'].map(name_map).fillna(party_seats['party'])

    # Only show top 10
    top10 = party_seats.head(10)

    colors = ['#1a6b3c' if p == 'TVK' else
              '#e63946' if p == 'DMK' else
              '#2196F3' if p == 'AIADMK' else
              '#ff9800' for p in top10['short_name']]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(top10['short_name'], top10['seats'], color=colors, edgecolor='white', width=0.6)

    # Add value labels on top of each bar
    for bar, val in zip(bars, top10['seats']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_title('Tamil Nadu Assembly Election 2026 — Seats Won by Party',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Party', fontsize=11)
    ax.set_ylabel('Seats Won', fontsize=11)
    ax.set_ylim(0, top10['seats'].max() + 15)
    plt.xticks(rotation=30, ha='right')
    plt.tight_layout()

    save_chart('01_seats_won_by_party.png')


# =============================================================================
# CHART 2 — Vote Share Distribution (Box Plot)
# WHY: Shows how spread out vote percentages are for each major party
# =============================================================================

def chart_vote_share_distribution(df):
    print("\n📊 Chart 2: Vote Share Distribution by Party")

    major = ['Tamilaga Vettri Kazhagam', 'Dravida Munnetra Kazhagam',
             'All India Anna Dravida Munnetra Kazhagam',
             'Naam Tamilar Katchi', 'Independent']

    df_major = df[df['party'].isin(major)].copy()

    short = {
        'Tamilaga Vettri Kazhagam'                   : 'TVK',
        'Dravida Munnetra Kazhagam'                  : 'DMK',
        'All India Anna Dravida Munnetra Kazhagam'   : 'AIADMK',
        'Naam Tamilar Katchi'                        : 'NTK',
        'Independent'                                : 'IND'
    }
    df_major['short_name'] = df_major['party'].map(short)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df_major, x='short_name', y='pct_votes',
                palette=['#1a6b3c','#e63946','#2196F3','#9c27b0','#9e9e9e'],
                width=0.5, ax=ax)

    ax.set_title('Vote Share (%) Distribution — Major Parties',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Party', fontsize=11)
    ax.set_ylabel('Vote Share (%)', fontsize=11)
    plt.tight_layout()

    save_chart('02_vote_share_distribution.png')


# =============================================================================
# CHART 3 — Winning Margin Distribution (Histogram)
# WHY: Tells us how competitive the election was overall
# =============================================================================

def chart_winning_margins(df):
    print("\n📊 Chart 3: Winning Margin Distribution")

    winners = df[df['is_winner'] == 1].copy()
    margins = winners['winning_margin']

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.hist(margins, bins=30, color='#1a6b3c', edgecolor='white', alpha=0.85)

    # Add mean and median lines
    ax.axvline(margins.mean(),   color='#e63946', linestyle='--', linewidth=2,
               label=f'Mean: {margins.mean():,.0f}')
    ax.axvline(margins.median(), color='#ff9800', linestyle='--', linewidth=2,
               label=f'Median: {margins.median():,.0f}')

    ax.set_title('Distribution of Winning Margins — Tamil Nadu 2026',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Winning Margin (Votes)', fontsize=11)
    ax.set_ylabel('Number of Constituencies', fontsize=11)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.legend(fontsize=10)
    plt.tight_layout()

    save_chart('03_winning_margin_distribution.png')


# =============================================================================
# CHART 4 — Top 10 Biggest Wins & Closest Contests
# WHY: Highlights the most dramatic results — great for LinkedIn storytelling
# =============================================================================

def chart_top_margins(df):
    print("\n📊 Chart 4: Biggest Wins & Closest Contests")

    winners = df[df['is_winner'] == 1][['constituency', 'candidate', 'party', 'winning_margin']].drop_duplicates('constituency')

    name_map = {
        'Tamilaga Vettri Kazhagam'                   : 'TVK',
        'Dravida Munnetra Kazhagam'                  : 'DMK',
        'All India Anna Dravida Munnetra Kazhagam'   : 'AIADMK',
    }

    top_wins    = winners.nlargest(10, 'winning_margin').copy()
    top_close   = winners.nsmallest(10, 'winning_margin').copy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Biggest wins ---
    top_wins['short_const'] = top_wins['constituency'].str.split(' - ').str[0].str[:20]
    top_wins['color'] = top_wins['party'].map(name_map).fillna('Others').map(
        {'TVK':'#1a6b3c','DMK':'#e63946','AIADMK':'#2196F3','Others':'#ff9800'})

    bars1 = ax1.barh(top_wins['short_const'], top_wins['winning_margin'],
                     color=top_wins['color'], edgecolor='white')
    for bar, val in zip(bars1, top_wins['winning_margin']):
        ax1.text(bar.get_width() + 500, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)
    ax1.set_title('🏆 Top 10 Biggest Wins', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Winning Margin (Votes)')
    ax1.invert_yaxis()
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))

    # --- Closest contests ---
    top_close['short_const'] = top_close['constituency'].str.split(' - ').str[0].str[:20]
    top_close['color'] = top_close['party'].map(name_map).fillna('Others').map(
        {'TVK':'#1a6b3c','DMK':'#e63946','AIADMK':'#2196F3','Others':'#ff9800'})

    bars2 = ax2.barh(top_close['short_const'], top_close['winning_margin'],
                     color=top_close['color'], edgecolor='white')
    for bar, val in zip(bars2, top_close['winning_margin']):
        ax2.text(bar.get_width() + 5, bar.get_y() + bar.get_height()/2,
                 f'{val:,}', va='center', fontsize=8)
    ax2.set_title('⚔️ Top 10 Closest Contests', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Winning Margin (Votes)')
    ax2.invert_yaxis()

    plt.suptitle('Tamil Nadu Assembly Election 2026 — Dramatic Results',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    save_chart('04_biggest_wins_closest_contests.png')


# =============================================================================
# CHART 5 — Winner vs Loser: Vote Share Comparison (KDE Plot)
# WHY: Shows the clear separation between winners and losers — this is
#      what motivates building an ML classifier
# =============================================================================

def chart_winner_vs_loser(df):
    print("\n📊 Chart 5: Winner vs Loser — Vote Share Comparison")

    fig, ax = plt.subplots(figsize=(11, 5))

    winners = df[df['is_winner'] == 1]['pct_votes']
    losers  = df[df['is_winner'] == 0]['pct_votes']

    ax.hist(winners, bins=40, alpha=0.6, color='#1a6b3c', label=f'Winners (n={len(winners)})', density=True)
    ax.hist(losers,  bins=40, alpha=0.6, color='#e63946',  label=f'Losers  (n={len(losers)})',  density=True)

    ax.set_title('Vote Share Distribution: Winners vs Losers',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Vote Share (%)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=11)
    plt.tight_layout()

    save_chart('05_winner_vs_loser_vote_share.png')


# =============================================================================
# CHART 6 — Candidates per Constituency (Histogram)
# WHY: Understand competition level — more candidates = vote splitting
# =============================================================================

def chart_candidates_per_constituency(df):
    print("\n📊 Chart 6: Candidates per Constituency")

    counts = df.groupby('constituency')['candidate'].count()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(counts, bins=20, color='#7b2d8b', edgecolor='white', alpha=0.85)
    ax.axvline(counts.mean(), color='#ff9800', linestyle='--', linewidth=2,
               label=f'Mean: {counts.mean():.1f} candidates')

    ax.set_title('Number of Candidates per Constituency',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Candidates', fontsize=11)
    ax.set_ylabel('Number of Constituencies', fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()

    save_chart('06_candidates_per_constituency.png')


# =============================================================================
# CHART 7 — Correlation Heatmap
# WHY: See which features are related to each other and to is_winner
#      This directly guides which features to use in ML
# =============================================================================

def chart_correlation_heatmap(df):
    print("\n📊 Chart 7: Correlation Heatmap")

    num_cols = ['evm_votes', 'postal_votes', 'total_votes', 'pct_votes',
                'winning_margin', 'candidates_in_constituency',
                'postal_vote_ratio', 'is_major_party', 'is_independent',
                'party_win_rate', 'is_winner']

    corr = df[num_cols].corr().round(2)

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # Hide upper triangle

    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0, linewidths=0.5, ax=ax,
                annot_kws={'size': 8})

    ax.set_title('Feature Correlation Heatmap\n(Focus on is_winner row)',
                 fontsize=13, fontweight='bold', pad=15)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()

    save_chart('07_correlation_heatmap.png')


# =============================================================================
# CHART 8 — Party Win Rate (Top 15 parties)
# WHY: Shows which parties are most efficient — wins per candidate fielded
# =============================================================================

def chart_party_win_rate(df):
    print("\n📊 Chart 8: Party Win Rate")

    # Only parties that fielded at least 5 candidates (avoid 1-win wonders)
    party_stats = df.groupby('party').agg(
        candidates=('candidate', 'count'),
        wins=('is_winner', 'sum')
    ).reset_index()
    party_stats = party_stats[party_stats['candidates'] >= 5].copy()
    party_stats['win_rate'] = (party_stats['wins'] / party_stats['candidates'] * 100).round(1)
    party_stats = party_stats.sort_values('win_rate', ascending=True).tail(12)

    name_map = {
        'Tamilaga Vettri Kazhagam'                   : 'TVK',
        'Dravida Munnetra Kazhagam'                  : 'DMK',
        'All India Anna Dravida Munnetra Kazhagam'   : 'AIADMK',
        'Indian National Congress'                   : 'INC',
        'Pattali Makkal Katchi'                      : 'PMK',
        'Viduthalai Chiruthaigal Katchi'             : 'VCK',
        'Communist Party of India (Marxist)'         : 'CPI(M)',
        'Communist Party of India'                   : 'CPI',
        'Indian Union Muslim League'                 : 'IUML',
        'Naam Tamilar Katchi'                        : 'NTK',
        'Tamizhaga Vaazhvurimai Katchi'              : 'TVK2',
        'Bahujan Samaj Party'                        : 'BSP'
    }
    party_stats['short'] = party_stats['party'].map(name_map).fillna(party_stats['party'])

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ['#1a6b3c' if r > 40 else '#2196F3' if r > 20 else '#ff9800'
              for r in party_stats['win_rate']]
    bars = ax.barh(party_stats['short'], party_stats['win_rate'],
                   color=colors, edgecolor='white')

    for bar, val in zip(bars, party_stats['win_rate']):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val}%', va='center', fontsize=9, fontweight='bold')

    ax.set_title('Party Win Rate — Wins per Candidates Fielded\n(Parties with 5+ candidates)',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Win Rate (%)', fontsize=11)
    ax.set_xlim(0, party_stats['win_rate'].max() + 10)
    plt.tight_layout()

    save_chart('08_party_win_rate.png')


# =============================================================================
# PRINT EDA SUMMARY STATS
# =============================================================================

def print_eda_summary(df):
    print("\n" + "=" * 60)
    print("EDA SUMMARY — KEY INSIGHTS")
    print("=" * 60)

    winners = df[df['is_winner'] == 1]

    print(f"\n🏆 Total Constituencies  : {df['constituency'].nunique()}")
    print(f"👥 Total Candidates      : {df['candidate'].nunique()}")
    print(f"🎯 Total Parties         : {df['party'].nunique()}")

    print(f"\n📊 Winner Party Summary (Top 5):")
    print(winners['party'].value_counts().head(5).to_string())

    print(f"\n📊 Winning Margin Stats:")
    print(f"   Average  : {winners['winning_margin'].mean():,.0f} votes")
    print(f"   Median   : {winners['winning_margin'].median():,.0f} votes")
    print(f"   Smallest : {winners['winning_margin'].min():,.0f} votes")
    print(f"   Largest  : {winners['winning_margin'].max():,.0f} votes")

    print(f"\n📊 Class Balance (for ML):")
    total = len(df)
    win   = df['is_winner'].sum()
    print(f"   Winners : {win}  ({win/total*100:.1f}%)")
    print(f"   Losers  : {total-win}  ({(total-win)/total*100:.1f}%)")
    print(f"   → Imbalanced dataset! We will handle this in ML step.")

    print(f"\n📊 Top correlations with is_winner:")
    corr = df[['evm_votes','total_votes','pct_votes','winning_margin',
               'party_win_rate','is_major_party','is_independent','is_winner']].corr()
    top_corr = corr['is_winner'].drop('is_winner').sort_values(key=abs, ascending=False)
    print(top_corr.round(3).to_string())


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    DATA_PATH = "data/processed/tn_election_cleaned.csv"

    print("\n🚀 TAMIL NADU ELECTION 2026 — EDA & VISUALIZATION")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded processed data: {df.shape[0]} rows x {df.shape[1]} columns")

    print_eda_summary(df)

    print("\n" + "=" * 60)
    print("Generating Charts...")
    print("=" * 60)

    chart_seats_won(df)
    chart_vote_share_distribution(df)
    chart_winning_margins(df)
    chart_top_margins(df)
    chart_winner_vs_loser(df)
    chart_candidates_per_constituency(df)
    chart_correlation_heatmap(df)
    chart_party_win_rate(df)

    print("\n" + "=" * 60)
    print("✅ ALL 8 CHARTS SAVED to outputs/figures/")
    print("=" * 60)