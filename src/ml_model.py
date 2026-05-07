# =============================================================================
# FILE: src/ml_model.py
# PROJECT: Tamil Nadu Election 2026 - Winner Prediction
# PURPOSE: Build, train, evaluate and save ML models
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve, accuracy_score)
from xgboost import XGBClassifier
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

plt.rcParams['figure.dpi']        = 150
plt.rcParams['axes.spines.top']   = False
plt.rcParams['axes.spines.right'] = False

OUTPUT_FIGURES = 'outputs/figures'
OUTPUT_REPORTS = 'outputs/reports'
MODELS_DIR     = 'outputs/models'

for d in [OUTPUT_FIGURES, OUTPUT_REPORTS, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)


# =============================================================================
# STEP 4.1 — LOAD & PREPARE DATA
# =============================================================================

def load_and_prepare(filepath):
    """
    Loads processed data and prepares features (X) and target (y).
    WHY: ML models only understand numbers. We select the right columns.
    """
    print("=" * 60)
    print("STEP 4.1: Loading & Preparing Data for ML")
    print("=" * 60)

    df = pd.read_csv(filepath)
    print(f"✅ Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    # --- Select feature columns ---
    # WHY these features?
    # - pct_votes         → direct vote share signal
    # - total_votes       → raw vote count
    # - evm_votes         → electronic votes
    # - postal_vote_ratio → mobilization signal
    # - party_win_rate    → historical party strength
    # - is_major_party    → big party advantage
    # - is_independent    → independents rarely win
    # - candidates_in_constituency → competition level
    # - winning_margin    → how decisive the result was
    # NOTE: We DO include total_votes/pct_votes here because in a
    # real deployment you'd have partial round data — this simulates
    # predicting from available counting data.

    FEATURES = [
        'pct_votes',
        'total_votes',
        'evm_votes',
        'postal_vote_ratio',
        'party_win_rate',
        'is_major_party',
        'is_independent',
        'candidates_in_constituency',
        'winning_margin'
    ]

    TARGET = 'is_winner'

    X = df[FEATURES]
    y = df[TARGET]

    print(f"\n📌 Features selected ({len(FEATURES)}):")
    for i, f in enumerate(FEATURES, 1):
        print(f"   {i}. {f}")

    print(f"\n📌 Target: '{TARGET}'")
    print(f"   Winners (1): {y.sum()}  ({y.mean()*100:.1f}%)")
    print(f"   Losers  (0): {(y==0).sum()}  ({(y==0).mean()*100:.1f}%)")
    print(f"\n   ⚠️  Class imbalance detected — will use class_weight='balanced'")

    return X, y, df, FEATURES


# =============================================================================
# STEP 4.2 — TRAIN / TEST SPLIT
# =============================================================================

def split_data(X, y):
    """
    Splits data into training set and test set.
    WHY: We train on 80% of data and test on the remaining 20% that the
    model has NEVER seen — this gives an honest accuracy score.
    """
    print("\n" + "=" * 60)
    print("STEP 4.2: Train / Test Split")
    print("=" * 60)

    # stratify=y ensures both train and test have same winner/loser ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"✅ Training set : {X_train.shape[0]} rows")
    print(f"✅ Test set     : {X_test.shape[0]} rows")
    print(f"   Train winners: {y_train.sum()} | Test winners: {y_test.sum()}")

    return X_train, X_test, y_train, y_test


# =============================================================================
# STEP 4.3 — SCALE FEATURES
# =============================================================================

def scale_features(X_train, X_test):
    """
    Standardizes numerical features to have mean=0 and std=1.
    WHY: Logistic Regression is sensitive to feature scales.
    Large values like total_votes (200,000) vs is_independent (0/1)
    would confuse the model without scaling.
    """
    print("\n" + "=" * 60)
    print("STEP 4.3: Feature Scaling")
    print("=" * 60)

    scaler = StandardScaler()

    # IMPORTANT: Fit ONLY on training data, then transform both
    # WHY: Fitting on test data = "data leakage" = cheating!
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    print(f"✅ Features scaled using StandardScaler")
    print(f"   Fit on training data only (no data leakage)")

    return X_train_scaled, X_test_scaled, scaler


# =============================================================================
# STEP 4.4 — TRAIN MULTIPLE MODELS
# =============================================================================

def train_models(X_train, y_train):
    """
    Trains 4 different ML models and returns all of them.
    WHY: We never know which model will perform best on a dataset.
    We try multiple and pick the winner based on cross-validation score.
    """
    print("\n" + "=" * 60)
    print("STEP 4.4: Training Models")
    print("=" * 60)

    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced',   # handles imbalanced classes
            max_iter=1000,
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100,          # 100 decision trees
            class_weight='balanced',
            random_state=42,
            n_jobs=-1                  # use all CPU cores
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            random_state=42
        ),
        'XGBoost': XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            scale_pos_weight=16,       # handles class imbalance
            random_state=42,
            eval_metric='logloss',
            verbosity=0
        )
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_results = {}

    for name, model in models.items():
        print(f"\n   Training: {name}...")
        model.fit(X_train, y_train)

        # 5-Fold Cross Validation score
        # WHY: More reliable than a single train/test split
        cv_scores = cross_val_score(model, X_train, y_train,
                                    cv=cv, scoring='roc_auc', n_jobs=-1)
        cv_results[name] = cv_scores

        print(f"   ✅ CV ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return models, cv_results


# =============================================================================
# STEP 4.5 — EVALUATE MODELS
# =============================================================================

def evaluate_models(models, X_test, y_test):
    """
    Evaluates each model on the unseen test set.
    WHY: Test set performance = real-world performance estimate.
    """
    print("\n" + "=" * 60)
    print("STEP 4.5: Model Evaluation on Test Set")
    print("=" * 60)

    results = {}

    for name, model in models.items():
        y_pred      = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)[:, 1]

        acc     = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)

        results[name] = {
            'accuracy' : acc,
            'roc_auc'  : roc_auc,
            'y_pred'   : y_pred,
            'y_prob'   : y_pred_prob
        }

        print(f"\n{'─'*40}")
        print(f"  {name}")
        print(f"{'─'*40}")
        print(f"  Accuracy : {acc*100:.2f}%")
        print(f"  ROC-AUC  : {roc_auc:.4f}")
        print(f"\n  Classification Report:")
        print(classification_report(y_test, y_pred,
              target_names=['Loser (0)', 'Winner (1)']))

    return results


# =============================================================================
# STEP 4.6 — FIND BEST MODEL
# =============================================================================

def find_best_model(results, models):
    """
    Picks the best model based on ROC-AUC score.
    WHY: ROC-AUC is better than accuracy for imbalanced datasets.
    Accuracy can be misleading — if 94% are losers, predicting
    everyone as loser gives 94% accuracy but is useless!
    """
    print("\n" + "=" * 60)
    print("STEP 4.6: Best Model Selection")
    print("=" * 60)

    best_name  = max(results, key=lambda x: results[x]['roc_auc'])
    best_score = results[best_name]['roc_auc']

    print(f"\n🏆 Best Model : {best_name}")
    print(f"   ROC-AUC   : {best_score:.4f}")

    print(f"\n📊 All Models Ranked by ROC-AUC:")
    ranked = sorted(results.items(), key=lambda x: x[1]['roc_auc'], reverse=True)
    for i, (name, res) in enumerate(ranked, 1):
        print(f"   {i}. {name:30s} → AUC: {res['roc_auc']:.4f}  Acc: {res['accuracy']*100:.2f}%")

    return best_name, models[best_name]


# =============================================================================
# CHART 9 — Model Comparison Bar Chart
# =============================================================================

def chart_model_comparison(results):
    print("\n📊 Chart 9: Model Comparison")

    names   = list(results.keys())
    aucs    = [results[n]['roc_auc'] for n in names]
    accs    = [results[n]['accuracy'] for n in names]

    x     = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, aucs, width, label='ROC-AUC',  color='#1a6b3c', alpha=0.85)
    bars2 = ax.bar(x + width/2, accs, width, label='Accuracy', color='#2196F3', alpha=0.85)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', fontsize=9, fontweight='bold')

    ax.set_title('ML Model Comparison — ROC-AUC vs Accuracy',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1.12)
    ax.set_ylabel('Score')
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(OUTPUT_FIGURES, '09_model_comparison.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   💾 Saved: {path}")


# =============================================================================
# CHART 10 — Confusion Matrix of Best Model
# =============================================================================

def chart_confusion_matrix(best_name, y_test, y_pred):
    print("\n📊 Chart 10: Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens',
                xticklabels=['Predicted Loser', 'Predicted Winner'],
                yticklabels=['Actual Loser', 'Actual Winner'],
                linewidths=1, ax=ax, annot_kws={'size': 14})

    ax.set_title(f'Confusion Matrix — {best_name}',
                 fontsize=13, fontweight='bold', pad=15)
    plt.tight_layout()

    path = os.path.join(OUTPUT_FIGURES, '10_confusion_matrix.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   💾 Saved: {path}")


# =============================================================================
# CHART 11 — ROC Curve for All Models
# =============================================================================

def chart_roc_curves(results, y_test):
    print("\n📊 Chart 11: ROC Curves")

    colors = ['#1a6b3c', '#e63946', '#2196F3', '#ff9800']
    fig, ax = plt.subplots(figsize=(9, 7))

    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {res['roc_auc']:.3f})")

    ax.plot([0,1], [0,1], 'k--', linewidth=1, label='Random Classifier')
    ax.set_title('ROC Curves — All Models', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.legend(fontsize=10)
    plt.tight_layout()

    path = os.path.join(OUTPUT_FIGURES, '11_roc_curves.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   💾 Saved: {path}")


# =============================================================================
# CHART 12 — Feature Importance (Best Model)
# =============================================================================

def chart_feature_importance(best_name, best_model, feature_names):
    print("\n📊 Chart 12: Feature Importance")

    # Only tree-based models have feature_importances_
    if not hasattr(best_model, 'feature_importances_'):
        print("   ⚠️  Logistic Regression selected — showing coefficients instead")
        importances = np.abs(best_model.coef_[0])
    else:
        importances = best_model.feature_importances_

    feat_df = pd.DataFrame({
        'feature'    : feature_names,
        'importance' : importances
    }).sort_values('importance', ascending=True)

    colors = ['#1a6b3c' if v > feat_df['importance'].median()
              else '#9e9e9e' for v in feat_df['importance']]

    fig, ax = plt.subplots(figsize=(10, 7))
    bars = ax.barh(feat_df['feature'], feat_df['importance'],
                   color=colors, edgecolor='white')

    for bar, val in zip(bars, feat_df['importance']):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.set_title(f'Feature Importance — {best_name}',
                 fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Importance Score', fontsize=11)
    plt.tight_layout()

    path = os.path.join(OUTPUT_FIGURES, '12_feature_importance.png')
    plt.savefig(path, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"   💾 Saved: {path}")


# =============================================================================
# STEP 4.7 — SAVE BEST MODEL
# =============================================================================

def save_best_model(best_name, best_model, scaler, feature_names):
    """
    Saves the trained model to disk so it can be reused without retraining.
    WHY: Training takes time. Save once, load anytime.
    """
    print("\n" + "=" * 60)
    print("STEP 4.7: Saving Best Model")
    print("=" * 60)

    model_path  = os.path.join(MODELS_DIR, 'best_model.pkl')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    joblib.dump(best_model, model_path)
    joblib.dump(scaler,     scaler_path)

    # Save feature list too — important for deployment
    feat_path = os.path.join(MODELS_DIR, 'features.txt')
    with open(feat_path, 'w') as f:
        f.write('\n'.join(feature_names))

    print(f"✅ Model  saved : {model_path}")
    print(f"✅ Scaler saved : {scaler_path}")
    print(f"✅ Features saved: {feat_path}")
    print(f"\n   Best Model: {best_name}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    DATA_PATH = "data/processed/tn_election_cleaned.csv"

    print("\n🚀 TAMIL NADU ELECTION 2026 — ML MODEL PIPELINE")
    print("=" * 60)

    # Run all steps
    X, y, df, FEATURES          = load_and_prepare(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_sc, X_test_sc, scaler    = scale_features(X_train, X_test)
    models, cv_results               = train_models(X_train_sc, y_train)
    results                          = evaluate_models(models, X_test_sc, y_test)
    best_name, best_model            = find_best_model(results, models)

    # Generate charts
    print("\n" + "=" * 60)
    print("Generating ML Charts...")
    print("=" * 60)
    chart_model_comparison(results)
    chart_confusion_matrix(best_name, y_test, results[best_name]['y_pred'])
    chart_roc_curves(results, y_test)
    chart_feature_importance(best_name, best_model, FEATURES)

    # Save best model
    save_best_model(best_name, best_model, scaler, FEATURES)

    print("\n" + "=" * 60)
    print("✅ ML PIPELINE COMPLETE!")
    print(f"   Best Model : {best_name}")
    print(f"   Charts saved to : outputs/figures/")
    print(f"   Model saved to  : outputs/models/")
    print("=" * 60)