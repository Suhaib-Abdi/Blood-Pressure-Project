import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (classification_report, roc_auc_score,accuracy_score, mean_squared_error, mean_absolute_error, r2_score)

base_path = os.path.dirname(os.path.abspath(__file__))
demo_df = pd.read_sas(os.path.join(base_path, "DEMO_I.xpt"))
bp_df = pd.read_sas(os.path.join(base_path, "BPX_I.xpt"))

merged_df = pd.merge(demo_df, bp_df, on='SEQN')

cols = ['SEQN', 'RIDRETH3', 'RIAGENDR', 'RIDAGEYR', 'BPXSY1', 'BPXDI1']
final_df = merged_df[cols].copy()

averages = final_df.groupby('RIDRETH3')[['BPXSY1', 'BPXDI1']].mean()

race_labels = {
    1.0: "Mexican American",
    2.0: "Other Hispanic",
    3.0: "Non-Hispanic White",
    4.0: "Non-Hispanic Black",
    6.0: "Non-Hispanic Asian",
    7.0: "Other/Multi-Racial"
}
averages.index = averages.index.map(race_labels)
print(averages)
print("\n" + "="*40 + "\n")

final_df['Hypertension_Risk'] = np.where((final_df['BPXSY1'] > 130) | (final_df['BPXDI1'] > 80), 1, 0)

df_clean = final_df.dropna()
X = df_clean[['RIAGENDR', 'RIDAGEYR', 'RIDRETH3']]
y = df_clean['Hypertension_Risk']
X = pd.get_dummies(X, columns=['RIAGENDR', 'RIDRETH3'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("Random Forest Model Results")
print(f"Overall Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred))

sns.set_theme(style="whitegrid")
averages.plot(kind='bar', figsize=(12, 6), color=['#d95f02', '#1b9e77'])
plt.title('Average Blood Pressure by Race / Ethnicity (2015-2016)', fontsize=16, fontweight='bold')
plt.xlabel('Race / Ethnicity', fontsize=12)
plt.ylabel('Blood Pressure (mmHg)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=11)
plt.legend(['Systolic', 'Diastolic'], title='Measurement', fontsize=11)
plt.axhline(y=130, color='red', linestyle='--', alpha=0.7, label='Hypertension Threshold (130)')
plt.legend()
plt.tight_layout()
plt.show()

feature_names = X_train.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
sorted_features = [feature_names[i] for i in indices]
sorted_importances = importances[indices]
clean_features = [f.replace('RIDRETH3_', 'Race: ').replace('RIAGENDR_', 'Gender: ') for f in sorted_features]
clean_features = ['Age' if f == 'RIDAGEYR' else f for f in clean_features]

plt.figure(figsize=(10, 6))
sns.barplot(x=sorted_importances, y=clean_features, hue=clean_features, palette="viridis", legend=False)
plt.title('Top Factors Driving Hypertension Risk Predictions', fontsize=16, fontweight='bold')
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Demographic Factor', fontsize=12)
plt.tight_layout()
plt.show()

# LINEAR REGRESSION 

print("\n" + "="*40)
print("  LINEAR REGRESSION RESULTS")
print("="*40)

y_sys = df_clean['BPXSY1']
y_dia = df_clean['BPXDI1']

X_train_lr, X_test_lr, ys_train, ys_test, yd_train, yd_test = train_test_split(
    X, y_sys, y_dia, test_size=0.2, random_state=42
)

model_sys = LinearRegression()
model_dia = LinearRegression()
model_sys.fit(X_train_lr, ys_train)
model_dia.fit(X_train_lr, yd_train)

ys_pred = model_sys.predict(X_test_lr)
yd_pred = model_dia.predict(X_test_lr)

def print_lr_metrics(name, y_true, y_pred, intercept):
    print(f"\n--- {name} ---")
    print(f"  Intercept : {intercept:.2f} mmHg")
    print(f"  R²        : {r2_score(y_true, y_pred):.4f}")
    print(f"  RMSE      : {np.sqrt(mean_squared_error(y_true, y_pred)):.2f} mmHg")
    print(f"  MAE       : {mean_absolute_error(y_true, y_pred):.2f} mmHg")

print_lr_metrics("Systolic BP (BPXSY1)",  ys_test, ys_pred, model_sys.intercept_)
print_lr_metrics("Diastolic BP (BPXDI1)", yd_test, yd_pred, model_dia.intercept_)

def coef_table(model, feature_names):
    rename = {f: f.replace('RIDRETH3_', 'Race: ').replace('RIAGENDR_', 'Gender: ') for f in feature_names}
    rename['RIDAGEYR'] = 'Age (years)'
    return pd.DataFrame({
        'Feature':     [rename.get(f, f) for f in feature_names],
        'Coefficient': model.coef_
    }).sort_values('Coefficient', key=abs, ascending=False)

print("\n--- Systolic Coefficients ---")
print(coef_table(model_sys, X.columns).to_string(index=False))
print("\n--- Diastolic Coefficients ---")
print(coef_table(model_dia, X.columns).to_string(index=False))

# Linear Regression Plots
fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle('Linear Regression – Blood Pressure Analysis (NHANES 2015-2016)',fontsize=16, fontweight='bold')

# Actual vs Predicted – Systolic
ax = axes[0, 0]
ax.scatter(ys_test, ys_pred, alpha=0.3, color='#1f77b4', edgecolors='none', s=15)
lims = [min(ys_test.min(), ys_pred.min()), max(ys_test.max(), ys_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual Systolic BP (mmHg)')
ax.set_ylabel('Predicted Systolic BP (mmHg)')
ax.set_title(f'Actual vs Predicted – Systolic\nR² = {r2_score(ys_test, ys_pred):.3f}')
ax.legend()

# Actual vs Predicted – Diastolic
ax = axes[0, 1]
ax.scatter(yd_test, yd_pred, alpha=0.3, color='#2ca02c', edgecolors='none', s=15)
lims = [min(yd_test.min(), yd_pred.min()), max(yd_test.max(), yd_pred.max())]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect fit')
ax.set_xlabel('Actual Diastolic BP (mmHg)')
ax.set_ylabel('Predicted Diastolic BP (mmHg)')
ax.set_title(f'Actual vs Predicted – Diastolic\nR² = {r2_score(yd_test, yd_pred):.3f}')
ax.legend()

# Coefficients – Systolic
ax = axes[1, 0]
coef_sys = coef_table(model_sys, X.columns)
colors = ['#d62728' if c > 0 else '#1f77b4' for c in coef_sys['Coefficient']]
ax.barh(coef_sys['Feature'], coef_sys['Coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (mmHg)')
ax.set_title('Regression Coefficients – Systolic BP')

# Coefficients – Diastolic
ax = axes[1, 1]
coef_dia = coef_table(model_dia, X.columns)
colors = ['#d62728' if c > 0 else '#1f77b4' for c in coef_dia['Coefficient']]
ax.barh(coef_dia['Feature'], coef_dia['Coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (mmHg)')
ax.set_title('Regression Coefficients – Diastolic BP')

plt.tight_layout()
plt.savefig('blood_pressure_regression.png', dpi=150, bbox_inches='tight')
plt.show()
