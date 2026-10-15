# Blood Pressure Analysis (NHANES 2015–2016)

A Python project that explores how age, gender, and race/ethnicity relate to blood pressure in the U.S. population, using public data from the CDC's National Health and Nutrition Examination Survey (NHANES).

The script does three things:

1. **Descriptive analysis**: average systolic and diastolic blood pressure by race/ethnicity.
2. **Classification**: a Random Forest that predicts whether a person falls into a "hypertension risk" group.
3. **Regression**: two Linear Regression models that predict systolic and diastolic blood pressure in mmHg.

## Data

Both files are SAS transport (`.xpt`) files from the NHANES 2015–2016 cycle, included in this repo:

| File | NHANES component | Used for |
|------|------------------|----------|
| `DEMO_I.xpt` | Demographics | Age, gender, race/ethnicity |
| `BPX_I.xpt` | Blood pressure examination | First systolic and diastolic readings |

The two files are merged on the participant ID (`SEQN`).

| Variable | Meaning |
|----------|---------|
| `RIDAGEYR` | Age in years |
| `RIAGENDR` | Gender (1 = male, 2 = female) |
| `RIDRETH3` | Race/ethnicity (see mapping in the script) |
| `BPXSY1` | Systolic blood pressure, first reading (mmHg) |
| `BPXDI1` | Diastolic blood pressure, first reading (mmHg) |

## How it works

- **Hypertension risk label:** a participant is labeled `1` if systolic is above 130 mmHg **or** diastolic is above 80 mmHg, otherwise `0`.
- **Features:** age, gender, and race/ethnicity (gender and race are one-hot encoded). Rows with missing values are dropped.
- **Train/test split:** 80/20 with `random_state=42` for reproducible results.
- **Random Forest:** `RandomForestClassifier` with 100 trees.
- **Linear Regression:** separate models for systolic and diastolic blood pressure.

## Results

Results from a run on the included data:

**Average blood pressure by race/ethnicity (mmHg)**

| Group | Systolic | Diastolic |
|-------|----------|-----------|
| Mexican American | 119.6 | 65.0 |
| Other Hispanic | 121.3 | 65.1 |
| Non-Hispanic White | 120.4 | 65.9 |
| Non-Hispanic Black | 123.1 | 67.0 |
| Non-Hispanic Asian | 118.1 | 69.1 |
| Other/Multi-Racial | 117.2 | 65.9 |

**Random Forest (hypertension risk)**

| Metric | Value |
|--------|-------|
| Accuracy | 0.706 |
| AUC-ROC | 0.735 |
| Precision / recall (at-risk class) | 0.48 / 0.46 |

**Linear Regression**

| Target | R² | RMSE (mmHg) | MAE (mmHg) |
|--------|-----|-------------|------------|
| Systolic | 0.366 | 14.10 | 10.53 |
| Diastolic | 0.071 | 13.67 | 10.25 |

Age is the most consistent predictor: in the regression, each additional year adds about 0.5 mmHg to systolic pressure on average.

### Interpreting these results

The models use only three demographic features, so they are meant for exploration, not diagnosis. Demographics explain a moderate share of systolic variation and very little of diastolic variation. The risk label is also based on a single blood pressure reading, which is a simplification of how hypertension is actually diagnosed. Race/ethnicity coefficients reflect differences in this sample and say nothing about causes.

## Getting started

### Requirements

- Python 3.8+
- `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run

```bash
git clone https://github.com/Suhaib-Abdi/Blood-Pressure-Project.git
cd Blood-Pressure-Project
python blood_pressure_python.py
```

The script reads the `.xpt` files from its own folder, prints the metrics to the console, and opens the plots in sequence (close each window to continue).

## Output

- Console summary of group averages, Random Forest metrics, and regression metrics and coefficients
- Bar chart of average blood pressure by race/ethnicity
- Bar chart of Random Forest feature importances
- `blood_pressure_regression.png`: a 2×2 figure with actual-vs-predicted plots and coefficient charts for both regression models
