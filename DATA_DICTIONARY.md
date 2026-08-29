# DATA_DICTIONARY.md
## Home Credit Default Risk — Dataset Reference

All 8 source tables, key columns, join relationships, missingness, and feature engineering decisions.

---

## Table Overview

| Table | Rows | Grain | Joins to |
|---|---|---|---|
| `application_train.csv` | 307,511 | 1 row per loan application (train) | — (root table) |
| `application_test.csv` | 48,744 | 1 row per loan application (test) | — (root table) |
| `bureau.csv` | 1,716,428 | 1 row per external credit account per applicant | `application` on SK_ID_CURR |
| `bureau_balance.csv` | 27,299,925 | 1 row per month per external credit account | `bureau` on SK_ID_BUREAU |
| `previous_application.csv` | 1,670,214 | 1 row per prior Home Credit application | `application` on SK_ID_CURR |
| `POS_CASH_balance.csv` | 10,001,358 | 1 row per month per prior POS/cash loan | `previous_application` on SK_ID_PREV |
| `installments_payments.csv` | 13,605,401 | 1 row per installment per prior loan | `previous_application` on SK_ID_PREV |
| `credit_card_balance.csv` | 3,840,312 | 1 row per month per prior credit card account | `previous_application` on SK_ID_PREV |

**Note:** All auxiliary tables join to `application` via SK_ID_CURR (many-to-one from auxiliary → application). POS_CASH, installments, and credit_card also carry SK_ID_PREV linking to `previous_application`.

---

## Table 1: `application_train.csv` / `application_test.csv`

### Key Identifiers
| Column | Type | Notes |
|---|---|---|
| `SK_ID_CURR` | int | Primary key. Never null. |
| `TARGET` | int {0,1} | Train only. 1 = defaulted. ~8.1% positive rate. |

### Target Variable Information
- Positive class (TARGET=1): 24,825 rows (8.07%)
- Negative class (TARGET=0): 282,686 rows (91.93%)
- Significant class imbalance — must use scale_pos_weight or is_unbalance

### Critical Financial Columns
| Column | Type | Missingness | Meaning | Feature Engineering |
|---|---|---|---|---|
| `AMT_INCOME_TOTAL` | float | 0% | Annual income | Used as denominator; cap outliers at 99th percentile |
| `AMT_CREDIT` | float | 0% | Loan amount requested | Divide by income → CREDIT_INCOME_RATIO |
| `AMT_ANNUITY` | float | 0.1% | Monthly payment amount | Divide by income; use to compute CREDIT_TERM |
| `AMT_GOODS_PRICE` | float | 9.5% | Price of goods (if applicable) | GOODS_CREDIT_RATIO = AMT_CREDIT / AMT_GOODS_PRICE |

### Client Demographics
| Column | Type | Missingness | Meaning | Feature Engineering |
|---|---|---|---|---|
| `DAYS_BIRTH` | int | 0% | Days before application (negative). Age = -DAYS_BIRTH/365 | Keep as-is; create AGE_YEARS |
| `DAYS_EMPLOYED` | int | 0% | Days employed before application (negative for employed) | **Anomaly: 55,374 rows = 365243 (≈ 1000 years) → means unemployed. Create EMPLOYED_ANOM flag, set DAYS_EMPLOYED=NaN for anomalies** |
| `DAYS_REGISTRATION` | float | 0% | Days since last registration change | Keep as-is |
| `DAYS_ID_PUBLISH` | int | 0% | Days since ID was published | Keep as-is |
| `CODE_GENDER` | object | 0% | M / F / XNA | Binary encode M=1, F=0; XNA → NaN |
| `NAME_FAMILY_STATUS` | object | 0% | Married/Single/etc | Label encode |
| `CNT_CHILDREN` | int | 0% | Number of children | Keep; create HAS_CHILDREN flag |
| `CNT_FAM_MEMBERS` | float | 2 nulls | Family members | Impute with CNT_CHILDREN + 1 |

### External Credit Scores
| Column | Type | Missingness | Meaning |
|---|---|---|---|
| `EXT_SOURCE_1` | float | 56.4% | External credit score 1 (bureau-derived) |
| `EXT_SOURCE_2` | float | 0.4% | External credit score 2 |
| `EXT_SOURCE_3` | float | 19.8% | External credit score 3 |

**EXT_SOURCE columns are the strongest predictors (top 3 by feature importance). Impute with median. Create EXT_SOURCE_MEAN, EXT_SOURCE_MIN, EXT_SOURCE_PRODUCT.**

### Document Flags (FLAG_DOCUMENT_*)
21 binary columns (0/1) indicating which documents were submitted.
Missingness: 0%. Feature engineering: sum all → DOCUMENTS_SUBMITTED_COUNT; keep individual flags.

### Social Circle Default Rates
| Column | Type | Missingness | Meaning |
|---|---|---|---|
| `OBS_30_CNT_SOCIAL_CIRCLE` | float | 0.3% | Observations in 30-day DPD social circle |
| `DEF_30_CNT_SOCIAL_CIRCLE` | float | 0.3% | Defaults in 30-day DPD social circle |
| `OBS_60_CNT_SOCIAL_CIRCLE` | float | 0.3% | Observations in 60-day DPD social circle |
| `DEF_60_CNT_SOCIAL_CIRCLE` | float | 0.3% | Defaults in 60-day DPD social circle |
Feature engineering: DEF_30_RATE = DEF_30 / OBS_30; same for 60-day.

### Columns Dropped (>60% missing)
```
APARTMENTS_AVG, BASEMENTAREA_AVG, YEARS_BEGINEXPLUATATION_AVG,
YEARS_BUILD_AVG, COMMONAREA_AVG, ELEVATORS_AVG, ENTRANCES_AVG,
FLOORSMAX_AVG, FLOORSMIN_AVG, LANDAREA_AVG, LIVINGAPARTMENTS_AVG,
LIVINGAREA_AVG, NONLIVINGAPARTMENTS_AVG, NONLIVINGAREA_AVG,
(and all corresponding _MODE, _MEDI variants)
```
Threshold: computed on train, applied identically to test.

---

## Table 2: `bureau.csv`

### Key Columns
| Column | Type | Missingness | Meaning |
|---|---|---|---|
| `SK_ID_CURR` | int | 0% | Links to application |
| `SK_ID_BUREAU` | int | 0% | Unique bureau account ID (primary key of this table) |
| `CREDIT_ACTIVE` | object | 0% | Active / Closed / Sold / Bad debt |
| `CREDIT_CURRENCY` | object | 0% | currency1/2/3/4 (anonymized) |
| `DAYS_CREDIT` | int | 0% | Days before application when credit was applied (negative) |
| `CREDIT_DAY_OVERDUE` | int | 0% | Days overdue on this credit at query time |
| `DAYS_CREDIT_ENDDATE` | float | 21.0% | Days until credit expires (can be negative = already expired) |
| `AMT_CREDIT_MAX_OVERDUE` | float | 65.3% | Maximum overdue amount ever |
| `CNT_CREDIT_PROLONG` | int | 0% | Number of times credit was prolonged |
| `AMT_CREDIT_SUM` | float | 12.8% | Total credit amount |
| `AMT_CREDIT_SUM_DEBT` | float | 33.2% | Current debt on this credit |
| `AMT_CREDIT_SUM_LIMIT` | float | 56.3% | Credit limit |
| `AMT_CREDIT_SUM_OVERDUE` | float | 0.1% | Current overdue amount |
| `CREDIT_TYPE` | object | 0% | 15 categories: Consumer credit, Credit card, Mortgage, etc. |
| `DAYS_CREDIT_UPDATE` | int | 0% | Days before application when info was last updated |
| `AMT_ANNUITY` | float | 82.6% | Annuity on this credit |

**Drop AMT_ANNUITY from bureau (82.6% missing — not worth aggregating).**

---

## Table 3: `bureau_balance.csv`

### Key Columns
| Column | Type | Missingness | Meaning |
|---|---|---|---|
| `SK_ID_BUREAU` | int | 0% | Links to bureau.csv |
| `MONTHS_BALANCE` | int | 0% | Month of observation relative to application (0 = month of application, negative = past) |
| `STATUS` | object | 0% | X=unknown, C=no loan, 0=no DPD, 1=1-30 DPD, 2=31-60 DPD, 3=61-90 DPD, 4=91-120 DPD, 5=120+ DPD |

**STATUS encoding for DPD features:** X→NaN, C→0, 0→0, 1→1, 2→2, 3→3, 4→4, 5→5.

---

## Table 4: `previous_application.csv`

### Key Columns
| Column | Type | Missingness | Meaning |
|---|---|---|---|
| `SK_ID_CURR` | int | 0% | Links to application |
| `SK_ID_PREV` | int | 0% | Unique previous application ID |
| `NAME_CONTRACT_TYPE` | object | 0% | Cash loans / Consumer loans / Revolving loans |
| `AMT_ANNUITY` | float | 0.1% | Annuity of previous loan |
| `AMT_APPLICATION` | float | 0% | Amount applied for |
| `AMT_CREDIT` | float | 0% | Final credit amount approved |
| `AMT_DOWN_PAYMENT` | float | 77.8% | Down payment (mostly null — skip) |
| `AMT_GOODS_PRICE` | float | 11.1% | Goods price |
| `NAME_CONTRACT_STATUS` | object | 0% | Approved / Refused / Canceled / Unused offer |
| `DAYS_DECISION` | int | 0% | Days before current application when decision was made (negative) |
| `NAME_PRODUCT_TYPE` | object | 0% | walk-in / XNA / x-sell |
| `CHANNEL_TYPE` | object | 0% | 8 categories for acquisition channel |
| `NAME_SELLER_INDUSTRY` | object | 0% | 11 categories |
| `CNT_PAYMENT` | float | 10.6% | Number of payments for the loan term |

---

## Table 5: `POS_CASH_balance.csv`

### Key Columns
| Column | Type | Missingness | Meaning |
|---|---|---|---|
| `SK_ID_PREV` | int | 0% | Links to previous_application |
| `SK_ID_CURR` | int | 0% | Links to application (also available directly) |
| `MONTHS_BALANCE` | int | 0% | Month of observation (negative = past) |
| `CNT_INSTALMENT` | float | 3.4% | Loan term in installments |
| `CNT_INSTALMENT_FUTURE` | float | 3.4% | Remaining installments |
| `NAME_CONTRACT_STATUS` | object | 0% | Active / Completed / Signed / etc. |
| `SK_DPD` | int | 0% | Days past due in this month |
| `SK_DPD_DEF` | int | 0% | DPD (defined tolerance) |

---

## Table 6: `installments_payments.csv`

### Key Columns
| Column | Type | Missingness | Meaning |
|---|---|---|---|
| `SK_ID_PREV` | int | 0% | Links to previous_application |
| `SK_ID_CURR` | int | 0% | Links to application |
| `NUM_INSTALMENT_VERSION` | float | 0% | Version of installment calendar |
| `NUM_INSTALMENT_NUMBER` | float | 0% | Which installment number this is |
| `DAYS_INSTALMENT` | float | 0% | When installment was scheduled (days from application; negative) |
| `DAYS_ENTRY_PAYMENT` | float | 4.9% | When actually paid (NaN = not yet paid at query time) |
| `AMT_INSTALMENT` | float | 0% | Amount scheduled to pay |
| `AMT_PAYMENT` | float | 4.9% | Amount actually paid |

**Key derived features:**
- `PAYMENT_DIFF = AMT_INSTALMENT - AMT_PAYMENT` — positive means underpaid
- `DAYS_PAST_DUE = DAYS_ENTRY_PAYMENT - DAYS_INSTALMENT` — positive means paid late
- When DAYS_ENTRY_PAYMENT is NaN: treat as "not yet paid" — set DAYS_PAST_DUE = current relative days past due if inferable, else drop from late-payment calculations

---

## Table 7: `credit_card_balance.csv`

### Key Columns
| Column | Type | Missingness | Meaning |
|---|---|---|---|
| `SK_ID_PREV` | int | 0% | Links to previous_application |
| `SK_ID_CURR` | int | 0% | Links to application |
| `MONTHS_BALANCE` | int | 0% | Month (negative = past) |
| `AMT_BALANCE` | float | 0% | Current balance |
| `AMT_CREDIT_LIMIT_ACTUAL` | int | 0% | Credit limit this month |
| `AMT_DRAWINGS_ATM_CURRENT` | float | 0.5% | ATM withdrawals this month |
| `AMT_DRAWINGS_CURRENT` | float | 0.5% | Total drawings |
| `AMT_DRAWINGS_TOTAL_CURRENT` | float | 0.5% | Total drawings (sum of all types) |
| `AMT_INST_MIN_REGULARITY` | float | 28.3% | Minimum regular installment |
| `AMT_PAYMENT_CURRENT` | float | 1.7% | Payment made this month |
| `AMT_PAYMENT_TOTAL_CURRENT` | float | 0.5% | Total payment this month |
| `AMT_RECEIVABLE_PRINCIPAL` | float | 0.5% | Principal owed |
| `AMT_RECIVABLE` | float | 0.5% | Total receivable |
| `AMT_TOTAL_RECEIVABLE` | float | 0.5% | Total receivable (another measure) |
| `CNT_DRAWINGS_ATM_CURRENT` | float | 0.5% | Count of ATM draws |
| `CNT_DRAWINGS_CURRENT` | float | 0.5% | Total draw count |
| `CNT_DRAWINGS_TOTAL_CURRENT` | float | 0.5% | Total draw count |
| `CNT_INSTALMENT_MATURE_CUM` | float | 0.5% | Cumulative mature instalments |
| `NAME_CONTRACT_STATUS` | object | 0% | Active / Completed / Signed / Demand / etc. |
| `SK_DPD` | int | 0% | Days past due |
| `SK_DPD_DEF` | int | 0% | DPD (defined) |

---

## Join Relationship Diagram

```
application_train/test  (root — SK_ID_CURR is primary key)
    │
    ├── bureau.csv  (many per SK_ID_CURR)
    │       └── bureau_balance.csv  (many per SK_ID_BUREAU)
    │
    └── previous_application.csv  (many per SK_ID_CURR)
            ├── POS_CASH_balance.csv  (many per SK_ID_PREV)
            ├── installments_payments.csv  (many per SK_ID_PREV)
            └── credit_card_balance.csv  (many per SK_ID_PREV)
```

All joins to application are LEFT JOINs — applicants with no history in auxiliary tables get NaN, which is then imputed by the pipeline.

---

## Missingness Summary (application_train.csv only)

| Missingness Range | Count of Columns | Action |
|---|---|---|
| 0% | ~120 cols | Keep as-is |
| 0.1% – 10% | ~30 cols | Impute median/mode |
| 10% – 60% | ~10 cols | Impute median/mode, flag missingness with indicator col |
| > 60% | ~45 cols | Drop (building attributes group) |

---

## Feature Engineering Decision Log

| Decision | Rationale |
|---|---|
| DAYS_EMPLOYED anomaly → flag + NaN | 365243 is a placeholder, not a real value. Including raw value as-is destroys the signal. |
| EXT_SOURCE_MEAN = mean(EXT_SOURCE_1,2,3) | Three sources together are stronger predictors; aggregating handles individual missingness |
| CREDIT_INCOME_RATIO | Standard creditworthiness metric; captures ability to repay relative to income |
| CREDIT_TERM = AMT_CREDIT / AMT_ANNUITY | Effective loan duration in months; tree models can't compute this from raw inputs |
| bureau aggs: active loan count | Active debt load is a stronger signal than total historical debt |
| installment PAYMENT_DIFF mean | Average underpayment is a leading indicator of distress |
| installment LATE_PAYMENT_RATE | Fraction of late payments integrates severity and frequency |
| Drop bureau AMT_ANNUITY | 82.6% missing — no useful information at this missingness level |
| Label encode all cats for trees | XGBoost/LightGBM handle integer categories natively |
| Separate cat/cont split for NN | Entity embeddings require integer indices; raw floats go through BatchNorm |
