import os
import pandas as pd
import numpy as np

def generate_mock_dataset():
    np.random.seed(42)
    raw_dir = "data/raw"
    os.makedirs(raw_dir, exist_ok=True)

    print("Generating mock dataset in data/raw...")

    # 1. application_train & application_test
    # We need to simulate around 500 train rows and 100 test rows
    n_train = 500
    n_test = 100
    sk_ids_train = np.arange(100001, 100001 + n_train)
    sk_ids_test = np.arange(100001 + n_train, 100001 + n_train + n_test)

    def make_app_df(sk_ids, is_train=True):
        n = len(sk_ids)
        df = pd.DataFrame({
            "SK_ID_CURR": sk_ids,
            "AMT_INCOME_TOTAL": np.random.exponential(150000, n) + 50000,
            "AMT_CREDIT": np.random.exponential(500000, n) + 100000,
            "AMT_ANNUITY": np.random.exponential(25000, n) + 5000,
            "AMT_GOODS_PRICE": np.random.exponential(450000, n) + 100000,
            "DAYS_BIRTH": np.random.randint(-25000, -7000, n),
            # Anomaly: 365243
            "DAYS_EMPLOYED": np.random.choice([-1000, -2000, -500, 365243], n, p=[0.4, 0.4, 0.1, 0.1]),
            "DAYS_REGISTRATION": np.random.uniform(-10000, 0, n),
            "DAYS_ID_PUBLISH": np.random.randint(-5000, 0, n),
            "CODE_GENDER": np.random.choice(["M", "F", "XNA"], n, p=[0.4, 0.59, 0.01]),
            "NAME_FAMILY_STATUS": np.random.choice(["Married", "Single / not married", "Civil marriage", "Widow", "Separated"], n),
            "CNT_CHILDREN": np.random.randint(0, 4, n),
            "CNT_FAM_MEMBERS": np.random.randint(1, 6, n),
            "EXT_SOURCE_1": np.random.uniform(0, 1, n),
            "EXT_SOURCE_2": np.random.uniform(0, 1, n),
            "EXT_SOURCE_3": np.random.uniform(0, 1, n),
            "FLAG_DOCUMENT_3": np.random.choice([0, 1], n, p=[0.2, 0.8]),
            "OBS_30_CNT_SOCIAL_CIRCLE": np.random.randint(0, 10, n),
            "DEF_30_CNT_SOCIAL_CIRCLE": np.random.randint(0, 3, n),
            "OBS_60_CNT_SOCIAL_CIRCLE": np.random.randint(0, 10, n),
            "DEF_60_CNT_SOCIAL_CIRCLE": np.random.randint(0, 3, n),
        })

        # Let's add building attributes (dropped > 60% missing)
        for col in ["APARTMENTS_AVG", "BASEMENTAREA_AVG", "YEARS_BEGINEXPLUATATION_AVG", "YEARS_BUILD_AVG", "COMMONAREA_AVG"]:
            df[col] = np.random.choice([np.nan, 0.5], n, p=[0.7, 0.3]) # 70% missing

        if is_train:
            df["TARGET"] = np.random.choice([0, 1], n, p=[0.92, 0.08])

        return df

    df_train = make_app_df(sk_ids_train, is_train=True)
    df_test = make_app_df(sk_ids_test, is_train=False)

    df_train.to_csv(os.path.join(raw_dir, "application_train.csv"), index=False)
    df_test.to_csv(os.path.join(raw_dir, "application_test.csv"), index=False)

    # Combine all sk_ids for child tables
    all_sk_ids = np.concatenate([sk_ids_train, sk_ids_test])

    # 2. bureau
    n_bureau = len(all_sk_ids) * 3
    bureau_ids = np.arange(5000001, 5000001 + n_bureau)
    df_bureau = pd.DataFrame({
        "SK_ID_CURR": np.random.choice(all_sk_ids, n_bureau),
        "SK_ID_BUREAU": bureau_ids,
        "CREDIT_ACTIVE": np.random.choice(["Closed", "Active", "Sold", "Bad debt"], n_bureau, p=[0.6, 0.38, 0.01, 0.01]),
        "CREDIT_CURRENCY": np.random.choice(["currency 1", "currency 2"], n_bureau, p=[0.99, 0.01]),
        "DAYS_CREDIT": np.random.randint(-2900, 0, n_bureau),
        "CREDIT_DAY_OVERDUE": np.random.choice([0, 10, 30], n_bureau, p=[0.98, 0.01, 0.01]),
        "DAYS_CREDIT_ENDDATE": np.random.randint(-1000, 2000, n_bureau),
        "AMT_CREDIT_MAX_OVERDUE": np.random.exponential(1000, n_bureau),
        "CNT_CREDIT_PROLONG": np.random.randint(0, 3, n_bureau),
        "AMT_CREDIT_SUM": np.random.exponential(200000, n_bureau) + 10000,
        "AMT_CREDIT_SUM_DEBT": np.random.exponential(50000, n_bureau),
        "AMT_CREDIT_SUM_LIMIT": np.random.exponential(20000, n_bureau),
        "AMT_CREDIT_SUM_OVERDUE": np.random.choice([0.0, 500.0], n_bureau, p=[0.99, 0.01]),
        "CREDIT_TYPE": np.random.choice(["Consumer credit", "Credit card", "Mortgage", "Car loan"], n_bureau, p=[0.7, 0.2, 0.08, 0.02]),
        "DAYS_CREDIT_UPDATE": np.random.randint(-100, 0, n_bureau),
        "AMT_ANNUITY": np.random.exponential(10000, n_bureau) # 82% missing will be simulated by features module dropping or ignoring it
    })
    df_bureau.to_csv(os.path.join(raw_dir, "bureau.csv"), index=False)

    # 3. bureau_balance
    # bureau balance relates to SK_ID_BUREAU
    n_bureau_bal = len(bureau_ids) * 5
    df_bureau_bal = pd.DataFrame({
        "SK_ID_BUREAU": np.random.choice(bureau_ids, n_bureau_bal),
        "MONTHS_BALANCE": np.random.randint(-50, 0, n_bureau_bal),
        "STATUS": np.random.choice(["C", "0", "1", "2", "3", "4", "5", "X"], n_bureau_bal, p=[0.4, 0.4, 0.1, 0.02, 0.02, 0.01, 0.01, 0.04])
    })
    df_bureau_bal.to_csv(os.path.join(raw_dir, "bureau_balance.csv"), index=False)

    # 4. previous_application
    n_prev = len(all_sk_ids) * 2
    prev_ids = np.arange(2000001, 2000001 + n_prev)
    df_prev = pd.DataFrame({
        "SK_ID_CURR": np.random.choice(all_sk_ids, n_prev),
        "SK_ID_PREV": prev_ids,
        "NAME_CONTRACT_TYPE": np.random.choice(["Cash loans", "Consumer loans", "Revolving loans"], n_prev),
        "AMT_ANNUITY": np.random.exponential(15000, n_prev) + 1000,
        "AMT_APPLICATION": np.random.exponential(100000, n_prev) + 5000,
        "AMT_CREDIT": np.random.exponential(100000, n_prev) + 5000,
        "AMT_DOWN_PAYMENT": np.random.exponential(10000, n_prev),
        "AMT_GOODS_PRICE": np.random.exponential(90000, n_prev) + 5000,
        "NAME_CONTRACT_STATUS": np.random.choice(["Approved", "Refused", "Canceled", "Unused offer"], n_prev, p=[0.6, 0.2, 0.15, 0.05]),
        "DAYS_DECISION": np.random.randint(-2900, 0, n_prev),
        "NAME_PRODUCT_TYPE": np.random.choice(["walk-in", "XNA", "x-sell"], n_prev),
        "CHANNEL_TYPE": np.random.choice(["Country-wide", "Contact center", "Credit and Cash offices"], n_prev),
        "NAME_SELLER_INDUSTRY": np.random.choice(["Connectivity", "Construction", "Consumer electronics"], n_prev),
        "CNT_PAYMENT": np.random.choice([6, 12, 18, 24, 36], n_prev)
    })
    df_prev.to_csv(os.path.join(raw_dir, "previous_application.csv"), index=False)

    # Helper function to get mapping SK_ID_PREV -> SK_ID_CURR
    prev_to_curr = dict(zip(df_prev["SK_ID_PREV"], df_prev["SK_ID_CURR"]))

    # 5. POS_CASH_balance
    n_pos = len(prev_ids) * 4
    chosen_prevs_pos = np.random.choice(prev_ids, n_pos)
    df_pos = pd.DataFrame({
        "SK_ID_PREV": chosen_prevs_pos,
        "SK_ID_CURR": [prev_to_curr[pid] for pid in chosen_prevs_pos],
        "MONTHS_BALANCE": np.random.randint(-70, 0, n_pos),
        "CNT_INSTALMENT": np.random.choice([6.0, 12.0, 24.0], n_pos),
        "CNT_INSTALMENT_FUTURE": np.random.choice([0.0, 6.0, 12.0], n_pos),
        "NAME_CONTRACT_STATUS": np.random.choice(["Active", "Completed", "Signed"], n_pos, p=[0.8, 0.18, 0.02]),
        "SK_DPD": np.random.choice([0, 5, 20], n_pos, p=[0.97, 0.02, 0.01]),
        "SK_DPD_DEF": np.random.choice([0, 5], n_pos, p=[0.99, 0.01])
    })
    df_pos.to_csv(os.path.join(raw_dir, "POS_CASH_balance.csv"), index=False)

    # 6. installments_payments
    n_installments = len(prev_ids) * 6
    chosen_prevs_inst = np.random.choice(prev_ids, n_installments)
    df_inst = pd.DataFrame({
        "SK_ID_PREV": chosen_prevs_inst,
        "SK_ID_CURR": [prev_to_curr[pid] for pid in chosen_prevs_inst],
        "NUM_INSTALMENT_VERSION": np.random.choice([0.0, 1.0, 2.0], n_installments),
        "NUM_INSTALMENT_NUMBER": np.random.randint(1, 24, n_installments),
        "DAYS_INSTALMENT": np.random.uniform(-2000, 0, n_installments),
        "DAYS_ENTRY_PAYMENT": np.random.uniform(-2000, 0, n_installments),
        "AMT_INSTALMENT": np.random.exponential(10000, n_installments) + 100,
        "AMT_PAYMENT": np.random.exponential(10000, n_installments) + 100
    })
    # Add some nulls to DAYS_ENTRY_PAYMENT and AMT_PAYMENT to simulate unpaid/missing installments
    null_idx = np.random.choice(n_installments, int(n_installments * 0.05), replace=False)
    df_inst.loc[null_idx, "DAYS_ENTRY_PAYMENT"] = np.nan
    df_inst.loc[null_idx, "AMT_PAYMENT"] = np.nan
    df_inst.to_csv(os.path.join(raw_dir, "installments_payments.csv"), index=False)

    # 7. credit_card_balance
    # Only some previous applications are credit cards, let's select a subset of prev_ids
    cc_prevs = np.random.choice(prev_ids, max(1, int(len(prev_ids) * 0.3)), replace=False)
    n_cc = len(cc_prevs) * 5
    chosen_prevs_cc = np.random.choice(cc_prevs, n_cc)
    df_cc = pd.DataFrame({
        "SK_ID_PREV": chosen_prevs_cc,
        "SK_ID_CURR": [prev_to_curr[pid] for pid in chosen_prevs_cc],
        "MONTHS_BALANCE": np.random.randint(-70, 0, n_cc),
        "AMT_BALANCE": np.random.exponential(20000, n_cc),
        "AMT_CREDIT_LIMIT_ACTUAL": np.random.choice([10000, 50000, 100000], n_cc),
        "AMT_DRAWINGS_ATM_CURRENT": np.random.exponential(5000, n_cc),
        "AMT_DRAWINGS_CURRENT": np.random.exponential(6000, n_cc),
        "AMT_DRAWINGS_TOTAL_CURRENT": np.random.exponential(6000, n_cc),
        "AMT_INST_MIN_REGULARITY": np.random.exponential(1000, n_cc),
        "AMT_PAYMENT_CURRENT": np.random.exponential(5000, n_cc),
        "AMT_PAYMENT_TOTAL_CURRENT": np.random.exponential(5000, n_cc),
        "AMT_RECEIVABLE_PRINCIPAL": np.random.exponential(18000, n_cc),
        "AMT_RECIVABLE": np.random.exponential(19000, n_cc),
        "AMT_TOTAL_RECEIVABLE": np.random.exponential(19000, n_cc),
        "CNT_DRAWINGS_ATM_CURRENT": np.random.randint(0, 5, n_cc),
        "CNT_DRAWINGS_CURRENT": np.random.randint(0, 10, n_cc),
        "CNT_DRAWINGS_TOTAL_CURRENT": np.random.randint(0, 10, n_cc),
        "CNT_INSTALMENT_MATURE_CUM": np.random.randint(0, 24, n_cc),
        "NAME_CONTRACT_STATUS": np.random.choice(["Active", "Completed", "Signed"], n_cc, p=[0.9, 0.08, 0.02]),
        "SK_DPD": np.random.choice([0, 5], n_cc, p=[0.98, 0.02]),
        "SK_DPD_DEF": np.random.choice([0, 5], n_cc, p=[0.99, 0.01])
    })
    df_cc.to_csv(os.path.join(raw_dir, "credit_card_balance.csv"), index=False)

    # 8. HomeCredit_columns_description.csv
    df_desc = pd.DataFrame({
        "Row": [1, 2],
        "Table": ["application_train", "application_test"],
        "Row_Name": ["SK_ID_CURR", "TARGET"],
        "Description": ["ID of loan", "Target variable"]
    })
    df_desc.to_csv(os.path.join(raw_dir, "HomeCredit_columns_description.csv"), index=False)

    print("Successfully generated all mock datasets in data/raw.")

if __name__ == "__main__":
    generate_mock_dataset()
