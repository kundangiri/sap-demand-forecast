"""
SAP-like Synthetic Data Generator
Simulates tables from SAP SD (Sales & Distribution) and MM (Materials Management) modules:
  - VBAK  : Sales Order Header
  - VBAP  : Sales Order Item
  - MARA  : General Material Data
  - KNA1  : Customer Master
  - T001W : Plant / Storage Location
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import datetime, timedelta
import os

fake = Faker()
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = os.path.dirname(__file__)

# ── Config ──────────────────────────────────────────────────────────────
N_CUSTOMERS   = 80
N_MATERIALS   = 40
N_PLANTS      = 5
N_ORDERS      = 6000
START_DATE    = datetime(2021, 1, 1)
END_DATE      = datetime(2024, 12, 31)

MATERIAL_GROUPS = ["ELEC", "MECH", "CHEM", "PACK", "CONS"]
SALES_ORGS      = ["1000", "2000", "3000"]
DIST_CHANNELS   = ["10", "20", "30"]
DIVISIONS       = ["00", "01", "02"]
ORDER_TYPES     = ["OR", "ZOR", "RE"]      # Standard, Special, Returns
CURRENCIES      = ["USD", "EUR", "GBP"]

# ── KNA1: Customer Master ────────────────────────────────────────────────
def gen_kna1(n=N_CUSTOMERS):
    records = []
    for i in range(n):
        records.append({
            "KUNNR": f"C{str(i+1).zfill(6)}",          # Customer number
            "NAME1": fake.company(),
            "ORT01": fake.city(),
            "LAND1": random.choice(["US", "DE", "GB", "FR", "SE"]),
            "KUKLA": random.choice(["A", "B", "C"]),    # Customer class
            "KDGRP": random.choice(["01","02","03"]),    # Customer group
        })
    return pd.DataFrame(records)

# ── MARA: General Material Data ─────────────────────────────────────────
def gen_mara(n=N_MATERIALS):
    records = []
    for i in range(n):
        mgroup = random.choice(MATERIAL_GROUPS)
        records.append({
            "MATNR": f"MAT{str(i+1).zfill(6)}",        # Material number
            "MAKTX": f"{mgroup} Material {i+1}",         # Description
            "MATKL": mgroup,                              # Material group
            "MEINS": random.choice(["EA","KG","L","M"]), # Base unit
            "BRGEW": round(random.uniform(0.1, 50), 2),  # Gross weight
            "NTGEW": round(random.uniform(0.1, 45), 2),  # Net weight
            "MTART": random.choice(["FERT","HAWA","ROH"]),# Material type
        })
    return pd.DataFrame(records)

# ── T001W: Plant Master ──────────────────────────────────────────────────
def gen_t001w(n=N_PLANTS):
    plants = []
    for i in range(n):
        plants.append({
            "WERKS": f"P{str(i+1).zfill(3)}",
            "NAME1": f"Plant {i+1} - {fake.city()}",
            "LAND1": random.choice(["US", "DE", "GB"]),
        })
    return pd.DataFrame(plants)

# ── VBAK + VBAP: Sales Order Header & Item ──────────────────────────────
def gen_sales_orders(kna1, mara, t001w, n=N_ORDERS):
    customers  = kna1["KUNNR"].tolist()
    materials  = mara["MATNR"].tolist()
    mat_groups = dict(zip(mara["MATNR"], mara["MATKL"]))
    plants     = t001w["WERKS"].tolist()

    vbak_rows = []
    vbap_rows = []

    date_range = (END_DATE - START_DATE).days

    for i in range(n):
        # Order date with seasonal pattern
        day_offset = int(np.random.triangular(0, date_range * 0.6, date_range))
        order_date = START_DATE + timedelta(days=day_offset)

        # Add seasonality: Q4 boost, summer dip
        month = order_date.month
        seasonal_factor = (
            1.4 if month in [10, 11, 12] else
            0.75 if month in [6, 7] else
            1.0
        )

        vbeln = f"OR{str(i+1).zfill(8)}"
        kunnr = random.choice(customers)
        werks = random.choice(plants)

        vbak_rows.append({
            "VBELN":   vbeln,
            "ERDAT":   order_date.strftime("%Y-%m-%d"),   # Creation date
            "AUART":   random.choice(ORDER_TYPES),        # Order type
            "KUNNR":   kunnr,                             # Sold-to party
            "VKORG":   random.choice(SALES_ORGS),         # Sales org
            "VTWEG":   random.choice(DIST_CHANNELS),      # Distribution channel
            "SPART":   random.choice(DIVISIONS),          # Division
            "WAERK":   random.choice(CURRENCIES),         # Currency
        })

        # 1–4 line items per order
        n_items = random.randint(1, 4)
        selected_mats = random.sample(materials, min(n_items, len(materials)))

        for j, matnr in enumerate(selected_mats):
            base_qty  = random.randint(1, 200)
            quantity  = int(base_qty * seasonal_factor * random.uniform(0.8, 1.2))
            unit_price = round(random.uniform(5, 500), 2)

            vbap_rows.append({
                "VBELN":  vbeln,
                "POSNR":  str((j+1) * 10).zfill(6),      # Item number
                "MATNR":  matnr,
                "MATKL":  mat_groups[matnr],
                "WERKS":  werks,
                "KWMENG": quantity,                        # Order quantity
                "NETPR":  unit_price,                     # Net price
                "NETWR":  round(quantity * unit_price, 2),# Net value
                "ERDAT":  order_date.strftime("%Y-%m-%d"),
                "LGORT":  f"L{random.randint(1,4):03d}",  # Storage location
            })

    return pd.DataFrame(vbak_rows), pd.DataFrame(vbap_rows)


# ── Main ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating SAP-like synthetic tables...")

    kna1  = gen_kna1()
    mara  = gen_mara()
    t001w = gen_t001w()
    vbak, vbap = gen_sales_orders(kna1, mara, t001w)

    for name, df in [("KNA1", kna1), ("MARA", mara), ("T001W", t001w),
                     ("VBAK", vbak), ("VBAP", vbap)]:
        path = os.path.join(OUTPUT_DIR, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"  {name:6s} → {len(df):>5,} rows  →  {path}")

    print("\nDone. All SAP tables generated.")
