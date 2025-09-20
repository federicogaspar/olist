# Olist (2016–2018) Recommender Challenge — Production-Ready CLI

**Objetiv:** deliver a “production-ready” package with:
- **Analytics**: clear metrics + insights and their business implications.
- **Modeling**: a simple and defensible recommender/predictor with an evaluation metric.
- **Code**: modular, with a CLI and at least one unit test.

---

## Contents

- [Project Structure](#project-structure)
- [Environment Setup](#environment-setup)
- [Data: Where to put the CSVs](#data-where-to-put-the-csvs)
- [How to Run the CLI](#how-to-run-the-cli)
- [Running Tests](#running-tests)
- [Modeling Overview](#modeling-overview)
- [Evaluation](#evaluation)
- [Extending to an API](#extending-to-an-api)
- [Assumptions & Notes](#assumptions--notes)
- [Troubleshooting](#troubleshooting)
- [Roadmap / Next Steps](#roadmap--next-steps)
- [License](#license)

---

## Project Structure

```
ds_challenge_olist/
├─ data/                     # Put CSVs here (or set DATA_DIR)
├─ notebooks/  
├─ reports 
│  ├─ analytics_summary.md   # Analytics summary
├─ src/
│  ├─ data_loader.py         # Loads and merges core Olist CSVs
│  ├─ model.py               # Hybrid recommender (CF + regional + global)
│  ├─ evaluate.py            # Precision@K and window comparison utilities
│  └─ main.py                # CLI entry point
├─ tests/
│  └─ test_model.py          # Minimal pytest unit tests
├─ requirements.txt
└─ README.md                 # This file
```

---

## Environment Setup

Use Python 3.10+.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows (PowerShell):
# .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

**Pinned requirements (excerpt):**
`pytest==8.4.2`, `pandas==2.2.3`, `numpy==2.2.2`, `python-dateutil==2.9.0.post0`, `rich==13.9.4`  
(See full list in `requirements.txt`.)

---

## Data: Where to put the CSVs

Download the Olist CSVs and place them under `./data/`, **or** point the code to your folder via an environment variable:

```bash
# Recommended: include a trailing slash!
export DATA_DIR=./data/
```

> **Important:** The current `DataLoader` expects `DATA_DIR` to end with a **trailing slash** (`/`) because it concatenates filenames like  
> `f"{DATA_DIR}olist_orders_dataset.csv"`.  
> Example expected filenames in that folder:
>
> - `olist_orders_dataset.csv`
> - `olist_order_items_dataset.csv`
> - `olist_customers_dataset.csv`
> - `olist_products_dataset.csv`

---

## How to Run the CLI

The CLI trains (in‑process) and prints recommendations for a given customer:

```bash
python -m src.main --customer_id <customer_unique_id> --top_k 5
```

**Example:**

```bash
export DATA_DIR=./data/
python -m src.main --customer_id 7c396fd4830fd04220f7ac84c7b01fbb --top_k 5
```

**Output (JSON):**
```json
{
  "8d50f5eadf13dc67e0ea3b5e1c8d7a7f": ["<product_id_1>", "<product_id_2>", "..."]
}
```

Tips to discover a valid `customer_unique_id`:

```bash
python -c "import pandas as pd; import os; d=os.getenv('DATA_DIR','./data/'); print(pd.read_csv(d+'olist_customers_dataset.csv')['customer_unique_id'].head().to_list())"
```

---

## Running Tests

```bash
pytest -q
```

The suite includes:
- Sanity checks for `HybridRecommender.recommend()` length/type.
- A toy precision@K calculation check.

---

## Modeling Overview

`src/model.py` implements a **hybrid strategy**:

1. **Collaborative filtering (co‑occurrence)**
   - Builds a simple user → list of purchased products mapping.
   - Recommends items co‑purchased by similar users (based on overlap).
2. **Regional popularity (last *window_days*)**
   - Ranks products by frequency **within the customer’s state**.
3. **Global popularity (last *window_days*)**
   - Fallback to top‑selling products globally.

**De‑duplication** is enforced across stages to ensure at most `top_k` unique items.

Config:
- `window_days` (default 90) controls the recency window for popularity signals.

---

## Evaluation

Utilities live in `src/evaluate.py`:

- `RecommendationEvaluator.calculate_precision_at_k(recommendations, actual_purchases, k)`
- `RecommendationEvaluator.evaluate_simple(train_data, test_data, model_class, k, window_days)`
- `RecommendationEvaluator.compare_window_days(...)` to benchmark sensitivity to the recency window.

**Current CLI** logs **only recommendations**.  
To evaluate offline:
- Import the evaluator in a notebook or script and call `evaluate_simple(...)` with a temporal split.
- Metrics reported: **Precision@K** per group (new vs returning users), hit counts, and group sizes.

---

## Extending to an API

The code is structured so an API layer can be added with minimal changes:

- Keep business logic in `model.py`.
- Create a thin FastAPI/Flask layer with endpoints like:
  - `GET /health`
  - `POST /recommendations` with JSON payload `{ "customer_id": "...", "top_k": 5 }`
- Reuse the same `DataLoader` and `HybridRecommender` instances (initialized at startup).

---

## Assumptions & Notes

- **Train/test split in the CLI**: The example uses a simple **index‑based 85/15 split** just to train quickly in‑process.  
  For a fair evaluation you should do a **time‑based split** (train on earlier dates, test on later ones) to avoid leakage.
- **Model persistence**: This baseline trains on the fly. Adding `save(path)` / `load(path)` in `HybridRecommender` for pickled artifacts is straightforward if needed.
- **Cold start**:
  - If a `customer_id` has no history, the recommender falls back to **regional** then **global** popularity.
  - If the state is unknown, the code uses the **most common state** as a fallback.
- **Data types**: Timestamps are parsed, and a `purchase_date` column is derived for filtering.

---

## Troubleshooting

**1) `AttributeError: 'DataLoader' object has no attribute '__init__'`**  
Make sure the `DataLoader` constructor uses **double underscores**: `def __init__(...)`, not `_init_`.

**2) `NameError: name '_file_' is not defined`**  
Use the correct dunder variable `__file__` when building default paths.

**3) File path errors (CSV not found)**  
Ensure `DATA_DIR` ends with a trailing slash, e.g., `./data/`.

**4) Empty recommendations**  
Try increasing `window_days` or ensure the customer exists in the training slice.

---

## Roadmap / Next Steps

- Replace index split with **time‑based** split in the CLI.
- Add **model persistence** (`save`/`load`) and a `models/` folder.
- Enhance CF with **item–item similarities** (Jaccard/lift) from `order_items`.
- Incorporate **product category** priors for better cold‑start behavior.
- Wire up **evaluation logging** into the CLI (Precision@K on a hold‑out).
- Add a minimal **FastAPI** server as a separate entry point.

---

## License

This code is provided for the purpose of the DS challenge and educational evaluation.