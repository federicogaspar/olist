# src/data_loader.py
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


class DataLoader:
    def __init__(self, data_path: str | None = None):
        # 1) DATA_DIR env > arg > ./data (relativo a este archivo)
        base_path = (
            os.getenv("DATA_DIR")
            or data_path
            or os.path.join(os.path.dirname(__file__), "..", "data")
        )

        # 2) Normaliza y asegura separador final
        base_path = os.path.abspath(base_path)
        if not base_path.endswith(os.sep):
            base_path += os.sep

        self.data_path = base_path

        # 3) Campos
        self.orders = None
        self.order_items = None
        self.customers = None
        self.products = None

    def _read(self, fname: str) -> pd.DataFrame:
        path = os.path.join(self.data_path, fname) if not self.data_path.endswith(
            os.sep) else self.data_path + fname
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)

    def load_data(self) -> pd.DataFrame:
        """Load and merge all necessary datasets."""
        # Load datasets
        self.orders = self._read("olist_orders_dataset.csv")
        self.order_items = self._read("olist_order_items_dataset.csv")
        self.customers = self._read("olist_customers_dataset.csv")
        self.products = self._read("olist_products_dataset.csv")

        # Convert timestamps
        self.orders["order_purchase_timestamp"] = pd.to_datetime(
            self.orders["order_purchase_timestamp"], errors="coerce"
        )
        self.orders["purchase_date"] = self.orders["order_purchase_timestamp"].dt.date

        # Create master dataset
        master_data = (
            self.orders.merge(self.order_items, on="order_id", how="inner")
            .merge(self.customers, on="customer_id", how="inner")
            [
                [
                    "order_id",
                    "customer_id",
                    "customer_unique_id",
                    "customer_state",
                    "product_id",
                    "purchase_date",
                    "order_purchase_timestamp",
                    "price",
                ]
            ]
        )

        return master_data

    def get_date_range(self, data: pd.DataFrame):
        """Get min and max dates from dataset."""
        return data["purchase_date"].min(), data["purchase_date"].max()
