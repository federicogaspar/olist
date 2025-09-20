import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import timedelta


class HybridRecommender:
    def __init__(self, window_days=90):
        self.window_days = window_days
        self.customer_history = defaultdict(list)
        self.regional_popularity = defaultdict(Counter)
        self.global_popularity = Counter()

    def fit(self, train_data, end_date):
        """Fit model on training data up to end_date"""
        # Filter training data to window
        start_date = end_date - timedelta(days=self.window_days)
        window_data = train_data[
            (train_data['purchase_date'] >= start_date) &
            (train_data['purchase_date'] <= end_date)
        ]

        # Build customer history (for collaborative filtering)
        self.customer_history = defaultdict(list)
        for _, row in train_data.iterrows():
            self.customer_history[row['customer_unique_id']].append(
                row['product_id'])

        # Build regional popularity (last 90 days)
        self.regional_popularity = defaultdict(Counter)
        for _, row in window_data.iterrows():
            self.regional_popularity[row['customer_state']
                                     ][row['product_id']] += 1

        # Build global popularity (last 90 days)
        self.global_popularity = Counter(window_data['product_id'])

    def get_collaborative_recommendations(self, customer_id, k=5):
        """Item-based collaborative filtering"""
        if customer_id not in self.customer_history:
            return []

        user_products = set(self.customer_history[customer_id])

        # Find products bought with user's products
        co_occurrence = Counter()
        for other_customer, other_products in self.customer_history.items():
            if other_customer == customer_id:
                continue

            other_products_set = set(other_products)
            if user_products & other_products_set:  # If they share products
                # Add products this customer hasn't bought
                new_products = other_products_set - user_products
                for product in new_products:
                    co_occurrence[product] += 1

        return [product for product, _ in co_occurrence.most_common(k)]

    def get_regional_recommendations(self, customer_state, k=5):
        """Regional popularity recommendations"""
        if customer_state in self.regional_popularity:
            return [product for product, _ in self.regional_popularity[customer_state].most_common(k)]
        return []

    def get_global_recommendations(self, k=5):
        """Global popularity recommendations"""
        return [product for product, _ in self.global_popularity.most_common(k)]

    def recommend(self, customer_id, customer_state, k=5):
        """Hybrid recommendation strategy with completion logic and no duplicates"""

        final_recs = []
        strategy_components = []

        # Strategy 1: Collaborative filtering if customer exists
        if customer_id in self.customer_history:
            cf_recs = self.get_collaborative_recommendations(customer_id, k)
            if cf_recs:
                final_recs.extend(cf_recs)
                strategy_components.append('collaborative')

                # Si CF da suficientes recomendaciones, retornar
                if len(final_recs) >= k:
                    return final_recs[:k], 'collaborative'

        # Strategy 2: Completar con Regional (evitando duplicados)
        if len(final_recs) < k:
            regional_recs = self.get_regional_recommendations(
                customer_state, k)

            # Filtrar duplicados
            regional_recs_filtered = [
                prod for prod in regional_recs if prod not in final_recs]

            needed_from_regional = k - len(final_recs)
            final_recs.extend(regional_recs_filtered[:needed_from_regional])

            if regional_recs_filtered:
                strategy_components.append('regional')

        # Strategy 3: Completar con Global (evitando duplicados)
        if len(final_recs) < k:
            global_recs = self.get_global_recommendations(k)

            # Filtrar duplicados
            global_recs_filtered = [
                prod for prod in global_recs if prod not in final_recs]

            needed_from_global = k - len(final_recs)
            final_recs.extend(global_recs_filtered[:needed_from_global])

            if global_recs_filtered:
                strategy_components.append('global')

        # Determinar strategy name
        if not strategy_components:
            strategy = 'no_recommendations'
        elif len(strategy_components) == 1:
            strategy = strategy_components[0]
        else:
            strategy = '_'.join(strategy_components)

        return final_recs[:k], strategy
