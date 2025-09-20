import pandas as pd
from collections import defaultdict


class RecommendationEvaluator:
    def __init__(self):
        pass

    def calculate_precision_at_k(self, recommendations, actual_purchases, k=5):
        """Calculate precision@k for single user"""
        if not recommendations or not actual_purchases:
            return 0.0

        recs_set = set(recommendations[:k])
        actual_set = set(actual_purchases)
        hits = len(recs_set & actual_set)

        return hits / min(len(recommendations), k)

    def evaluate_simple(self, train_data, test_data, model_class, k=5, window_days=90):
        """
        Simple evaluation: train model, test on test_data, return detailed metrics

        Args:
            train_data: Training DataFrame 
            test_data: Test DataFrame
            model_class: HybridRecommender class (not instance)
            k: Number of recommendations
            window_days: Window for popularity calculations

        Returns:
            dict with all metrics
        """

        # Train model
        model = model_class(window_days=window_days)
        train_end_date = train_data['purchase_date'].max()
        model.fit(train_data, train_end_date)

        # Get unique customers from test data
        test_customers = test_data[[
            'customer_unique_id', 'customer_state']].drop_duplicates()

        # Separate new vs returning customers
        train_customer_ids = set(train_data['customer_unique_id'].unique())

        new_customers = []
        returning_customers = []

        for _, row in test_customers.iterrows():
            customer_id = row['customer_unique_id']
            if customer_id in train_customer_ids:
                returning_customers.append(row)
            else:
                new_customers.append(row)

        # Convert to DataFrames for easier handling
        new_customers_df = pd.DataFrame(
            new_customers) if new_customers else pd.DataFrame()
        returning_customers_df = pd.DataFrame(
            returning_customers) if returning_customers else pd.DataFrame()

        # Count orders for each group
        new_users_orders = 0
        returning_customers_orders = 0

        if len(new_customers_df) > 0:
            new_users_orders = len(test_data[test_data['customer_unique_id'].isin(
                new_customers_df['customer_unique_id'])])

        if len(returning_customers_df) > 0:
            returning_customers_orders = len(test_data[test_data['customer_unique_id'].isin(
                returning_customers_df['customer_unique_id'])])

        # Evaluate new users
        hits_new_users = 0
        precisions_new = []

        for _, row in new_customers_df.iterrows():
            customer_id = row['customer_unique_id']
            customer_state = row['customer_state']

            # Get recommendations
            recommendations, _ = model.recommend(
                customer_id, customer_state, k)

            # Get actual purchases
            actual_purchases = test_data[
                test_data['customer_unique_id'] == customer_id
            ]['product_id'].tolist()

            # Calculate precision and hits
            precision = self.calculate_precision_at_k(
                recommendations, actual_purchases, k)
            precisions_new.append(precision)

            # Count hits (products that were actually bought)
            hits = len(set(recommendations[:k]) & set(actual_purchases))
            hits_new_users += hits

        # Evaluate returning customers
        hits_returning_customers = 0
        precisions_returning = []

        for _, row in returning_customers_df.iterrows():
            customer_id = row['customer_unique_id']
            customer_state = row['customer_state']

            # Get recommendations
            recommendations, _ = model.recommend(
                customer_id, customer_state, k)

            # Get actual purchases
            actual_purchases = test_data[
                test_data['customer_unique_id'] == customer_id
            ]['product_id'].tolist()

            # Calculate precision and hits
            precision = self.calculate_precision_at_k(
                recommendations, actual_purchases, k)
            precisions_returning.append(precision)

            # Count hits
            hits = len(set(recommendations[:k]) & set(actual_purchases))
            hits_returning_customers += hits

        # Calculate average precisions
        precision_new_users = sum(precisions_new) / \
            len(precisions_new) if precisions_new else 0.0
        precision_returning = sum(
            precisions_returning) / len(precisions_returning) if precisions_returning else 0.0

        # Results
        results = {
            'new_users': len(new_customers_df),
            'new_users_orders': new_users_orders,
            'returning_customers': len(returning_customers_df),
            'returning_customers_orders': returning_customers_orders,
            'hits_new_users_orders': hits_new_users,
            'hits_returning_customers_orders': hits_returning_customers,
            'calculate_precision_at_k_for_new_users': precision_new_users,
            'calculate_precision_at_k_for_returning': precision_returning
        }

        # Print results
        print(f"📊 EVALUATION RESULTS (k={k}, window_days={window_days})")
        print("=" * 60)
        print(f"New users: {results['new_users']:,}")
        print(f"New users orders: {results['new_users_orders']:,}")
        print(f"Returning customers: {results['returning_customers']:,}")
        print(
            f"Returning customers orders: {results['returning_customers_orders']:,}")
        print(f"Hits new users orders: {results['hits_new_users_orders']:,}")
        print(
            f"Hits returning customers orders: {results['hits_returning_customers_orders']:,}")
        print(
            f"Precision@{k} for new users: {results['calculate_precision_at_k_for_new_users']:.4f}")
        print(
            f"Precision@{k} for returning: {results['calculate_precision_at_k_for_returning']:.4f}")
        print()

        return results

    def compare_window_days(self, train_data, test_data, model_class, k=5, window_days_list=[30, 60, 90, 120]):
        """
        Compare different window_days values
        """
        print(f"🧪 COMPARING WINDOW_DAYS VALUES")
        print("=" * 80)

        all_results = []

        for window_days in window_days_list:
            print(f"\n🔄 Testing window_days = {window_days}")
            results = self.evaluate_simple(
                train_data, test_data, model_class, k, window_days)
            results['window_days'] = window_days
            all_results.append(results)

        # Summary table
        print("📋 SUMMARY COMPARISON")
        print("=" * 100)
        print(f"{'Window':>8} | {'New Users':>10} | {'Returning':>10} | {'Precision New':>13} | {'Precision Ret':>13}")
        print("-" * 100)

        for r in all_results:
            print(f"{r['window_days']:8d} | {r['new_users']:10d} | {r['returning_customers']:10d} | "
                  f"{r['calculate_precision_at_k_for_new_users']:13.4f} | {r['calculate_precision_at_k_for_returning']:13.4f}")

        return all_results
