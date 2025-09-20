import argparse
import pandas as pd
import json

from .data_loader import DataLoader
from .model import HybridRecommender


def get_customer_info(customer_id):
    """Get customer state from data"""
    loader = DataLoader()
    data = loader.load_data()

    customer_data = data[data['customer_unique_id'] == customer_id]
    if len(customer_data) > 0:
        return customer_data['customer_state'].iloc[0]

    # If not found, use most common state as fallback
    most_common_state = data['customer_state'].mode().iloc[0]
    return most_common_state


def main():
    parser = argparse.ArgumentParser(description='Olist Recommendation System')
    parser.add_argument('--customer_id', type=str, required=True,
                        help='Customer ID for recommendations')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of recommendations (default: 5)')

    args = parser.parse_args()

    # Load data and train model
    loader = DataLoader()
    data = loader.load_data()
    data['purchase_date'] = pd.to_datetime(data['purchase_date'])

    # Use 85% of data for training
    total_records = len(data)
    train_end_idx = int(total_records * 0.85)
    train_data = data.iloc[:train_end_idx]

    # Train model
    model = HybridRecommender(window_days=55)
    train_end_date = train_data['purchase_date'].max()
    model.fit(train_data, train_end_date)

    # Get customer state
    customer_state = get_customer_info(args.customer_id)

    # Generate recommendations
    recommendations, strategy = model.recommend(
        args.customer_id,
        customer_state,
        k=args.top_k
    )

    # Output JSON
    result = {args.customer_id: recommendations}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
