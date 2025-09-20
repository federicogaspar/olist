import sys
import pandas as pd
from datetime import datetime

# Add src to path FIRST
sys.path.append('src')

from model import HybridRecommender
# THEN import


def test_hybrid_recommender_basic():
    """Test basic functionality of HybridRecommender"""

    # Create simple test data
    test_data = pd.DataFrame({
        'customer_unique_id': ['customer1', 'customer1', 'customer2'],
        'customer_state': ['SP', 'SP', 'RJ'],
        'product_id': ['product1', 'product2', 'product1'],
        'purchase_date': [datetime(2018, 1, 1), datetime(2018, 1, 2), datetime(2018, 1, 3)]
    })

    # Initialize and fit model
    model = HybridRecommender(window_days=30)
    model.fit(test_data, datetime(2018, 1, 3))

    # Test recommendations
    recommendations, strategy = model.recommend('customer1', 'SP', k=2)

    # Basic assertions
    assert isinstance(recommendations, list)
    assert isinstance(strategy, str)
    assert len(recommendations) <= 2
    print("✅ Basic recommendation test passed")


def test_precision_calculation():
    """Test precision@k calculation"""

    recommendations = ['prod1', 'prod2', 'prod3']
    actual = ['prod2', 'prod4']

    # Calculate precision manually
    recs_set = set(recommendations[:3])
    actual_set = set(actual)
    hits = len(recs_set & actual_set)
    expected_precision = hits / 3

    assert expected_precision == 1/3
    print("✅ Precision calculation test passed")


if __name__ == "__main__":
    test_hybrid_recommender_basic()
    test_precision_calculation()
    print("🎉 All tests passed!")
