"""Real regression test for a genuine bug found in models/segmentation.py:
the active_clusters computation only excluded the "lost" cluster, not the
"champions" cluster, so the loyal-vs-new comparison silently operated on the
wrong pair of clusters. The result: two persona roles collapsed onto the same
cluster label (a dict literal with a duplicate key just keeps the last one),
"VIP Champions" vanished from the output, and that cluster's customers ended
up with a NaN segment_name -- invisible in a value_counts() print, not a
crash, so nothing caught it until this test.

Runs the real pipeline end-to-end (no mocking) against a real generated
database and checks the actual segment assignment output.
"""
import os
import sqlite3
import subprocess
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "sales.db")

EXPECTED_SEGMENTS = {"VIP Champions", "Loyal Customers", "New Customers", "At Risk / Hibernating"}


class TestSegmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # pipeline.py appends to the DB (if_exists="append"), so a stale DB
        # from a previous run would hit duplicate-email UNIQUE violations --
        # start from the documented fresh-checkout state.
        if os.path.exists(DB_PATH):
            os.remove(DB_PATH)
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "etl", "pipeline.py")], check=True)
        subprocess.run([sys.executable, os.path.join(PROJECT_ROOT, "models", "segmentation.py")], check=True)

    def test_all_four_personas_are_assigned(self):
        # Scoped to customers who actually have purchase history -- a
        # customer with zero transactions has no RFM to compute and
        # legitimately stays "Unsegmented" by the schema's default.
        conn = sqlite3.connect(DB_PATH)
        segments = [
            row[0] for row in conn.execute(
                "SELECT DISTINCT segment FROM dim_customers "
                "WHERE customer_id IN (SELECT DISTINCT customer_id FROM fact_sales)"
            ).fetchall()
        ]
        conn.close()
        self.assertEqual(
            set(segments), EXPECTED_SEGMENTS,
            "Not all four RFM personas were assigned -- a cluster's customers likely went unlabeled.",
        )

    def test_no_customer_is_left_without_a_segment(self):
        conn = sqlite3.connect(DB_PATH)
        total = conn.execute(
            "SELECT COUNT(*) FROM dim_customers WHERE customer_id IN (SELECT DISTINCT customer_id FROM fact_sales)"
        ).fetchone()[0]
        unlabeled = conn.execute(
            "SELECT COUNT(*) FROM dim_customers "
            "WHERE customer_id IN (SELECT DISTINCT customer_id FROM fact_sales) "
            "AND (segment IS NULL OR segment = '')"
        ).fetchone()[0]
        conn.close()
        self.assertGreater(total, 0)
        self.assertEqual(unlabeled, 0, "Some purchasing customers have no segment assigned.")


if __name__ == "__main__":
    unittest.main()
