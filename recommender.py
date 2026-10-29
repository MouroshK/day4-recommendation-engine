"""
recommender.py - Mama Mboga Online Customer Recommendation Engine
Uses Set Theory, Dictionaries, and Jaccard similarity.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

DATA_PATH = Path("data") / "purchases.csv"
OUTPUT_PATH = Path("output") / "recommendations.csv"


def read_purchases(path: Path = DATA_PATH) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Reads purchase records from CSV and separates valid rows from rejected rows.

    Cleaning rules:
      - Normalizes customer_id and product_id (strips whitespace, converts to uppercase).
      - Rejects rows with missing purchase_id, customer_id, or product_id.
      - Rejects duplicate purchase_ids.
      - Rejects non-positive (<= 0) or non-integer quantities.
    """
    valid_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    seen_purchases: set[str] = set()

    with open(path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line_no, raw_row in enumerate(reader, start=2):
            purchase_id = (raw_row.get("purchase_id") or "").strip()
            cust_id = (raw_row.get("customer_id") or "").strip().upper()
            cust_name = (raw_row.get("customer_name") or "").strip()
            prod_id = (raw_row.get("product_id") or "").strip().upper()
            prod_name = (raw_row.get("product_name") or "").strip()
            category = (raw_row.get("category") or "").strip()
            purchase_date = (raw_row.get("purchase_date") or "").strip()
            qty_raw = (raw_row.get("quantity") or "").strip()

            rejection_reason = None

            if not purchase_id:
                rejection_reason = "Missing purchase_id"
            elif purchase_id in seen_purchases:
                rejection_reason = f"Duplicate purchase_id: '{purchase_id}'"
            elif not cust_id:
                rejection_reason = "Missing customer_id"
            elif not prod_id:
                rejection_reason = "Missing product_id"
            else:
                try:
                    qty = int(qty_raw)
                    if qty <= 0:
                        rejection_reason = f"Quantity must be > 0 (found {qty})"
                except (ValueError, TypeError):
                    rejection_reason = f"Invalid integer quantity: '{qty_raw}'"

            if rejection_reason:
                rejected_rows.append({
                    "line_number": line_no,
                    "purchase_id": purchase_id,
                    "reason": rejection_reason,
                    "raw_data": raw_row
                })
            else:
                seen_purchases.add(purchase_id)
                valid_rows.append({
                    "purchase_id": purchase_id,
                    "customer_id": cust_id,
                    "customer_name": cust_name or "Unknown",
                    "product_id": prod_id,
                    "product_name": prod_name or "Unknown Item",
                    "category": category or "Uncategorized",
                    "purchase_date": purchase_date,
                    "quantity": qty
                })

    return valid_rows, rejected_rows




def jaccard(a: set, b: set) -> float:
    """
    Computes Jaccard similarity: |A ∩ B| / |A ∪ B|.
    Returns 0.0 when both sets are empty.
    """
    if not a and not b:
        return 0.0
    intersection_len = len(a & b)
    union_len = len(a | b)
    return intersection_len / union_len if union_len > 0 else 0.0



class RecommendationEngine:
    """Recommends products a customer has not bought, based on similar customers."""

    def __init__(self, purchases: list[dict[str, Any]], min_similarity: float = 0.0) -> None:
        self.min_similarity = min_similarity
        self.customer_baskets: dict[str, set[str]] = defaultdict(set)
        self.customer_names: dict[str, str] = {}
        self.product_metadata: dict[str, dict[str, str]] = {}
        self.product_popularity: dict[str, int] = defaultdict(int)

        # Build lookup tables once
        for row in purchases:
            cid = row["customer_id"]
            pid = row["product_id"]
            self.customer_baskets[cid].add(pid)
            self.customer_names[cid] = row["customer_name"]
            if pid not in self.product_metadata:
                self.product_metadata[pid] = {
                    "product_name": row["product_name"],
                    "category": row["category"]
                }

        # Popularity: number of distinct customer baskets containing the product
        for basket in self.customer_baskets.values():
            for pid in basket:
                self.product_popularity[pid] += 1

        self.all_customers = sorted(self.customer_baskets.keys())
        self.total_customers = len(self.all_customers)

    def history(self, customer_id: str) -> set[str]:
        """Returns the set of product IDs purchased by the customer."""
        if customer_id not in self.customer_baskets:
            raise KeyError(f"Customer '{customer_id}' not found in purchase records.")
        return self.customer_baskets[customer_id]

    def similarity(self, customer_a: str, customer_b: str) -> float:
        """Returns the Jaccard similarity between two customers."""
        return jaccard(self.history(customer_a), self.history(customer_b))

    def print_similarity_matrix(self, n: int = 5) -> None:
        """Prints an n x n Jaccard similarity matrix for the first n customers."""
        sample = self.all_customers[:n]
        header = f"{'Customer':<10}" + "".join(f"{cid:<10}" for cid in sample)
        print(header)
        print("-" * len(header))
        for r_id in sample:
            row_str = f"{r_id:<10}"
            for c_id in sample:
                sim = self.similarity(r_id, c_id)
                row_str += f"{sim:<10.2f}"
            print(row_str)


    def similar_customers(self, customer_id: str, k: int = 3) -> list[tuple[str, float]]:
        """
        Return the k most similar other customers as (customer_id, score), best first.
        Ties are broken deterministically by customer_id ascending.
        Excludes the customer themself and anyone with similarity <= min_similarity.
        """
        if customer_id not in self.customer_baskets:
            raise KeyError(f"Customer '{customer_id}' not found in purchase records.")

        candidates: list[tuple[str, float]] = []
        for other_id in self.all_customers:
            if other_id == customer_id:
                continue
            sim = self.similarity(customer_id, other_id)
            if sim > self.min_similarity:
                candidates.append((other_id, sim))

        # Sort: similarity descending (-sim), then customer_id ascending
        candidates.sort(key=lambda item: (-item[1], item[0]))
        return candidates[:k]

    
    def candidate_products(self, customer_id: str, k: int = 3) -> dict[str, dict]:
        """
        Products bought by similar customers but not by this one, with evidence.
        Evidence format:
          {
             product_id: {
                 "product_id": pid,
                 "supporting_customers": [cid1, cid2, ...],
                 "similarities": [sim1, sim2, ...]
             }
          }
        """
        target_basket = self.history(customer_id)
        neighbours = self.similar_customers(customer_id, k=k)
        candidates: dict[str, dict] = {}

        for n_id, sim in neighbours:
            neighbour_basket = self.history(n_id)
            unseen_products = neighbour_basket - target_basket
            for pid in unseen_products:
                if pid not in candidates:
                    candidates[pid] = {
                        "product_id": pid,
                        "supporting_customers": [],
                        "similarities": []
                    }
                candidates[pid]["supporting_customers"].append(n_id)
                candidates[pid]["similarities"].append(sim)

        return candidates
    

    def score(self, evidence: dict) -> float:
        """
        Ranking rule combining:
          1. Sum of similarity weights across all supporting neighbours
          2. Social proof multiplier: 1 + 0.15 * (count - 1)
          3. Global product popularity prior: 0.10 * (popularity / total_customers)
        """
        similarities = evidence["similarities"]
        pid = evidence["product_id"]
        count = len(similarities)
        sum_sim = sum(similarities)

        consensus_mult = 1.0 + 0.15 * (count - 1)
        pop_norm = self.product_popularity.get(pid, 0) / max(1, self.total_customers)

        return round((sum_sim * consensus_mult) + (0.10 * pop_norm), 3)
    



    def recommend(self, customer_id: str, n: int = 3, k: int = 3) -> list[dict]:
        """
        Returns up to n ranked recommendations for one customer.
        Falls back to global popularity if neighbours have no new products.
        """
        if customer_id not in self.customer_baskets:
            raise KeyError(f"Customer '{customer_id}' not found in purchase records.")

        target_basket = self.history(customer_id)
        candidates = self.candidate_products(customer_id, k=k)
        recommendations: list[dict] = []

        if candidates:
            for pid, ev in candidates.items():
                s = self.score(ev)
                meta = self.product_metadata[pid]
                recommendations.append({
                    "customer_id": customer_id,
                    "customer_name": self.customer_names[customer_id],
                    "product_id": pid,
                    "product_name": meta["product_name"],
                    "category": meta["category"],
                    "score": s,
                    "supporting_customers": ev["supporting_customers"]
                })
            # Deterministic sorting: score DESC, product_id ASC
            recommendations.sort(key=lambda r: (-r["score"], r["product_id"]))

            # Fallback to popularity if candidates < n
        if len(recommendations) < n:
            recommended_pids = target_basket | {r["product_id"] for r in recommendations}
            unseen_popular = [
                (pid, pop) for pid, pop in self.product_popularity.items()
                if pid not in recommended_pids
            ]
            unseen_popular.sort(key=lambda item: (-item[1], item[0]))

            for pid, pop in unseen_popular[: n - len(recommendations)]:
                meta = self.product_metadata[pid]
                fallback_score = round(0.10 * (pop / max(1, self.total_customers)), 3)
                recommendations.append({
                    "customer_id": customer_id,
                    "customer_name": self.customer_names[customer_id],
                    "product_id": pid,
                    "product_name": meta["product_name"],
                    "category": meta["category"],
                    "score": fallback_score,
                    "supporting_customers": []
                })

        # Edge case: customer bought everything
        if not recommendations:
            return [{
                "customer_id": customer_id,
                "customer_name": self.customer_names[customer_id],
                "product_id": "",
                "product_name": "",
                "category": "",
                "score": 0.0,
                "supporting_customers": []
            }]

        return recommendations[:n]

    def recommend_all(self, n: int = 3, k: int = 3) -> list[dict]:
        """Runs recommend() across all customers, assigning 1-based ranks."""
        all_rows: list[dict] = []
        for cid in self.all_customers:
            recs = self.recommend(cid, n=n, k=k)
            for rank, item in enumerate(recs, start=1):
                row = dict(item)
                row["rank"] = rank
                all_rows.append(row)
        return all_rows

    def export_csv(self, rows: list[dict], path: Path = OUTPUT_PATH) -> None:
        """Exports recommendations matching expected_output.md."""
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "customer_id",
            "customer_name",
            "rank",
            "product_id",
            "product_name",
            "category",
            "score",
            "supporting_customers"
        ]

        with open(path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "customer_id": row["customer_id"],
                    "customer_name": row["customer_name"],
                    "rank": row["rank"],
                    "product_id": row["product_id"],
                    "product_name": row["product_name"],
                    "category": row["category"],
                    "score": f"{row['score']:.3f}" if row["product_id"] else "",
                    "supporting_customers": ";".join(row["supporting_customers"])
                })
    