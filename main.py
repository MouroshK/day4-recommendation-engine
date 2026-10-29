"""
main.py - End-to-end execution script for Mama Mboga Online Recommender.
Matches the console layout and run summary from expected_output.md.
"""
from recommender import (
    DATA_PATH,
    OUTPUT_PATH,
    read_purchases,
    RecommendationEngine,
)


def print_customer_console(engine: RecommendationEngine, customer_id: str, k: int = 3, n: int = 3) -> None:
    """Prints the console inspection block matching expected_output.md."""
    name = engine.customer_names[customer_id]
    basket = sorted(engine.history(customer_id))
    print(f"Customer {customer_id} ({name}) bought: {{{', '.join(basket)}}}")

    print("Most similar customers:")
    similar = engine.similar_customers(customer_id, k=k)
    for other_id, sim in similar:
        print(f"  {other_id}  jaccard={sim:.2f}")

    print("Recommendations:")
    recs = engine.recommend(customer_id, n=n, k=k)
    for rank, rec in enumerate(recs, start=1):
        pname = rec["product_name"]
        cat = rec["category"]
        score = rec["score"]
        count = len(rec["supporting_customers"])
        if count > 0:
            reason = f"because {count} similar customers bought it"
        else:
            reason = "because it is popular across all stores"
        print(f"  {rank}. {pname} ({cat})  score={score:.3f}  {reason}")
    print()


def main() -> None:
    # 1. Ingest and validate data
    valid_rows, rejected_rows = read_purchases(DATA_PATH)
    total_read = len(valid_rows) + len(rejected_rows)

    # 2. Initialize engine
    engine = RecommendationEngine(valid_rows, min_similarity=0.0)

    # 3. Print similarity matrix sanity check
    print("Sanity Check: First 5 Customers Similarity Matrix")
    engine.print_similarity_matrix(n=5)
    print("\n" + "=" * 60 + "\n")

    # 4. Print inspection for two demonstration customers (as requested by starter.py)
    print("## Console (one customer)")
    print_customer_console(engine, "C101", k=3, n=3)
    print_customer_console(engine, "C113", k=3, n=3)

    # 5. Generate and export recommendations for all 20 customers
    all_recommendations = engine.recommend_all(n=3, k=3)
    engine.export_csv(all_recommendations, OUTPUT_PATH)

    # 6. Run summary matching expected_output.md
    print("## Run summary")
    print(f"Purchases read: {total_read} | used: {len(valid_rows)} | rejected: {len(rejected_rows)}")
    print(f"Customers: {len(engine.all_customers)} | Products: {len(engine.product_metadata)} | Recommendations written: {len(all_recommendations)}")


if __name__ == "__main__":
    main()