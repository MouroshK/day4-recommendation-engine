# Mama Mboga Online — Customer Recommendation Engine

A first-principles, explainable recommendation engine built for Mama Mboga Online's checkout flow using Python dictionaries, set theory, and Jaccard similarity.

---

## 1. Project Overview & Architecture

Mama Mboga Online delivers groceries across Nairobi. To suggest items at checkout without relying on complex, black-box machine-learning libraries, this engine:
1. Ingests raw historical transactions and cleans data defects.
2. Represents customer histories as sets of product IDs.
3. Calculates Jaccard similarity across customer baskets to identify nearest neighbours.
4. Generates candidate recommendations from neighbour purchase histories.
5. Scores candidates using an explainable formula combining collaborative similarity, neighbour consensus, and catalog popularity.
6. Handles cold-starts, saturated baskets, and edge cases gracefully.

---

## 2. Data Cleaning Rules (`read_purchases`)

The export in `data/purchases.csv` contains real-world flaws. `read_purchases()` processes the file with the following rules:

1. **Repeated Purchases / Duplicate IDs**:
   - Transaction `T9010` appears twice.
   - We track seen `purchase_id`s with a `set`. When a duplicate ID appears, it is rejected to avoid duplicate order counting.
2. **Missing Product Identifiers**:
   - Transaction `T9020` has an empty `product_id`.
   - Rows without a valid `product_id` cannot be added to a customer basket and are rejected.
3. **Non-Positive Quantities**:
   - Transaction `T9033` has a `quantity` of `0`.
   - Only transactions with an integer quantity strictly greater than zero (`quantity > 0`) are valid; zero or negative orders are rejected.
4. **ID Normalization**:
   - Row `T9041` lists the customer as lowercase `c106`.
   - All customer and product IDs are cleaned using `.strip().upper()`, ensuring `c106` correctly maps to `C106`.

**Ingestion Summary**:
- **Total read**: 63
- **Valid records used**: 60
- **Rejected records**: 3

---

## 3. Recommendation Scoring Rule & Worked Example

### Mathematical Formula

The engine ranks candidate products using a collaborative affinity score boosted by consensus and catalog popularity:

$$\text{Score}(p) = \left( \sum_{u \in N(p)} J(c, u) \right) \times \left(1 + 0.15 \times (\vert{}N(p)\vert{} - 1)\right) + 0.10 \times \left(\frac{\text{Pop}(p)}{M}\right)$$

Where:
- $J(c, u)$: Jaccard similarity between target customer $c$ and neighbour $u \in N(p)$
- $N(p)$: Set of top-$k$ neighbours who bought candidate product $p$
- $\vert{}N(p)\vert{}$: Number of supporting neighbours (consensus count)
- $\text{Pop}(p)$: Total number of distinct customer baskets containing product $p$
- $M$: Total active customers ($M = 20$)

---

### Why This Rule?
- **Weighted Affinity ($\sum J$)**: Recommendations from highly similar customers carry more weight than those from distant neighbours.
- **Consensus Multiplier (+15% per additional neighbour)**: If multiple similar customers independently purchased an item, that shared agreement strongly suggests the target customer will need it too.
- **Popularity Prior (10% weight)**: Acts as an explainable tie-breaker and baseline signal, favoring proven staple goods without drowning out personal taste.

---

### Worked Example: Customer C101 (Wanjiru)

- **Target Basket**: `{P04, P08}` (Blue Band 500g, Ketepa Tea 250g)
- **Top 3 Similar Neighbours ($k = 3$)**:
  1. **C117** ($J = 0.500$): owns `{P03, P04, P08, P10}`
  2. **C109** ($J = 0.333$): owns `{P03, P08}`
  3. **C113** ($J = 0.167$): owns `{P01, P02, P03, P04, P05}`

#### 1. Evaluating Candidate P03 (Festive Bread)
- **Supporting Neighbours**: C117, C109, and C113 ($\vert{}N(\text{P03})\vert{} = 3$)
- **Similarity Sum**:

$$\sum J = 0.500 + 0.3333 + 0.1667 = 1.000$$

- **Consensus Multiplier**:

$$\text{Consensus} = 1.0 + 0.15 \times (3 - 1) = 1.30$$

- **Popularity Prior**: Bought by 7 out of 20 customers ($\text{Pop} = 7$, $M = 20$):

$$\text{Prior} = 0.10 \times \frac{7}{20} = 0.035$$

- **Final Score**:

$$\text{Score}(\text{P03}) = (1.000 \times 1.30) + 0.035 = \mathbf{1.335}$$

#### 2. Evaluating Candidate P10 (Pampers Size 3)
- **Supporting Neighbours**: C117 only ($\vert{}N(\text{P10})\vert{} = 1$)
- **Similarity Sum**: $\sum J = 0.500$
- **Consensus Multiplier**: $1.0 + 0.15 \times (1 - 1) = 1.00$
- **Popularity Prior**: Bought by 6 out of 20 customers:

$$\text{Prior} = 0.10 \times \frac{6}{20} = 0.030$$

- **Final Score**:

$$\text{Score}(\text{P10}) = (0.500 \times 1.00) + 0.030 = \mathbf{0.530}$$

#### 3. Evaluating Candidate P01 (Kienyeji Eggs tray)
- **Supporting Neighbours**: C113 only ($\vert{}N(\text{P01})\vert{} = 1$, $J = 0.1667$)
- **Similarity Sum**: $\sum J = 0.1667$
- **Consensus Multiplier**: $1.00$
- **Popularity Prior**: Bought by 5 out of 20 customers:

$$\text{Prior} = 0.10 \times \frac{5}{20} = 0.025$$

- **Final Score**:

$$\text{Score}(\text{P01}) = 0.1667 + 0.025 = \mathbf{0.192}$$

---

### Resulting Recommendations for C101:
1. **Festive Bread (P03)** — Score: **1.335** (supported by 3 neighbours)
2. **Pampers Size 3 (P10)** — Score: **0.530** (supported by 1 neighbour)
3. **Kienyeji Eggs tray (P01)** — Score: **0.192** (supported by 1 neighbour)

---

## 4. Edge Cases Handled

1. **Saturated / Isolated Customer Baskets (e.g., `C107`, `C108`, `C120`)**:
   When a customer's nearest neighbours have no new products to suggest, the engine falls back to recommending the globally most popular items the customer does not yet own.
2. **Customers Who Bought Every Product**:
   If all catalog products are already owned, the engine outputs a placeholder row with blank product columns so the customer is not silently dropped from `recommendations.csv`.
3. **Deterministic Tie-Breaking**:
   All similarity and ranking sorts order by `(-score, customer_id / product_id)` to ensure identical outputs across multiple runs.
4. **Unknown Customer IDs**:
   Calling `engine.history("UNKNOWN")` or `engine.recommend("UNKNOWN")` raises an explicit `KeyError`.

---

## 5. How to Run

From the root project directory:

```bash
# Clone and enter repo
git clone [https://github.com/MouroshK/day4-recommendation-engine](https://github.com/MouroshK/day4-recommendation-engine)
cd day4-recommendation-engine

# Run pipeline end-to-end
python main.py
```


---

## 6. Execution Output & Verification

Below is the terminal output from running `python main.py` end-to-end, demonstrating the matrix sanity check, sample customer recommendation inspections, and the final dataset run summary.

```text
Sanity Check: First 5 Customers Similarity Matrix
Customer   C101      C102      C103      C104      C105      
------------------------------------------------------------
C101       1.00      0.00      0.00      0.00      0.00      
C102       0.00      1.00      0.00      0.00      0.00      
C103       0.00      0.00      1.00      0.00      0.00      
C104       0.00      0.00      0.00      1.00      0.00      
C105       0.00      0.00      0.00      0.00      1.00      

============================================================

## Console (one customer)
Customer C101 (Wanjiru) bought: {P04, P08}
Most similar customers:
  C117  jaccard=0.50
  C109  jaccard=0.33
  C113  jaccard=0.17
Recommendations:
  1. Festive Bread (Bakery)  score=1.335  because 3 similar customers bought it
  2. Pampers Size 3 (Baby)  score=0.530  because 1 similar customers bought it
  3. Kienyeji Eggs (tray) (Fresh)  score=0.192  because 1 similar customers bought it

Customer C113 (Wambui) bought: {P01, P02, P03, P04, P05}
Most similar customers:
  C118  jaccard=0.33
  C117  jaccard=0.29
  C101  jaccard=0.17
Recommendations:
  1. Ketepa Tea 250g (Beverages)  score=0.535  because 2 similar customers bought it
  2. Pishori Rice 5kg (Pantry)  score=0.353  because 1 similar customers bought it
  3. Pampers Size 3 (Baby)  score=0.316  because 1 similar customers bought it

## Run summary
Purchases read: 63 | used: 60 | rejected: 3
Customers: 20 | Products: 15 | Recommendations written: 60
```