# Customer Intelligence & Support Copilot

## Overview

A reproducible, fully offline Python/SQLite capstone built on deterministic synthetic data. The repository joins four pipelines: (A) audited data cleaning and SQL analytics, (B) RFM segmentation and pre-cutoff lapse prediction, (C) privacy-safe support-ticket triage, and (D) TF-IDF policy retrieval plus a tool-using refund agent. The Swiggy Instamart name is scenario framing only: all records, policies, orders and prices are fictional. Currency is INR. No network, API key, paid service or hosted model is used; every LLM-style call goes through the supplied `mock_llm.py`.

## Reproduce

Run these numbered commands from the repository root on Python 3.11+ (the generated database uses SQLite window functions):

1. `python -m venv .venv`
2. Activate it: macOS/Linux `source .venv/bin/activate`; Windows PowerShell `.venv\Scripts\Activate.ps1`.
3. `python -m pip install --upgrade pip`
4. `python -m pip install -r requirements.txt`
5. `python generate_data.py`
6. `python src/part_a.py`
7. `python src/validate.py`
8. `python src/part_b.py`
9. `python src/part_c.py`
10. `python src/part_d.py`

`generate_data.py` must end with `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. The scripts write all required CSV, SQLite, model, JSONL and checkpoint artifacts into `data/`, `outputs/` and `state/`. Run the scripts in the order shown: later parts depend on earlier outputs. The random seeds and deterministic synthetic generator are fixed. `requirements.txt` pins the four direct open-source dependencies used by the project; Python's standard library supplies SQLite, JSON, CSV, regex, time and filesystem functionality.

## Part A

### Raw audit

Shapes: customers `(1200, 5)`, products `(40, 4)`, orders `(13084, 9)`. Raw pandas dtypes and per-column missing counts/percentages are below. `tickets.csv` is read with `keep_default_na=False` in Part C so blank ticket bodies remain empty strings.

| Table / column | Raw dtype | Missing | Missing % |
|---|---:|---:|---:|
| customers.customer_id | object | 0 | 0.00 |
| customers.city | object | 0 | 0.00 |
| customers.signup_date | object | 0 | 0.00 |
| customers.age | float64 | 61 | 5.08 |
| customers.gender | object | 30 | 2.50 |
| products.product_id | object | 0 | 0.00 |
| products.product_name | object | 0 | 0.00 |
| products.category | object | 0 | 0.00 |
| products.price_inr | int64 | 0 | 0.00 |
| orders.order_id | object | 0 | 0.00 |
| orders.customer_id | object | 0 | 0.00 |
| orders.product_id | object | 0 | 0.00 |
| orders.quantity | int64 | 0 | 0.00 |
| orders.order_ts | object | 0 | 0.00 |
| orders.status | object | 0 | 0.00 |
| orders.discount_pct | int64 | 0 | 0.00 |
| orders.order_amount | object | 53 | 0.41 |
| orders.delivery_minutes | float64 | 2541 | 19.42 |

| Audit check | Result |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order IDs | 12,309 |
| Non-numeric order amounts (`to_numeric(errors='coerce')`) | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2,541 |
| Blank customer ages | 61 |
| Ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank raw gender spellings | 10 |
| Blank genders | 30 |

### Cleaning rules and changes

- **Customers:** trim city strings and apply title case (24 raw spellings collapse to the six canonical cities); map case/letter variants to `male` or `female`, and blanks/unrecognised values to `unknown` (10 spellings plus blanks become three categories). Convert age to numeric; set the six impossible values outside 10–100 to missing; fill all missing ages (61 blanks + 6 impossible = 67) with the city median using `groupby('city').age.transform('median')`; convert `signup_date` to datetime and persist as ISO date text. Clean city counts: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184. Gender counts: female 610, male 560, unknown 30. There are zero missing ages; all ages are 18–60.
- **Orders (in the specified order):** parse `order_ts` as datetime; remove 119 exact duplicate rows; sort by `order_id, order_ts` and keep the latest record per order ID, removing the 656 earlier pending rows and yielding 12,309 unique orders; coerce amount to numeric and fill the 112 remaining unparseable final-order amounts from `round(price_inr × quantity × (1 − discount_pct/100), 2)` after joining products; set all 105 negative delivery values that remain after final-order selection to missing; for completed orders fill 774 missing delivery times using the completed-order product-category median via `groupby(...).transform`; leave all 1,199 cancelled delivery times missing. `order_ts` is stored as `YYYY-MM-DD HH:MM:SS` text.
- **Products:** catalogue names, categories and prices are already clean and are retained unchanged.

Clean orders: 12,309 rows, 11,110 completed and 1,199 cancelled, no pending rows, zero missing amounts, zero missing completed delivery times, 1,199 missing cancelled delivery times. Completed-order revenue is **INR 2,998,668.80**. The three clean CSVs are written to `data/clean/`; the SQLite database `data/instamart.db` has `customers`, `products`, and `orders`. `src/validate.py` asserts nine checkpoints and prints exactly `ALL CHECKPOINTS PASSED`.

### SQL analysis

All seven queries are in `sql/queries.sql` and are executed with `pandas.read_sql_query` against SQLite.

| Query | Result |
|---|---|
| Q1 — completed orders over INR 500, grouped by customer; HAVING spend over INR 8,000 | 30 customers |
| Q2 — completed revenue by city and product category, three-table join | 30 cells; total INR 2,998,668.80; largest cell Bengaluru / Electronics = INR 233,909.45 |
| Q3 — products above average catalogue price (scalar subquery) | 11 products |
| Q4 — customers with no completed order (LEFT JOIN) | 25 customers |
| Q5 — customers spending above the average customer completed spend (two CTEs) | 550 customers; average INR 2,552.06 |
| Q6 — per-city spend with RANK, DENSE_RANK and ROW_NUMBER | 1,175 customers; 8 rows have RANK different from ROW_NUMBER |
| Q7 — daily completed revenue and cumulative window total | 183 days; final cumulative revenue INR 2,998,668.80 |

Top Q6 customer per city: Bengaluru C0082 (INR 12,856.20), Chennai C0930 (INR 10,151.40), Delhi C0291 (INR 13,742.60), Hyderabad C0743 (INR 12,730.05), Mumbai C0693 (INR 14,766.55), Pune C0426 (INR 10,377.10). Ties share a RANK value and therefore create gaps in the ranking, while ROW_NUMBER always assigns a unique sequential number.

### pandas cross-check and exploration

A `pivot_table(..., margins=True)` reconstruction of Q2 has a grand total of INR 2,998,668.80, equal to SQL. Named city aggregation (completed orders only): Bengaluru INR 523,643.40 / mean INR 276.47 / 1,894 orders; Chennai INR 463,629.15 / INR 272.72 / 1,700; Delhi INR 502,829.80 / INR 243.62 / 2,064; Hyderabad INR 528,589.90 / INR 284.95 / 1,855; Mumbai INR 479,058.50 / INR 297.00 / 1,613; Pune INR 500,918.05 / INR 252.48 / 1,984.

Completed `order_amount`: mean INR 269.91, median INR 135.00, skewness 4.61, Q1 INR 64.80, Q2 INR 135.00, Q3 INR 297.00, IQR INR 232.20, lower fence INR −283.50, upper fence INR 645.30, 0 lower outliers and 897 upper outliers. The data describes completed grocery, dairy, snack, personal-care and small-electronics orders across six cities during the synthetic six-month window. Spend is strongly right-skewed: the mean (INR 269.91) is much higher than the median (INR 135.00), with 897 high-value outliers. The outliers are plausible large baskets and electronics purchases, although they warrant validation rather than automatic deletion. The largest absolute correlation is order amount versus discount percentage (−0.095); order amount versus quantity is +0.087. Correlations this small describe weak linear association and do not establish causation.

## Part B

### RFM and clustering

RFM uses completed orders only, at snapshot `2025-10-01 00:00:00`: recency is whole days since the latest completed order, frequency is distinct completed order count, and monetary is completed spend. It covers 1,175 customers. Recency median/min/max = 19 / 0 / 180 days; frequency mean/max = 9.46 / 33; monetary mean/max = INR 2,552.06 / INR 14,766.55.

The three RFM fields are standardised before K-means. All runs use `n_init=10, random_state=42`.

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1,820.8 | 0.465 |
| 3 | 1,268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

K=4 is selected: it has the strongest silhouette, a pronounced elbow versus K=3, and yields four interpretable campaign groups. K=8 must have lower inertia than K=2 because adding clusters can only reduce within-cluster squared distances; this is a mechanical property, not proof of better business segmentation. Very large K can fragment actionable groups into tiny clusters and overfit noise. PCA on scaled RFM explains 0.634 and 0.218 of variance (total 0.852).

| Segment ID | Size | Mean recency (days) | Mean frequency | Mean monetary | Business name / action |
|---:|---:|---:|---:|---:|---|
| 0 | 436 | 28.86 | 3.77 | INR 1,057.62 | Developing Customers — low-cost cross-category discovery offers |
| 1 | 138 | 24.12 | 6.77 | INR 7,129.70 | **Premium Loyalists (Premium)** — personal concierge and early access |
| 2 | 419 | 8.77 | 19.40 | INR 3,410.02 | Frequent Essentials — subscription/reorder nudges |
| 3 | 182 | 109.04 | 2.20 | INR 685.99 | At-Risk Occasional — win-back coupon and service check |

Cluster IDs are arbitrary labels, not an ordinal scale. `outputs/customer_segments.csv` contains `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`; exactly the segment with the highest mean monetary value is marked Premium.

### Lapse target and leakage-safe features

The cutoff is `2025-09-01`. The cohort is 1,156 customers with at least one completed order before the cutoff; 443 (38.32%) have no completed order during September and are labelled lapsed. The split is 924 training rows and 232 test rows (`test_size=0.2, stratify=y, random_state=42`). Features are pre-cutoff recency, frequency, monetary, mean discount, cancellation rate, mean completed delivery time, favourite category, city, age, gender and tenure. No order dated on or after 2025-09-01 enters the feature table; September orders are used only to form the label. Segment labels from the whole-window RFM are not features. The feature matrix is asserted to contain no missing values. Numeric scaling and nominal one-hot encoding (`drop='first'`) live in a `ColumnTransformer` pipeline fitted only on training rows; no encoder, scaler or resampler sees test rows. The fitted frequency scaler mean is 8.115, versus the full-cohort feature mean 8.029, because the scaler is fitted on the training subset only and the stratified split has a different sample composition.

### Model tuning and evaluation

Always predicting not-lapsed gives test accuracy 61.6% and lapsed-class F1 0.000. Grid searches use 5-fold CV and lapsed-class F1: LogisticRegression `C ∈ {0.01, 0.1, 1, 10, 100}` (best C=100, CV F1=0.651); RandomForestClassifier `max_depth ∈ {3,5,8}` × `min_samples_leaf ∈ {5,10,20}` (45 fits; best depth=8, leaf=10, CV F1=0.681).

| Model | Confusion matrix `[[TN, FP], [FN, TP]]` | Lapsed precision | Lapsed recall | Lapsed F1 |
|---|---:|---:|---:|---:|
| Tuned logistic regression | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 |
| Tuned random forest | `[[113, 30], [31, 58]]` | 0.659 | 0.652 | 0.655 |

The random forest is selected because it has the higher cross-validated and test F1 while also giving slightly better precision at the same recall. Missing a customer who is about to lapse can cost repeat orders and future lifetime value, while wrongly targeting a loyal customer wastes a discount and can train them to wait for promotions. The model's balanced precision/recall is a practical compromise for a retention queue. The selected model reaches test F1 0.655, above the 0.55 requirement.

Training-only random oversampling duplicates lapsed training rows with replacement (`random_state=42`) until both classes have 570 rows; the untouched test set is unchanged. Logistic regression on the balanced training set gives precision 0.600, recall 0.809 and F1 0.689 versus 0.644 / 0.652 / 0.648 unbalanced. Recall moves most: more genuinely lapsing customers are caught, at the cost of more false-positive outreach; precision does not improve.

The five leading random-forest features by importance are frequency (0.280), monetary (0.271), recency (0.115), average delivery (0.061) and cancellation rate (0.052). In the logistic model, frequency has a negative coefficient (higher ordering frequency lowers predicted lapse odds), monetary is positive conditional on the other features (+0.283), and favourite-category Electronics is negative (−1.135) relative to the dropped reference category. The model is saved to `outputs/lapse_model.joblib`, reloaded, and its test predictions are asserted identical. `outputs/retention_list.csv` ranks test-set Premium customers by predicted lapse probability; 10 are listed and 1 truly lapsed. **This retention list is illustrative, not a leakage-free evaluation, because its Premium flag is a post-hoc whole-window label that includes the September outcome window.**

## Part C

### Privacy and input guardrails

Tickets are processed in original file order. Empty/whitespace tickets are skipped before any model call (4); tickets with `est_tokens(text) > 400` are flagged long before any model call (4; 734 tokens each). For eligible tickets, redaction runs in the required order: email → `[EMAIL]`, phone → `[PHONE]`, order ID → `[ORDER_ID]`. Substitutions: 55 emails, 91 phone numbers, 132 order IDs, 278 total. The pipeline asserts that reapplying all three patterns to each redacted message finds nothing. The user message passed to the model is exactly the redacted ticket text.

### Prompts, retries and validation

Both prompts specify CLASSIFY, the four allowed categories and JSON-only `{ "category": "...", "reason": "..." }`. The zero-shot system prompt has no `EXAMPLES:` marker; the few-shot prompt has labelled fraud and delivery examples after `EXAMPLES:`. Every chat call is wrapped in up to three retries; retry delay is `0.05 × 2**n × uniform(0.9, 1.15)` seconds and one retry log line includes ticket ID, retry number and wait. `mock_llm.reset()` is called before each full run. The full batch produces 19 transient errors, all recovered. JSON replies are accepted only for the four allowed category labels; non-JSON replies are accepted only when `re.findall(r'[a-z]+', text.lower())` finds exactly one allowed category word. Invalid JSON/category outputs are flagged for human review, never guessed.

| Run | Classified | Accuracy | Total tokens | Tokens / classified ticket | Status counts |
|---|---:|---:|---:|---:|---|
| Zero-shot | 147 | 85.7% (126/147) | 9,400 | 63.95 | 135 classified_json; 12 classified_keyword; 9 flagged_invalid_output; 4 skipped_empty; 4 flagged_long |
| Few-shot | 147 | 100% (147/147) | 12,520 | 85.17 | 135 classified_json; 12 classified_keyword; 9 flagged_invalid_output; 4 skipped_empty; 4 flagged_long |

Zero-shot confusion counts (true → predicted): billing→billing 38; delivery→delivery 36; technical→technical 29; technical→billing 10; fraud→fraud 23; fraud→billing 11. The few-shot prompt costs 3,120 extra tokens overall, or 21.22 more tokens per classified ticket, but raises accuracy by 14.3 percentage points and removes all 21 fraud/technical-to-billing errors in this deterministic run. That extra cost is worthwhile when category correctness and fraud escalation risk outweigh the small token increase; zero-shot remains a cheaper option for lower-risk queues with human review.

### Budget and routing

The zero-shot budget demonstration uses `TOKEN_BUDGET=6000`: 56 tickets are marked `budget_exhausted` (total consumed before stopping is 6,038 tokens). `outputs/triage_budget_demo.csv` records this run. The final few-shot run uses `TOKEN_BUDGET=100000` and writes 164 rows to `outputs/triage_results.csv`. `skipped_empty`, `flagged_long`, `flagged_invalid_output` and `budget_exhausted` route to `human_review` / NORMAL; valid fraud routes to `security_team` / HIGH; other valid tickets from Premium customers route to `premium_desk` / HIGH; all remaining valid tickets route to `support_queue` / NORMAL. Missing segment records are treated as non-premium. Categories are rule-validated against the allowed set; uncertain/invalid outputs and guardrail/budget cases are sent to humans rather than auto-routed as resolved categories.

## Part D

### Policy RAG

The five policy files are split into 40-word chunks with 10-word overlap. There are 15 chunks: refund 4, delivery 3, escalation 3, payment 3, privacy 2. `TfidfVectorizer()` is fitted on the chunk corpus; retrieval transforms the query with that same vectorizer and ranks cosine similarity. For `refund policy late delivery`, all top three chunks come from `refund_policy.txt`. TF (term frequency) measures how often a term occurs in a chunk; IDF downweights terms common across many chunks and upweights rarer, more discriminative terms. TF-IDF is lexical, so semantically equivalent wording with different vocabulary can rank poorly.

The answer function retrieves three chunks and sends a POLICY ASSISTANT system instruction plus a user message in the exact `CONTEXT:` / `[source] chunk` / `QUESTION:` format. It returns the mock model's answer and the retrieved source names.

| Chunking | Answerable questions with expected doc in top 3 | Answer contains expected fact | Unanswerable questions with fallback |
|---|---:|---:|---:|
| 40 words / 10 overlap | 12/12 | 12/12 | 2/2 |
| 40 words / 0 overlap | 12/12 | 10/12 | 2/2 |

With zero overlap, “What delivery time makes a completed order eligible for a full refund?” loses the expected phrase `more than 45 minutes`: the threshold is split across adjacent chunks, and neither retrieved sentence contains the full fact. The payment-failure/48-hours answer also breaks at a boundary. The overlapping chunks preserve these boundary-spanning facts.

### Refund tools and ReAct runs

`TOOLS_SPEC` defines descriptions and JSON parameter schemas for `get_order`, `search_policy`, `calculate_refund`, `issue_refund` and `escalate_to_human`; `TOOL_MAP` maps those names to Python functions. `get_order` reads the Part A SQLite database; `search_policy` reuses the retriever; `calculate_refund` applies the >45-minute full-refund rule; issue and escalation append timestamped JSON lines to `outputs/refund_ledger.jsonl`. Only completed orders are eligible. Amounts above INR 1,000 are escalated, not automatically issued. All simulated requests are treated as inside the 7-day window.

| Scenario | Status | Refund / escalation amount | Model calls | Tokens used |
|---|---|---:|---:|---:|
| S1 | REFUND | INR 128.25 | 5 | 892 |
| S2 | REFUND | INR 135.00 | 5 | 892 |
| S3 | ESCALATE | INR 1,999.00 | 5 | 879 |
| S4 | DENIED | INR 0.00 | 4 | 601 |
| S5 | DENIED | INR 0.00 | 3 | 334 |
| S6 | NOT_FOUND | INR 0.00 | 2 | 70 |
| S7 | FALLBACK_MAX_ITERATIONS | INR 0.00 | 6 | 2,994 |

S7 reaches all six model calls at the default `token_limit=6000` and stops safely with `FALLBACK_MAX_ITERATIONS`. The S1 demonstration with `token_limit=200` ends `FALLBACK_TOKEN_LIMIT`; the S1 demonstration with `time_limit_seconds=0` ends `FALLBACK_TIMEOUT` before the first model call. Both fallbacks retain completed steps and provide a human-handoff summary.

Every tool observation is checkpointed under `state/` with run ID, scenario ID, full messages, token count, step number and executed-tool history. The S1 crash is simulated immediately after the second tool observation has been checkpointed. Resume then completes with REFUND / INR 128.25. Tools executed across crash and resume, in order, exactly once each: `get_order` → `search_policy` → `calculate_refund` → `issue_refund`. `outputs/agent_runs.jsonl` stores each of the seven scenario runs, including tool names, arguments and observations.

### ReAct versus plan-and-execute (under 250 words)

ReAct uses one model call per decision and waits for each tool observation before deciding the next action. In these runs S1 took 5 calls (892 tokens), S5 took 3 (334 tokens), and S6 took 2 (70 tokens); it therefore spends more calls on the common multi-step refund path than a plan-and-execute design that asks for a complete sequence once. A plan can reduce latency and token overhead when the order is straightforward, but it may continue with an invalid sequence if a tool returns an unexpected observation. ReAct sees `NOT_FOUND` after `get_order` and stops immediately, and sees a cancelled status before calculating or issuing a refund. For this workflow I would keep ReAct for refunds because money movement and fraud-adjacent requests need observation-driven checks and safe stopping; a compact plan could still be used to propose a plan, with each consequential action gated by tool results and policy checks.

## Limitations and Next Steps

The data is synthetic, small, and generated from simple archetypes; results do not represent real customer behaviour, real Instamart systems or actual policy. TF-IDF is lexical rather than semantic, and the deterministic mock model does not measure production LLM quality. The model is evaluated on one stratified holdout and has no temporal backtest or calibrated-probability study. Add monitoring, drift checks, fairness review, human audit trails, idempotency keys for refund side effects, and a time-based validation split before any real deployment. Do not connect this prototype to live payments or customer records. **The Premium flag used to filter the retention list is a post-hoc whole-window label that includes September; consequently, the retention list is illustrative and not a leakage-free evaluation.**
