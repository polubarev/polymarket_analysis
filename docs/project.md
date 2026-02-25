
## 0) Goal of baseline (MVP)

**Goal:** Build a reproducible data pipeline that downloads:

1. **Event/market metadata** (incl. tags/categories and outcome token IDs)
2. **Historical price series** for each outcome token (or at least “YES” tokens for binary markets)
   …and produces a small analysis output:

* market coverage stats (how many markets have history, how long, missingness)
* grouping by “bet type” (tags)
* clustering of price-curve “shapes” (very simple baseline)

**Non-goals for baseline:** live trading, order placement, alpha production, or execution modeling.

---

## 1) APIs to use (public, no auth)

### Market discovery + metadata (Gamma API)

Use the **events endpoint** for bulk discovery (events include their markets) as recommended by Polymarket docs. ([Polymarket Documentation][1])
Example:

* `GET https://gamma-api.polymarket.com/events?active=true&closed=false&limit=100&offset=0` ([Polymarket Documentation][1])

Key behaviors:

* Pagination via `limit` and `offset` is standard. ([Polymarket Documentation][1])
* Filter/sort options exist (e.g., order by `volume_24hr`, `liquidity`, `end_date`, etc.). ([Polymarket Documentation][1])

### Outcome token IDs (“asset ids”)

For each market, you need the **token IDs (asset IDs)** for outcomes. Docs explicitly say the Gamma API market objects include outcome token IDs in a `tokens` array. ([Polymarket Documentation][2])

### Price history (CLOB API)

Use:

* `GET https://clob.polymarket.com/prices-history?market=<ASSET_ID>&interval=1h&startTs=...&endTs=...`

Docs note the `market` query param is the **asset id**. ([Polymarket Documentation][3])

Response schema:

* `{"history": [{"t": <unix_ts>, "p": <price>}, ...]}` ([Polymarket Documentation][3])

### Optional: trade prints (Data API) for validation

Use:

* `GET https://data-api.polymarket.com/trades?market=<CONDITION_ID>&limit=10000&offset=0`

Docs: `market` is a comma-separated list of **condition IDs**, and you can also filter by `eventId`, `side`, etc. ([Polymarket Documentation][4])

---

## 2) Rate limits + request behavior (must implement)

Polymarket rate limits are enforced via Cloudflare throttling (requests can be delayed/queued). ([Polymarket Documentation][5])

Baseline needs a simple limiter (token bucket or leaky bucket) tuned to:

* Gamma `/events`: 500 req / 10s ([Polymarket Documentation][5])
* CLOB `/prices-history`: 1,000 req / 10s ([Polymarket Documentation][5])
* Data `/trades`: 200 req / 10s ([Polymarket Documentation][5])

Also implement:

* retries with exponential backoff + jitter
* timeout defaults (e.g., 10–30s)
* caching (disk cache for metadata responses) to avoid re-hitting Gamma repeatedly during dev

---

## 3) Data model (minimum viable)

Store **raw** JSON and also a **normalized** form.

### A) Raw storage

* `raw/events_YYYYMMDD.jsonl` (each line: one event JSON from Gamma)
* `raw/prices/<asset_id>.parquet` (history points)
* optional: `raw/trades/<condition_id>.parquet`

### B) Normalized tables (Parquet recommended)

1. `events.parquet`

   * `event_id` (int)
   * `slug`
   * `title`
   * `start_ts` / `end_ts` (or whatever timestamps Gamma provides)
   * `tags` (list or JSON)
2. `markets.parquet`

   * `market_id` (Gamma market id)
   * `event_id`
   * `condition_id` (for joining to Data API trades)
   * `question/title`
   * `active`, `closed`
   * `market_type` (binary vs multi-outcome if available; else derive)
3. `tokens.parquet`

   * `market_id`
   * `outcome` (e.g., “Yes”, “No”, or outcome name)
   * `asset_id` (token id used by `/prices-history`) ([Polymarket Documentation][3])
4. `price_history.parquet`

   * `asset_id`
   * `ts` (int64 unix seconds)
   * `price` (float)
   * `interval` used
   * ingestion timestamp

---

## 4) Pipeline steps (baseline)

### Step 1 — Discover a working universe of markets

* Call Gamma `GET /events?active=true&closed=false&limit=...&offset=...` to fetch, say, the top ~2,000 active events (ordered by volume/liquidity is helpful). ([Polymarket Documentation][1])
* Extract markets from each event (events contain associated markets). ([Polymarket Documentation][1])
* For each market extract:

  * `conditionId`
  * `tokens[]` outcome token IDs (asset ids) ([Polymarket Documentation][2])
  * tags (event tags are your “bet types”)

Output: `events.parquet`, `markets.parquet`, `tokens.parquet`.

### Step 2 — Fetch price history for each token (asset_id)

For each `asset_id` (start with “YES” only for binary to keep it smaller):

* Call `GET /prices-history?market=<asset_id>&interval=1h&startTs=...&endTs=...` ([Polymarket Documentation][3])
* Store results to `price_history.parquet` (append mode)

Notes:

* For baseline, set:

  * `interval=1h` for manageable size
  * `startTs = now - 90 days` (or 180) and `endTs = now`
* If history is empty, record it (coverage metric).

### Step 3 — (Optional) Pull a small trades sample for validation

Select ~100 markets and fetch:

* `GET https://data-api.polymarket.com/trades?market=<condition_id>&limit=10000&offset=0` ([Polymarket Documentation][4])
  Use this only to sanity check that “price history looks plausible vs trade prints”.

---

## 5) “Bet types” grouping (baseline definition)

Define bet type as:

* primary = **top-level tag/category** from Gamma event tags (store as `event.tags`)
* fallback = derive from slug/title keywords if tags missing

This gets you:

* Politics / Sports / Crypto / etc. (depending on tags present)

---

## 6) Shape features + clustering (simple, first pass)

### Alignment / resampling

For each market (choose the YES token asset_id):

1. Load price series, keep last N days (e.g., 90d)
2. Resample to a fixed grid (1h)
3. Forward-fill small gaps (cap fill length; beyond that mark missing)
4. Normalize time axis:

   * baseline: “time since start of window” (easy)
   * better: “time to event end” (requires reliable `end_ts`)

### Feature extraction (very simple, robust)

Compute per series:

* `p_start`, `p_end`
* `max_price`, `min_price`
* `time_of_max` (index of max / length)
* `return_total = p_end - p_start`
* `volatility = std(diff(price))`
* `max_drawdown` (peak-to-trough)
* `slope` via linear regression on time vs price

### Clustering

Baseline options:

* **KMeans on features** (fast, easy)
* Choose k = 6–12, report silhouette score and cluster sizes

Deliverables:

* `clusters.parquet` with `market_id`, `asset_id`, `cluster_id`, feature columns
* a quick plot per cluster: median curve + IQR band (if you’re doing a notebook)

---

## 7) Sanity analysis outputs (must-have checks)

Produce a small report (JSON + a couple plots) with:

1. **Coverage**

* number of events/markets discovered
* number of tokens extracted
* percent of tokens with non-empty price history
* median number of points per history series

2. **By bet type**

* coverage per tag
* average volatility per tag
* average spread proxy if later added (optional)

3. **Clusters**

* cluster sizes
* per-cluster median curve description (e.g., “early pump then fade”, “monotonic drift up”, etc.)
* distribution of tags across clusters (do certain bet types dominate a cluster?)

---

## 8) Engineering checklist (baseline quality bar)

* Idempotent runs:

  * re-running ingestion should not duplicate identical `(asset_id, ts)` points
* Logging:

  * API request counts, retries, empty histories
* Config:

  * max events, time window days, interval
* Storage:

  * local parquet is fine; design so S3/GCS can be swapped in later

---

## 9) Known pitfalls to document in code comments

* `/prices-history` uses **asset_id**, while Data API `/trades` uses **conditionId** — you need both IDs in your schema. ([Polymarket Documentation][3])
* Gaps / thin markets: many series will be sparse; don’t over-interpret curves without liquidity filters (baseline can ignore liquidity, but record it if available).
* Cloudflare throttling means “success” might be delayed; implement backoff. ([Polymarket Documentation][5])


[1]: https://docs.polymarket.com/market-data/fetching-markets "Fetching Markets - Polymarket Documentation"
[2]: https://docs.polymarket.com/trading/ctf/overview "Conditional Token Framework - Polymarket Documentation"
[3]: https://docs.polymarket.com/api-reference/markets/get-prices-history "Get prices history - Polymarket Documentation"
[4]: https://docs.polymarket.com/api-reference/core/get-trades-for-a-user-or-markets "Get trades for a user or markets - Polymarket Documentation"
[5]: https://docs.polymarket.com/api-reference/rate-limits "Rate Limits - Polymarket Documentation"
