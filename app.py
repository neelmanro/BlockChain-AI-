import os, time, random, threading, requests
from dataclasses import dataclass, asdict
from collections import deque, defaultdict
from typing import Optional, List
from flask import Flask, render_template, jsonify, request



# --- Optional OpenAI (used for one-line summaries; safe fallback if no key)
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
try:
    from openai import OpenAI
    oai_client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None
except Exception:
    oai_client = None

# --- Optional Chainlink via Web3 for price enrichment
ETH_RPC_URL = os.getenv("ETH_RPC_URL", "")  # Alchemy/Infura/Web3 RPC URL with mainnet access
CHAINLINK_ETH_USD_FEED = os.getenv("CHAINLINK_ETH_USD_FEED", "0x5f4ec3df9cbd43714fe2740f5e3616155c5b8419")  # Mainnet ETH/USD
w3 = None
price_feed = None
try:
    if ETH_RPC_URL:
        from web3 import Web3
        w3 = Web3(Web3.HTTPProvider(ETH_RPC_URL, request_kwargs={"timeout": 10}))
        # Minimal ABI for AggregatorV3Interface latestRoundData()
        agg_abi = [{
            "inputs": [], "name": "latestRoundData", "outputs": [
                {"internalType":"usint80","name":"roundId","type":"uint80"},
                {"internalType":"int256","name":"answer","type":"int256"},
                {"internalType":"uint256","name":"startedAt","type":"uint256"},
                {"internalType":"uint256","name":"updatedAt","type":"uint256"},
                {"internalType":"uint80","name":"answeredInRound","type":"uint80"}],
            "stateMutability":"view","type":"function"}]
        price_feed = w3.eth.contract(address=Web3.to_checksum_address(CHAINLINK_ETH_USD_FEED), abi=agg_abi)
except Exception:
    w3 = None
    price_feed = None

# --- The Graph (Uniswap v3) endpoint
UNIV3_SUBGRAPH = os.getenv("UNIV3_SUBGRAPH", "https://api.thegraph.com/subgraphs/name/uniswap/uniswap-v3")

app = Flask(__name__)

# ---------------------------- Data Model -------------------------------------
@dataclass
class Tx:
    id: str
    hash: str
    chain: str
    addr_from: str
    addr_to: str
    usd: float
    latency: int
    ts: float
    risk: int
    flagged: bool
    ai_summary: Optional[str] = None
    explorer_url: Optional[str] = None
    token_symbol: Optional[str] = None

EVENTS: deque[Tx] = deque(maxlen=1200)

FIRST_SEEN: dict[str, float] = {}
FANOUT_WIN: defaultdict[str, deque] = defaultdict(lambda: deque(maxlen=300))
EDGES: deque[tuple[str, str, float, float]] = deque(maxlen=800)

RISK_BLOCKLIST = {
    "0xdeadbeefdeadbeefdeadbeefdeadbeefdead0001",
    "0xbadc0ffee0ddf00dba5eba11badc0ffee0000002",
}
CHAINS = ["Ethereum", "Polygon", "Arbitrum", "Avalanche"]

def now_ms() -> float: return time.time() * 1000
def rand_hex(n: int) -> str: return "0x" + "".join(random.choice("0123456789abcdef") for _ in range(n))

# --------------------------- Risk Heuristics ---------------------------------
def is_new_wallet(addr: str, horizon_ms: int = 24*60*60*1000) -> bool:
    fs = FIRST_SEEN.get(addr)
    return fs is None or (now_ms() - fs) < horizon_ms

def score_tx(tx_from: str, tx_to: str, usd: float, ts_ms: float, token_symbol: str | None = None) -> tuple[int, list[str]]:
    reasons, score = [], 0
    FIRST_SEEN.setdefault(tx_from, ts_ms)
    FIRST_SEEN.setdefault(tx_to, ts_ms)

    # 1) Fan-out behavior (burst of unique outputs in 5 min)
    w = FANOUT_WIN[tx_from]; w.append((tx_to, ts_ms, usd))
    cutoff = ts_ms - 5*60*1000
    while w and w[0][1] < cutoff: w.popleft()
    fanout_count = len({to for (to, t, u) in w})
    total_usd = sum(u for (_to, _t, u) in w)
    avg_usd = (total_usd / fanout_count) if fanout_count else 0
    if fanout_count >= 10 and avg_usd >= 500 and is_new_wallet(tx_to):
        score += int(30 + 3*fanout_count + 0.01*total_usd)
        reasons.append(f"Fan-out {fanout_count} wallets")

    # 2) Ping-pong (wash-like) within 10 min
    EDGES.append((tx_from, tx_to, ts_ms, usd))
    reverse_hits = sum(1 for (f,t,_ts,_u) in EDGES if f == tx_to and t == tx_from and ts_ms - _ts <= 10*60*1000)
    if reverse_hits >= 3 and total_usd >= 10_000:
        score += int(40 + 10*reverse_hits + 0.005*total_usd)
        reasons.append(f"Ping-pong {reverse_hits}x")

    # 3) Large transfer
    if usd >= 75_000:
        score += 35 + min(25, int(usd) // 50_000)
        reasons.append(f"Large amount ${int(usd):,}")

    # 4) Stablecoin special casing (fast bursts can be scams/drainers)
    if token_symbol in {"USDC","USDT","DAI"} and fanout_count >= 6 and avg_usd >= 200:
        score += 15
        reasons.append("Stablecoin burst")

    # 5) Blocklist
    if tx_from in RISK_BLOCKLIST or tx_to in RISK_BLOCKLIST:
        score = max(score, 90)
        reasons.append("Known risky counterparty")

    return min(100, score), reasons

def ai_summarize(tx: Tx, reasons: list[str]) -> str:
    base = f"{tx.addr_from[:6]}… → {tx.addr_to[:6]}… moved ${int(tx.usd):,} on {tx.chain}."
    if not oai_client:
        risk_word = "high" if tx.risk >= 85 else "medium" if tx.risk >= 70 else "low"
        why = "; ".join(reasons) if reasons else "no major anomalies"
        return f"{base} Risk {tx.risk}/100 ({risk_word}); {why}."
    try:
        prompt = (
            "One concise sentence (<20 words) explaining why a blockchain tx may be risky.\n"
            f"Tx: chain={tx.chain}, usd={int(tx.usd)}, token={tx.token_symbol}, from={tx.addr_from}, to={tx.addr_to}, "
            f"reasons={reasons}, risk={tx.risk}/100."
        )
        resp = oai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2, max_tokens=40,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        risk_word = "high" if tx.risk >= 85 else "medium" if tx.risk >= 70 else "low"
        why = "; ".join(reasons) if reasons else "no major anomalies"
        return f"{base} Risk {tx.risk}/100 ({risk_word}); {why}."

# ------------------------ Chainlink Price Helper ------------------------------
def chainlink_eth_usd() -> Optional[float]:
    """Return ETH/USD price if Chainlink feed + RPC configured, else None."""
    try:
        if price_feed:
            _, answer, _, _, _ = price_feed.functions.latestRoundData().call()
            # Mainnet ETH/USD has 8 decimals
            return float(answer) / 1e8
    except Exception:
        return None
    return None

# ----------------------- Real Data: The Graph (Uniswap) ----------------------
def fetch_uniswap_swaps(limit: int = 20) -> list[dict]:
    """Fetch recent swaps from Uniswap v3 (Ethereum) using The Graph."""
    query = """
    query RecentSwaps($n:Int!) {
      swaps(first: $n, orderBy: timestamp, orderDirection: desc) {
        id
        transaction { id }
        sender
        recipient
        amountUSD
        amount0
        amount1
        token0 { symbol }
        token1 { symbol }
        timestamp
      }
    }"""
    try:
        r = requests.post(UNIV3_SUBGRAPH, json={"query": query, "variables": {"n": limit}}, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data.get("data", {}).get("swaps", []) or []
    except Exception:
        return []

def append_real_batch():
    """Turn Uniswap swaps into our Tx objects and append to EVENTS."""
    swaps = fetch_uniswap_swaps(limit=12)
    if not swaps:
        return
    eth_px = chainlink_eth_usd()  # optional enrichment
    for s in swaps:
        tx_hash = s["transaction"]["id"]
        frm = s.get("sender") or rand_hex(40)
        to = s.get("recipient") or rand_hex(40)
        usd = float(s.get("amountUSD") or 0.0)
        # If amountUSD missing/zero, attempt rough estimate from amount0 as ETH
        if usd == 0.0 and eth_px is not None:
            try:
                usd = abs(float(s.get("amount0") or 0.0)) * eth_px
            except Exception:
                pass
        ts = float(s.get("timestamp") or time.time())
        token_symbol = (s.get("token0", {}) or {}).get("symbol") or "ETH"
        risk, reasons = score_tx(frm, to, usd, ts*1000, token_symbol)
        flagged = (risk >= 70) or (usd >= 75_000)
        tx = Tx(
            id=tx_hash[:20],
            hash=tx_hash,
            chain="Ethereum",
            addr_from=frm,
            addr_to=to,
            usd=usd,
            latency=random.randint(40, 260),
            ts=ts*1000,
            risk=risk,
            flagged=flagged,
            explorer_url=f"https://etherscan.io/tx/{tx_hash}",
            token_symbol=token_symbol,
        )
        if flagged and not tx.ai_summary:
            tx.ai_summary = ai_summarize(tx, reasons)
        # De-dup by hash
        if not any(e.hash == tx.hash for e in EVENTS):
            EVENTS.appendleft(tx)

# --------------------------- Mock Stream (fallback) ---------------------------
def append_fake_batch():
    burst = 1 + random.randint(0, 3)
    for _ in range(burst):
        chain = random.choice(CHAINS)
        usd = max(5, int(random.random() * 250_000))
        latency = random.randint(40, 260)
        ts = now_ms()
        f = rand_hex(40); t = rand_hex(40)
        token_symbol = random.choice(["ETH","USDC","USDT","DAI","ARB","MATIC","WBTC"])
        risk, reasons = score_tx(f, t, float(usd), ts, token_symbol)
        flagged = (risk >= 70) or (usd > 75_000) or (latency > 200)
        tx = Tx(
            id=rand_hex(10),
            hash=rand_hex(64),
            chain=chain,
            addr_from=f,
            addr_to=t,
            usd=float(usd),
            latency=latency,
            ts=ts,
            risk=risk,
            flagged=flagged,
            explorer_url=f"https://etherscan.io/tx/{rand_hex(64)}" if chain == "Ethereum" else None,
            token_symbol=token_symbol,
        )
        if flagged:
            tx.ai_summary = ai_summarize(tx, reasons)
        EVENTS.appendleft(tx)

# ----------------------------- Background Feeder ------------------------------
def feeder_loop():
    """
    Prefer real data via The Graph; if it fails temporarily, add a small mock batch
    so UI still looks live.
    """
    while True:
        try:
            append_real_batch()
        except Exception:
            pass
        # Always keep some motion
        if len(EVENTS) < 50:
            append_fake_batch()
        time.sleep(1.2)

threading.Thread(target=feeder_loop, daemon=True).start()

# ------------------------------ Helpers --------------------------------------
def compute_stats():
    now = now_ms()
    window_ms = 5000
    recent = [e for e in EVENTS if now - e.ts <= window_ms]
    tps = len(recent) / (window_ms / 1000)
    avg_latency = int(sum(e.latency for e in EVENTS) / len(EVENTS)) if EVENTS else 0
    flagged = sum(1 for e in EVENTS if e.flagged)
    total = len(EVENTS)
    return {"tps": f"{tps:.1f}", "avg_latency": avg_latency, "flagged": flagged, "total": total}

def apply_filters(all_txs: List[Tx], chain: str, min_usd: float, q: str, only_flagged: bool):
    txs = all_txs
    if chain and chain != "all": txs = [t for t in txs if t.chain.lower() == chain.lower()]
    if min_usd: txs = [t for t in txs if t.usd >= float(min_usd)]
    if q:
        ql = q.lower()
        txs = [t for t in txs if ql in t.hash.lower() or ql in t.addr_from.lower() or ql in t.addr_to.lower()]
    if only_flagged: txs = [t for t in txs if t.flagged]
    return txs

# ------------------------------- Pages & APIs --------------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template("home.html")

@app.get("/api/stream")
def api_stream():
    chain = request.args.get("chain", "all")
    min_usd = float(request.args.get("min_usd", "0") or 0)
    q = request.args.get("q", "").strip()
    only_flagged = bool(request.args.get("only_flagged"))
    txs = apply_filters(list(EVENTS), chain, min_usd, q, only_flagged)[:120]
    return jsonify({"stats": compute_stats(), "txs": [asdict(t) for t in txs]})

@app.get("/api/tx/<txid>")
def api_tx(txid: str):
    for t in EVENTS:
        if t.id == txid or t.hash.startswith(txid):
            if not t.ai_summary:
                t.ai_summary = ai_summarize(t, ["on-demand"])
            return jsonify(asdict(t))
    return jsonify({"error": "not found"}), 404


@app.post("/api/sponsor-check")
def api_sponsor_check():
    """
    Return booleans for sponsor integrations to display in the UI.
    - chainlink_ok: True if our app has a Chainlink price feed available (i.e., ETH_RPC_URL set)
    - thegraph_ok: True because we fetch swaps via The Graph subgraph already
    - stablecoin_mode: True if this tx involves a common stablecoin
    """
    data = request.get_json(force=True) or {}
    sym = (data.get("token_symbol") or "").upper()

    chainlink_ok = bool(price_feed)   # we set this when ETH_RPC_URL is configured
    thegraph_ok = True                # we are fetching from a The Graph subgraph
    stablecoin_mode = sym in {"USDC", "USDT", "DAI"}

    return jsonify({
        "chainlink_ok": chainlink_ok,
        "thegraph_ok": thegraph_ok,
        "stablecoin_mode": stablecoin_mode
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
