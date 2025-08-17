# BlockSense AI

Real-time cross-chain transaction intelligence for detecting risky or anomalous blockchain activity.

## 🚀 Overview
BlockSense AI is a live transaction monitoring dashboard that analyzes blockchain activity in real time.  
It integrates **The Graph** (Uniswap v3 subgraph) and **Chainlink** (ETH/USD price feed) with a Flask backend, TailwindCSS frontend, and optional **AI-generated insights**.

The platform scores transactions on a scale of 0–100 using heuristics like:
- Fan-out bursts of transfers
- Ping-pong behavior (wash-like activity)
- Large transfer volume
- Stablecoin bursts
- New or first-seen wallets
- Blocklisted addresses

Flagged transactions are highlighted in the dashboard, with human-readable AI summaries if an OpenAI API key is available.

## ✨ Features
- **Live blockchain data** from The Graph’s Uniswap v3 subgraph  
- **Price enrichment** with Chainlink ETH/USD feed  
- **AI summaries** of risky behavior using OpenAI (optional)  
- **Risk scoring engine** based on transaction patterns and heuristics  
- **Dashboard UI** with:
  - Real-time transaction feed  
  - Filtering by chain, minimum USD amount, search by hash/address, flagged-only mode  
  - Metrics: throughput, latency, flagged count, cached events  
  - Detailed panel with AI insights and sponsor integration checks  
- **Mock data fallback** to ensure the demo always runs smoothly  

## 🛠 Tech Stack
- **Backend**: Python, Flask  
- **Blockchain Data**: The Graph (Uniswap v3 subgraph), Chainlink price oracles  
- **Frontend**: HTML, Vanilla JavaScript, TailwindCSS  
- **AI**: OpenAI API
- **Other**: web3.py for Ethereum RPC connections  

## ⚡️ Project Architecture
1. **Background feeder** thread continuously fetches swaps from The Graph.  
2. Transactions are scored and appended to an in-memory store.  
3. Risk scores are computed using multiple heuristics.  
4. Flask serves REST endpoints for `/api/stream`, `/api/tx/<id>`, and `/api/sponsor-check`.  
5. Frontend polls the API every 2s and updates the live dashboard.  
6. AI summaries are generated on demand if an API key is present.  

## 📊 Example Dashboard
- View live transactions across Ethereum (real) and Polygon/Arbitrum/Avalanche (mock).  
- See risk ratings and flag status in real time. You can also filter transactions based on risk score, chain, or address.
- Click a row to inspect full transaction metadata and AI insights.  




