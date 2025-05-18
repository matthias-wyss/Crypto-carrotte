# 🪙 Crypto Carry Trade Project

**EPFL - Financial Engineering MA2 - FIN-413**  
**Spring Semester 2025**  
**Professor:** Dimitrios Karyampas  
**Group 7**  
**Authors:**  
- Matthias Wyss (SCIPER 329884)  
- Loris Tran (SCIPER 341214)  
- Massimo Berardi (SCIPER 345943)  
- Vincent Ventura (SCIPER 302810)  
- Alexandre Huou (SCIPER 342227)

---

## 📘 Project Overview

This project investigates the implementation and performance of various **delta-neutral crypto carry trade strategies** using perpetual futures and DeFi innovations like **staking** and **Pendle Finance**.

We analyze the **historical profitability**, **market resilience**, and **risk characteristics** of the strategies across different market regimes, including the 2021 bull market, the Luna collapse, the FTX crisis and the ETF bullrun.

<br>[📄 Read the full Project Report here](Project_report.pdf)<br>

---

## 📈 Strategies Implemented

### 1. Classical Carry Trade
- Long spot (BTC or ETH), short perpetuals  
- Captures funding rate as passive yield  
- Delta-neutral, relies on positive funding rate

### 2. Staking-Enhanced Carry (ETH + Lido)
- Stake ETH via **Lido** to earn staking APR  
- Hedge price exposure via shorting perpetuals  
- Combines funding rate + staking rewards

### 3. USD-Settled Carry via Pendle (PT-stETH)
- Buy **Pendle PT-stETH** (discounted staked ETH)  
- Short ETH perpetuals to neutralize price exposure  
- Realize fixed yield in USD terms at maturity  
- Only fully delta-neutral at maturity

---

## 🧪 Backtesting & Analysis

We conducted backtests from **2019 to 2024**, analyzing performance over different market regimes:

- 📈 **Bull Market (2021)**
- 💥 **Luna Collapse**
- 🧨 **FTX Collapse**
- 🚀 **ETF Bull Market (2024)**

Metrics include:
- Cumulative funding returns  
- Annualized funding rates 
- Funding rate distributions  
- Strategy resilience to market shocks
- Underlying asset drawdown
- Strategy drawdown

All data was collected from **Binance**, **CoinGlass API** and **Dune Analytics**.

---

## 📂 Repository Structure

```
├── data/ # Raw and processed datasets, includes some plots for Question 2
├── src/ # Python scripts implementing the crypto carry strategies
├── plots/ # Output plots for Questions 3 and 4
├── q2.ipynb # Notebook for plots and analysis related to Question 2
├── q3.ipynb # Main notebook generating data for Questions 3 and 4
├── q3_bis.ipynb # Supplementary visualizations for Question 3
├── q3_staking.ipynb # ETH staking analysis (not included in the final report)
├── q4.ipynb # Notebook for visualizations and insights for Question 4
├── Project_report.pdf # Final project report (PDF)
└── README.md # Project overview and structure
```

---

## 🔍 Key Findings

- **Funding rates for BTC and ETH were positive over 85% of the time**, especially during bull markets.
- **ETH carry strategies enhanced with staking** outperform pure funding-based strategies by approximately **3.9% on average**.
- **Pendle-based carry trades** provide **fixed yield in USD**, offering an attractive option for **risk-averse investors**.
- **Carry strategies remain resilient during market stress**, but **risk management is crucial**, particularly regarding **liquidation risk** and **funding rate volatility**.
- Simulations show that carry strategies with **dynamic leverage adjustment** reduce drawdowns while maintaining competitive returns.

---

## 🛠️ Dependencies

- Python ≥ 3.9  
- `pandas`, `numpy`, `matplotlib`, `requests`  
- Jupyter for notebooks  
- Dune API key
- CoinGlass API key

To install dependencies:

```bash
pip install -r requirements.txt
