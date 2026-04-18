# **IMPORTANT LEGAL DISCLAIMER AND RISK DISCLOSURE**

**1. GENERAL DISCLAIMER**

The content, signals, data, and software provided herein (collectively, the "System") are for informational, educational, and research purposes only. The Creator and associated contributors (the "Authors") are not registered financial advisors, broker-dealers, or investment professionals. Nothing in this System constitutes personalized investment advice, a recommendation to buy, sell, or hold any security or cryptocurrency, or a solicitation of any offer to buy or sell any financial instruments.

**2. NO FIDUCIARY DUTY**

You acknowledge that no fiduciary relationship exists between you and the Authors. All investment decisions are made solely by you at your own discretion and risk. You agree to consult with a qualified, licensed financial advisor or tax professional before making any financial decisions based on the outputs of this System.

**3. RISK OF LOSS (CRYPTOCURRENCY)**

Trading in cryptocurrency markets, particularly **Bitcoin (BTC)**, involves a high degree of risk and may not be suitable for all investors.

* Volatility Risk: Bitcoin is extremely volatile and can experience 50%+ drawdowns within weeks. You may sustain a total loss of your initial invested capital.

* Regulatory Risk: Cryptocurrency regulations vary by jurisdiction and may change without notice, potentially affecting your ability to trade, hold, or withdraw BTC.

* Custodial Risk: Digital assets may be lost due to exchange failures, hacking incidents, or loss of private keys.

* Market Manipulation: Cryptocurrency markets are less regulated than traditional securities markets and may be subject to manipulation, wash trading, and other fraudulent activities.

**4. HYPOTHETICAL PERFORMANCE DISCLOSURE**

The results presented in this dashboard, including backtests and historical simulations, are hypothetical. Hypothetical or simulated performance results have certain inherent limitations. Unlike an actual performance record, simulated results do not represent actual trading. Also, since the trades have not actually been executed, the results may have under- or over-compensated for the impact, if any, of certain market factors, such as lack of liquidity. No representation is being made that any account will or is likely to achieve profits or losses similar to those shown. Past performance is not necessarily indicative of future results.

**5. ALGORITHMIC AND TECHNICAL RISKS**

The System relies on artificial intelligence, machine learning models (specifically Gradient Boosting Classifiers), and third-party data feeds (Yahoo Finance).

The AI model is trained on historical data and may fail to predict future market regimes, "Black Swan" events, or structural market changes.

The System relies on external API data which may be delayed, inaccurate, or unavailable. The Authors do not guarantee the accuracy, timeliness, or completeness of the data.

The code is provided "AS IS" without warranty of any kind. There may be errors, bugs, or glitches in the logic that could result in incorrect signals or financial loss.

**6. LIMITATION OF LIABILITY**

**IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SYSTEM, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.**

=============================================================================
# **Dynamic Regime-Switching Algorithm for Bitcoin (BTC)**

Author: [PARVAUX]

Date: March 2026

Tech Stack: Python, Scikit-Learn (Gradient Boosting), Streamlit, yfinance

**1. Executive Summary**

This project implements a Machine Learning based trading strategy designed to capture the upside of **Bitcoin (BTC-USD)** while mitigating the inherent risks of extreme drawdowns during bear markets.

The system utilizes a Gradient Boosting Classifier to predict short-term market regimes, reinforced by a Hard Logic Regime Filter (200-Day SMA) to override AI signals during structural market downturns. Backtesting on data from 2021-2026 demonstrates a significant risk-adjusted return advantage over a passive "Buy & Hold" strategy, specifically by minimizing max drawdowns during the 2022 crypto crash.

**2. Investment Thesis**

Bitcoin offers exceptional upside potential but suffers from extreme volatility, with historical drawdowns of 50-80% during bear markets. A passive "Buy & Hold" strategy exposes the investor to the full brunt of these drawdowns.

Hypothesis: A "Blind Buy & Hold" strategy on BTC is psychologically and financially challenging due to extreme volatility.

Solution: A Regime-Switching Model is required to identify "Safe" (Bull) vs. "Dangerous" (Bear) periods. The algorithm moves to 100% Cash (Risk-Free Rate) during dangerous regimes to preserve capital, re-entering BTC only when conditions favor momentum.

**3. Methodology & Architecture**
**A. Feature Engineering**

The model inputs 8 technical and volatility derivatives rather than raw price, normalizing data for machine learning:

* Market Fear: VIX Index (Normalized), VIX 5-Day Momentum (early warning of volatility shifts), & Rolling Volatility (20-Day).

* Trend Extension: Distance from 200-Day SMA, 50/200 SMA Cross Ratio (golden/death cross signal).

* Momentum: RSI (14-Day Relative Strength Index) and 20-Day Price Momentum to detect overbought/oversold conditions and trend strength.

* Volume: Volume Ratio (current volume vs 20-day average) to detect unusual trading activity.

**B. The AI Core (Gradient Boosting)**

A Gradient Boosting Classifier was selected over Neural Networks (LSTM/Transformers) due to its robustness with tabular data and resistance to noise.

* Decision Stumps: The model uses max_depth=1 (decision stumps), meaning each tree can only make a single split. This prevents the model from memorizing multi-feature noise patterns and forces it to learn only the strongest individual signals. This was a critical design choice — deeper trees (max_depth=3) were found to overfit on BTC's noisy data.

* Hyperparameters: Tuned for robustness (n_estimators=60, max_depth=1, learning_rate=0.03, subsample=0.5, min_samples_leaf=120, random_state=42).

* Class Balancing: Applied sample_weights = compute_sample_weight(class_weight='balanced', y=y_train) to penalize the model heavily for missing Sell signals, countering the dataset's inherent Bullish bias.

* Expanding Window Retraining: The model retrains every 60 days on all available data up to 6 months before the prediction date. This allows the model to adapt to evolving market conditions while maintaining a gap to prevent label leakage from the 10-day forward return target.

**C. The "Safety Valve" (Regime Filter)**

To prevent "AI Hallucinations" during black swan events, a hard-coded logic layer acts as a circuit breaker:

Rule 1 (Trend Filter): If BTC < 200-Day SMA, the system is forced to CASH, regardless of AI confidence.

Rule 2 (Hysteresis): To minimize transaction costs (churn), the model requires a confidence buffer (>51% to Buy, <46% to Sell). Signals in the "Gray Zone" result in holding the current position.

D. Overfitting Mitigation & Robustness

To prevent the model from memorizing historical noise ("looking back"), the following constraints were architected into the system:

* Walk-Forward Validation: Unlike standard cross-validation which shuffles time, this project utilized **Time Series Split** validation (5 folds). This simulates real-world conditions by training only on past data and testing on future data, strictly preventing look-ahead bias.

* Decision Stumps (max_depth=1): The single most impactful overfitting mitigation. Each tree makes only one binary split, forcing the ensemble to learn simple, robust signals rather than complex noise patterns. Combined with min_samples_leaf=120, this prevents the algorithm from creating hyper-specific rules based on outlier days. The training-to-CV gap was reduced from 16.8% (Overfitting) to 9.4% (Robust) by switching from max_depth=3 to max_depth=1.

* Stochastic Gradient Boosting: A subsample=0.5 parameter was implemented, forcing the model to train on random 50% subsets of data for each tree. This introduces randomness that penalizes variance and improves generalization.

* Periodic Retraining: The backtest simulation retrains the model every 60 days using an expanding window, mirroring the live system's behavior. This prevents look-ahead bias in the performance results and demonstrates the strategy's robustness across different training periods.

**4. Performance Validation**

The strategy was validated using Walk-Forward Cross-Validation (Time Series Split) to prevent look-ahead bias.

Backtesting on 2021–2026 data (including the 2022 crash and 2023–2025 recovery) with 0.1% transaction costs per trade:

| Metric | AI Strategy | Passive Buy & Hold |
|---|---|---|
| Total Return | +215% | +144% |
| Max Drawdown | -22% | -77% |
| Sharpe Ratio | 1.03 | 0.58 |
| Total Trades | 46 | N/A |

Model Health (Walk-Forward CV):

| Metric | Value |
|---|---|
| Training Accuracy | 55.5% |
| Walk-Forward CV | 46.2% |
| Gap | 9.4% |
| Status | Robust |


**5. Conclusion**

The algorithm successfully acts as a "Crypto Regime Detector." It does not attempt to predict exact daily price movements (which is stochastic) but rather identifies the underlying state of the market. The strategy outperforms passive Buy & Hold by +71 percentage points while reducing maximum drawdown from -77% to -22% — a 3.5x improvement in downside protection. The Sharpe ratio of 1.03 demonstrates strong risk-adjusted returns for a crypto strategy.

=============================================================================
# Development Methodology
**The core financial strategy—including the Regime-Switching logic, Hysteresis filters, and drawdown mitigation—was conceptualized and architected by the author.**

This project was built using an AI-Accelerated Workflow.

Large Language Models (Gemini) were utilized to accelerate syntax generation and boilerplate implementation, allowing the focus to remain on quantitative logic, parameter tuning, and risk management validation.
