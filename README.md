# **IMPORTANT LEGAL DISCLAIMER AND RISK DISCLOSURE**

**1. GENERAL DISCLAIMER** 

The content, signals, data, and software provided herein (collectively, the "System") are for informational, educational, and research purposes only. The Creator and associated contributors (the "Developers") are not registered financial advisors, broker-dealers, or investment professionals. Nothing in this System constitutes personalized investment advice, a recommendation to buy, sell, or hold any security, or a solicitation of any offer to buy or sell any financial instruments.

**2. NO FIDUCIARY DUTY** 

You acknowledge that no fiduciary relationship exists between you and the Developers. All investment decisions are made solely by you at your own discretion and risk. You agree to consult with a qualified, licensed financial advisor or tax professional before making any financial decisions based on the outputs of this System.

**3. RISK OF LOSS AND LEVERAGE**

Trading in financial markets, particularly with the use of leveraged instruments (such as ProShares Ultra S&P500 - SSO), involves a high degree of risk and may not be suitable for all investors.

* Leverage Risk: This System utilizes 2x leveraged ETFs. While leverage can magnify gains, it also magnifies losses. A relatively small market movement can have a disproportionately large impact on the funds deposited. You may sustain a total loss of your initial invested capital and, in certain cases, typically involving margin, liable for amounts exceeding your initial deposit.

* Volatility Decay: Leveraged ETFs are designed for short-term trading. Holding them for extended periods in volatile markets can result in significant value decay, even if the underlying index (S&P 500) remains flat.

**4. HYPOTHETICAL PERFORMANCE DISCLOSURE** 

The results presented in this dashboard, including backtests and historical simulations, are hypothetical. Hypothetical or simulated performance results have certain inherent limitations. Unlike an actual performance record, simulated results do not represent actual trading. Also, since the trades have not actually been executed, the results may have under- or over-compensated for the impact, if any, of certain market factors, such as lack of liquidity. No representation is being made that any account will or is likely to achieve profits or losses similar to those shown. Past performance is not necessarily indicative of future results.

**5. ALGORITHMIC AND TECHNICAL RISKS** 

The System relies on artificial intelligence, machine learning models (specifically Gradient Boosting Classifiers), and third-party data feeds (Yahoo Finance).

The AI model is trained on historical data and may fail to predict future market regimes, "Black Swan" events, or structural market changes.

The System relies on external API data which may be delayed, inaccurate, or unavailable. The Developers do not guarantee the accuracy, timeliness, or completeness of the data.

The code is provided "AS IS" without warranty of any kind. There may be errors, bugs, or glitches in the logic that could result in incorrect signals or financial loss.

**6. LIMITATION OF LIABILITY** 

IN NO EVENT SHALL THE DEVELOPERS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SYSTEM, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

=============================================================================
# **Dynamic Regime-Switching Algorithm for Leveraged ETFs**

Author: [PEHC] 

Date: January 2026 

Tech Stack: Python, Scikit-Learn (Gradient Boosting), Streamlit, yfinance

**1. Executive Summary**

This project implements a Machine Learning based trading strategy designed to capture the upside of leveraged equity ETFs (ProShares Ultra S&P500 - SSO) while mitigating the inherent risks of volatility decay and drawdowns during bear markets.

The system utilizes a Gradient Boosting Classifier to predict short-term market regimes, reinforced by a Hard Logic Regime Filter (200-Day SMA) to override AI signals during structural market downturns. Backtesting on data from 2020–2026 demonstrates a significant risk-adjusted return advantage over a passive "Buy & Hold" strategy, specifically by minimizing max drawdowns during the 2020 and 2022 crashes.

**2. Investment Thesis**

Leveraged ETFs (LETFs) like SSO (2x S&P 500) offer amplified returns but suffer from Beta Decay (Volatility Drag). In choppy or declining markets, the daily rebalancing of LETFs causes the asset value to erode faster than the underlying index.

Hypothesis: A "Blind Buy & Hold" strategy on SSO is mathematically flawed due to decay.

Solution: A Regime-Switching Model is required to identify "Safe" (Low Vol) vs. "Dangerous" (High Vol) periods. The algorithm moves to 100% Cash (Risk-Free Rate) during dangerous regimes to preserve capital, re-entering leverage only when conditions favor momentum.

**3. Methodology & Architecture**
**A. Feature Engineering**

The model inputs technical and volatility derivatives rather than raw price, normalizing data for machine learning:

* Market Fear: VIX Index (Normalized) & Rolling Volatility (20-Day).

* Trend Extension: Distance from 200-Day SMA.

* Momentum: RSI (Relative Strength Index) to detect overbought/oversold conditions.

**B. The AI Core (Gradient Boosting)**

A Gradient Boosting Classifier was selected over Neural Networks (LSTM/Transformers) due to its robustness with tabular data and resistance to noise.

* Hyperparameters: Tuned for a "Sniper" approach (n_estimators=90, max_depth=3, learning_rate=0.025, ubsample=0.7, min_samples_leaf=60, and random_state=42).

* Class Balancing: Applied sample_weights = compute_sample_weight(class_weight='balanced', y=y_train) and model.fit(X_train, y_train, sample_weight=sample_weights) to penalize the model heavily for missing Sell signals, countering the dataset's inherent Bullish bias.

**C. The "Safety Valve" (Regime Filter)**

To prevent "AI Hallucinations" during black swan events, a hard-coded logic layer acts as a circuit breaker:

Rule 1 (Trend Filter): If SPY < 200-Day SMA, the system is forced to CASH, regardless of AI confidence.

Rule 2 (Hysteresis): To minimize transaction costs (churn), the model requires a confidence buffer (>55% to Buy, <45% to Sell). Signals in the "Gray Zone" result in holding the current position.

D. Overfitting Mitigation & Robustness

To prevent the model from memorizing historical noise ("looking back"), the following constraints were architected into the system:

* Walk-Forward Validation: Unlike standard cross-validation which shuffles time, this project utilized **Time Series Split** validation. This simulates real-world conditions by training only on past data and testing on future data, strictly preventing look-ahead bias.
  
* Tree Constraints: The model is limited to shallow learners (max_depth=3) and requires a high minimum sample size per leaf (min_samples_leaf=60). This prevents the algorithm from creating hyper-specific rules based on outlier days.

* Stochastic Gradient Boosting: A subsample=0.7 parameter was implemented, forcing the model to train on random subsets of data for each tree. This introduces randomness that penalizes variance and improves generalization.

**4. Performance Validation**

The strategy was validated using Walk-Forward Cross-Validation (Time Series Split) to prevent look-ahead bias.

Metric 1: Offense (Precision on Buy Signals): Achieved a Precision of ~66%, indicating a strong statistical edge when deploying leverage.

Metric 2: Net Profitability: The backtest includes a simulated 0.001% transaction cost per trade to reflect real-world slippage and commissions.


**5. Conclusion**

The algorithm successfully acts as a "Market Regime Detector." It does not attempt to predict exact daily price movements (which is stochastic) but rather identifies the underlying state of the market. This approach allows for the responsible use of 2x leverage by neutralizing the primary risk factor: prolonged exposure to bear markets.

=============================================================================
## Development Methodology 
This project was built using an AI-Accelerated Workflow. The core financial strategy—including the Regime-Switching logic, Hysteresis filters, and Volatility Decay mitigation—was conceptualized and architected by the author. Large Language Models (Gemini) were utilized to accelerate syntax generation and boilerplate implementation, allowing the focus to remain on quantitative logic, parameter tuning, and risk management validation.
