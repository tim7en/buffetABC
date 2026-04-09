# Analysis Variable Inventory

| Variable | Label | Role | Source | Input Frequency | Availability Treatment | Supervised | GMM | Cycle |
|---|---|---|---|---|---|---:|---:|---:|
| market_cap_to_gdp_anchor_level | Official market cap to GDP anchor | base reference series | World Bank / FRED market cap to GDP | annual input aligned to daily | lagged annual release, then forward-filled on trading days |  |  |  |
| wilshire_level | Wilshire total-market index | base reference series | Yahoo Wilshire total-market proxy + FRED GDP | daily | same-day daily market observation |  |  |  |
| nominal_gdp_level | Nominal GDP | base reference series | FRED GDP | quarterly input aligned to daily | lagged quarterly GDP release, then forward-filled on trading days |  |  |  |
| qqq_feedback_score | QQQ feedback score | composite model input | QQQ daily parquet | daily | same-day composite from lag-safe inputs | Y |  |  |
| external_shock_score | External shock score | composite model input | Composite from aligned market and macro features | daily composite | same-day composite from lag-safe inputs | Y | Y |  |
| latent_sentiment_index | Latent sentiment | composite model input | Composite from aligned market and macro features | daily composite | same-day composite from lag-safe inputs | Y | Y | Y |
| market_cap_to_gdp_anchor_252d_drift | Market Cap To Gdp Anchor 252D Drift | diagnostic feature | World Bank / FRED market cap to GDP | annual input aligned to daily | lagged annual release, then forward-filled on trading days |  |  |  |
| dxy_level | Dxy Level | diagnostic feature | Yahoo DXY | daily | same-day daily market observation |  |  |  |
| gold_level | Gold Level | diagnostic feature | Yahoo GC=F gold futures | daily | same-day daily market observation |  |  |  |
| qqq_close | Qqq Close | diagnostic feature | QQQ daily parquet | daily | same-day daily market observation |  |  |  |
| us2y_level | Us2Y Level | diagnostic feature | FRED Treasury yields | daily | same-day daily market observation |  |  |  |
| us30y_level | Us30Y Level | diagnostic feature | FRED Treasury yields | daily | same-day daily market observation |  |  |  |
| wti_level | Wti Level | diagnostic feature | FRED WTI | daily | same-day daily market observation |  |  |  |
| cape_rolling_z | Shiller CAPE rolling z-score | diagnostic feature | Multpl Shiller CAPE | monthly input aligned to daily | market-based monthly observation forward-filled on trading days |  |  | Y |
| cpi_mom_pct | Cpi Mom Pct | diagnostic feature | FRED CPI | monthly input aligned to daily | lagged monthly release, then forward-filled on trading days |  |  |  |
| buffett_indicator_proxy_63d_change | Buffett Indicator Proxy 63D Change | diagnostic feature | Yahoo Wilshire total-market proxy + FRED GDP | quarterly input aligned to daily | lagged quarterly GDP release, then forward-filled on trading days |  |  |  |
| buffett_indicator_proxy_rolling_z | Wilshire / GDP rolling z-score | diagnostic feature | Yahoo Wilshire total-market proxy + FRED GDP | quarterly input aligned to daily | lagged quarterly GDP release, then forward-filled on trading days |  |  | Y |
| curve_10y2y_level | Yield curve 10Y-2Y | model feature | FRED Treasury yields | daily | same-day daily market observation | Y | Y | Y |
| dxy_63d_return | US dollar 3-month return | model feature | Yahoo DXY | daily | same-day daily market observation | Y | Y |  |
| gold_63d_return | Gold 3-month return | model feature | Yahoo GC=F gold futures | daily | same-day daily market observation | Y | Y |  |
| hy_oas_63d_change_pp | High-yield spread 3-month change | model feature | FRED BAMLH0A0HYM2 | daily | same-day daily market observation | Y |  |  |
| hy_oas_level | High-yield spread | model feature | FRED BAMLH0A0HYM2 | daily | same-day daily market observation | Y | Y | Y |
| nfci_63d_change | Financial conditions 3-month change | model feature | FRED NFCI | daily | same-day daily market observation | Y |  |  |
| nfci_level | Financial conditions level | model feature | FRED NFCI | daily | same-day daily market observation | Y | Y | Y |
| qqq_21d_return | QQQ 1-month return | model feature | QQQ daily parquet | daily | same-day daily market observation | Y |  |  |
| qqq_63d_return | QQQ 3-month return | model feature | QQQ daily parquet | daily | same-day daily market observation | Y | Y |  |
| qqq_drawdown_252d | QQQ 1-year drawdown | model feature | QQQ daily parquet | daily | same-day daily market observation | Y |  | Y |
| qqq_realized_vol_21d | QQQ 1-month realized volatility | model feature | QQQ daily parquet | daily | same-day daily market observation | Y | Y |  |
| qqq_sma222 | QQQ 222-day trend level | model feature | QQQ daily parquet | daily | same-day daily market observation | Y | Y |  |
| qqq_sma65 | QQQ 65-day trend level | model feature | QQQ daily parquet | daily | same-day daily market observation | Y | Y |  |
| qqq_volume | QQQ volume | model feature | QQQ daily parquet | daily | same-day daily market observation | Y | Y |  |
| qqq_vs_sma200 | QQQ vs 200-day trend | model feature | QQQ daily parquet | daily | same-day daily market observation | Y | Y | Y |
| t10y3m_level | 10Y-3M curve | model feature | FRED T10Y3M | daily | same-day daily market observation | Y |  |  |
| us10y_63d_change_pp | 10Y yield 3-month change | model feature | FRED Treasury yields | daily | same-day daily market observation | Y | Y |  |
| us10y_level | 10Y Treasury yield level | model feature | FRED Treasury yields | daily | same-day daily market observation | Y |  |  |
| vix_21d_change | VIX 1-month change | model feature | FRED VIXCLS | daily | same-day daily market observation | Y |  |  |
| vix_level | VIX level | model feature | FRED VIXCLS | daily | same-day daily market observation | Y | Y | Y |
| wti_63d_return | Oil 3-month return | model feature | FRED WTI | daily | same-day daily market observation | Y | Y |  |
| cape_63d_change | Shiller CAPE 3-month change | model feature | Multpl Shiller CAPE | monthly input aligned to daily | market-based monthly observation forward-filled on trading days | Y |  |  |
| cape_level | Shiller CAPE | model feature | Multpl Shiller CAPE | monthly input aligned to daily | market-based monthly observation forward-filled on trading days | Y | Y |  |
| cpi_yoy_3m_change_pp | Inflation 3-month change | model feature | FRED CPI | monthly input aligned to daily | lagged monthly release, then forward-filled on trading days | Y |  | Y |
| cpi_yoy_pct | Inflation YoY | model feature | FRED CPI | monthly input aligned to daily | lagged monthly release, then forward-filled on trading days | Y | Y | Y |
| unemployment_6m_change_pp | Unemployment 6-month change | model feature | FRED UNRATE | monthly input aligned to daily | lagged monthly release, then forward-filled on trading days | Y |  | Y |
| unemployment_rate_pct | Unemployment rate | model feature | FRED UNRATE | monthly input aligned to daily | lagged monthly release, then forward-filled on trading days | Y | Y | Y |
| buffett_indicator_proxy_252d_drift | Wilshire / GDP 1-year drift | model feature | Yahoo Wilshire total-market proxy + FRED GDP | quarterly input aligned to daily | lagged quarterly GDP release, then forward-filled on trading days | Y | Y |  |
| buffett_indicator_proxy_level | Wilshire / GDP valuation proxy | model feature | Yahoo Wilshire total-market proxy + FRED GDP | quarterly input aligned to daily | lagged quarterly GDP release, then forward-filled on trading days | Y | Y |  |