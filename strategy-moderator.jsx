import { useState, useEffect } from "react";

const TABS = [
  { id: "overview", label: "Weekly Review" },
  { id: "scoring", label: "Asset Scoring" },
  { id: "rotation", label: "Rotation Engine" },
  { id: "macro", label: "Macro Overlay" },
  { id: "sensitivity", label: "Sensitivity" },
  { id: "thresholds", label: "Thresholds & Rules" },
  { id: "failures", label: "Failure Modes" },
];

// ── Utility ──────────────────────────────────────────────
const cn = (...classes) => classes.filter(Boolean).join(" ");
const Badge = ({ children, color = "neutral" }) => {
  const colors = {
    green: "bg-emerald-900/60 text-emerald-300 border-emerald-700/50",
    yellow: "bg-amber-900/60 text-amber-300 border-amber-700/50",
    red: "bg-red-900/60 text-red-300 border-red-700/50",
    blue: "bg-sky-900/60 text-sky-300 border-sky-700/50",
    neutral: "bg-zinc-800/60 text-zinc-300 border-zinc-700/50",
    purple: "bg-violet-900/60 text-violet-300 border-violet-700/50",
  };
  return (
    <span className={cn("px-2 py-0.5 rounded text-xs border font-medium", colors[color])}>
      {children}
    </span>
  );
};

const Card = ({ title, subtitle, children, accent }) => (
  <div
    className="rounded-lg border border-zinc-800/80 bg-zinc-900/50 backdrop-blur-sm overflow-hidden"
    style={accent ? { borderTopColor: accent, borderTopWidth: 2 } : {}}
  >
    {(title || subtitle) && (
      <div className="px-5 pt-4 pb-2">
        {title && <h3 className="text-sm font-semibold text-zinc-100 tracking-wide">{title}</h3>}
        {subtitle && <p className="text-xs text-zinc-500 mt-0.5">{subtitle}</p>}
      </div>
    )}
    <div className="px-5 pb-4">{children}</div>
  </div>
);

const Metric = ({ label, value, sub, status }) => {
  const statusColor = { good: "text-emerald-400", warn: "text-amber-400", bad: "text-red-400", neutral: "text-zinc-300" };
  return (
    <div className="flex flex-col">
      <span className="text-[10px] uppercase tracking-widest text-zinc-500 mb-1">{label}</span>
      <span className={cn("text-lg font-bold tabular-nums", statusColor[status || "neutral"])}>{value}</span>
      {sub && <span className="text-[10px] text-zinc-600 mt-0.5">{sub}</span>}
    </div>
  );
};

const Section = ({ number, title, children }) => (
  <div className="mb-6">
    <div className="flex items-center gap-2 mb-3">
      <span className="w-6 h-6 rounded bg-zinc-800 text-zinc-400 text-xs font-bold flex items-center justify-center border border-zinc-700/50">
        {number}
      </span>
      <h4 className="text-sm font-semibold text-zinc-200">{title}</h4>
    </div>
    {children}
  </div>
);

const RuleBlock = ({ title, rules, type = "rule" }) => {
  const icons = { rule: "◆", check: "✓", warn: "⚠", info: "●" };
  const colors = { rule: "text-sky-400", check: "text-emerald-400", warn: "text-amber-400", info: "text-violet-400" };
  return (
    <div className="mb-4">
      {title && <p className="text-xs font-semibold text-zinc-300 mb-2 uppercase tracking-wider">{title}</p>}
      <div className="space-y-1.5">
        {rules.map((r, i) => (
          <div key={i} className="flex items-start gap-2 text-xs text-zinc-400 leading-relaxed">
            <span className={cn("mt-0.5 text-[10px]", colors[type])}>{icons[type]}</span>
            <span>{r}</span>
          </div>
        ))}
      </div>
    </div>
  );
};

const Table = ({ headers, rows, compact }) => (
  <div className="overflow-x-auto">
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-zinc-800">
          {headers.map((h, i) => (
            <th key={i} className={cn("py-2 text-left text-[10px] uppercase tracking-widest text-zinc-500 font-medium", i > 0 && "text-right", compact && "px-2")}>
              {h}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row, ri) => (
          <tr key={ri} className="border-b border-zinc-800/50 hover:bg-zinc-800/30 transition-colors">
            {row.map((cell, ci) => (
              <td key={ci} className={cn("py-2 text-zinc-300", ci > 0 && "text-right tabular-nums", compact && "px-2")}>
                {cell}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

// ── TAB CONTENT COMPONENTS ───────────────────────────────

function WeeklyReview() {
  const [weekNum, setWeekNum] = useState(4);
  return (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-bold text-zinc-100">Weekly Strategy Review</h2>
          <p className="text-xs text-zinc-500 mt-0.5">Systematic checklist — execute every Friday close or Monday pre-open</p>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-zinc-500">Week</span>
          <select value={weekNum} onChange={e => setWeekNum(+e.target.value)} className="bg-zinc-800 border border-zinc-700 rounded px-2 py-1 text-xs text-zinc-300">
            {[1,2,3,4,5,6,7,8].map(w => <option key={w} value={w}>W{w}</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3">
        <Card><Metric label="Strategy Age" value={`${weekNum} wk`} sub="Min 8 wk for param changes" status="neutral" /></Card>
        <Card><Metric label="Health Score" value="—" sub="Populate after scoring" status="neutral" /></Card>
        <Card><Metric label="Live vs Backtest" value="—" sub="Drift %" status="neutral" /></Card>
        <Card><Metric label="Next Review" value="Weekly" sub="Full rotation: monthly" status="neutral" /></Card>
      </div>

      <Card title="WEEKLY REVIEW PROTOCOL" subtitle="7-step process — do NOT skip steps or act on partial data">
        <Section number="1" title="Execution Audit">
          <RuleBlock type="check" rules={[
            "Verify all fills matched signal prices within slippage budget (0.05% crypto, 0.02% FX, 0.01% futures)",
            "Confirm no missed entries or exits — log every deviation with reason code",
            "Check that position sizes matched ATR-based calc at time of signal",
            "Record actual costs vs assumed costs — flag if actual > 1.3× assumed",
          ]} />
        </Section>

        <Section number="2" title="Live vs Backtest Drift">
          <RuleBlock type="check" rules={[
            "Calculate cumulative live return vs cumulative backtest return for same period",
            "Drift = |live_return − backtest_return| / |backtest_return|",
            "GREEN: drift < 15%  |  YELLOW: 15–30%  |  RED: > 30%",
            "If RED for 2 consecutive weeks → halt new entries, investigate data/execution issues",
            "Common causes: look-ahead bias in backtest, fill assumptions, fee model errors",
          ]} />
        </Section>

        <Section number="3" title="Per-Asset Signal Quality">
          <RuleBlock type="check" rules={[
            "Count wins/losses this week per asset — compare to backtest win rate ± 2 standard deviations",
            "Flag any asset with 0 trades in 3+ weeks if backtest expects ≥ 2/month",
            "Check if Donchian channel width (high−low of N bars) is compressing or expanding",
            "If channel width < 30th percentile of last 252 bars → asset entering chop, raise caution flag",
          ]} />
        </Section>

        <Section number="4" title="Risk & Drawdown Check">
          <RuleBlock type="check" rules={[
            "Current portfolio drawdown from equity peak",
            "GREEN: DD < 50% of max backtest DD  |  YELLOW: 50–80%  |  RED: > 80%",
            "If RED → reduce position sizes by 50% until DD recovers to YELLOW",
            "If live DD exceeds backtest max DD → halt strategy, full parameter review required",
            "Check correlation of concurrent positions — flag if > 0.7 pairwise",
          ]} />
        </Section>

        <Section number="5" title="Macro Regime Tag (see Macro tab)">
          <RuleBlock type="check" rules={[
            "Classify current week: TRENDING / CHOPPY / HIGH-VOL / LOW-VOL",
            "Cross-check: is the strategy performing as expected for this regime?",
            "If regime shifted this week → note it, but do NOT change parameters yet",
            "Regime shifts only trigger action after 2 consecutive weeks of confirmed shift",
          ]} />
        </Section>

        <Section number="6" title="Decision Gate">
          <RuleBlock type="warn" rules={[
            "Weeks 1–4: OBSERVATION ONLY — no parameter changes, no asset removals",
            "Weeks 5–8: May reduce weight on assets scoring < 40th percentile composite",
            "Weeks 9+: Full rotation eligible — keep/reduce/remove/add per Rotation Engine",
            "NEVER change Donchian length or EMA filters based on < 8 weeks live data",
            "Policy action shifts: log them, tag affected assets, but wait for price confirmation before acting",
          ]} />
        </Section>

        <Section number="7" title="Weekly Log Entry">
          <RuleBlock type="info" rules={[
            "Record: date, equity, DD%, # trades, # assets active, regime tag, drift %, any flags",
            "Record any discretionary overrides with justification — review monthly for pattern",
            "If > 2 overrides per month → tighten rules or acknowledge the strategy doesn't match your conviction",
          ]} />
        </Section>
      </Card>

      <Card title="DECISION TIMELINE" accent="#3b82f6">
        <div className="space-y-2 text-xs text-zinc-400">
          <div className="flex gap-3 items-start">
            <Badge color="blue">W1–W4</Badge>
            <span>Observe only. Build the log. No changes. Accept that variance ≠ broken strategy.</span>
          </div>
          <div className="flex gap-3 items-start">
            <Badge color="yellow">W5–W8</Badge>
            <span>First scoring pass. Flag underperformers. May reduce weights (not remove). Begin walk-forward comparison.</span>
          </div>
          <div className="flex gap-3 items-start">
            <Badge color="green">W9+</Badge>
            <span>Full rotation cycle. Monthly scoring. Add/remove eligible. Parameter sensitivity check quarterly only.</span>
          </div>
          <div className="flex gap-3 items-start">
            <Badge color="purple">Quarterly</Badge>
            <span>Full backtest refresh. Sensitivity analysis. Macro overlay review. Strategy health audit.</span>
          </div>
        </div>
      </Card>
    </div>
  );
}

function AssetScoring() {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-zinc-100">Composite Asset Scoring</h2>
        <p className="text-xs text-zinc-500 mt-0.5">Score each asset 0–100 using four pillars — only out-of-sample metrics count</p>
      </div>

      <Card title="SCORING PILLARS" subtitle="Each pillar 0–25 points → composite 0–100">
        <div className="grid grid-cols-2 gap-4 mb-4">
          {[
            { name: "Risk-Adjusted Return", weight: "25 pts", metrics: "OOS Sharpe, Sortino, MAR/Calmar, net return after costs", color: "#10b981" },
            { name: "Stability", weight: "25 pts", metrics: "Rolling 13-week Sharpe std dev, DD recovery time consistency, win-rate stability", color: "#3b82f6" },
            { name: "Diversification", weight: "25 pts", metrics: "Avg correlation of returns with other held assets, marginal risk contribution", color: "#a855f7" },
            { name: "Implementation", weight: "25 pts", metrics: "Turnover cost ratio, slippage impact, liquidity score, % time in market", color: "#f59e0b" },
          ].map((p, i) => (
            <div key={i} className="rounded border border-zinc-800 p-3" style={{ borderLeftColor: p.color, borderLeftWidth: 3 }}>
              <div className="flex justify-between items-center mb-1">
                <span className="text-xs font-semibold text-zinc-200">{p.name}</span>
                <Badge>{p.weight}</Badge>
              </div>
              <p className="text-[11px] text-zinc-500 leading-relaxed">{p.metrics}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card title="PER-ASSET METRICS (calculate for each)" subtitle="All metrics must use walk-forward / rolling OOS windows only">
        <Table
          headers={["Metric", "Calculation", "Good", "Warning", "Fail"]}
          rows={[
            ["Net Return (OOS)", "Cumulative return after all costs, OOS only", "> 0", "−5% to 0%", "< −5%"],
            ["Sharpe Ratio", "Mean excess return / σ (annualized, OOS)", "> 0.8", "0.3–0.8", "< 0.3"],
            ["Sortino Ratio", "Mean excess / downside σ", "> 1.0", "0.5–1.0", "< 0.5"],
            ["MAR / Calmar", "CAGR / Max DD", "> 0.5", "0.2–0.5", "< 0.2"],
            ["Max Drawdown", "Peak-to-trough max", "< 15%", "15–25%", "> 25%"],
            ["Expectancy/Trade", "(Win% × AvgWin) − (Loss% × AvgLoss)", "> 0.3R", "0–0.3R", "< 0"],
            ["Turnover (annual)", "Avg # round trips / year", "< 50", "50–120", "> 120"],
            ["Cost Sensitivity", "Return change if costs +50%", "< 20% reduction", "20–40%", "> 40%"],
            ["% Time in Market", "Bars with position / total bars", "30–70%", "< 30 or > 70%", "< 15 or > 85%"],
          ]}
        />
      </Card>

      <Card title="REGIME PERFORMANCE MATRIX" subtitle="Score each asset's strategy across 4 regimes separately">
        <Table
          headers={["Regime", "Detection", "Expected Behavior", "Score If Matches"]}
          rows={[
            ["Trending", "ADX > 25 or Donchian expanding for 10+ bars", "Positive expectancy, higher win rate on breakouts", "+6 pts"],
            ["Choppy / Mean-rev", "ADX < 20 and price crossing mid-channel frequently", "Reduced activity or flat — acceptable. Losses expected.", "+3 if losses controlled"],
            ["High Volatility", "ATR > 75th percentile (252-bar lookback)", "Wider stops, fewer trades, larger R-multiples", "+5 if R-mult > 1.5"],
            ["Low Volatility", "ATR < 25th percentile", "May skip trades (channel too narrow). Acceptable.", "+3 if no false breakout losses"],
          ]}
        />
        <div className="mt-3 p-2 rounded bg-zinc-800/50 text-[11px] text-zinc-500">
          <strong className="text-zinc-400">Critical rule:</strong> Do NOT penalize an asset because its price is declining.
          Penalize only if the <em>strategy's edge on that asset</em> is deteriorating out-of-sample.
          A trending short signal on a declining asset is valuable, not a reason to remove it (if long/short).
        </div>
      </Card>

      <Card title="COMPOSITE SCORE → TIER" accent="#10b981">
        <div className="space-y-2">
          {[
            { tier: "A", range: "75–100", action: "Full weight. Priority allocation.", color: "green" },
            { tier: "B", range: "55–74", action: "Standard weight. Monitor for improvement or decay.", color: "blue" },
            { tier: "C", range: "40–54", action: "Reduced weight (50%). On watch list.", color: "yellow" },
            { tier: "D", range: "25–39", action: "Minimal weight (25%) or remove if diversification value < threshold.", color: "yellow" },
            { tier: "F", range: "0–24", action: "Remove from live universe. Re-evaluate quarterly.", color: "red" },
          ].map((t, i) => (
            <div key={i} className="flex items-center gap-3 text-xs">
              <Badge color={t.color}>Tier {t.tier}</Badge>
              <span className="text-zinc-500 w-16 tabular-nums">{t.range}</span>
              <span className="text-zinc-400">{t.action}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function RotationEngine() {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-zinc-100">Rotation Decision Engine</h2>
        <p className="text-xs text-zinc-500 mt-0.5">Monthly full rotation — weekly only flags and weight adjustments</p>
      </div>

      <Card title="ROTATION DECISION TREE" subtitle="Follow top-to-bottom for each asset at monthly review">
        <div className="space-y-3">
          {[
            {
              step: "1", q: "Is the OOS composite score ≥ 75?",
              yes: "→ KEEP at full weight",
              no: "→ Continue to step 2",
            },
            {
              step: "2", q: "Is the OOS composite score 55–74 AND rolling 13-wk Sharpe stable (σ < 0.3)?",
              yes: "→ KEEP at standard weight",
              no: "→ Continue to step 3",
            },
            {
              step: "3", q: "Is the OOS composite score 40–54?",
              yes: "→ REDUCE to 50% weight. Set 4-week probation. If still < 55 after probation → step 4",
              no: "→ Continue to step 4",
            },
            {
              step: "4", q: "Is the OOS composite score < 40 AND diversification benefit (correlation < 0.3 with portfolio) exists?",
              yes: "→ REDUCE to 25% weight. Keep for diversification. Re-score next month.",
              no: "→ Continue to step 5",
            },
            {
              step: "5", q: "Is the OOS composite score < 40 AND diversification benefit absent (corr ≥ 0.3)?",
              yes: "→ REMOVE from live universe",
              no: "→ Edge case — review manually",
            },
          ].map((s, i) => (
            <div key={i} className="rounded border border-zinc-800 p-3">
              <div className="flex items-start gap-2 mb-2">
                <span className="w-5 h-5 rounded bg-zinc-800 text-zinc-400 text-[10px] font-bold flex items-center justify-center shrink-0 mt-0.5">{s.step}</span>
                <span className="text-xs text-zinc-200 font-medium">{s.q}</span>
              </div>
              <div className="ml-7 space-y-1 text-[11px]">
                <div className="text-emerald-400/80">{s.yes}</div>
                <div className="text-zinc-500">{s.no}</div>
              </div>
            </div>
          ))}
        </div>
      </Card>

      <Card title="ADD DECISION TREE" subtitle="For new candidate assets">
        <RuleBlock type="rule" rules={[
          "Step 1: Run full walk-forward backtest with same Donchian/EMA parameters as existing universe",
          "Step 2: OOS composite score must be ≥ 55 (Tier B minimum) to be considered",
          "Step 3: Pairwise correlation with every existing held asset must be < 0.5 (at least one < 0.3)",
          "Step 4: Cost-adjusted expectancy must be > 0 in at least 3 of 4 regimes",
          "Step 5: Paper-trade for minimum 4 weeks (or use last 4 weeks OOS segment) before live allocation",
          "Step 6: Initial weight = 50% of target weight → scale to full over 4 weeks if tracking backtest",
        ]} />
      </Card>

      <Card title="WEIGHT CALCULATION" accent="#a855f7">
        <div className="space-y-3 text-xs text-zinc-400">
          <div className="p-3 rounded bg-zinc-800/50 font-mono text-[11px] leading-relaxed">
            <div className="text-zinc-500 mb-1"># Base weight by inverse volatility</div>
            <div>w_i = (1 / σ_i) / Σ(1 / σ_j)  for all j in universe</div>
            <div className="text-zinc-500 mt-2 mb-1"># Adjust by composite score</div>
            <div>w_adj_i = w_i × (score_i / avg_score)</div>
            <div className="text-zinc-500 mt-2 mb-1"># Apply constraints</div>
            <div>w_final_i = min(w_adj_i, max_weight_per_asset)</div>
            <div>Σ w_final = 1.0  (re-normalize after constraints)</div>
          </div>
          <RuleBlock type="warn" title="CONSTRAINTS" rules={[
            "Max weight per asset: 25% (or 1/N if universe < 4 assets)",
            "Max correlated cluster: 40% (assets with pairwise corr > 0.6)",
            "Min weight to stay live: 5% — below this, remove or don't bother",
            "Cash buffer: always keep 5–10% in reserve for new entries / margin",
          ]} />
        </div>
      </Card>

      <Card title="CRITICAL SAFEGUARD" accent="#ef4444">
        <div className="p-3 rounded bg-red-950/30 border border-red-900/50 text-xs text-red-300/80 leading-relaxed space-y-2">
          <p><strong className="text-red-300">Do not remove assets just because recent price performance was weak.</strong></p>
          <p>A trend-following strategy profits from trends. If a formerly trending asset enters chop, the strategy should naturally reduce exposure (fewer signals, tighter stops). That is the system working correctly.</p>
          <p>Remove only when: (1) OOS edge metrics (Sharpe, expectancy, Calmar) have decayed below thresholds for 8+ weeks, AND (2) the asset no longer provides diversification value to the portfolio.</p>
          <p>Ask: "Has the strategy's ability to extract returns from this asset's price action degraded?" — not "Has this asset's price gone down?"</p>
        </div>
      </Card>
    </div>
  );
}

function MacroOverlay() {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-zinc-100">Macro Regime Overlay</h2>
        <p className="text-xs text-zinc-500 mt-0.5">Active config: Two-layer (VIX daily + Crypto F&G) + per-asset EMA hard stop. All inputs lagged 1 day.</p>
      </div>

      <Card title="ACTIVE TWO-LAYER OVERLAY" subtitle="Session Turtle Core x3 — ema_hard_fg_half configuration" accent="#10b981">
        <div className="grid grid-cols-2 gap-3 mb-3">
          {[
            {
              name: "Layer 1 — VIX Daily Macro",
              scope: "All non-crypto assets",
              color: "#3b82f6",
              rules: [
                "VIX ≤ 15 = risk-on  → longs 1.0x, shorts suppressed (~0x)",
                "VIX 15–25 = neutral → longs 1.0x, shorts 1.0x",
                "VIX ≥ 25 = risk-off → longs 0.5x, shorts 1.0x",
                "Lag: use prior-day close. Never use same-day VIX.",
              ],
            },
            {
              name: "Layer 2 — Crypto Fear & Greed",
              scope: "Crypto bucket only",
              color: "#a855f7",
              rules: [
                "F&G ≥ 60 = greed/risk-on → longs 1.0x, shorts suppressed (~0x)",
                "F&G 30–60 = neutral    → longs 1.0x, shorts 1.0x",
                "F&G ≤ 30 = fear/risk-off → longs 0.5x (half entry), shorts 1.0x",
                "Lag: use prior-day score. Never use same-day F&G.",
              ],
            },
          ].map((l, i) => (
            <div key={i} className="rounded border border-zinc-800 p-3" style={{ borderLeftColor: l.color, borderLeftWidth: 3 }}>
              <div className="mb-2">
                <span className="text-xs font-semibold text-zinc-200">{l.name}</span>
                <span className="ml-2 text-[10px] text-zinc-500">{l.scope}</span>
              </div>
              {l.rules.map((r, j) => (
                <div key={j} className="flex items-start gap-1.5 text-[11px] text-zinc-400 leading-relaxed mb-0.5">
                  <span className="text-sky-500 mt-0.5 text-[9px]">◆</span><span>{r}</span>
                </div>
              ))}
            </div>
          ))}
        </div>
        <div className="p-3 rounded bg-zinc-800/50 text-[11px] text-zinc-400 space-y-1">
          <div><span className="text-zinc-300 font-medium">Multipliers stack multiplicatively.</span> VIX risk-off + F&G fear on a crypto asset = 0.5 × 0.5 = 0.25× effective size.</div>
          <div><span className="text-zinc-300 font-medium">VIXY intraday layer not active</span> in this configuration — tested and excluded for simplicity (marginal improvement, more data dependencies).</div>
        </div>
      </Card>

      <Card title="CRYPTO EMA HARD STOP" subtitle="Per-asset daily EMA(200) — crypto bucket only, 1-day lag" accent="#ef4444">
        <div className="grid grid-cols-2 gap-3 mb-3">
          <div className="rounded border border-zinc-800 p-3">
            <p className="text-xs font-semibold text-emerald-400 mb-2">Price &gt; EMA(200) — With-Trend Long</p>
            <div className="space-y-1 text-[11px] text-zinc-400">
              <div>Long entries: <span className="text-emerald-400 font-medium">1.0× (full size)</span></div>
              <div>Short entries: <span className="text-red-400 font-medium">0× (hard stop — blocked)</span></div>
            </div>
          </div>
          <div className="rounded border border-zinc-800 p-3">
            <p className="text-xs font-semibold text-red-400 mb-2">Price &lt; EMA(200) — With-Trend Short</p>
            <div className="space-y-1 text-[11px] text-zinc-400">
              <div>Short entries: <span className="text-emerald-400 font-medium">1.0× (full size)</span></div>
              <div>Long entries: <span className="text-red-400 font-medium">0× (hard stop — blocked)</span></div>
            </div>
          </div>
        </div>
        <RuleBlock type="warn" rules={[
          "EMA hard stop is crypto-only. Non-crypto uses VIX overlay for regime scaling, not EMA blocking.",
          "1-day lag: always use yesterday's daily close vs yesterday's EMA. Never today's intrabar value.",
          "Tested against soft-half (0.5×): hard stop (0×) produced better CAGR and lower DD in backtest.",
          "Do not apply EMA hard stop to equity/gold/metals — they use the VIX layer instead.",
        ]} />
      </Card>

      <Card title="MACRO REGIME CLASSIFICATION" subtitle="Use only variables known at time of decision — no revisions">
        <Table
          headers={["Variable", "Source", "Lag", "How to Use"]}
          rows={[
            ["VIX / implied vol", "Options market", "1-day prior close", "VIX ≤ 15: risk-on (longs full, shorts suppressed). VIX 15–25: neutral. VIX ≥ 25: risk-off (longs 0.5×)."],
            ["Crypto Fear & Greed", "Alternative.me index", "1-day prior score", "F&G ≥ 60: greed/risk-on (shorts suppressed). F&G 30–60: neutral. F&G ≤ 30: fear (longs 0.5× half entry)."],
            ["Per-asset EMA(200)", "Daily price data", "1-day prior close", "Crypto only. Below EMA: longs hard-stopped (0×). Above EMA: shorts hard-stopped (0×)."],
            ["Yield curve slope", "10Y − 2Y spread", "Real-time", "Monitoring only. Inversion → log event, do not override signals."],
            ["ISM / PMI", "Survey data", "Monthly release", "< 50 → contraction. Log for regime tag. Strategy adapts via price."],
          ]}
        />
      </Card>

      <Card title="OVERLAY IMPLEMENTATION RULES" accent="#a855f7">
        <RuleBlock type="rule" title="ACCEPTANCE CRITERIA (all must be met to keep macro overlay)" rules={[
          "OOS Sharpe must improve by ≥ 0.1 vs baseline (not just in-sample)",
          "OOS max DD must decrease by ≥ 10% relative (e.g., 20% DD → ≤ 18% DD)",
          "Turnover must not increase by > 30% (macro filter shouldn't whipsaw positions)",
          "Improvement must persist across ≥ 3 of 4 walk-forward windows",
          "If macro overlay meets 3 of 4 criteria → keep with reduced confidence weight",
          "If < 3 of 4 → discard macro overlay, use baseline",
        ]} />
        <RuleBlock type="warn" title="OVERFIT GUARDRAILS" rules={[
          "Macro features add degrees of freedom — each new input doubles overfit risk",
          "Max 3 macro variables in any model — prefer 1–2",
          "Use only binary or tercile regime tags — not continuous macro variables as direct inputs",
          "Test with randomized macro labels — if 'strategy' still 'works', your macro signal is noise",
        ]} />
      </Card>

      <Card title="POLICY ACTION RESPONSE PROTOCOL" subtitle="For live policy shifts during the week">
        <div className="space-y-2">
          {[
            { event: "Surprise rate cut/hike", response: "Log event. Do NOT override signals. Wait 2 sessions for price to absorb. Then check if regime tag changed.", urgency: "yellow" },
            { event: "QE/QT announcement", response: "Flag affected asset classes. Reduce conviction weight by 20% for 2 weeks while market reprices.", urgency: "yellow" },
            { event: "Geopolitical shock", response: "If VIX spikes > 30: auto-reduce all positions by 30%. Let strategy re-enter at new levels.", urgency: "red" },
            { event: "Tariff / trade policy shift", response: "Tag affected sectors/currencies. Monitor for 1 week. Strategy signals should naturally adapt via price.", urgency: "yellow" },
            { event: "Data surprise (NFP, CPI)", response: "No action. Strategy already captures via price. Log for regime classification only.", urgency: "green" },
          ].map((e, i) => (
            <div key={i} className="flex gap-3 items-start p-2 rounded bg-zinc-800/30">
              <Badge color={e.urgency}>{e.event}</Badge>
              <span className="text-[11px] text-zinc-400 leading-relaxed">{e.response}</span>
            </div>
          ))}
        </div>
      </Card>
    </div>
  );
}

function Sensitivity() {
  const [showExample, setShowExample] = useState(false);
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-zinc-100">Parameter Sensitivity Analysis</h2>
        <p className="text-xs text-zinc-500 mt-0.5">Run quarterly — not weekly. Classify each parameter combination.</p>
      </div>

      <Card title="DONCHIAN LENGTH SENSITIVITY" subtitle="Test N = base ± 30% in steps of 10%">
        <div className="space-y-3">
          <div className="p-3 rounded bg-zinc-800/50 font-mono text-[11px] leading-relaxed text-zinc-400">
            <div className="text-zinc-500 mb-1"># Example: if your base Donchian = 20</div>
            <div>Test: N = 14, 16, 18, <span className="text-emerald-400">20</span>, 22, 24, 26</div>
            <div className="text-zinc-500 mt-2 mb-1"># For each N, record OOS:</div>
            <div>Sharpe, MaxDD, Calmar, Expectancy, Turnover, #Trades</div>
          </div>
          <RuleBlock type="rule" title="ROBUSTNESS CLASSIFICATION" rules={[
            "ROBUST: Sharpe varies < 20% across all N values. Profitable in ≥ 6 of 7 tests. DD stable.",
            "UNSTABLE: Sharpe varies 20–50%. Some N values unprofitable. Performance cliff at edges.",
            "LIKELY OVERFIT: Only 1–2 N values profitable. Sharp performance cliff. In-sample >> out-of-sample.",
          ]} />
        </div>
      </Card>

      <Card title="EMA FILTER SENSITIVITY" subtitle="Current base: EMA(200) daily hard stop on crypto — test ± 30%">
        <RuleBlock type="rule" rules={[
          "Base per-asset EMA = 200. Test: 140, 160, 180, 200, 220, 240, 260",
          "Also compare hard stop (0× counter-trend) vs soft-half (0.5× counter-trend) vs no EMA overlay",
          "Hard stop (0×) outperformed soft-half (0.5×) in the full 4.1-year backtest — but verify in walk-forward",
          "Cross-test: for each EMA period, test across all Donchian lengths → creates a parameter surface",
          "The parameter surface should be smooth — a single peak surrounded by gradual decline",
          "Multiple peaks or jagged surface = overfit warning",
          "EMA hard stop is crypto-only. For equity/gold/metals, VIX macro overlay is the regime filter.",
        ]} />
      </Card>

      <Card title="PARAMETER SURFACE INTERPRETATION" accent="#f59e0b">
        <div className="grid grid-cols-3 gap-3">
          {[
            {
              label: "Robust",
              color: "green",
              visual: "▓▓▓▓▓\n▓████▓\n▓████▓\n▓▓▓▓▓",
              desc: "Smooth hill. Wide profitable region. Your base params are near the center.",
            },
            {
              label: "Unstable",
              color: "yellow",
              visual: "░░▓▓░\n░▓██░░\n▓██░░░\n░░░░░░",
              desc: "Tilted or narrow ridge. Profitable region exists but is asymmetric or thin.",
            },
            {
              label: "Likely Overfit",
              color: "red",
              visual: "░░░░░\n░░█░░\n░░░░░\n░░░░░",
              desc: "Isolated spike. Only one param combo works. Everything else fails.",
            },
          ].map((p, i) => (
            <div key={i} className="rounded border border-zinc-800 p-3 text-center">
              <Badge color={p.color}>{p.label}</Badge>
              <pre className="text-[10px] font-mono text-zinc-500 my-2 leading-tight">{p.visual}</pre>
              <p className="text-[10px] text-zinc-500 leading-relaxed">{p.desc}</p>
            </div>
          ))}
        </div>
      </Card>

      <Card title="COST SENSITIVITY" subtitle="Re-run best parameter set with costs multiplied by 1×, 1.5×, 2×, 3×">
        <RuleBlock type="check" rules={[
          "If strategy is profitable at 2× costs → ROBUST to execution quality variance",
          "If strategy breaks even at 1.5× costs → FRAGILE — implementation must be tight",
          "If strategy is unprofitable at 1.5× costs → likely not viable live — too cost-sensitive",
          "Record the 'cost breakeven multiplier' for each asset — assets with multiplier < 1.5× are candidates for removal",
        ]} />
      </Card>

      <Card title="WALK-FORWARD VALIDATION PROTOCOL" accent="#3b82f6">
        <RuleBlock type="rule" rules={[
          "Split data into 5 equal segments (or rolling 70/30 if data is short)",
          "Train on segments 1–3, test on 4. Train on 2–4, test on 5. Etc.",
          "Never use test segment data for any decision — including regime classification",
          "Report: mean OOS return, std of OOS returns across folds, worst fold performance",
          "If worst fold return < −10% annualized → strategy may not survive bad regimes",
          "Final params = median of best params across all folds (not the single best)",
        ]} />
      </Card>
    </div>
  );
}

function Thresholds() {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-zinc-100">Exact Thresholds & Rules</h2>
        <p className="text-xs text-zinc-500 mt-0.5">Hard-coded decision boundaries — no discretion unless noted</p>
      </div>

      <Card title="KEEP / REDUCE / REMOVE THRESHOLDS">
        <Table
          headers={["Action", "Composite Score", "OOS Sharpe", "Correlation", "Min Tenure"]}
          rows={[
            [<Badge color="green">KEEP — Full</Badge>, "≥ 75", "≥ 0.8", "Any", "—"],
            [<Badge color="green">KEEP — Standard</Badge>, "55–74", "≥ 0.5", "Any", "4 weeks"],
            [<Badge color="yellow">REDUCE — 50%</Badge>, "40–54", "0.3–0.5", "Any", "4 wk probation"],
            [<Badge color="yellow">REDUCE — 25%</Badge>, "25–39", "< 0.3", "< 0.3 w/ portfolio", "Diversification hold"],
            [<Badge color="red">REMOVE</Badge>, "< 25 OR < 40 w/ corr ≥ 0.3", "< 0.3", "≥ 0.3", "8 weeks min live"],
            [<Badge color="blue">ADD</Badge>, "≥ 55 (new asset)", "≥ 0.5", "< 0.5 w/ all held", "4 wk paper/OOS"],
          ]}
        />
      </Card>

      <Card title="HALT / CIRCUIT BREAKER RULES" accent="#ef4444">
        <RuleBlock type="warn" rules={[
          "HALT ALL NEW ENTRIES if portfolio DD > 80% of backtest max DD (backtest max DD: 32.35%)",
          "REDUCE ALL SIZES 50% if portfolio DD > 60% of backtest max DD",
          "STOP STRATEGY if live DD exceeds backtest max DD for > 2 consecutive weeks",
          "HALT ASSET if live vs backtest drift > 30% for 2 consecutive weeks",
          "AUTO-REDUCE ASSET 50% if Donchian channel width < 30th percentile for 3+ weeks (chop filter)",
          "EMA HARD STOP (crypto): automatically blocks new longs below EMA(200) and new shorts above EMA(200) — this is a system rule, not a circuit breaker",
          "F&G HALF SIZE (crypto): automatically halves long entries when F&G ≤ 30 — system rule, not discretionary",
          "VIX HALF SIZE (non-crypto): automatically halves long entries when VIX ≥ 25 — system rule",
          "Resume normal sizing only after DD recovers to < 40% of backtest max DD",
        ]} />
      </Card>

      <Card title="PORTFOLIO CONSTRAINTS">
        <Table
          headers={["Constraint", "Hard Limit", "Soft Target", "Action if Breached"]}
          rows={[
            ["Max assets held", "12", "6–8", "Remove lowest-scored until compliant"],
            ["Max weight per asset", "25%", "15%", "Re-normalize weights"],
            ["Max correlated cluster", "40%", "30%", "Reduce highest-correlated asset"],
            ["Max sector exposure", "40%", "30%", "Diversify or reduce cluster"],
            ["Min weight to hold", "5%", "8%", "Remove if below — not worth execution cost"],
            ["Cash reserve", "5%", "10%", "Hold for new entries / margin buffer"],
            ["Max portfolio leverage", "3.0× (exposure_mult)", "2.5×", "Scale all weights proportionally"],
          ]}
        />
      </Card>

      <Card title="POSITION SIZING FORMULA" accent="#10b981">
        <div className="p-3 rounded bg-zinc-800/50 font-mono text-[11px] leading-relaxed text-zinc-400 space-y-2">
          <div>
            <span className="text-zinc-500"># ATR-based fixed-risk sizing</span>
          </div>
          <div>risk_per_trade = account_equity × risk_pct</div>
          <div>stop_distance = ATR(14) × atr_multiplier</div>
          <div>position_size = risk_per_trade / stop_distance</div>
          <div className="text-zinc-500 mt-2"># Apply weight and constraint adjustments</div>
          <div>adjusted_size = position_size × w_final_i × regime_scalar</div>
          <div>final_size = min(adjusted_size, max_position_limit)</div>
        </div>
        <div className="mt-3">
          <RuleBlock type="info" rules={[
            "risk_pct: 5% per trade (base_risk_pct=0.05), directional cap 7%",
            "atr_multiplier: via fixed_stop_pct=10% hard stop on the position",
            "regime_scalar (longs): 0.5× if VIX ≥ 25 (non-crypto) | 0.5× if F&G ≤ 30 (crypto) | 0.0× if below EMA(200) (crypto — hard stop) | 1.0× otherwise",
            "regime_scalar (shorts): 0.0× if VIX ≤ 15 (non-crypto, risk-on) | 0.0× if F&G ≥ 60 (crypto, greed) | 0.0× if above EMA(200) (crypto — hard stop) | 1.0× otherwise",
            "exposure_mult: 3.0× — total portfolio exposure capacity multiplier",
          ]} />
        </div>
      </Card>

      <Card title="REVIEW FREQUENCY MATRIX">
        <Table
          headers={["Review Type", "Frequency", "What Changes", "What Doesn't"]}
          rows={[
            ["Execution audit", "Weekly", "Flag fills, drift", "Nothing — observation only"],
            ["Weight adjustment", "Weekly (W5+)", "Asset weights ±25%", "No asset adds/removes"],
            ["Full rotation", "Monthly (W9+)", "Keep/reduce/remove/add", "Donchian/EMA params"],
            ["Parameter review", "Quarterly", "Donchian length, EMA, stops", "Core strategy logic"],
            ["Strategy audit", "Semi-annual", "Everything on the table", "—"],
          ]}
        />
      </Card>
    </div>
  );
}

function FailureModes() {
  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-lg font-bold text-zinc-100">Key Failure Modes</h2>
        <p className="text-xs text-zinc-500 mt-0.5">Know what kills the strategy before it happens</p>
      </div>

      <Card title="STRATEGY-LEVEL FAILURES">
        {[
          {
            mode: "Regime Shift: Trending → Choppy",
            prob: "High",
            impact: "Severe",
            detect: "ADX declining, Donchian channel narrowing, consecutive losing breakouts (3+)",
            mitigate: "Channel width filter auto-reduces size. Accept DD. Do not curve-fit shorter Donchian.",
            color: "red",
          },
          {
            mode: "Overfitting to Backtest",
            prob: "Medium",
            impact: "Fatal",
            detect: "Live vs backtest drift > 30%. OOS Sharpe < 50% of IS Sharpe.",
            mitigate: "Use walk-forward only. Never optimize on full dataset. Prefer median params across folds.",
            color: "red",
          },
          {
            mode: "Correlated Drawdown",
            prob: "Medium",
            impact: "Severe",
            detect: "Multiple assets in DD simultaneously. Portfolio DD exceeding any single-asset DD.",
            mitigate: "Correlation monitoring. Cluster weight caps. Diversification pillar in scoring.",
            color: "yellow",
          },
          {
            mode: "Cost Erosion",
            prob: "Medium",
            impact: "Moderate",
            detect: "Net return declining while gross return stable. Actual costs > 1.5× assumed.",
            mitigate: "Weekly cost tracking. Cost sensitivity analysis. Remove assets with breakeven multiplier < 1.5×.",
            color: "yellow",
          },
          {
            mode: "Liquidity Withdrawal",
            prob: "Low",
            impact: "Severe",
            detect: "Slippage increasing. Wider spreads. Partial fills.",
            mitigate: "Reduce position size. Switch to limit orders. Remove illiquid assets.",
            color: "yellow",
          },
          {
            mode: "Behavioral Override",
            prob: "High",
            impact: "Variable",
            detect: "> 2 discretionary overrides per month. 'Feeling' the market instead of following signals.",
            mitigate: "Log all overrides. Compare override P&L vs system P&L monthly. If overrides lose, enforce rules.",
            color: "red",
          },
          {
            mode: "Parameter Decay",
            prob: "Low",
            impact: "Moderate",
            detect: "Quarterly sensitivity check shows parameter surface shifting. Optimal N drifting from chosen N.",
            mitigate: "Use robust parameters (center of plateau). Only change quarterly. Prefer median of walk-forward results.",
            color: "green",
          },
        ].map((f, i) => (
          <div key={i} className="mb-3 p-3 rounded border border-zinc-800">
            <div className="flex items-center gap-2 mb-2 flex-wrap">
              <span className="text-xs font-semibold text-zinc-200">{f.mode}</span>
              <Badge color={f.color}>Prob: {f.prob}</Badge>
              <Badge color={f.impact === "Fatal" ? "red" : f.impact === "Severe" ? "yellow" : "neutral"}>Impact: {f.impact}</Badge>
            </div>
            <div className="grid grid-cols-2 gap-2 text-[11px]">
              <div><span className="text-zinc-500">Detection: </span><span className="text-zinc-400">{f.detect}</span></div>
              <div><span className="text-zinc-500">Mitigation: </span><span className="text-zinc-400">{f.mitigate}</span></div>
            </div>
          </div>
        ))}
      </Card>

      <Card title="MACRO OVERLAY FAILURE MODES" accent="#a855f7">
        <RuleBlock type="warn" rules={[
          "False regime signal: macro says 'risk-off' but market trends up → missed profits. Limit macro to position sizing, not signal veto.",
          "Lagging indicators: PMI is monthly, GDP is quarterly. Price leads macro. If macro conflicts with price, trust price for signal, use macro only for sizing.",
          "Overfitting macro features: adding 5 macro variables to 2 years of data guarantees overfit. Max 2–3 variables. Binary/tercile only.",
          "Confirmation bias: searching for macro justification of price moves after the fact. Pre-commit to macro rules before seeing results.",
        ]} />
      </Card>

      <Card title="RECOVERY PROTOCOL" accent="#3b82f6">
        <Section number="1" title="When Portfolio DD Hits 50% of Backtest Max">
          <RuleBlock type="rule" rules={[
            "Reduce all position sizes by 30%",
            "Tighten stops by 20% (reduce ATR multiplier by 0.5)",
            "No new asset additions until DD recovers to < 30%",
            "Continue executing all signals — do not stop trading",
          ]} />
        </Section>
        <Section number="2" title="When Portfolio DD Hits 80% of Backtest Max">
          <RuleBlock type="rule" rules={[
            "Halt all new entries",
            "Reduce existing positions by 50%",
            "Review execution quality immediately",
            "If execution is clean → this is the strategy's tail risk. Ride it out.",
            "If execution drift detected → fix execution first before resuming",
          ]} />
        </Section>
        <Section number="3" title="When Live DD Exceeds Backtest Max">
          <RuleBlock type="rule" rules={[
            "Full strategy halt for 1 week minimum",
            "Complete audit: data integrity, fill quality, parameter drift, regime classification",
            "If root cause found → fix and resume at 25% size",
            "If no root cause → the backtest was wrong. Re-run walk-forward with updated data. New baseline.",
            "Resume at 25% size → 50% after 2 weeks → 100% after 4 weeks if tracking new backtest",
          ]} />
        </Section>
      </Card>
    </div>
  );
}

// ── MAIN APP ──────────────────────────────────────────────

export default function StrategyModerator() {
  const [activeTab, setActiveTab] = useState("overview");

  const content = {
    overview: <WeeklyReview />,
    scoring: <AssetScoring />,
    rotation: <RotationEngine />,
    macro: <MacroOverlay />,
    sensitivity: <Sensitivity />,
    thresholds: <Thresholds />,
    failures: <FailureModes />,
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-200" style={{ fontFamily: "'DM Sans', 'Helvetica Neue', sans-serif" }}>
      <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet" />

      {/* Header */}
      <div className="border-b border-zinc-800/80 bg-zinc-950/90 backdrop-blur-md sticky top-0 z-50">
        <div className="max-w-6xl mx-auto px-4 py-3">
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded bg-zinc-800 border border-zinc-700 flex items-center justify-center text-xs font-bold text-zinc-400">SM</div>
              <div>
                <h1 className="text-sm font-bold text-zinc-100 tracking-tight">STRATEGY MODERATOR</h1>
                <p className="text-[10px] text-zinc-600 uppercase tracking-widest">Session Turtle Core x3 · EMA Hard Stop + F&amp;G Half Entry · Donchian(20) · VIX/F&amp;G Two-Layer Overlay</p>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <Badge color="green">LIVE</Badge>
              <span className="text-[10px] text-zinc-600">Week 4 of observation</span>
            </div>
          </div>

          {/* Tabs */}
          <div className="flex gap-1 overflow-x-auto pb-0.5 -mb-px">
            {TABS.map(tab => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "px-3 py-1.5 text-xs font-medium rounded-t transition-colors whitespace-nowrap",
                  activeTab === tab.id
                    ? "bg-zinc-800/80 text-zinc-100 border border-zinc-700/50 border-b-zinc-950"
                    : "text-zinc-500 hover:text-zinc-300 hover:bg-zinc-800/30"
                )}
              >
                {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-6xl mx-auto px-4 py-6">
        {content[activeTab]}
      </div>

      {/* Footer */}
      <div className="border-t border-zinc-800/50 mt-8">
        <div className="max-w-6xl mx-auto px-4 py-4 text-[10px] text-zinc-700 flex justify-between">
          <span>Strategy Moderator v2.0 — Session Turtle Core x3 · ema_hard_fg_half · Backtest 2022-02-22 → present</span>
          <span>Review frequency: Weekly execution · Monthly rotation · Quarterly parameters</span>
        </div>
      </div>
    </div>
  );
}
