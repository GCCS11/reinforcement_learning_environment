import os
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, HRFlowable, KeepTogether
)
from reportlab.platypus.flowables import HRFlowable
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY

output_path = "report.pdf"
artifacts   = "artifacts"

accent      = colors.HexColor("#2c3e50")
accent_light= colors.HexColor("#3498db")
row_a       = colors.white
row_b       = colors.HexColor("#f0f4f8")
grid_c      = colors.HexColor("#d0d7de")


def add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#888888"))
    canvas.drawRightString(7.5 * inch, 0.5 * inch, f"Page {doc.page}")
    canvas.drawString(inch, 0.5 * inch, "RL Trading Environment — BTC/USDT")
    canvas.restoreState()


def get_styles():
    styles    = getSampleStyleSheet()
    title     = ParagraphStyle("title",  parent=styles["Title"],
                               fontSize=26, spaceAfter=8, textColor=colors.white,
                               fontName="Helvetica-Bold")
    subtitle  = ParagraphStyle("sub",    parent=styles["Normal"],
                               fontSize=13, spaceAfter=4, textColor=colors.HexColor("#bdc3c7"))
    h1        = ParagraphStyle("h1",     parent=styles["Heading1"],
                               fontSize=13, spaceAfter=8, spaceBefore=14,
                               textColor=accent, fontName="Helvetica-Bold",
                               borderPad=4)
    h2        = ParagraphStyle("h2",     parent=styles["Heading2"],
                               fontSize=11, spaceAfter=5, spaceBefore=10,
                               textColor=accent_light, fontName="Helvetica-Bold")
    body      = ParagraphStyle("body",   parent=styles["Normal"],
                               fontSize=10, spaceAfter=6, leading=15, alignment=TA_JUSTIFY)
    cell      = ParagraphStyle("cell",   parent=styles["Normal"],
                               fontSize=8,  leading=11, alignment=TA_JUSTIFY)
    cell_bold = ParagraphStyle("cellb",  parent=styles["Normal"],
                               fontSize=8,  leading=11,
                               fontName="Helvetica-Bold", textColor=colors.white)
    caption   = ParagraphStyle("cap",    parent=styles["Normal"],
                               fontSize=8, spaceAfter=10, spaceBefore=2,
                               textColor=colors.HexColor("#666666"), alignment=TA_CENTER)
    return dict(title=title, subtitle=subtitle, h1=h1, h2=h2,
                body=body, cell=cell, cell_bold=cell_bold, caption=caption)


def tbl_style():
    return TableStyle([
        ("BACKGROUND",     (0, 0), (-1, 0),  accent),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [row_a, row_b]),
        ("GRID",           (0, 0), (-1, -1), 0.4, grid_c),
        ("VALIGN",         (0, 0), (-1, -1), "TOP"),
        ("PADDING",        (0, 0), (-1, -1), 5),
        ("LINEBELOW",      (0, 0), (-1, 0),  1.2, accent_light),
    ])


def p(text, style):
    return Paragraph(str(text), style)


def section_header(text, S):
    return KeepTogether([
        HRFlowable(width="100%", thickness=2, color=accent, spaceAfter=4),
        p(text, S["h1"]),
    ])


def two_images(path1, path2, cap1, cap2, S, w=3.1*inch, h=2.3*inch):
    img_row = [[Image(path1, width=w, height=h), Image(path2, width=w, height=h)]]
    cap_row = [[p(cap1, S["caption"]), p(cap2, S["caption"])]]
    t1 = Table(img_row, colWidths=[w+0.1*inch, w+0.1*inch])
    t1.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER"), ("VALIGN", (0,0), (-1,-1), "TOP")]))
    t2 = Table(cap_row, colWidths=[w+0.1*inch, w+0.1*inch])
    t2.setStyle(TableStyle([("ALIGN", (0,0), (-1,-1), "CENTER")]))
    return [t1, t2, Spacer(1, 0.1*inch)]


def build():
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=inch, rightMargin=inch,
        topMargin=0.75*inch, bottomMargin=0.85*inch
    )
    S     = get_styles()
    story = []
    metrics = pd.read_csv(f"{artifacts}/metrics_summary.csv")

    def row(label):
        return metrics[metrics["label"] == label].iloc[0]

    s1 = row("strategy_fold_1"); s2 = row("strategy_fold_2")
    s3 = row("strategy_fold_3"); s4 = row("strategy_fold_4")
    sm = row("strategy_mean");   ss = row("strategy_std")
    b1 = row("btc_fold_1");      b2 = row("btc_fold_2")
    b3 = row("btc_fold_3");      b4 = row("btc_fold_4")
    p1 = row("sp500_fold_1");    p2 = row("sp500_fold_2")
    p3 = row("sp500_fold_3");    p4 = row("sp500_fold_4")

    # ── Cover ──────────────────────────────────────────────────────────
    cover_data = [[Paragraph(
        "<para alignment='center'>"
        "<font color='white' size=26><b>Reinforcement Learning</b></font><br/>"
        "<font color='white' size=26><b>Trading Environment</b></font><br/><br/>"
        "<font color='#bdc3c7' size=13>Executive Report — BTC/USDT Hourly Agent</font><br/><br/>"
        "<font color='#bdc3c7' size=10>Walk-Forward Validation | PPO | Stable Baselines 3</font>"
        "</para>",
        getSampleStyleSheet()["Normal"]
    )]]
    cover = Table(cover_data, colWidths=[6.5*inch])
    cover.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), accent),
        ("PADDING",    (0,0), (-1,-1), 40),
        ("ALIGN",      (0,0), (-1,-1), "CENTER"),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(Spacer(1, 1.2*inch))
    story.append(cover)
    story.append(Spacer(1, 0.4*inch))

    meta_data = [
        [p("Dataset",   S["cell_bold"]), p("BTC/USDT 1h | Binance | Aug 2017 – Apr 2026", S["cell"])],
        [p("Algorithm", S["cell_bold"]), p("PPO (Proximal Policy Optimization) — Stable Baselines 3", S["cell"])],
        [p("Folds",     S["cell_bold"]), p("4-fold expanding-window walk-forward validation", S["cell"])],
        [p("Features",  S["cell_bold"]), p("17 technical indicators, causally computable", S["cell"])],
        [p("Seed",      S["cell_bold"]), p("42 (fully reproducible)", S["cell"])],
    ]
    meta = Table(meta_data, colWidths=[1.3*inch, 5.2*inch])
    meta.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (0,-1), accent),
        ("ROWBACKGROUNDS", (1,0), (1,-1), [row_a, row_b]),
        ("GRID",           (0,0), (-1,-1), 0.4, grid_c),
        ("PADDING",        (0,0), (-1,-1), 6),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
    ]))
    story.append(meta)
    story.append(PageBreak())

    # ── 1. Dataset Analysis ────────────────────────────────────────────
    story.append(section_header("1. Dataset Analysis", S))
    story.append(p(
        "The dataset contains 75,558 hourly OHLCV bars for BTC/USDT from Binance, spanning "
        "August 17, 2017 to April 26, 2026. After computing features with rolling windows up to "
        "168 bars, 75,420 rows remain. The data covers the full lifecycle of Bitcoin as a liquid "
        "asset, including multiple boom-bust cycles of increasing amplitude.",
        S["body"]
    ))

    story.append(p("1.1 Market Regimes", S["h2"]))
    story.append(p(
        "Seven distinct regimes are identified based on price trend, realized volatility, and "
        "maximum drawdown observed in the data:",
        S["body"]
    ))

    regime_data = [
        [p("Regime", S["cell_bold"]), p("Period", S["cell_bold"]),
         p("Trend", S["cell_bold"]), p("Ann. Vol", S["cell_bold"]),
         p("Max DD", S["cell_bold"])],
        [p("Bull Run I",    S["cell"]), p("Aug 2017 – Jan 2018", S["cell"]),
         p("Strong uptrend +2,000%", S["cell"]), p("~200%", S["cell"]), p("–30%",  S["cell"])],
        [p("Crypto Winter", S["cell"]), p("Jan 2018 – Jan 2019", S["cell"]),
         p("Sustained decline –84%",  S["cell"]), p("~120%", S["cell"]), p("–84%",  S["cell"])],
        [p("Accumulation",  S["cell"]), p("Jan 2019 – Oct 2020", S["cell"]),
         p("Range-bound, slow recovery", S["cell"]), p("~60%", S["cell"]), p("–57%", S["cell"])],
        [p("Bull Run II",   S["cell"]), p("Oct 2020 – Nov 2021", S["cell"]),
         p("Strong uptrend +1,400%", S["cell"]), p("~80%", S["cell"]), p("–53%",  S["cell"])],
        [p("Bear Market",   S["cell"]), p("Nov 2021 – Dec 2022", S["cell"]),
         p("Sustained decline –77%",  S["cell"]), p("~90%", S["cell"]), p("–77%",  S["cell"])],
        [p("Recovery",      S["cell"]), p("Jan 2023 – Mar 2024", S["cell"]),
         p("Strong uptrend +300%",   S["cell"]), p("~50%", S["cell"]), p("–22%",  S["cell"])],
        [p("Bull Run III",  S["cell"]), p("Mar 2024 – Apr 2026", S["cell"]),
         p("New ATH cycle, high vol", S["cell"]), p("~60%", S["cell"]), p("–36%",  S["cell"])],
    ]
    story.append(Table(regime_data,
                       colWidths=[1.1*inch, 1.5*inch, 1.7*inch, 0.9*inch, 0.9*inch],
                       style=tbl_style()))
    story.append(Spacer(1, 0.1*inch))

    story.append(p("1.2 Statistical Properties", S["h2"]))
    story.append(p(
        "Log returns exhibit a kurtosis of 35.74, confirming heavy tails relative to the normal "
        "distribution — visible in both the histogram and Q-Q plot of the EDA. Stationarity tests "
        "confirm that price levels are non-stationary (ADF p-value = 0.756, KPSS rejects "
        "stationarity) while log returns are stationary (ADF p-value < 0.001, KPSS p-value = 0.10). "
        "This motivates using returns rather than raw prices as model inputs.",
        S["body"]
    ))
    story.append(p(
        "The ACF of log returns shows no significant autocorrelation beyond lag zero, consistent "
        "with weak-form market efficiency. However, the ACF of squared and absolute returns shows "
        "persistent autocorrelation across all 48 lags tested, confirming volatility clustering. "
        "This justifies including multi-scale rolling volatility features (vol_24h, vol_72h, "
        "vol_168h) in the state representation.",
        S["body"]
    ))

    story.append(PageBreak())

    # ── 2. SDP ─────────────────────────────────────────────────────────
    story.append(section_header("2. Sequential Decision Modeling", S))
    story.append(p(
        "The trading problem is formulated as a sequential decision problem in Powell's framework. "
        "The agent interacts with the market at discrete decision epochs, which are not evenly "
        "spaced in calendar time — this is a semi-Markov formulation where the decision frequency "
        "is endogenous and depends on when positions close.",
        S["body"]
    ))

    story.append(p("2.1 State Variables", S["h2"]))
    state_data = [
        [p("Component", S["cell_bold"]), p("Contents", S["cell_bold"]),
         p("Justification", S["cell_bold"])],
        [p("Physical S^phys_t",    S["cell"]),
         p("Position status: always flat at decision time (d = 0). Agent decides only when no position is open.", S["cell"]),
         p("Semi-Markov design: position management is delegated to the environment.", S["cell"])],
        [p("Information S^info_t", S["cell"]),
         p("17-dimensional normalized feature vector: log returns at 1h/2h/4h/8h/24h lags, "
           "rolling volatility at 24h/72h/168h, RSI(14), RSI(28), MACD diff, Bollinger band "
           "percentile and width, ATR%, volume ratio, hour sin/cos.", S["cell"]),
         p("Features are causally computable from data available at or before decision time. "
           "Normalization uses training-fold statistics only.", S["cell"])],
        [p("Belief S^belief_t",    S["cell"]),
         p("Implicit in the MLP policy weights. No explicit probabilistic belief state.", S["cell"]),
         p("A POMDP with explicit beliefs was considered but rejected: adds architectural "
           "complexity with uncertain benefit given the feature set size.", S["cell"])],
    ]
    story.append(Table(state_data, colWidths=[1.3*inch, 2.7*inch, 2.5*inch], style=tbl_style()))
    story.append(Spacer(1, 0.1*inch))

    story.append(p("2.2 Decision Variables", S["h2"]))
    story.append(p(
        "The decision variable x_t belongs to a discrete action space A = {0, 1, 2}, where "
        "0 = stay flat (advance one bar with zero reward), 1 = open long position, "
        "2 = open short position. A continuous action space encoding position size was considered "
        "but rejected: it would require reward normalization by size and add a dimension to the "
        "credit assignment problem without clear benefit given the binary direction signal in "
        "hourly crypto data. An explicit close action was not included because position management "
        "is handled entirely by the environment through predefined exit rules.",
        S["body"]
    ))

    story.append(p("2.3 Exogenous Information", S["h2"]))
    story.append(p(
        "The exogenous information process W_{t+1} consists of the OHLCV data for all bars from "
        "t+1 until position close. Specifically, for a position opened at t, the relevant "
        "exogenous information is the sequence {(O_s, H_s, L_s, C_s) : s = t+1, ..., t+τ} "
        "where τ is the first bar at which a closing condition is met. The agent cannot observe "
        "this information at decision time — it is revealed sequentially as the environment "
        "simulates the position forward.",
        S["body"]
    ))

    story.append(p("2.4 Transition Function", S["h2"]))
    story.append(p(
        "The physical-state transition is governed by four closing conditions evaluated in order "
        "of priority for each bar s after entry: "
        "(1) Stop-loss: L_s ≤ entry × (1 − sl) for long, H_s ≥ entry × (1 + sl) for short — "
        "exit price = stop-loss level. "
        "(2) Take-profit: H_s ≥ entry × (1 + tp) for long, L_s ≤ entry × (1 − tp) for short — "
        "exit price = take-profit level. "
        "(3) Maximum holding period: s = t + 1 + max_hold, exit at C_s. "
        "(4) Episode boundary: s = episode_end, exit at C_s. "
        "OHLC assumption: if both stop-loss and take-profit are touched within the same bar, "
        "the stop-loss fills first (conservative default). "
        "Entry price is the open of bar t+1, avoiding within-bar lookahead on the decision bar.",
        S["body"]
    ))

    story.append(p("2.5 Contribution Function", S["h2"]))
    story.append(p(
        "The contribution (reward) is terminal — paid at position close rather than accruing "
        "per bar. For a position opened at t with direction d ∈ {+1, −1}:",
        S["body"]
    ))

    contrib_data = [
        [p("Condition", S["cell_bold"]), p("Contribution", S["cell_bold"])],
        [p("x_t ∈ {1, 2}  (open position)", S["cell"]),
         p("C = d × log(P_exit / P_entry) − 2 × 0.00125", S["cell"])],
        [p("x_t = 0  (flat)", S["cell"]),
         p("C = 0", S["cell"])],
    ]
    story.append(Table(contrib_data, colWidths=[2.8*inch, 3.7*inch], style=tbl_style()))
    story.append(Spacer(1, 0.1*inch))
    story.append(p(
        "The log return formulation was chosen over raw PnL because it is scale-invariant across "
        "BTC price regimes. Raw PnL was rejected because it produces rewards two orders of "
        "magnitude larger in later folds, destabilizing training. Per-step rewards during a "
        "position were rejected due to poor credit assignment. Failure mode: the log return reward "
        "does not penalize intrabar drawdown, only the final outcome. A Sharpe-based reward was "
        "considered but rejected because episode-level Sharpe is non-Markovian.",
        S["body"]
    ))

    story.append(p("2.6 Objective", S["h2"]))
    story.append(p(
        "The objective is to find the policy π* that maximizes the expected sum of contributions "
        "over the episode: π* = argmax_π E[ Σ_t C(S_t, x_t, W_{t+1}) ]. "
        "No discounting (γ = 0.99 ≈ 1 for 720-bar episodes) is applied, consistent with the "
        "portfolio objective of maximizing total return over the episode.",
        S["body"]
    ))

    story.append(p("2.7 Episode Definition", S["h2"]))
    story.append(p(
        "An episode is a 720-bar (30-day) window with a randomly sampled start index within the "
        "training fold. BTC trades 24/7, so there is no natural session boundary. The 30-day "
        "window was chosen to be long enough for the agent to experience multiple trade "
        "opportunities while short enough to allow diverse sampling. A single long episode "
        "covering the full training fold was rejected because it prevents frequent exposure "
        "to early training data during optimization.",
        S["body"]
    ))

    story.append(PageBreak())

    # ── 3. Training ────────────────────────────────────────────────────
    story.append(section_header("3. Training", S))
    story.append(p(
        "The agent is trained using PPO (Proximal Policy Optimization) with an MLP policy. "
        "PPO was selected over SAC (requires continuous actions), DDPG (requires continuous "
        "actions), and A2C (PPO's clipped objective is empirically more stable). The default "
        "MlpPolicy uses two hidden layers of 64 units each with tanh activations.",
        S["body"]
    ))

    story.append(p("3.1 Hyperparameter Search", S["h2"]))
    story.append(p(
        "Three configurations are evaluated on the validation fold for 50,000 steps each. "
        "The best by cumulative validation reward is retrained for 500,000 steps. "
        "The grid varies learning rate, entropy coefficient, stop-loss, and take-profit:",
        S["body"]
    ))

    hp_data = [
        [p("Config", S["cell_bold"]), p("lr", S["cell_bold"]),
         p("ent_coef", S["cell_bold"]), p("stop_loss", S["cell_bold"]),
         p("take_profit", S["cell_bold"])],
        [p("1", S["cell"]), p("3e-4", S["cell"]), p("0.010", S["cell"]),
         p("2%", S["cell"]), p("4%", S["cell"])],
        [p("2", S["cell"]), p("3e-4", S["cell"]), p("0.010", S["cell"]),
         p("3%", S["cell"]), p("6%", S["cell"])],
        [p("3", S["cell"]), p("1e-4", S["cell"]), p("0.005", S["cell"]),
         p("2%", S["cell"]), p("6%", S["cell"])],
    ]
    story.append(Table(hp_data, colWidths=[0.9*inch, 0.9*inch, 0.9*inch, 1.0*inch, 1.0*inch],
                       style=tbl_style()))
    story.append(Spacer(1, 0.1*inch))

    best_data = [
        [p("Fold", S["cell_bold"]), p("Best Config", S["cell_bold"]),
         p("sl", S["cell_bold"]), p("tp", S["cell_bold"]),
         p("lr", S["cell_bold"]), p("Val Reward", S["cell_bold"])],
        [p("1", S["cell"]), p("3", S["cell"]), p("2%", S["cell"]),
         p("6%", S["cell"]), p("1e-4", S["cell"]), p("-0.691", S["cell"])],
        [p("2", S["cell"]), p("1", S["cell"]), p("2%", S["cell"]),
         p("4%", S["cell"]), p("3e-4", S["cell"]), p("-0.064", S["cell"])],
        [p("3", S["cell"]), p("3", S["cell"]), p("2%", S["cell"]),
         p("6%", S["cell"]), p("1e-4", S["cell"]), p("-0.341", S["cell"])],
        [p("4", S["cell"]), p("2", S["cell"]), p("3%", S["cell"]),
         p("6%", S["cell"]), p("3e-4", S["cell"]), p("-0.265", S["cell"])],
    ]
    story.append(Table(best_data,
                       colWidths=[0.7*inch, 1.0*inch, 0.7*inch, 0.7*inch, 0.9*inch, 1.0*inch],
                       style=tbl_style()))
    story.append(Spacer(1, 0.1*inch))
    story.append(p(
        "All validation rewards are negative, confirming that the problem is not simply a matter "
        "of configuration tuning — the features do not contain sufficient predictive signal for "
        "short-term direction at the 500,000-step training budget.",
        S["body"]
    ))

    story.append(p("3.2 Training Curves", S["h2"]))
    story.append(p(
        "All four folds show a consistent upward trend in training reward from approximately "
        "-0.10 to +0.10 to +0.20. This confirms the agent is learning on training data. "
        "The gap between positive training reward and negative test performance indicates "
        "overfitting to regime-specific patterns.",
        S["body"]
    ))

    for paths, caps in [
        (("training_curves_fold_1.png", "training_curves_fold_2.png"),
         ("Fold 1 (sl=0.02, tp=0.06, lr=1e-4)", "Fold 2 (sl=0.02, tp=0.04, lr=3e-4)")),
        (("training_curves_fold_3.png", "training_curves_fold_4.png"),
         ("Fold 3 (sl=0.02, tp=0.06, lr=1e-4)", "Fold 4 (sl=0.03, tp=0.06, lr=3e-4)")),
    ]:
        p1_path = f"{artifacts}/{paths[0]}"
        p2_path = f"{artifacts}/{paths[1]}"
        if os.path.exists(p1_path) and os.path.exists(p2_path):
            for item in two_images(p1_path, p2_path, caps[0], caps[1], S):
                story.append(item)

    story.append(PageBreak())

    # ── 4. Results ─────────────────────────────────────────────────────
    story.append(section_header("4. Results", S))
    story.append(p("4.1 Metrics Summary", S["h2"]))

    def fmt(val, pct=False):
        if pd.isna(val): return "—"
        return f"{val*100:.1f}%" if pct else f"{val:.3f}"

    btc_mean_ann = np.mean([b1.annualized_return, b2.annualized_return,
                            b3.annualized_return, b4.annualized_return])
    spy_mean_ann = np.mean([p1.annualized_return, p2.annualized_return,
                            p3.annualized_return, p4.annualized_return])
    btc_mean_sh  = np.mean([b1.sharpe, b2.sharpe, b3.sharpe, b4.sharpe])
    spy_mean_sh  = np.mean([p1.sharpe, p2.sharpe, p3.sharpe, p4.sharpe])
    btc_mean_tr  = np.mean([b1.total_return, b2.total_return, b3.total_return, b4.total_return])
    spy_mean_tr  = np.mean([p1.total_return, p2.total_return, p3.total_return, p4.total_return])

    metrics_data = [
        [p("Metric", S["cell_bold"]),
         p("Fold 1", S["cell_bold"]), p("Fold 2", S["cell_bold"]),
         p("Fold 3", S["cell_bold"]), p("Fold 4", S["cell_bold"]),
         p("Mean ± Std", S["cell_bold"]),
         p("BTC mean", S["cell_bold"]),
         p("SPY mean", S["cell_bold"])],
        [p("Total Return", S["cell"]),
         p(fmt(s1.total_return, True), S["cell"]), p(fmt(s2.total_return, True), S["cell"]),
         p(fmt(s3.total_return, True), S["cell"]), p(fmt(s4.total_return, True), S["cell"]),
         p(f"{sm.total_return*100:.1f}±{ss.total_return*100:.1f}%", S["cell"]),
         p(fmt(btc_mean_tr, True), S["cell"]), p(fmt(spy_mean_tr, True), S["cell"])],
        [p("Ann. Return", S["cell"]),
         p(fmt(s1.annualized_return, True), S["cell"]), p(fmt(s2.annualized_return, True), S["cell"]),
         p(fmt(s3.annualized_return, True), S["cell"]), p(fmt(s4.annualized_return, True), S["cell"]),
         p(f"{sm.annualized_return*100:.1f}±{ss.annualized_return*100:.1f}%", S["cell"]),
         p(fmt(btc_mean_ann, True), S["cell"]), p(fmt(spy_mean_ann, True), S["cell"])],
        [p("Ann. Vol", S["cell"]),
         p(fmt(s1.annualized_vol, True), S["cell"]), p(fmt(s2.annualized_vol, True), S["cell"]),
         p(fmt(s3.annualized_vol, True), S["cell"]), p(fmt(s4.annualized_vol, True), S["cell"]),
         p(f"{sm.annualized_vol*100:.1f}±{ss.annualized_vol*100:.1f}%", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
        [p("Sharpe", S["cell"]),
         p(fmt(s1.sharpe), S["cell"]), p(fmt(s2.sharpe), S["cell"]),
         p(fmt(s3.sharpe), S["cell"]), p(fmt(s4.sharpe), S["cell"]),
         p(f"{sm.sharpe:.3f}±{ss.sharpe:.3f}", S["cell"]),
         p(fmt(btc_mean_sh), S["cell"]), p(fmt(spy_mean_sh), S["cell"])],
        [p("Sortino", S["cell"]),
         p(fmt(s1.sortino), S["cell"]), p(fmt(s2.sortino), S["cell"]),
         p(fmt(s3.sortino), S["cell"]), p(fmt(s4.sortino), S["cell"]),
         p(f"{sm.sortino:.3f}±{ss.sortino:.3f}", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
        [p("Max Drawdown", S["cell"]),
         p(fmt(s1.max_drawdown, True), S["cell"]), p(fmt(s2.max_drawdown, True), S["cell"]),
         p(fmt(s3.max_drawdown, True), S["cell"]), p(fmt(s4.max_drawdown, True), S["cell"]),
         p(f"{sm.max_drawdown*100:.1f}±{ss.max_drawdown*100:.1f}%", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
        [p("Calmar", S["cell"]),
         p(fmt(s1.calmar), S["cell"]), p(fmt(s2.calmar), S["cell"]),
         p(fmt(s3.calmar), S["cell"]), p(fmt(s4.calmar), S["cell"]),
         p(f"{sm.calmar:.3f}±{ss.calmar:.3f}", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
        [p("Hit Rate", S["cell"]),
         p(fmt(s1.hit_rate, True), S["cell"]), p(fmt(s2.hit_rate, True), S["cell"]),
         p(fmt(s3.hit_rate, True), S["cell"]), p(fmt(s4.hit_rate, True), S["cell"]),
         p(f"{sm.hit_rate*100:.1f}±{ss.hit_rate*100:.1f}%", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
        [p("Avg Duration (h)", S["cell"]),
         p(f"{s1.avg_duration_hours:.1f}", S["cell"]), p(f"{s2.avg_duration_hours:.1f}", S["cell"]),
         p(f"{s3.avg_duration_hours:.1f}", S["cell"]), p(f"{s4.avg_duration_hours:.1f}", S["cell"]),
         p(f"{sm.avg_duration_hours:.1f}±{ss.avg_duration_hours:.1f}", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
        [p("Turnover", S["cell"]),
         p(fmt(s1.turnover, True), S["cell"]), p(fmt(s2.turnover, True), S["cell"]),
         p(fmt(s3.turnover, True), S["cell"]), p(fmt(s4.turnover, True), S["cell"]),
         p(f"{sm.turnover*100:.2f}±{ss.turnover*100:.2f}%", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
        [p("Alpha vs BTC", S["cell"]),
         p(fmt(s1.alpha_vs_btc, True), S["cell"]), p(fmt(s2.alpha_vs_btc, True), S["cell"]),
         p(fmt(s3.alpha_vs_btc, True), S["cell"]), p(fmt(s4.alpha_vs_btc, True), S["cell"]),
         p(f"{sm.alpha_vs_btc*100:.1f}±{ss.alpha_vs_btc*100:.1f}%", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
        [p("Alpha vs SPY", S["cell"]),
         p(fmt(s1.alpha_vs_sp500, True), S["cell"]), p(fmt(s2.alpha_vs_sp500, True), S["cell"]),
         p(fmt(s3.alpha_vs_sp500, True), S["cell"]), p(fmt(s4.alpha_vs_sp500, True), S["cell"]),
         p(f"{sm.alpha_vs_sp500*100:.1f}±{ss.alpha_vs_sp500*100:.1f}%", S["cell"]),
         p("—", S["cell"]), p("—", S["cell"])],
    ]
    story.append(Table(metrics_data,
                       colWidths=[1.05*inch, 0.6*inch, 0.6*inch, 0.6*inch, 0.6*inch,
                                  1.15*inch, 0.85*inch, 0.85*inch],
                       style=tbl_style()))
    story.append(Spacer(1, 0.1*inch))

    story.append(p("4.2 Discussion", S["h2"]))
    story.append(p(
        "The strategy produces negative returns across all four test folds with a mean annualized "
        "return of -50.0% ± 26.5% and a mean Sharpe ratio of -1.116 ± 0.530. The results are "
        "consistently negative — there is no fold where the strategy approaches breakeven after "
        "costs. This consistency rules out bad luck as the primary explanation. "
        "Fold 1 (2023) shows the least negative performance (-14.6%, Sharpe -0.39) in a year "
        "when BTC returned +155%. Folds 2 and 3 are worst (-54% and -52%). The hit rate is "
        "consistently below 40%, meaning the agent loses more than 60% of trades. "
        "The training curves show positive rewards on training data, confirming the agent learns "
        "regime-specific patterns that do not generalize — the classic overfitting signature "
        "in financial RL.",
        S["body"]
    ))

    story.append(p("4.3 Equity Curves", S["h2"]))
    for paths, caps in [
        (("equity_curves_test_fold_1.png", "equity_curves_test_fold_2.png"),
         ("Fold 1 — Test 2023", "Fold 2 — Test 2024")),
        (("equity_curves_test_fold_3.png", "equity_curves_test_fold_4.png"),
         ("Fold 3 — Test 2025", "Fold 4 — Test Jan-Apr 2026")),
    ]:
        p1_path = f"{artifacts}/{paths[0]}"
        p2_path = f"{artifacts}/{paths[1]}"
        if os.path.exists(p1_path) and os.path.exists(p2_path):
            for item in two_images(p1_path, p2_path, caps[0], caps[1], S):
                story.append(item)

    story.append(PageBreak())

    # ── 5. Trade Analysis ──────────────────────────────────────────────
    story.append(section_header("5. Trade Analysis", S))

    story.append(p("5.1 Three Largest Losing Trades", S["h2"]))
    story.append(p(
        "All three largest losing trades occur in Fold 1 (2023), all long positions hitting "
        "their stop-loss. The identical net PnL of -2.27% is expected: 2% gross loss plus "
        "0.25% round-trip cost.",
        S["body"]
    ))

    trade_data = [
        [p("", S["cell_bold"]), p("Trade 1", S["cell_bold"]),
         p("Trade 2", S["cell_bold"]), p("Trade 3", S["cell_bold"])],
        [p("Entry",       S["cell"]), p("2023-06-29 15:00", S["cell"]),
         p("2023-10-05 01:00", S["cell"]), p("2023-12-29 03:00", S["cell"])],
        [p("Exit",        S["cell"]), p("2023-06-30 13:00", S["cell"]),
         p("2023-10-06 12:00", S["cell"]), p("2023-12-29 17:00", S["cell"])],
        [p("Exit Reason", S["cell"]), p("stop_loss", S["cell"]),
         p("stop_loss", S["cell"]), p("stop_loss", S["cell"])],
        [p("Direction",   S["cell"]), p("long", S["cell"]),
         p("long", S["cell"]), p("long", S["cell"])],
        [p("Entry Price", S["cell"]), p("$30,450", S["cell"]),
         p("$27,812", S["cell"]), p("$42,575", S["cell"])],
        [p("Exit Price",  S["cell"]), p("$29,841", S["cell"]),
         p("$27,256", S["cell"]), p("$41,723", S["cell"])],
        [p("Net PnL",     S["cell"]), p("-2.27%", S["cell"]),
         p("-2.27%", S["cell"]), p("-2.27%", S["cell"])],
        [p("Duration",    S["cell"]), p("22h", S["cell"]),
         p("35h", S["cell"]), p("14h", S["cell"])],
    ]
    t = Table(trade_data, colWidths=[1.3*inch, 1.7*inch, 1.7*inch, 1.7*inch])
    t.setStyle(TableStyle([
        ("BACKGROUND",     (0, 0), (0, -1), accent),
        ("TEXTCOLOR",      (0, 0), (0, -1), colors.white),
        ("FONTNAME",       (0, 0), (0, -1), "Helvetica-Bold"),
        ("BACKGROUND",     (1, 0), (-1, 0), accent),
        ("TEXTCOLOR",      (1, 0), (-1, 0), colors.white),
        ("FONTNAME",       (1, 0), (-1, 0), "Helvetica-Bold"),
        ("ROWBACKGROUNDS", (1, 1), (-1, -1), [row_a, row_b]),
        ("GRID",           (0, 0), (-1, -1), 0.4, grid_c),
        ("VALIGN",         (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING",        (0, 0), (-1, -1), 5),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.1*inch))
    story.append(p(
        "In all three cases the agent opened a long during apparent upward momentum but entered "
        "at a local peak followed by a reversal large enough to trigger the stop-loss. "
        "The failure mode is momentum chasing without sufficient context to identify "
        "overextended conditions. To avoid these trades, the state could include a "
        "longer-horizon trend feature (e.g., 7-day MA slope) or the reward could penalize "
        "entries during high ATR to discourage opening in volatile conditions.",
        S["body"]
    ))

    story.append(p("5.2 Largest Drawdown Period", S["h2"]))
    story.append(p(
        "The largest drawdown occurs in Fold 2 (2024), where equity falls from ~1.10 in "
        "February 2024 to ~0.47 by mid-May 2024 (-57%). During this period BTC surged from "
        "$40,000 to over $70,000 (+75%). The agent systematically opened short positions "
        "in a rising market, accumulating stop-loss exits. The equity curve shows a staircase "
        "pattern of discrete losses — each step is a short position stopped out by the "
        "continuing uptrend. This is the most damaging regime for this agent: a strong "
        "unidirectional trend that the agent consistently fades.",
        S["body"]
    ))

    story.append(p("5.3 Correlation with BTC Returns", S["h2"]))
    corr_data = [
        [p("Fold", S["cell_bold"]), p("Test Period", S["cell_bold"]),
         p("Correlation", S["cell_bold"]), p("Interpretation", S["cell_bold"])],
        [p("1", S["cell"]), p("2023", S["cell"]), p("0.401", S["cell"]),
         p("Moderate positive: agent partially tracks BTC direction.", S["cell"])],
        [p("2", S["cell"]), p("2024", S["cell"]), p("0.209", S["cell"]),
         p("Weak positive: agent less correlated with BTC.", S["cell"])],
        [p("3", S["cell"]), p("2025", S["cell"]), p("-0.140", S["cell"]),
         p("Weak negative: agent slightly contrarian to BTC.", S["cell"])],
        [p("4", S["cell"]), p("Jan-Apr 2026", S["cell"]), p("-0.369", S["cell"]),
         p("Moderate negative: agent fades BTC moves.", S["cell"])],
    ]
    story.append(Table(corr_data, colWidths=[0.6*inch, 1.0*inch, 1.0*inch, 3.9*inch],
                       style=tbl_style()))
    story.append(Spacer(1, 0.1*inch))
    story.append(p(
        "The correlation shifts from positive in early folds to negative in later folds. "
        "The agent trained on 2017-2021 bull data learned long-biased signals that transferred "
        "to the 2023 recovery. As training expanded to include the 2022 bear market, the agent's "
        "signals became contrarian. This is not a leveraged long position — it is inconsistent "
        "and unprofitable behavior driven by overfitted regime-specific patterns.",
        S["body"]
    ))

    story.append(PageBreak())

    # ── 6. Production Failure Analysis ────────────────────────────────
    story.append(section_header("6. Production Failure Analysis", S))

    story.append(p("6.1 Execution Assumptions", S["h2"]))
    story.append(p(
        "The simulator fills orders at the open of the bar immediately after the decision bar. "
        "In production, the true fill price will differ due to bid-ask spread (0.01–0.05% on "
        "liquid BTC pairs), market impact for size beyond a few thousand dollars, and queue "
        "position for limit orders. The 0.125% per-side cost assumption may underestimate real "
        "costs for any meaningful position size, particularly during high-volatility periods.",
        S["body"]
    ))

    story.append(p("6.2 Latency", S["h2"]))
    story.append(p(
        "The model runs inference in under one second (MLP with 17 inputs). The more critical "
        "concern is data feed delay. If the OHLCV data for bar t arrives with a one-bar delay "
        "— common with aggregated feeds — the agent effectively uses bar t-1 features to trade "
        "at bar t+1 open. In this hourly framework one-bar latency is less catastrophic than "
        "in high-frequency settings, but signal decay is nonzero given the low autocorrelation "
        "in returns documented in the EDA.",
        S["body"]
    ))

    story.append(p("6.3 Capacity and Crowding", S["h2"]))
    story.append(p(
        "The strategy trades ~3.5% of available bars. BTC/USDT on Binance has daily volume "
        "exceeding $1B, so retail-scale positions have negligible market impact. However, if "
        "50 similar agents ran simultaneously, correlated entry signals would create order flow "
        "clustering at bar open, increasing slippage. Given the negative test performance, the "
        "counterparty is likely better-informed — market makers and directional participants "
        "exploiting the agent's predictable signal.",
        S["body"]
    ))

    story.append(p("6.4 Regime Change", S["h2"]))
    story.append(p(
        "The agent performs worst in sustained bull markets (Folds 1-2, BTC +120-155%) and "
        "relatively better in volatile recovery phases. A regime resembling 2018 (slow sustained "
        "decline with bear rallies) would cause losses on long entries that look like momentum. "
        "A 2022-like regime (rapid decline then sideways) would challenge both stop-loss "
        "calibration and trend-following features. Features most likely to degrade in novel "
        "regimes: vol_24h and vol_168h (calibrated to historical BTC volatility levels), "
        "RSI thresholds (regime-dependent), and MACD (fails in ranging markets). The agent has "
        "no explicit regime detection — it extrapolates from the nearest seen regime with no "
        "guarantee of relevance.",
        S["body"]
    ))

    doc.build(story, onFirstPage=add_page_number, onLaterPages=add_page_number)
    print(f"report saved to {output_path}")


if __name__ == "__main__":
    build()