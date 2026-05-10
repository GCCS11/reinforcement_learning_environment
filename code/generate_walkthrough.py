import os
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors

output_path = "code/walkthrough.pdf"


def build():
    os.makedirs("code", exist_ok=True)
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=inch,
        rightMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle("title", parent=styles["Title"], fontSize=18, spaceAfter=20)
    h1_style    = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=14, spaceAfter=10, spaceBefore=16)
    h2_style    = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=11, spaceAfter=6, spaceBefore=10)
    body_style  = ParagraphStyle("body", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14)
    code_style  = ParagraphStyle("code", parent=styles["Code"], fontSize=8, spaceAfter=6,
                                  backColor=colors.HexColor("#f5f5f5"), leading=12)

    story = []

    story.append(Paragraph("Code Walkthrough", title_style))
    story.append(Paragraph("Reinforcement Learning Trading Environment — BTC/USDT", styles["Normal"]))
    story.append(Spacer(1, 0.2 * inch))

    # 1. Overview
    story.append(Paragraph("1. Overview", h1_style))
    story.append(Paragraph(
        "This project trains a PPO-based reinforcement learning agent to trade BTC/USDT on hourly data "
        "from August 2017 to April 2026. The agent operates in a semi-Markov environment where it decides "
        "whether to open a long, short, or stay flat at each decision point. Positions are managed "
        "automatically by the environment using stop-loss, take-profit, and maximum holding period rules. "
        "Walk-forward validation across four expanding folds is used to evaluate generalization.",
        body_style
    ))

    # 2. Code Structure
    story.append(Paragraph("2. Code Structure", h1_style))
    story.append(Paragraph(
        "The entire pipeline is in the code/ directory and is run with a single entrypoint:", body_style
    ))
    story.append(Paragraph("python -m code.main", code_style))

    table_data = [
        ["File", "Responsibility"],
        ["eda.py", "Loads raw CSV, computes regime analysis, return distributions, autocorrelation, "
                   "and stationarity tests. Outputs artifacts/btcusdt_eda.pdf."],
        ["features.py", "Computes all technical indicators and returns. Defines the four walk-forward "
                        "fold splits and the normalization routine."],
        ["environment.py", "Implements the gymnasium-compatible semi-Markov trading environment. "
                           "Contains the position simulation logic with OHLC-based stop/take-profit resolution."],
        ["train.py", "Runs the hyperparameter search on the validation set and retrains the best "
                     "configuration for the full timestep budget. Saves models and training curves."],
        ["benchmarks.py", "Computes BTC buy-and-hold and SPY equity curves for the test periods."],
        ["evaluate.py", "Loads saved models, runs deterministic evaluation on test sets, computes "
                        "all required metrics, and generates equity curves and trade logs."],
        ["main.py", "Single entrypoint. Sets global random seeds and calls run_training() then run_evaluation()."],
        ["generate_walkthrough.py", "Generates this document."],
    ]

    table = Table(table_data, colWidths=[1.6 * inch, 4.8 * inch])
    table.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("PADDING",     (0, 0), (-1, -1), 6),
    ]))
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))

    # 3. Environment Design
    story.append(Paragraph("3. Environment Design", h1_style))

    story.append(Paragraph("3.1 Semi-Markov Formulation", h2_style))
    story.append(Paragraph(
        "The environment is semi-Markov: the agent only makes a decision when it has no open position. "
        "Once a position is opened, the environment simulates forward bar by bar until a closing condition "
        "is met. Each gymnasium step therefore covers a variable number of hourly bars. "
        "PPO handles this naturally since it treats each step as a single transition regardless of "
        "the real-world duration.",
        body_style
    ))

    story.append(Paragraph("3.2 Action Space", h2_style))
    story.append(Paragraph(
        "Discrete(3): action 0 = stay flat (advance one bar), action 1 = open long, action 2 = open short. "
        "A continuous action space (e.g., position size) was considered but rejected because it would "
        "require a more complex reward normalization and the project focuses on direction prediction "
        "rather than sizing.",
        body_style
    ))

    story.append(Paragraph("3.3 Entry and Exit Price Assumptions", h2_style))
    story.append(Paragraph(
        "Entry price: the open of the bar immediately following the decision bar. This avoids any "
        "within-bar lookahead on the decision bar. Exit price: exactly at the stop-loss or take-profit "
        "level if triggered within a bar, or at the close of the bar if the horizon is reached. "
        "OHLC assumption: if both stop-loss and take-profit are touched within the same bar, the "
        "stop-loss fills first. This is the conservative default and is explicitly stated in the environment docstring.",
        body_style
    ))

    story.append(Paragraph("3.4 Episode Definition", h2_style))
    story.append(Paragraph(
        "Each training episode is a 720-bar window (30 days) with a randomly sampled start index "
        "within the training fold. This gives the agent diverse market conditions across episodes "
        "without memorizing a fixed sequence. A fixed-start single-episode approach was considered "
        "but would have led to severe overfitting to the specific trajectory seen during training.",
        body_style
    ))

    # 4. Feature Engineering
    story.append(Paragraph("4. Feature Engineering", h1_style))
    story.append(Paragraph(
        "All 17 features are causally computable at decision time — no future data is used. "
        "Features fall into five categories:",
        body_style
    ))

    feature_data = [
        ["Category", "Features", "Justification"],
        ["Returns",     "log_return, log_return_2h/4h/8h/24h",
         "Captures recent momentum. Autocorrelation in the EDA showed returns are near-uncorrelated "
         "at lag 1, but multi-period returns carry some signal (MI > 0.05)."],
        ["Volatility",  "vol_24h, vol_72h, vol_168h",
         "Justified by strong ACF in squared returns (volatility clustering). "
         "ATR-based feature also included for position sizing context."],
        ["Momentum",    "rsi_14, rsi_28, macd_diff",
         "Standard momentum indicators. Low MI scores suggest limited predictive power "
         "but included for completeness."],
        ["Structure",   "bb_pct, bb_width, atr_pct",
         "Bollinger band position and width capture mean-reversion and expansion regimes."],
        ["Time",        "hour_sin, hour_cos",
         "Crypto trades 24/7. Cyclical encoding preserves continuity at hour 23 to 0. "
         "Low MI suggests intraday patterns are weak in this dataset."],
    ]

    ftable = Table(feature_data, colWidths=[1.1 * inch, 1.9 * inch, 3.4 * inch])
    ftable.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 8),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("VALIGN",      (0, 0), (-1, -1), "TOP"),
        ("PADDING",     (0, 0), (-1, -1), 5),
    ]))
    story.append(ftable)
    story.append(Spacer(1, 0.15 * inch))

    story.append(Paragraph(
        "Normalization: means and standard deviations are computed on the training fold only and "
        "applied to validation and test folds. This prevents any information leakage from future data "
        "into the feature scaling.",
        body_style
    ))

    # 5. Training
    story.append(Paragraph("5. Training Procedure", h1_style))
    story.append(Paragraph(
        "Algorithm: PPO (Proximal Policy Optimization) with MlpPolicy. PPO was chosen over SAC "
        "because our action space is discrete — SAC is designed for continuous actions. DDPG was "
        "rejected for the same reason. A2C was considered but PPO's clipped objective consistently "
        "outperforms A2C in practice.",
        body_style
    ))
    story.append(Paragraph(
        "Hyperparameter search: three configurations are evaluated on the validation fold for "
        "50,000 steps each. The configuration with the highest cumulative reward on the validation "
        "set is retrained for 500,000 steps. The search grid varies learning rate, entropy coefficient, "
        "stop-loss, and take-profit simultaneously.",
        body_style
    ))
    story.append(Paragraph(
        "Fixed hyperparameters across all configurations: batch_size=64, n_epochs=10, gamma=0.99, "
        "gae_lambda=0.95, clip_range=0.2, n_steps=2048. These are the SB3 defaults and were not "
        "tuned to keep the search space manageable.",
        body_style
    ))

    # 6. Walk-Forward Evaluation
    story.append(Paragraph("6. Walk-Forward Evaluation", h1_style))
    story.append(Paragraph(
        "Four expanding-window folds are used. Training always starts from August 2017. "
        "The validation set is used only for hyperparameter selection. The test set is touched "
        "exactly once after all hyperparameter decisions are finalized. Models are saved per fold "
        "and evaluation is run separately from training to allow re-evaluation without retraining.",
        body_style
    ))

    fold_data = [
        ["Fold", "Train End",   "Validation", "Test",          "Train Rows", "Test Rows"],
        ["1",    "Dec 2021",    "2022",        "2023",          "38,054",     "8,759"],
        ["2",    "Dec 2022",    "2023",        "2024",          "46,814",     "8,784"],
        ["3",    "Dec 2023",    "2024",        "2025",          "55,573",     "8,579"],
        ["4",    "Dec 2024",    "2025",        "Jan-Apr 2026",  "64,357",     "2,484"],
    ]

    foldtable = Table(fold_data, colWidths=[0.5*inch, 1.0*inch, 1.0*inch, 1.1*inch, 1.1*inch, 1.0*inch])
    foldtable.setStyle(TableStyle([
        ("BACKGROUND",  (0, 0), (-1, 0),  colors.HexColor("#2c3e50")),
        ("TEXTCOLOR",   (0, 0), (-1, 0),  colors.white),
        ("FONTNAME",    (0, 0), (-1, 0),  "Helvetica-Bold"),
        ("FONTSIZE",    (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f2f2")]),
        ("GRID",        (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ALIGN",       (0, 0), (-1, -1), "CENTER"),
        ("PADDING",     (0, 0), (-1, -1), 5),
    ]))
    story.append(foldtable)
    story.append(Spacer(1, 0.15 * inch))

    # 7. Reproducibility
    story.append(Paragraph("7. Reproducibility", h1_style))
    story.append(Paragraph(
        "All random seeds are fixed at 42 via set_seeds() in main.py, which sets Python random, "
        "NumPy, and PyTorch seeds. SB3 receives seed=42 directly. Running python -m code.main "
        "from the project root with the dataset at data/Binance_BTCUSDT_1h.csv reproduces all "
        "artifacts in a single pass.",
        body_style
    ))

    doc.build(story)
    print(f"walkthrough saved to {output_path}")


if __name__ == "__main__":
    build()