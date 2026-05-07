import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from code.environment import BTCTradingEnv
from code.features import compute_features, folds, get_fold_splits, normalize_features

data_path   = "data/Binance_BTCUSDT_1h.csv"
models_dir  = "artifacts/models"
figures_dir = "artifacts"


def load_and_prepare():
    df = pd.read_csv(data_path, skiprows=1)
    df.columns = [
        "unix", "date", "symbol", "open", "high",
        "low", "close", "volume_btc", "volume_usdt", "tradecount"
    ]
    df["date"] = pd.to_datetime(df["date"], format="mixed")
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop(columns=["symbol"])
    df = compute_features(df)
    df = df.dropna().reset_index(drop=True)
    return df


class RewardCallback(BaseCallback):
    def __init__(self, log_freq=2048):
        super().__init__()
        self.log_freq        = log_freq
        self.episode_rewards = []
        self.logged_rewards  = []
        self.logged_steps    = []

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for info, done in zip(infos, dones):
            ep_info = info.get("episode")
            if ep_info is not None:
                self.episode_rewards.append(ep_info["r"])

        if self.n_calls % self.log_freq == 0 and self.episode_rewards:
            self.logged_steps.append(self.num_timesteps)
            self.logged_rewards.append(np.mean(self.episode_rewards[-50:]))

        return True


def evaluate_on_split(model, df, stop_loss, take_profit):
    env = BTCTradingEnv(df, stop_loss=stop_loss, take_profit=take_profit,
                        episode_length=len(df) - 50)
    env.reset(seed=0)
    env.current_step = 0
    env.episode_end  = len(df) - 2
    obs = env._obs()

    total_reward = 0.0
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated

    return total_reward


hyperparams_grid = [
    {"learning_rate": 3e-4, "ent_coef": 0.01,  "n_steps": 2048, "stop_loss": 0.02, "take_profit": 0.04},
    {"learning_rate": 3e-4, "ent_coef": 0.01,  "n_steps": 2048, "stop_loss": 0.03, "take_profit": 0.06},
    {"learning_rate": 1e-4, "ent_coef": 0.005, "n_steps": 2048, "stop_loss": 0.02, "take_profit": 0.06},
]

search_timesteps = 50_000
final_timesteps  = 500_000


def train_fold(fold_config, df, seed=42):
    fold_num = fold_config["fold"]
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    print(f"training fold {fold_num}...")

    train_df, val_df, test_df = get_fold_splits(df, fold_config)
    train_df, val_df, test_df, means, stds = normalize_features(train_df, val_df, test_df)

    norm_stats = pd.DataFrame({"mean": means, "std": stds})
    norm_stats.to_csv(f"{models_dir}/norm_stats_fold_{fold_num}.csv")

    val_results = []

    for i, hp in enumerate(hyperparams_grid):
        env = Monitor(BTCTradingEnv(
            train_df,
            stop_loss    = hp["stop_loss"],
            take_profit  = hp["take_profit"],
        ))
        model = PPO(
            "MlpPolicy", env,
            learning_rate = hp["learning_rate"],
            ent_coef      = hp["ent_coef"],
            n_steps       = hp["n_steps"],
            batch_size    = 64,
            n_epochs      = 10,
            gamma         = 0.99,
            gae_lambda    = 0.95,
            clip_range    = 0.2,
            seed          = seed,
            verbose       = 0,
        )
        model.learn(total_timesteps=search_timesteps)
        val_reward = evaluate_on_split(model, val_df, hp["stop_loss"], hp["take_profit"])
        val_results.append({**hp, "val_reward": val_reward, "config_idx": i})
        print(f"  config {i+1}/{len(hyperparams_grid)} val_reward={val_reward:.4f}")

    val_df_results = pd.DataFrame(val_results)
    val_df_results.to_csv(f"{figures_dir}/validation_results_fold_{fold_num}.csv", index=False)

    best_idx  = val_df_results["val_reward"].idxmax()
    best_hp   = hyperparams_grid[best_idx]
    print(f"  best config: sl={best_hp['stop_loss']} tp={best_hp['take_profit']} lr={best_hp['learning_rate']}")

    callback = RewardCallback()
    env = Monitor(BTCTradingEnv(
        train_df,
        stop_loss   = best_hp["stop_loss"],
        take_profit = best_hp["take_profit"],
    ))
    model = PPO(
        "MlpPolicy", env,
        learning_rate = best_hp["learning_rate"],
        ent_coef      = best_hp["ent_coef"],
        n_steps       = best_hp["n_steps"],
        batch_size    = 64,
        n_epochs      = 10,
        gamma         = 0.99,
        gae_lambda    = 0.95,
        clip_range    = 0.2,
        seed          = seed,
        verbose       = 0,
    )
    model.learn(total_timesteps=final_timesteps, callback=callback)
    model.save(f"{models_dir}/ppo_fold_{fold_num}")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(callback.logged_steps, callback.logged_rewards, linewidth=1)
    ax.set_title(
        f"Training Curve - Fold {fold_num}  "
        f"(sl={best_hp['stop_loss']} tp={best_hp['take_profit']} lr={best_hp['learning_rate']})"
    )
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Mean Episode Reward (last 50 eps)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(f"{figures_dir}/training_curves_fold_{fold_num}.png", dpi=150)
    plt.close(fig)

    print(f"fold {fold_num} done")
    return model, train_df, val_df, test_df


def run_training(seed=42):
    df = load_and_prepare()
    trained = []
    for fold_config in folds:
        result = train_fold(fold_config, df, seed=seed)
        trained.append(result)
    return trained


if __name__ == "__main__":
    run_training()