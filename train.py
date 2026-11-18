#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""End-to-end Conv1D regression project for Kaggle House Prices."""
from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "SalePrice"
ID_COLUMN = "Id"
CLASS_NAMES = ["قیمت پایین", "قیمت متوسط", "قیمت بالا"]


def set_seed(seed: int = RANDOM_STATE) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def get_feature_column_groups(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "category", "string", "bool"]
    ).columns.tolist()
    return numeric_cols, categorical_cols


def coerce_feature_dtypes(
    X: pd.DataFrame, numeric_cols: list[str], categorical_cols: list[str]
) -> pd.DataFrame:
    """Keep train/validation/test dtypes consistent, including all-null test columns."""
    X = X.copy()
    for col in numeric_cols:
        if col in X.columns:
            X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("object")
    return X


def make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a leakage-safe numeric/categorical preprocessing pipeline."""
    numeric_cols, categorical_cols = get_feature_column_groups(X)

    transformers = []
    if numeric_cols:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipe, numeric_cols))
    if categorical_cols:
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(
                    strategy="constant", fill_value="Missing", keep_empty_features=True
                )),
                ("onehot", OneHotEncoder(
                    handle_unknown="ignore", sparse_output=False, dtype=np.float32
                )),
            ]
        )
        transformers.append(("categorical", categorical_pipe, categorical_cols))
    if not transformers:
        raise ValueError("هیچ ستون ویژگی عددی یا دسته‌ای برای آموزش پیدا نشد.")
    return ColumnTransformer(transformers=transformers, remainder="drop", verbose_feature_names_out=True)


def build_cnn(
    input_shape: tuple[int, int],
    num_blocks: int = 3,
    use_batch_norm: bool = True,
    dropout_rate: float = 0.25,
    learning_rate: float = 1e-3,
):
    """Create a 1D CNN regressor for the encoded tabular feature vector."""
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow نصب نیست. دستور `pip install -r requirements.txt` را اجرا کنید."
        ) from exc

    if num_blocks < 1 or num_blocks > 4:
        raise ValueError("num_blocks باید بین ۱ و ۴ باشد.")

    inputs = tf.keras.Input(shape=input_shape, name="tabular_features")
    x = inputs
    filters = [32, 64, 128, 256]
    current_length = input_shape[0]

    for block_index in range(num_blocks):
        x = tf.keras.layers.Conv1D(
            filters=filters[block_index],
            kernel_size=3,
            padding="same",
            use_bias=not use_batch_norm,
            kernel_regularizer=tf.keras.regularizers.l2(1e-5),
            name=f"conv1d_block_{block_index + 1}",
        )(x)
        if use_batch_norm:
            x = tf.keras.layers.BatchNormalization(name=f"batch_norm_{block_index + 1}")(x)
        x = tf.keras.layers.Activation("relu", name=f"relu_{block_index + 1}")(x)
        if current_length >= 2:
            x = tf.keras.layers.MaxPooling1D(pool_size=2, name=f"max_pool_{block_index + 1}")(x)
            current_length = current_length // 2
        if dropout_rate > 0:
            x = tf.keras.layers.Dropout(dropout_rate, name=f"dropout_{block_index + 1}")(x)

    x = tf.keras.layers.GlobalAveragePooling1D(name="global_average_pooling")(x)
    x = tf.keras.layers.Dense(64, activation="relu", name="dense_64")(x)
    if dropout_rate > 0:
        x = tf.keras.layers.Dropout(dropout_rate, name="dense_dropout")(x)
    outputs = tf.keras.layers.Dense(1, activation="linear", name="log_sale_price")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"cnn_{num_blocks}_blocks")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae_log")],
    )
    return model


def make_callbacks(patience: int = 14):
    import tensorflow as tf
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=patience, restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(4, patience // 3),
            min_lr=1e-6, verbose=0
        ),
    ]


def to_model_input(X: np.ndarray) -> np.ndarray:
    return np.asarray(X, dtype=np.float32).reshape((X.shape[0], X.shape[1], 1))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64).reshape(-1), 0.0)
    log_true = np.log1p(y_true)
    log_pred = np.log1p(y_pred)
    return {
        "MSE_price": float(mean_squared_error(y_true, y_pred)),
        "RMSE_price": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE_price": float(mean_absolute_error(y_true, y_pred)),
        "R2_price": float(r2_score(y_true, y_pred)),
        "MSE_log1p": float(mean_squared_error(log_true, log_pred)),
        "RMSE_log1p_Kaggle": float(np.sqrt(mean_squared_error(log_true, log_pred))),
    }


def price_band_metrics(
    y_train: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[dict[str, float], np.ndarray, list[float]]:
    """Supplementary 3-class metrics after converting prices to train-defined bands."""
    q1, q2 = np.quantile(np.asarray(y_train, dtype=float), [1 / 3, 2 / 3])
    if not q1 < q2:
        raise ValueError("مرزهای سه گروه قیمتی یکسان شده‌اند؛ امکان طبقه‌بندی گروهی وجود ندارد.")
    true_class = np.digitize(np.asarray(y_true, dtype=float), bins=[q1, q2], right=True)
    pred_class = np.digitize(np.maximum(np.asarray(y_pred, dtype=float), 0.0), bins=[q1, q2], right=True)
    labels = [0, 1, 2]
    cm = confusion_matrix(true_class, pred_class, labels=labels)
    metrics = {
        "Accuracy_price_bands": float(accuracy_score(true_class, pred_class)),
        "Precision_macro_price_bands": float(precision_score(
            true_class, pred_class, labels=labels, average="macro", zero_division=0
        )),
        "Recall_macro_price_bands": float(recall_score(
            true_class, pred_class, labels=labels, average="macro", zero_division=0
        )),
        "F1_macro_price_bands": float(f1_score(
            true_class, pred_class, labels=labels, average="macro", zero_division=0
        )),
    }
    return metrics, cm, [float(q1), float(q2)]


def save_history(history: Any, out_dir: Path) -> pd.DataFrame:
    hist = pd.DataFrame(history.history)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(out_dir / "training_history.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(9, 5))
    plt.plot(hist["epoch"], hist["loss"], label="Training MSE")
    if "val_loss" in hist:
        plt.plot(hist["epoch"], hist["val_loss"], label="Validation MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE on log1p(SalePrice)")
    plt.title("CNN Training and Validation MSE")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "mse_by_epoch.png", dpi=180)
    plt.close()
    return hist


def save_confusion_matrix(cm: np.ndarray, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(3), yticks=np.arange(3),
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        xlabel="گروه قیمتی پیش‌بینی‌شده", ylabel="گروه قیمتی واقعی",
        title="Confusion Matrix - Price Bands",
    )
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right", rotation_mode="anchor")
    threshold = cm.max() / 2 if cm.size and cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold else "black")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_price_bands.png", dpi=180)
    plt.close(fig)
    pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        out_dir / "confusion_matrix_price_bands.csv", encoding="utf-8-sig"
    )


def save_prediction_plots(y_true: np.ndarray, y_pred: np.ndarray, out_dir: Path) -> None:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    plt.figure(figsize=(7, 6))
    plt.scatter(y_true, y_pred, alpha=0.65, s=22)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    plt.plot([low, high], [low, high], "r--", label="Ideal prediction")
    plt.xlabel("Actual SalePrice ($)")
    plt.ylabel("Predicted SalePrice ($)")
    plt.title("Actual vs Predicted House Prices")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "actual_vs_predicted.png", dpi=180)
    plt.close()

    residuals = y_true - y_pred
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=30, edgecolor="black", alpha=0.8)
    plt.axvline(0, color="red", linestyle="--")
    plt.xlabel("Residual (Actual - Predicted) ($)")
    plt.ylabel("Frequency")
    plt.title("Validation Residual Distribution")
    plt.grid(axis="y", alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_distribution.png", dpi=180)
    plt.close()


def train_one_model(
    X_fit: np.ndarray,
    y_fit_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    epochs: int,
    batch_size: int,
    num_blocks: int = 3,
    use_batch_norm: bool = True,
    dropout_rate: float = 0.25,
    patience: int = 14,
    verbose: int = 0,
):
    model = build_cnn(
        input_shape=(X_fit.shape[1], 1), num_blocks=num_blocks,
        use_batch_norm=use_batch_norm, dropout_rate=dropout_rate,
    )
    history = model.fit(
        to_model_input(X_fit), y_fit_log,
        validation_data=(to_model_input(X_val), y_val_log),
        epochs=epochs, batch_size=batch_size, shuffle=True,
        callbacks=make_callbacks(patience=patience), verbose=verbose,
    )
    val_pred_log = model.predict(to_model_input(X_val), verbose=0).reshape(-1)
    val_pred_price = np.maximum(np.expm1(np.clip(val_pred_log, -10, 20)), 0.0)
    return model, history, val_pred_price


def write_report(
    out_dir: Path,
    metrics: dict[str, float],
    cls_metrics: dict[str, float],
    thresholds: list[float],
    n_rows: int,
    n_features: int,
    n_numeric: int,
    n_categorical: int,
    best_epoch: int,
    ablation_df: pd.DataFrame | None,
) -> None:
    lines = [
        "# گزارش پروژه پیش‌بینی قیمت خانه با شبکه عصبی CNN",
        "",
        "## ۱. معرفی مسئله",
        "هدف، پیش‌بینی مقدار پیوسته SalePrice با یک شبکه عصبی کانولوشنی یک‌بعدی (Conv1D) است.",
        "در این پروژه SalePrice با تبدیل log1p آموزش داده شده تا تابع خطا با معیار مسابقه Kaggle هم‌راستا باشد.",
        "",
        "## ۲. مشخصات داده و آزمایش",
        f"- تعداد ردیف‌های فایل آموزش: {n_rows}",
        f"- تعداد ویژگی‌های ورودی پس از کدگذاری: {n_features}",
        f"- تعداد ویژگی‌های عددی خام: {n_numeric}",
        f"- تعداد ویژگی‌های دسته‌ای خام: {n_categorical}",
        "- تقسیم آموزش/اعتبارسنجی: ۸۰٪ / ۲۰٪ با seed برابر ۴۲",
        f"- بهترین epoch مدل اصلی براساس کمترین val_loss: {best_epoch}",
        "",
        "## ۳. معیارهای رگرسیون روی مجموعه اعتبارسنجی",
        "| معیار | مقدار |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| {key} | {value:,.6f} |")
    lines += [
        "",
        "توجه: معیار رسمی Kaggle برابر RMSE روی لگاریتم قیمت واقعی و پیش‌بینی‌شده است؛ در این گزارش با RMSE_log1p_Kaggle نشان داده شده است.",
        "",
        "## ۴. معیارهای طبقه‌بندی گروه‌های قیمتی (تحلیل تکمیلی)",
        "این معیارها معیار اصلی رگرسیون نیستند. قیمت‌ها فقط برای این تحلیل جانبی به سه گروه پایین، متوسط و بالا تبدیل شده‌اند.",
        f"- مرز گروه پایین/متوسط (ثلث اول داده آموزش): {thresholds[0]:,.2f} دلار",
        f"- مرز گروه متوسط/بالا (ثلث دوم داده آموزش): {thresholds[1]:,.2f} دلار",
        "| معیار | مقدار |",
        "|---|---:|",
    ]
    for key, value in cls_metrics.items():
        lines.append(f"| {key} | {value:,.6f} |")
    lines += [
        "",
        "![ماتریس ابهام گروه‌های قیمتی](confusion_matrix_price_bands.png)",
        "",
        "## ۵. نمودارها",
        "- [MSE برحسب Epoch](mse_by_epoch.png)",
        "- [قیمت واقعی در برابر پیش‌بینی‌شده](actual_vs_predicted.png)",
        "- [توزیع باقیمانده‌ها](residual_distribution.png)",
        "- [ماتریس ابهام گروه‌های قیمتی](confusion_matrix_price_bands.png)",
        "",
        "## ۶. بررسی اثر لایه‌ها و اجزای CNN",
    ]
    if ablation_df is not None and not ablation_df.empty:
        lines += [
            "| معماری | تعداد بلوک | BatchNorm | Dropout | RMSE لگاریتمی | R² قیمت |",
            "|---|---:|---|---|---:|---:|",
        ]
        for _, row in ablation_df.iterrows():
            lines.append(
                f"| {row['experiment']} | {int(row['num_blocks'])} | {row['batch_norm']} | "
                f"{row['dropout']} | {row['RMSE_log1p_Kaggle']:.6f} | {row['R2_price']:.6f} |"
            )
        lines += [
            "",
            "- [مقایسه RMSE لگاریتمی معماری‌ها](ablation_rmse_log.png)",
            "- [مقایسه R² معماری‌ها](ablation_r2.png)",
        ]
    else:
        lines.append("آزمایش ablation در این اجرا غیرفعال بوده است.")
    lines += [
        "",
        "## ۷. نکات روش‌شناختی و محدودیت‌ها",
        "1. فایل test.csv در مسابقه Kaggle ستون SalePrice ندارد؛ بنابراین محاسبه معیار خطا روی آن ممکن نیست. ارزیابی معتبر این پروژه روی validation جداشده از train.csv انجام شده است.",
        "2. Accuracy، Precision، Recall، F1 و ماتریس ابهام به خود مسئله رگرسیون تعلق ندارند و فقط پس از تبدیل هدف به سه گروه قیمتی گزارش شده‌اند.",
        "3. ویژگی‌های جدولی ذاتاً ساختار مکانی یا ترتیبی طبیعی ندارند؛ Conv1D در این پروژه یک انتخاب آموزشی مطابق صورت تمرین است و لزوماً از مدل‌های تخصصی داده جدولی بهتر نیست.",
        "4. مقادیر این گزارش فقط پس از اجرای واقعی train.py و روی فایل‌های رسمی کاربر تولید می‌شوند.",
        "",
        "## ۸. فایل خروجی مسابقه",
        "فایل submission.csv با ستون‌های Id و SalePrice در همین پوشه تولید شده است.",
    ]
    (out_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")


def run_ablation(
    X_fit: np.ndarray,
    y_fit_log: np.ndarray,
    X_val: np.ndarray,
    y_val_log: np.ndarray,
    y_val_price: np.ndarray,
    epochs: int,
    batch_size: int,
    base_metrics: dict[str, float],
    patience: int,
    out_dir: Path,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    experiments = [
        {"experiment": "CNN_2_blocks", "num_blocks": 2, "batch_norm": True, "dropout": 0.25},
        {"experiment": "CNN_4_blocks", "num_blocks": 4, "batch_norm": True, "dropout": 0.25},
        {"experiment": "without_BatchNorm", "num_blocks": 3, "batch_norm": False, "dropout": 0.25},
        {"experiment": "without_Dropout", "num_blocks": 3, "batch_norm": True, "dropout": 0.0},
    ]
    results = [{
        "experiment": "CNN_3_blocks_baseline", "num_blocks": 3,
        "batch_norm": True, "dropout": 0.25,
        "RMSE_log1p_Kaggle": base_metrics["RMSE_log1p_Kaggle"],
        "R2_price": base_metrics["R2_price"],
        "MAE_price": base_metrics["MAE_price"],
    }]
    for i, config in enumerate(experiments, start=1):
        print(f"[Ablation {i}/{len(experiments)}] {config['experiment']}")
        set_seed(seed)  # same initialization seed for a fairer architecture comparison
        _, _, pred = train_one_model(
            X_fit, y_fit_log, X_val, y_val_log,
            epochs=epochs, batch_size=batch_size,
            num_blocks=config["num_blocks"],
            use_batch_norm=config["batch_norm"],
            dropout_rate=config["dropout"],
            patience=patience,
            verbose=0,
        )
        result_metrics = regression_metrics(y_val_price, pred)
        results.append({
            **config,
            "RMSE_log1p_Kaggle": result_metrics["RMSE_log1p_Kaggle"],
            "R2_price": result_metrics["R2_price"],
            "MAE_price": result_metrics["MAE_price"],
        })

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "ablation_results.csv", index=False, encoding="utf-8-sig")

    for metric, filename, title, ylabel in [
        ("RMSE_log1p_Kaggle", "ablation_rmse_log.png", "CNN Architecture Comparison - Log RMSE", "Log RMSE (lower is better)"),
        ("R2_price", "ablation_r2.png", "CNN Architecture Comparison - R-squared", "R² (higher is better)"),
    ]:
        plt.figure(figsize=(10, 5))
        plt.bar(df["experiment"], df[metric])
        plt.xticks(rotation=20, ha="right")
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(axis="y", alpha=0.2)
        plt.tight_layout()
        plt.savefig(out_dir / filename, dpi=180)
        plt.close()
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 1D CNN for Kaggle House Prices regression")
    parser.add_argument("--train", default="data/train.csv", help="مسیر train.csv")
    parser.add_argument("--test", default="data/test.csv", help="مسیر test.csv")
    parser.add_argument("--output", default="outputs", help="پوشه ذخیره خروجی‌ها")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=14)
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-ablation", action="store_true", help="رد کردن آزمایش مقایسه معماری‌ها")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", category=FutureWarning)
    set_seed(args.seed)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_path = Path(args.train)
    test_path = Path(args.test)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"فایل‌ها پیدا نشدند. مسیرهای فعلی: train={train_path.resolve()} و test={test_path.resolve()}. "
            "فایل‌های رسمی Kaggle را در پوشه data قرار دهید یا مسیرشان را با آرگومان مشخص کنید."
        )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    if TARGET not in train_df.columns:
        raise ValueError(f"ستون هدف {TARGET} در train.csv وجود ندارد.")
    if len(train_df) < 30:
        raise ValueError("تعداد ردیف‌های train.csv برای تقسیم آموزش/اعتبارسنجی کافی نیست.")

    y_all = pd.to_numeric(train_df[TARGET], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(y_all).all() or (y_all <= 0).any():
        raise ValueError("ستون SalePrice باید عددی، مثبت و بدون مقدار گمشده باشد.")

    train_ids = train_df[ID_COLUMN].copy() if ID_COLUMN in train_df.columns else pd.Series(np.arange(1, len(train_df) + 1), name=ID_COLUMN)
    test_ids = test_df[ID_COLUMN].copy() if ID_COLUMN in test_df.columns else pd.Series(np.arange(1, len(test_df) + 1), name=ID_COLUMN)

    X_all = train_df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    X_test_raw = test_df.drop(columns=[TARGET, ID_COLUMN], errors="ignore")
    # Guarantee identical feature columns/order and consistent dtypes, including all-null test columns.
    X_test_raw = X_test_raw.reindex(columns=X_all.columns)
    numeric_cols, categorical_cols = get_feature_column_groups(X_all)
    X_all = coerce_feature_dtypes(X_all, numeric_cols, categorical_cols)
    X_test_raw = coerce_feature_dtypes(X_test_raw, numeric_cols, categorical_cols)

    indices = np.arange(len(X_all))
    fit_idx, val_idx = train_test_split(
        indices, test_size=args.validation_size, random_state=args.seed, shuffle=True
    )
    X_fit_raw = X_all.iloc[fit_idx].copy()
    X_val_raw = X_all.iloc[val_idx].copy()
    y_fit = y_all[fit_idx]
    y_val = y_all[val_idx]
    y_fit_log = np.log1p(y_fit).astype(np.float32)
    y_val_log = np.log1p(y_val).astype(np.float32)

    preprocessor = make_preprocessor(X_fit_raw)
    X_fit = np.asarray(preprocessor.fit_transform(X_fit_raw), dtype=np.float32)
    X_val = np.asarray(preprocessor.transform(X_val_raw), dtype=np.float32)
    feature_names = preprocessor.get_feature_names_out().tolist()
    if X_fit.shape[1] < 2:
        raise ValueError("پس از پیش‌پردازش کمتر از دو ویژگی باقی مانده است.")

    print(f"Train rows: {len(fit_idx)} | Validation rows: {len(val_idx)} | Encoded features: {X_fit.shape[1]}")
    set_seed(args.seed)
    model, history, val_pred = train_one_model(
        X_fit, y_fit_log, X_val, y_val_log,
        epochs=args.epochs, batch_size=args.batch_size,
        num_blocks=3, use_batch_norm=True, dropout_rate=0.25,
        patience=args.patience, verbose=1,
    )
    hist_df = save_history(history, out_dir)
    best_epoch = int(hist_df.loc[hist_df["val_loss"].idxmin(), "epoch"])

    val_metrics = regression_metrics(y_val, val_pred)
    cls_metrics, cm, thresholds = price_band_metrics(y_fit, y_val, val_pred)
    save_confusion_matrix(cm, out_dir)
    save_prediction_plots(y_val, val_pred, out_dir)

    val_predictions = pd.DataFrame({
        "Id": train_ids.iloc[val_idx].to_numpy(),
        "SalePrice_actual": y_val,
        "SalePrice_predicted": val_pred,
        "residual_actual_minus_predicted": y_val - val_pred,
    })
    val_predictions.to_csv(out_dir / "validation_predictions.csv", index=False, encoding="utf-8-sig")
    all_metrics = {**val_metrics, **cls_metrics}
    (out_dir / "evaluation_metrics.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ablation_df = None
    if not args.skip_ablation:
        ablation_df = run_ablation(
            X_fit, y_fit_log, X_val, y_val_log, y_val,
            epochs=args.epochs, batch_size=args.batch_size,
            base_metrics=val_metrics, patience=args.patience, out_dir=out_dir, seed=args.seed,
        )

    # Refit preprocessing and the baseline architecture on every labeled row for Kaggle submission.
    print("Training final CNN on all labeled rows for Kaggle test predictions...")
    final_preprocessor = make_preprocessor(X_all)
    X_all_processed = np.asarray(final_preprocessor.fit_transform(X_all), dtype=np.float32)
    X_kaggle_test = np.asarray(final_preprocessor.transform(X_test_raw), dtype=np.float32)
    set_seed(args.seed)
    final_model = build_cnn(
        input_shape=(X_all_processed.shape[1], 1), num_blocks=3,
        use_batch_norm=True, dropout_rate=0.25,
    )
    final_model.fit(
        to_model_input(X_all_processed), np.log1p(y_all).astype(np.float32),
        epochs=max(1, best_epoch), batch_size=args.batch_size, shuffle=True, verbose=0,
    )
    test_pred_log = final_model.predict(to_model_input(X_kaggle_test), verbose=0).reshape(-1)
    test_pred = np.maximum(np.expm1(np.clip(test_pred_log, -10, 20)), 0.0)
    submission = pd.DataFrame({ID_COLUMN: test_ids.to_numpy(), TARGET: test_pred})
    submission.to_csv(out_dir / "submission.csv", index=False)

    model.save(out_dir / "final_cnn_model.keras")
    joblib.dump(final_preprocessor, out_dir / "preprocessor.joblib")
    (out_dir / "encoded_feature_names.json").write_text(
        json.dumps(feature_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame({"Id": test_ids.to_numpy(), "SalePrice_predicted": test_pred}).to_csv(
        out_dir / "kaggle_test_predictions.csv", index=False
    )

    numeric_cols, categorical_cols = get_feature_column_groups(X_all)
    write_report(
        out_dir, val_metrics, cls_metrics, thresholds,
        n_rows=len(train_df), n_features=X_fit.shape[1],
        n_numeric=len(numeric_cols), n_categorical=len(categorical_cols),
        best_epoch=best_epoch, ablation_df=ablation_df,
    )

    print("\nDone. Outputs saved to:", out_dir.resolve())
    print("Validation metrics:")
    for key, value in all_metrics.items():
        print(f"  {key}: {value:.6f}")
    print("Kaggle submission:", (out_dir / "submission.csv").resolve())
    print("Report:", (out_dir / "REPORT.md").resolve())


if __name__ == "__main__":
    main()
