
import os
import json
import math
import itertools
import random
import time
import gc
import warnings
from datetime import datetime, timedelta
from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False
    print("Warning: VADER not available. Install with: pip install vaderSentiment")

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks, regularizers
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Install with: pip install tensorflow")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
if TENSORFLOW_AVAILABLE:
    tf.random.set_seed(SEED)

STRESS_KEYWORDS = [
    "stress", "anxiety", "depressed", "depression", "tired", "hopeless", "alone", "worthless",
    "panic", "overwhelmed", "can't cope", "exhausted", "helpless", "sad", "numb", "empty",
    "hurt", "broken", "struggle", "alone", "suicidal thought", "give up"
]

NEUTRAL_WORDS = [
    "work", "coffee", "movie", "football", "game", "music", "lecture", "project", "lunch", "travel",
    "happy", "love", "family", "friend", "weather", "tech", "code", "study", "shopping", "birthday"
]

POSITIVE_PHRASES = [
    "Had a great day!", "Feeling good today.", "Loved the movie I watched.", "Excited for the weekend.",
    "Grateful for my friends.", "Learning new things is fun.", "Productive day at work."
]

NEGATIVE_PHRASES = [
    "This day has been rough.", "Feeling down.", "Really tired of everything.", "Not my day.",
    "Things are hard right now.", "I can't focus.", "Feeling low and drained."
]

TWEET_TEMPLATES = [
    "{prefix} {middle} {suffix}",
    "{middle} {suffix}",
    "{prefix} {middle}",
    "{middle}",
    "{prefix} {suffix}",
    "{suffix}"
]


def rand_date_within(days_back=730):
    now = datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow()
    delta = timedelta(days=random.randint(0, days_back), seconds=random.randint(0, 86400))
    return (now - delta).isoformat() + "Z"


def make_tweet_text(is_at_risk):
    parts = []
    stress_prob = 0.35 if is_at_risk else 0.06
    negative_prob = 0.30 if is_at_risk else 0.08
    positive_prob = 0.08 if is_at_risk else 0.25
    neutral_prob = 0.30 if not is_at_risk else 0.20

    if random.random() < 0.4:
        parts.append(random.choice(["FYI", "Update:", "Note:", ""]))
    
    mid_roll = random.random()
    if mid_roll < stress_prob:
        kw = random.choice(STRESS_KEYWORDS)
        addon = random.choice(NEGATIVE_PHRASES) if random.random() < 0.6 else random.choice(NEUTRAL_WORDS)
        parts.append(f"{kw} {addon}")
    elif mid_roll < stress_prob + negative_prob:
        parts.append(random.choice(NEGATIVE_PHRASES))
    elif mid_roll < stress_prob + negative_prob + positive_prob:
        parts.append(random.choice(POSITIVE_PHRASES))
    else:
        parts.append(random.choice(NEUTRAL_WORDS) + " " + random.choice(NEUTRAL_WORDS))
    
    if random.random() < 0.25:
        parts.append(random.choice(["#life", "#mood", "#work", "#tired", ":-(", ":)"]))
    
    if random.random() < 0.05:
        parts.append(random.choice(["Thanks!", "Totally", "Agreed", "Same here"]))
    
    text = " ".join([p for p in parts if p]).strip()
    if len(text) > 280:
        text = text[:277] + "..."
    return text


def generate_synthetic_dataset(n_users=8000, min_tweets_per_user=20, max_tweets_per_user=150,
                               self_harm_prevalence=0.15, output_dir="."):
    users = []
    for uid in range(1, n_users + 1):
        self_harm_flag = np.random.choice([0, 1], p=[1 - self_harm_prevalence, self_harm_prevalence])
        signup_days_ago = random.randint(30, 3650)
        now = datetime.now(datetime.UTC) if hasattr(datetime, 'UTC') else datetime.utcnow()
        signup_date = (now - timedelta(days=signup_days_ago)).date().isoformat()
        followers = max(0, int(np.random.exponential(scale=50)))
        total_tweets = random.randint(50, 5000)
        users.append({
            "user_id": uid,
            "self_harm_flag": int(self_harm_flag),
            "signup_date": signup_date,
            "followers": followers,
            "total_tweets_estimate": total_tweets
        })

    users_df = pd.DataFrame(users)

    tweets_records = []
    tweet_id_counter = 1

    for _, u in users_df.iterrows():
        uid = int(u.user_id)
        is_at_risk = bool(u.self_harm_flag)
        n_tweets = np.random.randint(min_tweets_per_user, max_tweets_per_user + 1)
        for i in range(n_tweets):
            text = make_tweet_text(is_at_risk)
            created_at = rand_date_within(days_back=730)
            is_reply = random.random() < 0.12
            is_retweet = random.random() < 0.08
            like_count = int(np.random.poisson(2))
            retweet_count = int(np.random.poisson(0.5))
            tweets_records.append({
                "tweet_id": tweet_id_counter,
                "user_id": uid,
                "created_at": created_at,
                "text": text,
                "is_reply": int(is_reply),
                "is_retweet": int(is_retweet),
                "like_count": like_count,
                "retweet_count": retweet_count
            })
            tweet_id_counter += 1

    tweets_df = pd.DataFrame(tweets_records)

    users_path = os.path.join(output_dir, f"synthetic_users_{n_users}.csv")
    tweets_path = os.path.join(output_dir, f"synthetic_tweets_{n_users}_users.csv")
    users_df.to_csv(users_path, index=False)
    tweets_df.to_csv(tweets_path, index=False)

    print(f"Saved files:\n  {users_path}\n  {tweets_path}")
    print(f"Users: {len(users_df)}, Tweets: {len(tweets_df)}")

    return users_df, tweets_df, users_path, tweets_path



def compute_sentiment_scores(tweets_df):
    if not VADER_AVAILABLE:
        raise ImportError("VADER sentiment analyzer not available. Install with: pip install vaderSentiment")
    
    analyzer = SentimentIntensityAnalyzer()

    def safe_sentiment(text):
        try:
            return analyzer.polarity_scores(str(text))["compound"]
        except:
            return 0.0

    tweets_df["sentiment_score"] = tweets_df["text"].apply(safe_sentiment)
    return tweets_df


def compute_avg_sentiment(tweets_df):
    """Compute average sentiment per user"""
    if "sentiment_score" not in tweets_df.columns:
        tweets_df = compute_sentiment_scores(tweets_df)
    
    avg_sentiment = (
        tweets_df.groupby("user_id")["sentiment_score"]
        .mean()
        .reset_index()
    )
    avg_sentiment.columns = ["user_id", "avg_sentiment"]
    return avg_sentiment


def compute_neg_tweet_ratio(tweets_df):
    if "sentiment_score" not in tweets_df.columns:
        tweets_df = compute_sentiment_scores(tweets_df)
    
    tweets_df["is_negative"] = tweets_df["sentiment_score"] <= -0.05
    neg_tweet_ratio = (
        tweets_df.groupby("user_id")["is_negative"]
        .mean()
        .reset_index()
    )
    neg_tweet_ratio.columns = ["user_id", "neg_tweet_ratio"]
    return neg_tweet_ratio


def compute_stress_keywords_freq(tweets_df):
    stress_keywords = [
        "stress", "tired", "depressed", "depression", "anxiety", "worthless",
        "panic", "alone", "sad", "hopeless", "fail", "overwhelmed",
        "empty", "hurt", "struggle", "helpless", "broken", "give up", "suicidal"
    ]
    stress_keywords = [w.lower() for w in stress_keywords]

    def count_stress_words(text):
        text = str(text).lower()
        return sum(text.count(w) for w in stress_keywords)

    tweets_df["stress_word_count"] = tweets_df["text"].apply(count_stress_words)
    tweets_df["has_stress_word"] = tweets_df["stress_word_count"] > 0

    stress_keywords_total = (
        tweets_df.groupby("user_id")["stress_word_count"]
        .sum()
        .reset_index()
    )
    stress_keywords_total.columns = ["user_id", "stress_keywords_freq_total"]

    stress_keywords_tweetcount = (
        tweets_df.groupby("user_id")["has_stress_word"]
        .sum()
        .reset_index()
    )
    stress_keywords_tweetcount.columns = ["user_id", "stress_keywords_freq_tweets"]

    return stress_keywords_total, stress_keywords_tweetcount


def compute_past_month_activity(tweets_df):
    """Compute number of tweets posted in the last 30 days per user"""
    tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], errors="coerce")
    
    # Find the latest tweet timestamp in the dataset (use as "today")
    latest_time = tweets_df["created_at"].max()
    cutoff_date = latest_time - timedelta(days=30)
    
    # Mark tweets posted in the last 30 days
    tweets_df["is_recent"] = tweets_df["created_at"] >= cutoff_date
    
    # Count recent tweets per user
    past_month_activity = (
        tweets_df.groupby("user_id")["is_recent"]
        .sum()
        .reset_index()
    )
    past_month_activity.columns = ["user_id", "past_month_activity"]
    return past_month_activity


def create_final_ml_dataset(users_df, tweets_df, output_path=None):
    tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"], errors="coerce")

    avg_sentiment = compute_avg_sentiment(tweets_df)

    neg_tweet_ratio = compute_neg_tweet_ratio(tweets_df)

    stress_keywords_total, stress_keywords_tweetcount = compute_stress_keywords_freq(tweets_df)

    past_month_activity = compute_past_month_activity(tweets_df)

    frames = [
        users_df,
        avg_sentiment,
        neg_tweet_ratio,
        stress_keywords_total,
        stress_keywords_tweetcount,
        past_month_activity
    ]

    final_df = reduce(lambda left, right: pd.merge(left, right, on="user_id", how="left"), frames)

    final_df["avg_sentiment"] = final_df["avg_sentiment"].fillna(0.0)
    final_df["neg_tweet_ratio"] = final_df["neg_tweet_ratio"].fillna(0.0)
    final_df["stress_keywords_freq_total"] = final_df["stress_keywords_freq_total"].fillna(0).astype(int)
    final_df["stress_keywords_freq_tweets"] = final_df["stress_keywords_freq_tweets"].fillna(0).astype(int)
    final_df["past_month_activity"] = final_df["past_month_activity"].fillna(0).astype(int)

    final_df["tweet_activity_ratio"] = (
        final_df["past_month_activity"] / final_df["total_tweets_estimate"].replace(0, np.nan)
    )
    final_df["tweet_activity_ratio"] = final_df["tweet_activity_ratio"].fillna(0.0)

    if output_path:
        final_df.to_csv(output_path, index=False)
        print(f"Saved final ML dataset to: {output_path}")

    print(f"\nFinal dataset shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")
    print(f"\nTarget balance (self_harm_flag):")
    print(final_df["self_harm_flag"].value_counts(dropna=False))

    return final_df



def train_traditional_ml_models(df, feature_cols=None, target="self_harm_flag", test_size=0.25):
    if feature_cols is None:
        feature_cols = [
            "avg_sentiment",
            "neg_tweet_ratio",
            "stress_keywords_freq_total",
            "stress_keywords_freq_tweets",
            "past_month_activity",
            "followers",
            "tweet_activity_ratio"
        ]

    X = df[feature_cols]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    print("\n================ LOGISTIC REGRESSION ================")
    lr = LogisticRegression(max_iter=3000)
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_acc = accuracy_score(y_test, lr_pred)
    results["Logistic Regression"] = (lr, lr_pred, lr_acc)
    print(f"Accuracy: {lr_acc:.4f}")
    print(classification_report(y_test, lr_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, lr_pred))

    print("\n====================== SVM (RBF) =====================")
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X_train_scaled, y_train)
    svm_pred = svm.predict(X_test_scaled)
    svm_acc = accuracy_score(y_test, svm_pred)
    results["SVM"] = (svm, svm_pred, svm_acc)
    print(f"Accuracy: {svm_acc:.4f}")
    print(classification_report(y_test, svm_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, svm_pred))

    print("\n==================== RANDOM FOREST ===================")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        random_state=SEED,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    results["Random Forest"] = (rf, rf_pred, rf_acc)
    print(f"Accuracy: {rf_acc:.4f}")
    print(classification_report(y_test, rf_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

    if XGBOOST_AVAILABLE:
        print("\n======================== XGBOOST =====================")
        xgb_model = xgb.XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.07,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=SEED
        )
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        results["XGBoost"] = (xgb_model, xgb_pred, xgb_acc)
        print(f"Accuracy: {xgb_acc:.4f}")
        print(classification_report(y_test, xgb_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, xgb_pred))

    print("\n\n================== ACCURACY COMPARISON ==================")
    for name, (_, _, acc) in results.items():
        print(f"{name:25s}: {acc:.4f}")

    return results, X_test, y_test


if TENSORFLOW_AVAILABLE:
    def is_id_like(n):
        n = str(n).lower().strip()
        return n == "id" or n.endswith("_id") or n.startswith("id_")

    def add_datetime_parts(df, target):
        df = df.copy()
        dtcols = []
        for c in df.columns:
            if c == target:
                continue
            if df[c].dtype == object:
                try:
                    df[c] = pd.to_datetime(df[c], errors="raise")
                    dtcols.append(c)
                except:
                    pass
            elif np.issubdtype(df[c].dtype, np.datetime64):
                dtcols.append(c)
        for c in dtcols:
            df[c + "_year"] = df[c].dt.year
            df[c + "_month"] = df[c].dt.month
            df[c + "_day"] = df[c].dt.day
            df[c + "_hour"] = df[c].dt.hour
            df[c + "_dow"] = df[c].dt.dayofweek
            df.drop(columns=[c], inplace=True)
        return df

    def auto_feature_crosses(Xdf, max_pairs=20):
        nums = Xdf.select_dtypes(include=[np.number]).columns.tolist()
        if len(nums) < 2:
            return Xdf, []
        vari = Xdf[nums].var().sort_values(ascending=False)
        tops = vari.index[:max_pairs + 2]
        pairs = list(itertools.combinations(tops, 2))[:max_pairs]
        for a, b in pairs:
            Xdf[f"{a}*{b}"] = Xdf[a] * Xdf[b]
        return Xdf, pairs

    def se_block(x, ratio=8, name="se"):
        channels = int(x.shape[-1])
        s = layers.Reshape((1, channels), name=f"{name}_reshape_in")(x)
        s = layers.GlobalAveragePooling1D(name=f"{name}_gap")(s)
        s = layers.Dense(max(1, channels // ratio), activation="relu", name=f"{name}_fc1")(s)
        s = layers.Dense(channels, activation="sigmoid", name=f"{name}_fc2")(s)
        s = layers.Reshape((channels,), name=f"{name}_reshape_out")(s)
        return layers.Multiply(name=f"{name}_scale")([x, s])

    def make_cosine_lr_fn(initial_lr, decay_epochs=40, alpha=0.0002):
        def lr_fn(epoch):
            t = min(epoch, decay_epochs)
            cos_val = 0.5 * (1 + math.cos(math.pi * t / decay_epochs))
            lr = initial_lr * (alpha + (1 - alpha) * cos_val)
            return lr
        return lr_fn

    def build_widedeep(input_dim, num_classes, hidden_sizes=[512, 256, 128], 
                       dropout_rate=0.10, l2_reg=1e-5, use_bn=True):
        inputs = keras.Input(shape=(input_dim,))
        wide_logits = layers.Dense(num_classes, kernel_regularizer=regularizers.l2(l2_reg))(inputs)

        x = inputs
        for i, h in enumerate(hidden_sizes, 1):
            x = layers.Dense(h, kernel_initializer="he_normal",
                             kernel_regularizer=regularizers.l2(l2_reg))(x)
            if use_bn:
                x = layers.BatchNormalization()(x)
            x = layers.Activation("swish")(x)
            x = layers.Dropout(dropout_rate)(x)
            x = se_block(x, ratio=8, name=f"se_{i}")

        deep_logits = layers.Dense(num_classes, kernel_regularizer=regularizers.l2(l2_reg))(x)
        combined = layers.Add()([wide_logits, deep_logits])
        out = layers.Activation("softmax", dtype="float32")(combined)
        return keras.Model(inputs, out)

    def train_deep_learning_model(df, target="self_harm_flag", val_size=0.20,
                                   hidden_sizes=[512, 256, 128], dropout_rate=0.10,
                                   batch_size=256, initial_lr=1e-3, l2_reg=1e-5,
                                   epochs=100, use_mixed_precision=False,
                                   save_dir="./models", model_name="widedeep_model"):
        print("TF version:", tf.__version__)
        print("GPUs:", tf.config.list_physical_devices("GPU"))

        from tensorflow.keras import mixed_precision
        if use_mixed_precision:
            try:
                mixed_precision.set_global_policy("mixed_float16")
                print("Mixed precision enabled.")
            except Exception as e:
                print("Could not enable mixed precision:", e)
        else:
            try:
                mixed_precision.set_global_policy("float32")
                print("Using float32 policy (stable numerics).")
            except:
                pass

        strategy = tf.distribute.MirroredStrategy()
        print("Strategy replicas:", strategy.num_replicas_in_sync)

        df_processed = df.copy()
        
        df_processed = df_processed.drop(
            columns=[c for c in df_processed.columns if is_id_like(c)], errors="ignore"
        )

        nunique = df_processed.nunique(dropna=False)
        df_processed = df_processed.drop(columns=nunique[nunique <= 1].index.tolist())

        df_processed = add_datetime_parts(df_processed, target)

        if target not in df_processed.columns:
            raise ValueError(f"Target column '{target}' not found in dataset")

        y_raw = df_processed[target].values
        classes = np.unique(y_raw)
        class_map = {str(c): i for i, c in enumerate(classes)}
        y = np.array([class_map[str(v)] for v in y_raw], dtype=int)

        X = df_processed.drop(columns=[target]).copy()

        for c in X.columns:
            if X[c].dtype == object:
                _, inv = np.unique(X[c].astype(str), return_inverse=True)
                X[c] = inv.astype(np.float32)

        X, used_pairs = auto_feature_crosses(X, max_pairs=20)
        print("Added feature crosses:", used_pairs)

        X_temp, X_test, y_temp, y_test = train_test_split(
            X.values, y, test_size=val_size, random_state=SEED, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=SEED, stratify=y_temp
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train.astype(np.float32))
        X_val = scaler.transform(X_val.astype(np.float32))
        X_test = scaler.transform(X_test.astype(np.float32))

        input_dim = X_train.shape[1]
        num_classes = len(classes)
        print(f"Input dim: {input_dim}, Num classes: {num_classes}")
        print(f"Train/Val/Test shapes: {X_train.shape} {X_val.shape} {X_test.shape}")

        os.makedirs(save_dir, exist_ok=True)

        with strategy.scope():
            model = build_widedeep(input_dim, num_classes, hidden_sizes, dropout_rate, l2_reg)
            opt = keras.optimizers.Adam(learning_rate=initial_lr)
            model.compile(
                optimizer=opt,
                loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
                metrics=["accuracy"]
            )

        y_train_oh = keras.utils.to_categorical(y_train, num_classes)
        y_val_oh = keras.utils.to_categorical(y_val, num_classes)

        lr_schedule = make_cosine_lr_fn(initial_lr)
        ckpt_path = os.path.join(save_dir, f"{model_name}_best.keras")
        cb_list = [
            callbacks.ModelCheckpoint(
                ckpt_path, save_best_only=True, monitor="val_accuracy", mode="max", verbose=1
            ),
            callbacks.EarlyStopping(
                monitor="val_accuracy", patience=12, restore_best_weights=True, verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=6, verbose=1, min_lr=1e-6
            ),
            callbacks.LearningRateScheduler(lambda ep: lr_schedule(ep), verbose=0)
        ]

        history = model.fit(
            X_train, y_train_oh,
            validation_data=(X_val, y_val_oh),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=cb_list,
            verbose=2
        )

        try:
            model.load_weights(ckpt_path)
        except Exception:
            pass

        preds = model.predict(X_test, batch_size=1024).argmax(axis=1)
        test_accuracy = accuracy_score(y_test, preds)
        print(f"\nTest accuracy: {test_accuracy:.6f}")

        return model, history, test_accuracy



def main():
    print("=" * 80)
    print("SENTIMENT ANALYSIS FOR SELF-HARM DETECTION")
    print("=" * 80)

    N_USERS = 8000
    OUTPUT_DIR = "."

    print("\n[Step 1] Generating synthetic dataset...")
    users_df, tweets_df, users_path, tweets_path = generate_synthetic_dataset(
        n_users=N_USERS, output_dir=OUTPUT_DIR
    )

    print("\n[Step 2] Creating final ML dataset with features...")
    final_df = create_final_ml_dataset(
        users_df, tweets_df,
        output_path=os.path.join(OUTPUT_DIR, f"final_ml_dataset_{N_USERS}.csv")
    )

    print("\n[Step 3] Training traditional ML models...")
    ml_results, X_test, y_test = train_traditional_ml_models(final_df)

    if TENSORFLOW_AVAILABLE:
        print("\n[Step 4] Training deep learning model...")
        try:
            dl_model, history, dl_acc = train_deep_learning_model(
                final_df, epochs=50, save_dir="./models"
            )
            print(f"Deep Learning Model Accuracy: {dl_acc:.6f}")
        except Exception as e:
            print(f"Error training deep learning model: {e}")
    else:
        print("\n[Step 4] Skipping deep learning (TensorFlow not available)")

    print("\n" + "=" * 80)
    print("EXECUTION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

