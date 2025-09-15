# src/models/train_transformer.py
import os, json
import numpy as np, pandas as pd
from datetime import datetime
from datasets import Dataset, Features, ClassLabel, Value
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from src.common.paths import pjoin, load_config
from src.common.logging import get_logger

log = get_logger("models.transformer")


def f1_macro(eval_pred):
    from sklearn.metrics import f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"f1": f1_score(labels, preds, average="macro")}


def make_training_args(ts, epochs, batch):
    """
    Construye TrainingArguments compatible con versiones nuevas y viejas.
    Si tu transformers es viejo, cae a un set mínimo de argumentos.
    """
    try:
        # API moderna
        return (
            TrainingArguments(
                output_dir=f"out/{ts}",
                evaluation_strategy="epoch",
                save_strategy="epoch",
                num_train_epochs=epochs,
                per_device_train_batch_size=batch,
                per_device_eval_batch_size=max(8, batch),
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                weight_decay=0.01,
                learning_rate=2e-5,
                save_total_limit=2,
                logging_steps=50,
                report_to="none",
                seed=42,
            ),
            True,
            "new_api",
        )
    except TypeError:
        # API vieja (sin evaluation_strategy / save_strategy / metric_for_best_model)
        try:
            return (
                TrainingArguments(
                    output_dir=f"out/{ts}",
                    do_eval=True,
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch,
                    per_device_eval_batch_size=max(8, batch),
                    weight_decay=0.01,
                    learning_rate=2e-5,
                    save_total_limit=2,
                    logging_steps=50,
                    seed=42,
                    save_steps=500,
                    eval_steps=500,
                ),
                False,
                "old_api",
            )
        except TypeError:
            # Mínimo-mínimo
            return (
                TrainingArguments(
                    output_dir=f"out/{ts}",
                    num_train_epochs=epochs,
                    per_device_train_batch_size=batch,
                    per_device_eval_batch_size=max(8, batch),
                    weight_decay=0.01,
                    learning_rate=2e-5,
                    logging_steps=50,
                    seed=42,
                ),
                False,
                "minimal_api",
            )


def run():
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    cfg = load_config()
    model_id = cfg["model"].get(
        "transformer_model", "dccuchile/bert-base-spanish-wwm-cased"
    )
    max_len = int(cfg["model"].get("max_length", 256))
    epochs = int(cfg["model"].get("epochs", 3))
    batch = int(cfg["model"].get("batch_size", 16))

    # === Datos ===
    df = pd.read_csv(pjoin("data", "interim", "limpio_final.csv")).dropna(
        subset=["texto_raw", "sentimiento"]
    )
    if df.empty:
        raise RuntimeError("limpio_final.csv está vacío o sin columnas necesarias.")

    le = LabelEncoder()
    y = le.fit_transform(df["sentimiento"])
    class_names = list(le.classes_)
    log.info("Clases: %s", class_names)

    ds_full = Dataset.from_dict(
        {"text": df["texto_raw"].astype(str).tolist(), "label": y.astype(int).tolist()}
    )
    features = Features(
        {"text": Value("string"), "label": ClassLabel(names=class_names)}
    )
    ds_full = ds_full.cast(features)

    # Split estratificado con fallback
    try:
        ds = ds_full.train_test_split(
            test_size=0.2, stratify_by_column="label", seed=42
        )
        split_note = "stratified(ClassLabel)"
    except Exception as e:
        log.warning("Fallo split estratificado: %s. Uso split aleatorio.", e)
        ds = ds_full.train_test_split(test_size=0.2, seed=42)
        split_note = "random(fallback)"

    # Tokenización
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    def tok_fn(batch):
        return tok(batch["text"], padding=True, truncation=True, max_length=max_len)

    ds = ds.map(tok_fn, batched=True)

    # Modelo
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, num_labels=len(class_names)
    )

    # TrainingArguments compatibles
    targs, can_early_stop, api_note = make_training_args(ts, epochs, batch)
    callbacks = (
        [EarlyStoppingCallback(early_stopping_patience=2)] if can_early_stop else []
    )

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=f1_macro,
        tokenizer=tok,
        callbacks=callbacks,
    )

    log.info("Entrenando… API=%s | split=%s", api_note, split_note)
    trainer.train()

    # Evaluación
    logits = trainer.predict(ds["test"]).predictions
    y_pred = logits.argmax(axis=1)
    y_true = np.array(ds["test"]["label"])

    from sklearn.metrics import classification_report, confusion_matrix

    rep = classification_report(
        y_true, y_pred, target_names=class_names, digits=3, output_dict=True
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    os.makedirs("reports", exist_ok=True)
    pd.DataFrame(rep).to_csv(pjoin("reports", f"transformer_report_{ts}.csv"))
    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        pjoin("reports", f"transformer_confusion_{ts}.csv")
    )

    pred_df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    pred_df["y_true_lbl"] = [class_names[i] for i in y_true]
    pred_df["y_pred_lbl"] = [class_names[i] for i in y_pred]
    pred_df.to_csv(pjoin("reports", f"transformer_test_preds_{ts}.csv"), index=False)

    meta = {
        "model_id": model_id,
        "labels": class_names,
        "max_length": max_len,
        "split": split_note,
        "api": api_note,
    }
    with open(
        pjoin("reports", f"transformer_meta_{ts}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    log.info("Listo. Reportes en reports/. Split=%s | API=%s", split_note, api_note)
    print("Macro-F1:", round(rep["macro avg"]["f1-score"], 3))
    print("Split:", split_note, "| API:", api_note)


if __name__ == "__main__":
    run()
