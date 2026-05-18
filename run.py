"""Run Brownian composite detector with 4D features on ShallowWater_50km."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc

from brownian_4d import (
    CompositeBrownianDetector,
    load_experiment_data,
    define_experiments,
)


def evaluate(trajs_a, trajs_b):
    """Evaluate composite Brownian detector: hard decision (k=2, 3-class) + soft decision (AUC)."""
    if len(trajs_a) < 2 or len(trajs_b) < 1:
        return None

    detector = CompositeBrownianDetector()
    detector.fit(trajs_a)
    detector.fit_score_calibration(trajs_a)

    all_trajs = trajs_a + trajs_b
    labels = np.concatenate([np.ones(len(trajs_a)), np.zeros(len(trajs_b))])
    scores = np.array([detector.score_trajectory(t) for t in all_trajs])

    scores_a = scores[: len(trajs_a)]
    results = {}

    # Hard decision: k=2 (mean + 2*std threshold)
    thr2 = np.mean(scores_a) + 2 * np.std(scores_a)
    preds2 = (scores <= thr2).astype(int)
    results["hard_k2"] = {
        "acc": accuracy_score(labels, preds2),
        "f1": f1_score(labels, preds2, zero_division=0),
        "threshold": float(thr2),
    }

    # Three-class decision
    preds_3c = np.array(
        [1 if detector.predict(t)[0] == "Normal" else 0 for t in all_trajs]
    )
    results["hard_3c"] = {
        "acc": accuracy_score(labels, preds_3c),
        "f1": f1_score(labels, preds_3c, zero_division=0),
    }

    # Soft decision: normalized confidence + ROC AUC
    s_max = max(np.max(scores), 1.0)
    confidence = 1.0 - scores / s_max
    fpr, tpr, _ = roc_curve(labels, confidence)
    roc_auc = auc(fpr, tpr)

    best_f1, best_thr = 0, 0.5
    for thr in np.linspace(0, 1, 200):
        pred = (confidence > thr).astype(int)
        f1 = f1_score(labels, pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    results["soft"] = {
        "auc": roc_auc,
        "acc_opt": accuracy_score(labels, (confidence > best_thr).astype(int)),
        "f1_opt": best_f1,
        "thr_opt": best_thr,
        "conf_legit": float(np.mean(confidence[: len(trajs_a)])),
        "conf_attack": float(np.mean(confidence[len(trajs_a) :])),
    }

    return results


def main():
    print("=" * 70)
    print("Brownian Composite Detector + 4D Features")
    print("4D: amplitude mean/var, CFO mean/var")
    print("Detector: absolute position (KDE + convex hull) + increment (Gaussian)")
    print("=" * 70)

    exps = define_experiments()
    all_rows = []

    for ek, ec in exps.items():
        data = load_experiment_data(ec)
        if data is None:
            print(f"\n  [{ek}] Skipped (no data)")
            continue

        X_4d, y_4d = data["X_4d"], data["y_4d"]
        trajs_a = [X_4d[y_4d == 1][i : i + 1] for i in range(int(np.sum(y_4d == 1)))]
        trajs_b = [X_4d[y_4d == 0][i : i + 1] for i in range(int(np.sum(y_4d == 0)))]

        r = evaluate(trajs_a, trajs_b)
        if r is None:
            print(f"\n  [{ek}] Skipped (insufficient data)")
            continue

        s = r["soft"]
        print(f"\n  [{ek}] {data['label_a']} vs {data['label_b']}")
        print(f"    Hard : k=2 Acc={r['hard_k2']['acc']:.4f},  3c Acc={r['hard_3c']['acc']:.4f}")
        print(f"    Soft : AUC={s['auc']:.4f},  Acc_opt={s['acc_opt']:.4f},  F1_opt={s['f1_opt']:.4f}")
        print(f"    Conf : legit={s['conf_legit']:.3f},  attack={s['conf_attack']:.3f}")

        all_rows.append(
            {
                "exp": ek,
                "label": f"{data['label_a']} vs {data['label_b']}",
                "acc_k2": r["hard_k2"]["acc"],
                "acc_3c": r["hard_3c"]["acc"],
                **r["soft"],
            }
        )

    # Summary table
    if all_rows:
        print("\n" + "=" * 80)
        print("Summary")
        print("=" * 80)
        print(
            f"{'Exp':>4}  {'k=2 Acc':>8}  {'3c Acc':>8}  {'AUC':>8}  "
            f"{'Acc_opt':>8}  {'F1_opt':>8}  {'Conf_L':>7}  {'Conf_A':>7}"
        )
        print("-" * 80)
        for row in all_rows:
            print(
                f"{row['exp']:>4}  {row['acc_k2']:.4f}    {row['acc_3c']:.4f}    "
                f"{row['auc']:.4f}    {row['acc_opt']:.4f}    {row['f1_opt']:.4f}    "
                f"{row['conf_legit']:.3f}   {row['conf_attack']:.3f}"
            )
        aucs = [r["auc"] for r in all_rows]
        print(f"\n  Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
