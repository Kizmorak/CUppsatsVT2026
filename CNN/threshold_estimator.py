import numpy as np
from sklearn.metrics import confusion_matrix
import torch
import model_maker
# -------------------------
# Threshold auto-tuning for 3 class classification
# -------------------------


class ThresholdEstimator:
    def __init__(self, model, nomov_probs, nomov_labels, expected_noMov_ratio=0.0, threshold_sweep_steps=20, threshold_sweep_objective="macro_f1"):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nomov_probs = nomov_probs
        self.nomov_labels = nomov_labels
        self.expected_noMov_ratio = expected_noMov_ratio
        self.threshold_sweep_steps = threshold_sweep_steps
        self.threshold_sweep_objective = threshold_sweep_objective
        self.threshold_tuning_strategy = "sweep" if expected_noMov_ratio > 0 else "prior_quantile"

    # -------------------------
    # Threshold auto-tuning for 3 class classification
    # -------------------------

        # Compute metrics for given thresholds (used for auto-tuning)
        def compute_metrics_for_thresholds(low, high):

            preds = [model_maker.predict_open_set(prob, low, high) for prob in nomov_probs]
            cm = confusion_matrix(nomov_labels, preds, labels=[0, 1, 2])

            tp_local = np.diag(cm).astype(float)
            support_local = cm.sum(axis=1).astype(float)
            pred_count_local = cm.sum(axis=0).astype(float)

            recall_local = np.divide(tp_local, support_local, out=np.zeros_like(tp_local), where=support_local > 0)
            precision_local = np.divide(tp_local, pred_count_local, out=np.zeros_like(tp_local), where=pred_count_local > 0)
            f1_local = np.divide(
                2 * precision_local * recall_local,
                precision_local + recall_local,
                out=np.zeros_like(tp_local),
                where=(precision_local + recall_local) > 0,
            )

            macro_f1_local = float(np.mean(f1_local))
            balanced_accuracy_local = float(np.mean(recall_local))
            return macro_f1_local, balanced_accuracy_local

        # Auto-tune NOMOV thresholds based on backtesting set performance if enabled
        if len(nomov_probs) > 0:
            if self.threshold_tuning_strategy == "prior_quantile":
                tail_ratio = max(0.0, min(0.49, (1.0 - expected_noMov_ratio) / 2.0))
                auto_low_threshold = float(np.quantile(nomov_probs, tail_ratio))
                auto_high_threshold = float(np.quantile(nomov_probs, 1.0 - tail_ratio))
                print(
                    f"Threshold auto-tune (prior_quantile): expected_noMov_ratio={self.expected_noMov_ratio:.4f}, "
                    f"tail_ratio={tail_ratio:.4f}"
                )
            elif self.threshold_tuning_strategy == "sweep":
                probs_np_for_grid = np.array(nomov_probs, dtype=float)
                prob_min = float(np.quantile(probs_np_for_grid, 0.01))
                prob_max = float(np.quantile(probs_np_for_grid, 0.99))

                # Guard against degenerate probability distributions.
                if prob_max <= prob_min:
                    prob_min = float(np.min(probs_np_for_grid))
                    prob_max = float(np.max(probs_np_for_grid))

                grid = np.linspace(prob_min, prob_max, self.threshold_sweep_steps)

                best_score = -1.0
                best_balanced = -1.0
                best_low = auto_low_threshold
                best_high = auto_high_threshold

                for low in grid:
                    for high in grid:
                        if high <= low:
                            continue

                        macro_f1_candidate, balanced_candidate = compute_metrics_for_thresholds(float(low), float(high))
                        candidate_score = macro_f1_candidate if self.threshold_sweep_objective == "macro_f1" else balanced_candidate

                        # Tie-break with balanced accuracy to avoid unstable picks.
                        if (candidate_score > best_score) or (
                            abs(candidate_score - best_score) <= 1e-12 and balanced_candidate > best_balanced
                        ):
                            best_score = candidate_score
                            best_balanced = balanced_candidate
                            best_low = float(low)
                            best_high = float(high)

                auto_low_threshold = best_low
                auto_high_threshold = best_high
                print(
                    f"\nThreshold auto-tune (sweep): objective={self.threshold_sweep_objective}, "
                    f"best_score={best_score:.4f}, best_balanced_acc={best_balanced:.4f}, "
                    f"best_low={auto_low_threshold:.4f}, best_high={auto_high_threshold:.4f}"
                )
                return auto_low_threshold, auto_high_threshold
            else:
                raise ValueError(
                    f"Invalid threshold_tuning_strategy: {self.threshold_tuning_strategy}. "
                    f"Use 'prior_quantile' or 'sweep'."
                )
