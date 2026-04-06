import numpy as np
from sklearn.metrics import confusion_matrix
# -------------------------
# Threshold auto-tuning for 3 class classification
# -------------------------


class ThresholdEstimator:
    def __init__(self, model, nomov_probs, nomov_labels, noMov_ratio=0.0, threshold_sweep_steps=61, threshold_sweep_objective="macro_f1"):
        self.model = model
        self.nomov_probs = nomov_probs
        self.nomov_labels = nomov_labels
        self.noMov_ratio = noMov_ratio
        self.threshold_sweep_steps = threshold_sweep_steps
        self.threshold_sweep_objective = threshold_sweep_objective
        self.threshold_tuning_strategy = "prior_quantile" if noMov_ratio > 0 else "sweep"

    def _predict_open_set(self, prob, low, high):
        if prob <= low:
            return 0
        if prob >= high:
            return 1
        return 2

    def _compute_metrics_for_thresholds(self, low, high):
        preds = [self._predict_open_set(prob, low, high) for prob in self.nomov_probs]
        cm = confusion_matrix(self.nomov_labels, preds, labels=[0, 1, 2])

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

    def estimate_thresholds(self):
        if len(self.nomov_probs) == 0:
            return 0.0, 1.0

        if self.threshold_tuning_strategy == "prior_quantile":
            tail_ratio = max(0.0, min(0.49, (1.0 - self.noMov_ratio) / 2.0))
            auto_low_threshold = float(np.quantile(self.nomov_probs, tail_ratio))
            auto_high_threshold = float(np.quantile(self.nomov_probs, 1.0 - tail_ratio))
            print(
                f"Threshold auto-tune (prior_quantile): noMov_ratio={self.noMov_ratio:.4f}, "
                f"tail_ratio={tail_ratio:.4f}"
            )
            return auto_low_threshold, auto_high_threshold

        probs_np_for_grid = np.array(self.nomov_probs, dtype=float)
        prob_min = float(np.quantile(probs_np_for_grid, 0.01))
        prob_max = float(np.quantile(probs_np_for_grid, 0.99))
        if prob_max <= prob_min:
            prob_min = float(np.min(probs_np_for_grid))
            prob_max = float(np.max(probs_np_for_grid))

        grid = np.linspace(prob_min, prob_max, self.threshold_sweep_steps)
        best_score = -1.0
        best_balanced = -1.0
        best_low = float(grid[0])
        best_high = float(grid[-1])

        for low in grid:
            for high in grid:
                if high <= low:
                    continue

                macro_f1_candidate, balanced_candidate = self._compute_metrics_for_thresholds(float(low), float(high))
                candidate_score = macro_f1_candidate if self.threshold_sweep_objective == "macro_f1" else balanced_candidate

                if (candidate_score > best_score) or (
                    abs(candidate_score - best_score) <= 1e-12 and balanced_candidate > best_balanced
                ):
                    best_score = candidate_score
                    best_balanced = balanced_candidate
                    best_low = float(low)
                    best_high = float(high)

        print(
            f"\nThreshold auto-tune (sweep): objective={self.threshold_sweep_objective}, "
            f"best_score={best_score:.4f}, best_balanced_acc={best_balanced:.4f}, "
            f"best_low={best_low:.4f}, best_high={best_high:.4f}"
        )
        return best_low, best_high
