"""Simplified honest causal forest for CATE estimation.

Educational implementation inspired by Athey & Wager (2018):
- Trees split to maximize treatment-effect heterogeneity
- Honest estimation: separate samples for structure vs leaf effects
- Forest averages leaf-level CATEs

Not a production econml substitute — designed for clarity and synthetic demos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.utils import check_random_state


@dataclass
class _Leaf:
    index: np.ndarray
    cate: float
    n: int


@dataclass
class _Tree:
    feature: int | None = None
    threshold: float | None = None
    left: _Tree | None = None
    right: _Tree | None = None
    leaf: _Leaf | None = None


@dataclass
class CausalForest:
    n_trees: int = 80
    max_depth: int = 5
    min_leaf: int = 40
    max_features: float = 0.6
    random_state: int = 42
    trees_: list[_Tree] = field(default_factory=list)
    model_version: str = "causal-forest-v1"

    def fit(self, x: np.ndarray, t: np.ndarray, y: np.ndarray) -> CausalForest:
        rng = check_random_state(self.random_state)
        n = len(y)
        self.trees_ = []
        for _ in range(self.n_trees):
            idx = rng.randint(0, n, size=n)
            # Honest split: half for structure, half for estimation
            perm = rng.permutation(idx)
            mid = len(perm) // 2
            struct_idx, est_idx = perm[:mid], perm[mid:]
            tree = self._build_tree(
                x, t, y, struct_idx, est_idx, depth=0, rng=rng
            )
            self.trees_.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if not self.trees_:
            raise RuntimeError("Forest not fitted")
        preds = np.zeros(len(x))
        for i in range(len(x)):
            vals = [self._predict_row(tree, x[i]) for tree in self.trees_]
            preds[i] = float(np.mean(vals))
        return preds

    def _predict_row(self, tree: _Tree, row: np.ndarray) -> float:
        node = tree
        while node.leaf is None:
            assert node.feature is not None and node.threshold is not None
            if row[node.feature] <= node.threshold:
                node = node.left  # type: ignore[assignment]
            else:
                node = node.right  # type: ignore[assignment]
            if node is None:
                return 0.0
        return node.leaf.cate

    def _build_tree(
        self,
        x: np.ndarray,
        t: np.ndarray,
        y: np.ndarray,
        struct_idx: np.ndarray,
        est_idx: np.ndarray,
        depth: int,
        rng: np.random.RandomState,
    ) -> _Tree:
        if (
            depth >= self.max_depth
            or len(struct_idx) < 2 * self.min_leaf
            or len(est_idx) < self.min_leaf
        ):
            return _Tree(leaf=self._make_leaf(t, y, est_idx))

        feat_ids = self._sample_features(x.shape[1], rng)
        best = None
        best_score = -np.inf
        best_split: tuple[int, float, np.ndarray, np.ndarray] | None = None

        for f in feat_ids:
            vals = np.unique(x[struct_idx, f])
            if len(vals) < 2:
                continue
            # Candidate thresholds at quantiles
            qs = np.quantile(vals, [0.25, 0.5, 0.75])
            for thr in qs:
                left = struct_idx[x[struct_idx, f] <= thr]
                right = struct_idx[x[struct_idx, f] > thr]
                if len(left) < self.min_leaf or len(right) < self.min_leaf:
                    continue
                score = self._split_score(t, y, left) + self._split_score(t, y, right)
                # Prefer splits that increase heterogeneity
                score += abs(self._naive_cate(t, y, left) - self._naive_cate(t, y, right))
                if score > best_score:
                    best_score = score
                    best_split = (f, float(thr), left, right)

        if best_split is None:
            return _Tree(leaf=self._make_leaf(t, y, est_idx))

        f, thr, _, _ = best_split
        left_struct = struct_idx[x[struct_idx, f] <= thr]
        right_struct = struct_idx[x[struct_idx, f] > thr]
        left_est = est_idx[x[est_idx, f] <= thr]
        right_est = est_idx[x[est_idx, f] > thr]
        if len(left_est) < max(5, self.min_leaf // 2) or len(right_est) < max(
            5, self.min_leaf // 2
        ):
            return _Tree(leaf=self._make_leaf(t, y, est_idx))

        node = _Tree(feature=f, threshold=thr)
        node.left = self._build_tree(x, t, y, left_struct, left_est, depth + 1, rng)
        node.right = self._build_tree(x, t, y, right_struct, right_est, depth + 1, rng)
        return node

    def _sample_features(self, n_features: int, rng: np.random.RandomState) -> np.ndarray:
        k = max(1, int(n_features * self.max_features))
        return rng.choice(n_features, size=k, replace=False)

    def _make_leaf(self, t: np.ndarray, y: np.ndarray, idx: np.ndarray) -> _Leaf:
        return _Leaf(index=idx, cate=self._naive_cate(t, y, idx), n=len(idx))

    @staticmethod
    def _naive_cate(t: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
        """Leaf CATE as difference in loss rates (risk reduction)."""
        if len(idx) == 0:
            return 0.0
        tt = t[idx]
        yy = y[idx]
        treated = yy[tt == 1]
        control = yy[tt == 0]
        if len(treated) == 0 or len(control) == 0:
            return 0.0
        return float(control.mean() - treated.mean())

    @staticmethod
    def _split_score(t: np.ndarray, y: np.ndarray, idx: np.ndarray) -> float:
        if len(idx) < 4:
            return 0.0
        # Pseudo-outcome variance reduction proxy
        tt = t[idx].astype(float)
        yy = y[idx].astype(float)
        # Horvitz-Thompson style transformed outcome with e=0.5 prior
        e = 0.5
        pseudo = ((tt - e) / (e * (1 - e))) * yy
        return float(np.var(pseudo) * len(idx))
