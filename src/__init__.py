"""
Prediction-only expert identification via class-conditional disagreement geometry.

Importable core of the method, baselines, diagnostics, and run pipeline. See the
top-level README for how the modules map to the paper.
"""

from disagreement import class_disagreement_matrix
from dunn import find_dunn_experts, model_total_intragroup_distance
from voting import (
    experts_weighted_voting,
    all_weighted_voting,
    majority_voting_experts,
    predictions_to_vector,
)
from baselines import majority_voting_all_models, em_dawid_skene, sml_predict
from signals import (
    model_centrality,
    centrality_dispersion,
    under_bias,
    over_bias,
    separation_gap,
    class_signals,
)
from metrics import per_class_confusion, summarize_method_performance
from gating import gate_class, gated_predictions
from io_utils import load_dataset_predictions, make_test_sizes
from pipeline import run_dataset, run_single_subsample
