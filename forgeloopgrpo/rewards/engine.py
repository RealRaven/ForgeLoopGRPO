"""Multi-Objective Policy Reward Orchestration Engine

Manages the real-time evaluation, mathematical normalization, and alignment gating
of a 15-component asynchronous reward matrix during GRPO optimization loops.
"""

import numpy as np
from typing import List, Dict, Tuple
from ..embeddings import CPUEmbeddingStore
from .components import VocabularyAlignmentScorer, FluencyScorer
from .tone_scorer import ToneScorer
from .semantic_diversity import SemanticDiversityScorer
from .behavioral_alignment_scorers import BehavioralAlignmentScorers
from .soft_cull import SoftCull
from .degeneracy_auditor import DegeneracyAuditor
from .gate import AdaptiveGate
from .normalizer import BoundedNormalizer


class PolicyRewardOrchestrationEngine:
    """Core evaluation runtime orchestrating multi-variable heuristic scoring matrices.

    Applies bounded distribution normalization and target policy constraints.
    """

    def __init__(self, config, embedding_store: CPUEmbeddingStore):
        self.config = config
        self.embedding_store = embedding_store

        self.vocabulary_alignment_scorer = VocabularyAlignmentScorer(config)
        self.tone_scorer = ToneScorer(config)
        self.fluency_scorer = FluencyScorer()
        self.semantic_diversity_scorer = SemanticDiversityScorer(embedding_store, floor=config.semantic_diversity_floor)
        self.behavioral_alignment = BehavioralAlignmentScorers()
        self.soft_cull = SoftCull(
            percentage=config.soft_cull.cull_percentage,
            penalty=config.soft_cull.cull_penalty
        )
        self.degeneracy_auditor = DegeneracyAuditor(config.degeneracy_auditor.model_dump())
        self.gate = AdaptiveGate(alpha=config.gate.historical_mean_alpha, enabled=config.gate.enabled)
        self.normalizer = BoundedNormalizer(alpha=0.1)

        # ==========================================================================
        # RUNTIME WEIGHT NORMALIZATION ENGINE
        # ==========================================================================
        raw_weights = config.reward_weights.model_dump()
        
        # Extract enabled weights only (supports new {enabled, weight} dicts and legacy flat floats)
        enabled_weights = {}
        for k, v in raw_weights.items():
            if isinstance(v, dict):
                if v.get("enabled", True):
                    enabled_weights[k] = v.get("weight", 0.0)
            else:
                enabled_weights[k] = float(v)

        weight_sum = sum(enabled_weights.values())
        
        if not np.isclose(weight_sum, 1.0) and weight_sum > 0:
            print(f"[REWARD ENGINE INITIALIZATION] Enabled weights sum to {weight_sum:.4f} instead of 1.0. Normalizing...")
            self.weights = {k: v / weight_sum for k, v in enabled_weights.items()}
        else:
            self.weights = enabled_weights

        self.all_component_keys = [
            "thematic_consistency", "tone_consistency", "semantic_diversity", "fluency",
            "reasoning_depth", "lexical_diversity", "efficiency_coefficient", "directive_clarity",
            "context_alignment", "input_adaptation", "style_preservation", "creative_problem_solving", 
            "cognitive_richness", "style_coherence", "exploratory_boldness"
        ]

    def score_group(self, texts: List[str], prompts: List[str], metainfos: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """Score a full group of G generations. Returns raw rewards and diagnostics."""
        G = len(texts)
        rewards = np.zeros(G)
        component_scores = {k: np.zeros(G) for k in self.all_component_keys}

        # 1. Thematic Consistency scores
        if "thematic_consistency" in self.weights:
            for i, text in enumerate(texts):
                component_scores["thematic_consistency"][i] = self.vocabulary_alignment_scorer.score(text)

        # 2. Tone Consistency scores
        if "tone_consistency" in self.weights:
            for i, (text, meta) in enumerate(zip(texts, metainfos)):
                target = meta.get("target_mode", "") or meta.get("tone", "")
                component_scores["tone_consistency"][i] = self.tone_scorer.score(text, target)

        # 3. Semantic Diversity scores (group-level)
        if "semantic_diversity" in self.weights:
            semantic_diversity_scores = self.semantic_diversity_scorer.score(texts)
            for i, score in enumerate(semantic_diversity_scores):
                component_scores["semantic_diversity"][i] = score

        # 4. Fluency scores
        if "fluency" in self.weights:
            for i, text in enumerate(texts):
                component_scores["fluency"][i] = self.fluency_scorer.score(text)

        # 5-15. Behavioral Structural Alignment scores
        behavioral_keys = [
            "reasoning_depth", "lexical_diversity", "efficiency_coefficient",
            "directive_clarity", "context_alignment", "input_adaptation",
            "style_preservation", "creative_problem_solving",
            "cognitive_richness", "style_coherence", "exploratory_boldness"
        ]
        if any(k in self.weights for k in behavioral_keys):
            for i, (text, prompt) in enumerate(zip(texts, prompts)):
                if "reasoning_depth" in self.weights:
                    component_scores["reasoning_depth"][i] = self.behavioral_alignment.reasoning_depth(text)
                if "lexical_diversity" in self.weights:
                    component_scores["lexical_diversity"][i] = self.behavioral_alignment.lexical_diversity(text)
                if "efficiency_coefficient" in self.weights:
                    component_scores["efficiency_coefficient"][i] = self.behavioral_alignment.efficiency_coefficient(text)
                if "directive_clarity" in self.weights:
                    component_scores["directive_clarity"][i] = self.behavioral_alignment.directive_clarity(text)
                if "context_alignment" in self.weights:
                    component_scores["context_alignment"][i] = self.behavioral_alignment.thematic_consistency(text)
                if "input_adaptation" in self.weights:
                    component_scores["input_adaptation"][i] = self.behavioral_alignment.input_adaptation(text, prompt)
                if "style_preservation" in self.weights:
                    component_scores["style_preservation"][i] = self.behavioral_alignment.tone_consistency(text, prompt)
                if "creative_problem_solving" in self.weights:
                    component_scores["creative_problem_solving"][i] = self.behavioral_alignment.creative_problem_solving(text)
                if "cognitive_richness" in self.weights:
                    component_scores["cognitive_richness"][i] = self.behavioral_alignment.cognitive_richness(text)
                if "style_coherence" in self.weights:
                    component_scores["style_coherence"][i] = self.behavioral_alignment.style_coherence(text)
                if "exploratory_boldness" in self.weights:
                    component_scores["exploratory_boldness"][i] = self.behavioral_alignment.exploratory_boldness(text, texts)

        # Weighted sum using mathematically normalized attributes
        for comp_name, weight in self.weights.items():
            if comp_name in component_scores:
                rewards += component_scores[comp_name] * weight

        # Update the Normalizer BEFORE squashing
        self.normalizer.update(rewards)

        # Normalize initial distribution structures
        for i in range(G):
            rewards[i] = self.normalizer.normalize(rewards[i])

        # Capital offenses auditing pass
        penalties, offense_report = self.degeneracy_auditor.audit(texts)
        rewards += penalties

        # Soft cull filtering loops
        rewards, culled_indices = self.soft_cull.apply_to_advantages(rewards)

        diagnostics = {
            "component_scores": {k: v.tolist() for k, v in component_scores.items()},
            "degeneracy_auditor": offense_report,
            "culled_indices": culled_indices,
            "raw_rewards": rewards.tolist(),
            "gate_weight": self.gate.weight(np.mean(rewards))
        }

        return rewards, diagnostics

    def compute_advantages(self, rewards: np.ndarray) -> np.ndarray:
        """GRPO advantage computation. Forces strict zero-sum alignment."""
        mean = np.mean(rewards)
        std = np.std(rewards) + 1e-6
        raw_advantages = (rewards - mean) / std

        # Signed power transform
        advantages = np.sign(raw_advantages) * np.power(np.abs(raw_advantages), 0.7) * 2.0

        # Re-center to restore zero-sum constraint after nonlinear distortion
        advantages = advantages - np.mean(advantages)

        # Apply gate calibration
        gate_w = self.gate.weight(mean)
        advantages *= gate_w

        self.gate.update(mean)

        return advantages