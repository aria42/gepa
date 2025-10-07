# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import os
from typing import Any, Callable, Generic, Iterable

from gepa.core.adapter import RolloutOutput
from gepa.gepa_utils import idxmax, json_default


ValScores = dict[int, float]
ValOutputs = dict[int, RolloutOutput]
_VALIDATION_SCHEMA_VERSION = 2


class GEPAState(Generic[RolloutOutput]):
    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[int | None]]

    program_full_scores_val_set: list[float]
    program_val_scores: list[ValScores]
    program_val_coverage_counts: list[int]

    pareto_front_valset: dict[int, float]
    program_at_pareto_front_valset: dict[int, set[int]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    known_val_ids: set[int]
    unevaluated_val_ids: set[int]

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list

    per_program_tracked_scores: list[float]

    best_outputs_valset: dict[int, list[tuple[int, RolloutOutput]]] | None = None

    validation_schema_version: int

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_valset_eval_output: tuple[list[RolloutOutput], list[float]],
        track_best_outputs: bool = False,
    ):
        outputs, scores = base_valset_eval_output
        base_scores_dict: ValScores = {idx: score for idx, score in enumerate(scores)}
        valset_base_score = sum(base_scores_dict.values()) / max(1, len(base_scores_dict))

        self.program_candidates = [seed_candidate]
        self.program_full_scores_val_set = [valset_base_score]
        self.program_val_scores = [base_scores_dict]
        self.program_val_coverage_counts = [len(base_scores_dict)]

        self.per_program_tracked_scores = [valset_base_score]

        self.pareto_front_valset = {idx: score for idx, score in base_scores_dict.items()}
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = {idx: {0} for idx in base_scores_dict.keys()}

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_metric_calls_by_discovery = [0]

        if track_best_outputs:
            self.best_outputs_valset = {idx: [(0, outputs[idx])] for idx in base_scores_dict.keys()}
        else:
            self.best_outputs_valset = None

        self.known_val_ids = set(base_scores_dict.keys())
        self.unevaluated_val_ids = set()

        self.full_program_trace = []
        self.validation_schema_version = _VALIDATION_SCHEMA_VERSION

    def is_consistent(self):
        assert len(self.program_candidates) == len(self.program_full_scores_val_set)
        assert len(self.program_candidates) == len(self.per_program_tracked_scores)
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        assert len(self.program_candidates) == len(self.program_val_scores)
        assert len(self.program_candidates) == len(self.program_val_coverage_counts)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        for front in self.program_at_pareto_front_valset.values():
            for prog_idx in front:
                assert prog_idx < len(self.program_candidates), "Program index in valset pareto front exceeds number of program candidates"

        assert set(self.pareto_front_valset.keys()) == set(self.program_at_pareto_front_valset.keys())
        assert self.known_val_ids.issuperset(self.pareto_front_valset.keys())

        return True

    def save(self, run_dir: str | None):
        if run_dir is None:
            return
        with open(os.path.join(run_dir, "gepa_state.bin"), "wb") as f:
            import pickle
            d = dict(self.__dict__.items())
            d["validation_schema_version"] = _VALIDATION_SCHEMA_VERSION
            pickle.dump(d, f)

    @staticmethod
    def load(run_dir: str) -> "GEPAState":
        with open(os.path.join(run_dir, "gepa_state.bin"), "rb") as f:
            import pickle
            d = pickle.load(f)

        version = d.get("validation_schema_version")
        if version != _VALIDATION_SCHEMA_VERSION:
            GEPAState._migrate_legacy_state_dict(d)

        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)

        state.validation_schema_version = _VALIDATION_SCHEMA_VERSION
        assert len(state.program_candidates) == len(state.program_full_scores_val_set)
        assert set(state.pareto_front_valset.keys()) == set(state.program_at_pareto_front_valset.keys())
        assert len(state.program_candidates) == len(state.program_val_scores)
        assert len(state.program_candidates) == len(state.program_val_coverage_counts)
        assert len(state.program_candidates) == len(state.num_metric_calls_by_discovery)
        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)

        return state

    @staticmethod
    def _migrate_legacy_state_dict(d: dict[str, Any]) -> None:
        legacy_scores: list[list[float]] = d.pop("prog_candidate_val_subscores", [])
        program_val_scores: list[ValScores] = [
            {idx: score for idx, score in enumerate(scores)} for scores in legacy_scores
        ]
        if "program_val_scores" not in d:
            d["program_val_scores"] = program_val_scores

        d["program_val_coverage_counts"] = [len(scores) for scores in d["program_val_scores"]]

        pareto_front = d.get("pareto_front_valset")
        if isinstance(pareto_front, list):
            d["pareto_front_valset"] = {idx: score for idx, score in enumerate(pareto_front)}

        program_at_front = d.get("program_at_pareto_front_valset")
        if isinstance(program_at_front, list):
            d["program_at_pareto_front_valset"] = {idx: set(front) for idx, front in enumerate(program_at_front)}

        best_outputs = d.get("best_outputs_valset")
        if isinstance(best_outputs, list):
            d["best_outputs_valset"] = {idx: list(outputs) for idx, outputs in enumerate(best_outputs)}

        d["known_val_ids"] = set(d["pareto_front_valset"].keys())
        d.setdefault("unevaluated_val_ids", set())
        d["validation_schema_version"] = _VALIDATION_SCHEMA_VERSION

    def register_new_val_ids(self, new_ids: Iterable[int]):
        for val_id in new_ids:
            if val_id in self.known_val_ids:
                continue
            self.known_val_ids.add(val_id)
            self.unevaluated_val_ids.add(val_id)
            self.pareto_front_valset[val_id] = float("-inf")
            self.program_at_pareto_front_valset[val_id] = set()
            if self.best_outputs_valset is not None:
                self.best_outputs_valset[val_id] = []

    def ensure_valset_size(self, total_val_count: int):
        self.register_new_val_ids(idx for idx in range(total_val_count))

    def _compute_average(self, scores: ValScores) -> float:
        if not scores:
            return float("-inf")
        return sum(scores.values()) / len(scores)

    def get_program_average(self, program_idx: int) -> tuple[float | None, int]:
        scores = self.program_val_scores[program_idx]
        if not scores:
            return None, 0
        return self._compute_average(scores), len(scores)

    def missing_val_ids_for_program(self, program_idx: int) -> set[int]:
        return set(self.known_val_ids).difference(self.program_val_scores[program_idx].keys())

    def record_val_scores(
        self,
        program_idx: int,
        scores: ValScores,
        outputs: dict[int, RolloutOutput] | None,
        run_dir: str | None,
        iteration: int,
    ) -> None:
        program_scores = self.program_val_scores[program_idx]
        program_scores.update(scores)
        self.program_val_coverage_counts[program_idx] = len(program_scores)

        avg = self._compute_average(program_scores)
        self.program_full_scores_val_set[program_idx] = avg
        self.per_program_tracked_scores[program_idx] = avg

        for val_id, score in scores.items():
            self._update_pareto_front_for_val_id(val_id, score, program_idx, outputs, run_dir, iteration)

    def _update_pareto_front_for_val_id(
        self,
        val_id: int,
        score: float,
        program_idx: int,
        outputs: dict[int, RolloutOutput] | None,
        run_dir: str | None,
        iteration: int,
    ) -> None:
        prev_score = self.pareto_front_valset.get(val_id, float("-inf"))
        if score > prev_score:
            self.pareto_front_valset[val_id] = score
            self.program_at_pareto_front_valset[val_id] = {program_idx}
            if self.best_outputs_valset is not None and outputs is not None and val_id in outputs:
                self.best_outputs_valset[val_id] = [(program_idx, outputs[val_id])]
            if run_dir is not None and outputs is not None and val_id in outputs:
                os.makedirs(os.path.join(run_dir, "generated_best_outputs_valset", f"task_{val_id}"), exist_ok=True)
                with open(
                    os.path.join(
                        run_dir,
                        "generated_best_outputs_valset",
                        f"task_{val_id}",
                        f"iter_{iteration}_prog_{program_idx}.json",
                    ),
                    "w",
                ) as f:
                    json.dump(outputs[val_id], f, indent=4, default=json_default)
        elif score == prev_score:
            self.program_at_pareto_front_valset[val_id].add(program_idx)
            if self.best_outputs_valset is not None and outputs is not None and val_id in outputs:
                self.best_outputs_valset[val_id].append((program_idx, outputs[val_id]))

    def update_state_with_new_program(
        self,
        parent_program_idx: list[int],
        new_program: dict[str, str],
        valset_scores: ValScores,
        valset_outputs: dict[int, RolloutOutput] | None,
        run_dir: str | None,
        num_metric_calls_by_discovery_of_new_program: int,
    ) -> tuple[int, int]:
        new_program_idx = len(self.program_candidates)
        self.program_candidates.append(new_program)
        self.num_metric_calls_by_discovery.append(num_metric_calls_by_discovery_of_new_program)

        max_predictor_id = max(
            [self.named_predictor_id_to_update_next_for_program_candidate[p] for p in parent_program_idx],
            default=0,
        )
        self.named_predictor_id_to_update_next_for_program_candidate.append(max_predictor_id)
        self.parent_program_for_candidate.append(list(parent_program_idx))

        self.program_val_scores.append(dict(valset_scores))
        coverage = len(valset_scores)
        self.program_val_coverage_counts.append(coverage)

        avg = self._compute_average(valset_scores)
        self.program_full_scores_val_set.append(avg)
        self.per_program_tracked_scores.append(avg)

        for val_id in valset_scores:
            if val_id in self.unevaluated_val_ids:
                self.unevaluated_val_ids.discard(val_id)

        for val_id, score in valset_scores.items():
            self._update_pareto_front_for_val_id(val_id, score, new_program_idx, valset_outputs, run_dir, self.i + 1)

        linear_pareto_front_program_idx = self._best_program_idx()
        return new_program_idx, linear_pareto_front_program_idx

    def _best_program_idx(self) -> int:
        best_idx = 0
        best_avg, best_cov = self._best_program_score_and_cov(0)
        for idx in range(1, len(self.program_candidates)):
            avg, cov = self._best_program_score_and_cov(idx)
            if avg > best_avg or (avg == best_avg and cov > best_cov):
                best_idx = idx
                best_avg, best_cov = avg, cov
        return best_idx

    def _best_program_score_and_cov(self, idx: int) -> tuple[float, int]:
        scores = self.program_val_scores[idx]
        if not scores:
            return float("-inf"), 0
        avg = self._compute_average(scores)
        return avg, len(scores)


def write_eval_output_to_directory(
    eval_out: tuple[list[RolloutOutput], list[float]],
    output_dir: str
):
    outputs, scores = eval_out
    for task_idx, _score in enumerate(scores):
        os.makedirs(os.path.join(output_dir, f"task_{task_idx}"), exist_ok=True)
        with open(os.path.join(output_dir, f"task_{task_idx}", f"iter_{0}_prog_0.json"), "w") as f:
            json.dump(outputs[task_idx], f, indent=4, default=json_default)


def initialize_gepa_state(
    run_dir: str | None,
    logger,
    seed_candidate: dict[str, str],
    valset_evaluator: Callable[[dict[str, str]], tuple[list[RolloutOutput], list[float]]],
    track_best_outputs: bool = False,
    current_valset_size: int | None = None,
):
    if run_dir is not None and os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        logger.log("Loading gepa state from run dir")
        gepa_state = GEPAState.load(run_dir)
        if current_valset_size is not None:
            gepa_state.ensure_valset_size(current_valset_size)
    else:
        num_evals_run = 0

        valset_out = valset_evaluator(seed_candidate)
        if run_dir is not None:
            write_eval_output_to_directory(valset_out, os.path.join(run_dir, "generated_best_outputs_valset"))
        num_evals_run += len(valset_out[1])

        gepa_state = GEPAState(
            seed_candidate,
            valset_out,
            track_best_outputs=track_best_outputs,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

        if current_valset_size is not None:
            gepa_state.ensure_valset_size(current_valset_size)

    return gepa_state
