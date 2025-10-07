# Copyright (c) 2025 Lakshya A Agrawal and the GEPA contributors
# https://github.com/gepa-ai/gepa

import json
import os
from typing import Any, Callable, Generic

from gepa.core.adapter import RolloutOutput
from gepa.gepa_utils import json_default

ProgramIdx = int
ValId = int
ValScores = dict[ValId, float]
ValOutputs = dict[ValId, RolloutOutput]
_VALIDATION_SCHEMA_VERSION = 2


class GEPAState(Generic[RolloutOutput]):
    program_candidates: list[dict[str, str]]
    parent_program_for_candidate: list[list[ProgramIdx | None]]

    program_val_scores: list[ValScores]

    pareto_front_valset: ValScores
    program_at_pareto_front_valset: dict[ValId, set[ProgramIdx]]

    list_of_named_predictors: list[str]
    named_predictor_id_to_update_next_for_program_candidate: list[int]

    valset_size: int

    i: int
    num_full_ds_evals: int

    total_num_evals: int

    num_metric_calls_by_discovery: list[int]

    full_program_trace: list
    best_outputs_valset: dict[ValId, list[tuple[ProgramIdx, RolloutOutput]]] | None = None

    validation_schema_version: int

    def __init__(
        self,
        seed_candidate: dict[str, str],
        base_valset_eval_output: tuple[ValOutputs, ValScores],
        valset_size: int | None = None,
        track_best_outputs: bool = False,
    ):
        base_outputs, base_scores = base_valset_eval_output
        self.program_candidates = [seed_candidate]
        self.program_val_scores = [base_scores]

        self.pareto_front_valset = {val_id: score for val_id, score in base_scores.items()}
        self.parent_program_for_candidate = [[None]]
        self.program_at_pareto_front_valset = {val_id: {0} for val_id in base_scores.keys()}

        self.list_of_named_predictors = list(seed_candidate.keys())
        self.named_predictor_id_to_update_next_for_program_candidate = [0]
        self.i = -1

        self.num_metric_calls_by_discovery = [0]

        self.best_outputs_valset = (
            {val_id: [(0, output)] for val_id, output in base_outputs.items()} if track_best_outputs else None
        )

        if valset_size is None:
            # if not told valset size, assume we observe largest id
            valset_size = max(val_id for val_id in base_scores.keys()) + 1
        self.valset_size = valset_size

        self.full_program_trace = []
        self.validation_schema_version = _VALIDATION_SCHEMA_VERSION

    def is_consistent(self):
        assert len(self.program_candidates) == len(self.parent_program_for_candidate)
        assert len(self.program_candidates) == len(self.named_predictor_id_to_update_next_for_program_candidate)
        assert len(self.program_candidates) == len(self.program_val_scores)
        assert len(self.program_candidates) == len(self.num_metric_calls_by_discovery)

        for front in self.program_at_pareto_front_valset.values():
            for prog_idx in front:
                assert prog_idx < len(self.program_candidates), (
                    "Program index in valset pareto front exceeds number of program candidates"
                )

        assert set(self.pareto_front_valset.keys()) == set(self.program_at_pareto_front_valset.keys())
        assert max(self.program_at_pareto_front_valset.keys()) < self.valset_size

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
        if version is None:
            GEPAState._migrate_legacy_state_dict(d)

        state = GEPAState.__new__(GEPAState)
        state.__dict__.update(d)

        state.validation_schema_version = _VALIDATION_SCHEMA_VERSION
        assert set(state.pareto_front_valset.keys()) == set(state.program_at_pareto_front_valset.keys())
        assert len(state.program_candidates) == len(state.program_val_scores)
        assert len(state.program_candidates) == len(state.num_metric_calls_by_discovery)
        assert len(state.program_candidates) == len(state.parent_program_for_candidate)
        assert len(state.program_candidates) == len(state.named_predictor_id_to_update_next_for_program_candidate)
        assert max(state.pareto_front_valset.keys()) < state.valset_size
        return state

    @staticmethod
    def _migrate_legacy_state_dict(d: dict[str, Any]) -> None:
        legacy_scores: list[list[float]] = d.pop("prog_candidate_val_subscores", [])
        program_val_scores: list[ValScores] = [
            {idx: score for idx, score in enumerate(scores)} for scores in legacy_scores
        ]
        if "program_val_scores" not in d:
            d["program_val_scores"] = program_val_scores

        pareto_front = d.get("pareto_front_valset")
        if isinstance(pareto_front, list):
            d["pareto_front_valset"] = {idx: score for idx, score in enumerate(pareto_front)}

        program_at_front = d.get("program_at_pareto_front_valset")
        if isinstance(program_at_front, list):
            d["program_at_pareto_front_valset"] = {idx: set(front) for idx, front in enumerate(program_at_front)}

        best_outputs = d.get("best_outputs_valset")
        if isinstance(best_outputs, list):
            d["best_outputs_valset"] = {idx: list(outputs) for idx, outputs in enumerate(best_outputs)}

        d["valset_size"] = len(best_outputs)
        d["validation_schema_version"] = _VALIDATION_SCHEMA_VERSION

    def get_program_average(self, program_idx: int) -> tuple[float, int]:
        scores = self.program_val_scores[program_idx]
        if not scores:
            return float("-inf"), 0
        num_samples = len(scores)
        avg = sum(scores.values()) / num_samples
        return avg, num_samples

    def missing_val_ids_for_program(self, program_idx: int) -> set[int]:
        return set(range(self.valset_size)).difference(self.program_val_scores[program_idx].keys())

    @property
    def unevaluated_val_ids(self) -> set[int]:
        """ Validation examples not evaluated for any program """
        unevaluated_val_ids = set(range(self.valset_size))
        for val_scores in self.program_val_scores:
            unevaluated_val_ids = unevaluated_val_ids.difference(val_scores.keys())
        return unevaluated_val_ids

    @property
    def program_full_scores_val_set(self) -> list[float]:
        return [self.get_program_average(program_idx)[0] for program_idx in range(len(self.program_val_scores))]

    @property
    def per_program_tracked_scores(self) -> list[float]:
        # TODO(aria42): This is same as valset program average scores, but this was already the case
        return [self.get_program_average(program_idx)[0] for program_idx in range(len(self.program_val_scores))]

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
            if self.best_outputs_valset is not None and outputs is not None and (output := outputs[val_id]):
                self.best_outputs_valset[val_id] = [(program_idx, output)]
                if run_dir is not None and outputs is not None:
                    task_path = os.path.join(run_dir, "generated_best_outputs_valset", f"task_{val_id}")
                    os.makedirs(task_path, exist_ok=True)
                    with open(task_path.join(f"iter_{iteration}_prog_{program_idx}.json")) as fout:
                        json.dump(output, fout, indent=4, default=json_default)
        elif score == prev_score:
            pareto_front = self.program_at_pareto_front_valset.setdefault(val_id, set())
            pareto_front.add(program_idx)
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

        for val_id, score in valset_scores.items():
            self._update_pareto_front_for_val_id(val_id, score, new_program_idx, valset_outputs, run_dir, self.i + 1)

        linear_pareto_front_program_idx = self._best_program_idx()
        return new_program_idx, linear_pareto_front_program_idx

    def _best_program_idx(self) -> int:
        best_idx = 0
        best_avg, best_cov = self.get_program_average(0)
        for idx in range(1, len(self.program_candidates)):
            avg, cov = self.get_program_average(idx)
            if avg > best_avg or (avg == best_avg and cov > best_cov):
                best_idx = idx
                best_avg, best_cov = avg, cov
        return best_idx


def write_eval_output_to_directory(outputs: ValOutputs, output_dir: str):
    for task_idx, output in outputs.items():
        os.makedirs(os.path.join(output_dir, f"task_{task_idx}"), exist_ok=True)
        with open(os.path.join(output_dir, f"task_{task_idx}", f"iter_{0}_prog_0.json"), "w") as f:
            json.dump(output, f, indent=4, default=json_default)


def initialize_gepa_state(
    run_dir: str | None,
    logger,
    seed_candidate: dict[str, str],
    valset_evaluator: Callable[[dict[str, str]], tuple[ValOutputs, ValScores]],
    track_best_outputs: bool = False,
    valset_size: int | None = None,
):
    if run_dir is not None and os.path.exists(os.path.join(run_dir, "gepa_state.bin")):
        logger.log("Loading gepa state from run dir")
        gepa_state = GEPAState.load(run_dir)
        if valset_size:
            if gepa_state.valset_size > valset_size:
                raise ValueError("Cannot shrink valset")
            gepa_state.valset_size = max(gepa_state.valset_size, valset_size)
    else:
        num_evals_run = 0

        seed_val_outputs, seed_val_scores = valset_evaluator(seed_candidate)
        if run_dir is not None:
            write_eval_output_to_directory(seed_val_outputs, os.path.join(run_dir, "generated_best_outputs_valset"))
        num_evals_run += len(seed_val_scores)

        gepa_state = GEPAState(
            seed_candidate,
            (seed_val_outputs, seed_val_scores),
            track_best_outputs=track_best_outputs,
        )

        gepa_state.num_full_ds_evals = 1
        gepa_state.total_num_evals = num_evals_run

    return gepa_state
