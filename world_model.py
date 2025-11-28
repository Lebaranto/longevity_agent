from __future__ import annotations

from typing import Dict, List, Optional, Any, Set
import yaml
from pydantic import BaseModel, Field, ValidationError


# ---------- СУЩНОСТИ ОНТОЛОГИИ ----------

class Species(BaseModel):
    id: str
    name: str


class Pathway(BaseModel):
    id: str
    name: str
    conserved_human_mouse: bool = False
    hallmarks: List[str] = Field(default_factory=list)


class Gene(BaseModel):
    id: str
    species: List[str]
    pathways: List[str] = Field(default_factory=list)
    longevity_evidence: Dict[str, str] = Field(default_factory=dict)


class InterventionEffect(BaseModel):
    species: str
    lifespan_change_pct: float
    evidence_level: str  # можно позже сузить до Literal["strong","medium","mixed","weak"]


class Intervention(BaseModel):
    id: str
    name: str
    type: str
    targets: List[str] = Field(default_factory=list)
    effects: List[InterventionEffect] = Field(default_factory=list)


# ---------- ШАГИ ВЫВОДА (TRACE) ----------

class InferenceStep(BaseModel):
    rule_name: str
    applied: bool
    delta: float
    explanation: str
    preconditions: Dict[str, Any] = Field(default_factory=dict)


# ---------- МИР ----------

class WorldModel:
    def __init__(self, yaml_path: str):
        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        try:
            self.species: Dict[str, Species] = {
                s["id"]: Species.model_validate(s) for s in data.get("species", [])
            }
            self.pathways: Dict[str, Pathway] = {
                p["id"]: Pathway.model_validate(p) for p in data.get("pathways", [])
            }
            self.genes: Dict[str, Gene] = {
                g["id"]: Gene.model_validate(g) for g in data.get("genes", [])
            }

            self.interventions: Dict[str, Intervention] = {}
            for i in data.get("interventions", []):
                iv = Intervention.model_validate(i)
                self.interventions[iv.id] = iv

            self.scoring_rules: Dict[str, float] = data.get("rules", {}).get(
                "translation_scoring", {}
            ) or {}

        except ValidationError as e:
            # Это удобно, потому что сразу увидишь, где YAML сломан
            raise RuntimeError(f"Ошибка валидации онтологии: {e}") from e

    # ---------- ГЕТТЕРЫ / ПОИСК ----------

    def get_intervention(self, intervention_id: str) -> Optional[Intervention]:
        return self.interventions.get(intervention_id)

    def find_intervention_by_name(self, name: str) -> Optional[Intervention]:
        name = name.strip().lower()
        for iv in self.interventions.values():
            if iv.name.lower() == name:
                return iv
        return None

    # ---------- ЛОГИКА ОЦЕНКИ ----------

    def compute_translation_score(self, intervention_id: str) -> Dict[str, Any]:
        """
        Возвращает dict:
        {
          "intervention": {...},
          "score": float,
          "steps": [InferenceStep-as-dict, ...],
          "warnings": [str, ...],
        }
        """
        iv = self.get_intervention(intervention_id)
        if iv is None:
            return {
                "intervention": None,
                "score": 0.0,
                "steps": [],
                "warnings": [f"Интервенция '{intervention_id}' не найдена в World Model."],
            }

        rules = self.scoring_rules
        score = float(rules.get("base", 0.0))
        steps: List[InferenceStep] = []

        # --- R1: multi-species сигнал ---
        species_set: Set[str] = {eff.species for eff in iv.effects}
        n_species = len(species_set)
        multi_bonus = float(rules.get("multi_species_bonus", 2.0))
        if n_species >= 2:
            score += multi_bonus
            steps.append(
                InferenceStep(
                    rule_name="multi_species_bonus",
                    applied=True,
                    delta=multi_bonus,
                    explanation=f"Эффект показан у нескольких видов ({', '.join(sorted(species_set))}).",
                    preconditions={"n_species": n_species},
                )
            )
        else:
            steps.append(
                InferenceStep(
                    rule_name="multi_species_bonus",
                    applied=False,
                    delta=0.0,
                    explanation="Эффект показан только у одного вида.",
                    preconditions={"n_species": n_species},
                )
            )

        # --- R2: консервативные пути через таргеты ---
        conserved_bonus = float(rules.get("conserved_pathway_bonus", 2.0))
        conserved_hit = False
        conserved_paths: List[str] = []

        for g_id in iv.targets:
            gene = self.genes.get(g_id)
            if gene is None:
                continue
            for pw_id in gene.pathways:
                pw = self.pathways.get(pw_id)
                if pw and pw.conserved_human_mouse:
                    conserved_hit = True
                    conserved_paths.append(pw.name)

        if conserved_hit:
            score += conserved_bonus
            steps.append(
                InferenceStep(
                    rule_name="conserved_pathway_bonus",
                    applied=True,
                    delta=conserved_bonus,
                    explanation=(
                        "Интервенция таргетирует консервативные между мышью и человеком пути: "
                        + ", ".join(sorted(set(conserved_paths)))
                    ),
                    preconditions={"conserved_paths": conserved_paths},
                )
            )
        else:
            steps.append(
                InferenceStep(
                    rule_name="conserved_pathway_bonus",
                    applied=False,
                    delta=0.0,
                    explanation="Не обнаружено консервативных между мышью и человеком путей среди таргетов.",
                    preconditions={},
                )
            )

        # --- R3: human longevity evidence по таргетам ---
        human_bonus = float(rules.get("human_longevity_gene_bonus", 2.0))
        human_ev_targets: List[str] = []
        for g_id in iv.targets:
            gene = self.genes.get(g_id)
            if gene is None:
                continue
            ev = gene.longevity_evidence.get("human")
            if ev and ev.lower() in {"candidate", "strong"}:
                human_ev_targets.append(f"{g_id} ({ev})")

        if human_ev_targets:
            score += human_bonus
            steps.append(
                InferenceStep(
                    rule_name="human_longevity_gene_bonus",
                    applied=True,
                    delta=human_bonus,
                    explanation=(
                        "Среди таргетов есть гены с evidence по человеческому долголетию: "
                        + ", ".join(human_ev_targets)
                    ),
                    preconditions={"targets_with_human_evidence": human_ev_targets},
                )
            )
        else:
            steps.append(
                InferenceStep(
                    rule_name="human_longevity_gene_bonus",
                    applied=False,
                    delta=0.0,
                    explanation="Нет прямых таргетов с evidence по человеческому долголетию.",
                    preconditions={},
                )
            )

        # --- R4: штраф за weak/mixed evidence ---
        weak_penalty = float(rules.get("weak_evidence_penalty", -1.0))
        weak_cases: List[str] = []
        evidence_levels = set()

        for eff in iv.effects:
            lvl = eff.evidence_level.lower()
            evidence_levels.add(lvl)
            if lvl in {"weak", "mixed"}:
                score += weak_penalty
                weak_cases.append(f"{eff.species} ({eff.evidence_level})")

        if weak_cases:
            steps.append(
                InferenceStep(
                    rule_name="weak_evidence_penalty",
                    applied=True,
                    delta=weak_penalty * len(weak_cases),
                    explanation=(
                        "Есть слабый/смешанный evidence по видам: "
                        + ", ".join(weak_cases)
                    ),
                    preconditions={"weak_cases": weak_cases},
                )
            )
        else:
            steps.append(
                InferenceStep(
                    rule_name="weak_evidence_penalty",
                    applied=False,
                    delta=0.0,
                    explanation="Слабого или смешанного evidence не обнаружено.",
                    preconditions={},
                )
            )

        # --- валидация вывода ---
        warnings = self.validate_inference(iv, score, steps, evidence_levels, species_set)

        return {
            "intervention": {
                "id": iv.id,
                "name": iv.name,
                "type": iv.type,
            },
            "score": score,
            "steps": [s.model_dump() for s in steps],
            "warnings": warnings,
        }

    # ---------- МИНИ-ВАЛИДАТОР ----------

    def validate_inference(
        self,
        iv: Intervention,
        score: float,
        steps: List[InferenceStep],
        evidence_levels: set[str],
        species_set: set[str],
    ) -> List[str]:
        warnings: List[str] = []

        # Было ли human longevity evidence использовано?
        has_human_ev = any(
            s.rule_name == "human_longevity_gene_bonus" and s.applied for s in steps
        )

        only_invertebrates = species_set.issubset({"fly", "worm"})

        # 1) высокий score без human evidence
        if score >= 4.0 and not has_human_ev:
            warnings.append(
                "Высокий интегральный score при отсутствии прямого evidence по человеку. "
                "Интерпретировать результат с осторожностью."
            )

        # 2) эффект только у беспозвоночных, но score высокий
        if score >= 3.0 and only_invertebrates:
            warnings.append(
                "Эффекты показаны только у беспозвоночных моделей (fly/worm). "
                "Риск низкой транслируемости на млекопитающих."
            )

        # 3) гетерогенный профиль evidence
        if "strong" in evidence_levels and ("weak" in evidence_levels or "mixed" in evidence_levels):
            warnings.append(
                "Наблюдается гетерогенный профиль evidence (сильные и слабые/смешанные данные по разным видам). "
                "Возможна контекст-зависимая эффективность."
            )

        return warnings
