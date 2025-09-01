You are a senior engineer refactoring a research codebase into a small, testable Python package with clear CLIs. Preserve current functionality while making the system modular, reproducible, and easy to extend to **multiple datasets** (initially: CholecSeg8k and EndoScape).

> **Package name:** use **`endopoint`** (dataset-agnostic).  
> Keep backwards compatibility for existing outputs, but add `schema_version` and a config/hash to all new results.

---

## Objectives

1. Extract notebook logic into a proper package `endopoint/` with clean module boundaries.
    
2. Add a **DatasetAdapter** interface + registry so CholecSeg8k and EndoScape are drop-ins.
    
3. Unify prompt variants and few-shot logic behind registries; support planned few-shot (with hard negatives).
    
4. Unify model calls behind a `ModelAdapter` interface.
    
5. Deterministic caching + JSON schemas for results.
    
6. CLIs for each stage (cache, select, few-shot, eval, summarize, visualize) that work across datasets.
    
7. Tests for geometry, parser, metrics, caching, few-shot plans, and dataset adapters.
    
8. Config-driven runs (YAML), reproducible seeds, clear logs.
    

---

## Target Layout

```
repo/
├─ pyproject.toml
├─ README.md
├─ configs/
│  ├─ dataset_cholecseg8k.yaml
│  ├─ dataset_endoscape.yaml
│  ├─ selection.yaml
│  ├─ prompts.yaml
│  └─ runs/                  # frozen experiment configs
├─ data_info/                # balanced_indices_*.json, fewshot_plan_*.json
├─ cache/                    # presence caches
├─ results/
│  └─ pointing/<dataset>/<exp_id>/<prompt>/<model>/...
├─ src/endopoint/
│  ├─ __init__.py
│  ├─ utils/
│  │  ├─ io.py               # load_json, save_json, file hashing
│  │  ├─ logging.py
│  │  └─ rng.py              # seed_all
│  ├─ datasets/
│  │  ├─ base.py             # DatasetAdapter protocol + registry
│  │  ├─ cholecseg8k.py      # concrete adapter
│  │  └─ endoscape.py        # concrete adapter
│  ├─ data/
│  │  └─ presence.py         # compute_presence_matrix_cached(adapter,...)
│  ├─ geometry/
│  │  └─ canvas.py           # letterbox_to_canvas, canvas↔orig
│  ├─ prompts/
│  │  ├─ builders.py         # system/user builders
│  │  └─ registry.py         # PROMPT_ABLATIONS
│  ├─ fewshot/
│  │  ├─ plan.py             # build_fewshot_plan_excluding(...), hard negatives
│  │  └─ providers.py        # PlannedFewshotProvider
│  ├─ models/
│  │  ├─ base.py             # ModelAdapter protocol
│  │  ├─ openai_gpt.py
│  │  ├─ anthropic_claude.py
│  │  └─ google_gemini.py
│  ├─ eval/
│  │  ├─ parser.py           # parse_pointing_json(...)
│  │  ├─ runner.py           # PointingEvaluator
│  │  └─ summarize.py        # summarize_pointing(...)
│  ├─ metrics/
│  │  └─ pointing.py         # presence acc, hit@present, gated acc
│  └─ vis/
│     ├─ balance.py          # stacked bars, heatmaps
│     └─ fewshot.py          # grids for pos/neg/hard-neg
├─ cli/
│  ├─ build_presence_cache.py
│  ├─ select_balanced.py
│  ├─ build_fewshot_plan.py
│  ├─ run_pointing.py
│  ├─ summarize_pointing.py
│  └─ visualize_balance.py
└─ tests/
   ├─ test_canvas.py
   ├─ test_parser.py
   ├─ test_presence.py
   ├─ test_metrics.py
   └─ test_datasets.py       # adapters’ contracts
```

---

## Dataset Adapter (implement exactly)

```
# src/endopoint/datasets/base.py
from typing import Protocol, Tuple, Dict, Any, Sequence
import torch
from PIL import Image

class DatasetAdapter(Protocol):
    # Identity & schema
    @property
    def dataset_tag(self) -> str: ...
    @property
    def version(self) -> str: ...            # e.g., 'v1' or HF revision
    @property
    def id2label(self) -> Dict[int, str]: ...
    @property
    def label_ids(self) -> Sequence[int]: ... # e.g., [1..K], excludes background
    @property
    def ignore_index(self) -> int: ...        # label ignore id or -1
    @property
    def recommended_canvas(self) -> Tuple[int, int]: ...  # default (768,768)

    # Data access
    def total(self, split: str) -> int: ...
    def get_example(self, split: str, index: int) -> Any: ...
    def example_to_tensors(self, example: Any) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          img_t: FloatTensor [C,H,W] in [0,1]
          lab_t: LongTensor  [H,W] with class ids
        """

    # Semantics
    def labels_to_presence_vector(self, lab_t: torch.Tensor, min_pixels: int) -> torch.Tensor:
        """Return LongTensor [K] over self.label_ids with {0,1}."""

    # Optional utilities
    def sample_point_in_mask(self, lab_t: torch.Tensor, class_id: int, strategy: str = "centroid") -> Tuple[int,int] | None: ...
```

### Dataset Registry

```
# src/endopoint/datasets/base.py
DATASET_REGISTRY = {}  # name -> callable(**cfg) -> DatasetAdapter

def register_dataset(name: str):
    def deco(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return deco

def build_dataset(name: str, **cfg) -> DatasetAdapter:
    return DATASET_REGISTRY[name](**cfg)
```

### Concrete adapters

- `cholecseg8k.py` implements the adapter using your current logic (HF `load_dataset`, same ID2LABEL, etc.).
    
- `endoscape.py` implements the adapter analogously; don’t hardcode label count—allow **K to vary**.
    

> All downstream code must use only the **adapter interface**; never import dataset-specific symbols directly.

---

## Model Adapter (as before)

```
# src/endopoint/models/base.py
from typing import Protocol, Sequence, Union, Tuple
from PIL import Image

PromptPart = Union[str, Image.Image]
OneQuery   = Tuple[PromptPart, ...]
Batch      = Sequence[OneQuery]

class ModelAdapter(Protocol):
    def __call__(self, prompts: Batch, *, system_prompt: str) -> Sequence[str]:
        """Return raw model strings, one per query."""
```

---

## Prompt Registry (unchanged in spirit)

- `prompts/builders.py`: `sys_prompt_strict(CW, CH)`, `user_prompt_base(organ)`, variants.
    
- `prompts/registry.py`: maps prompt names → dicts of builder callables (+ optional few-shot provider).
    

---

## Few-Shot Plan & Provider (multi-dataset aware)

- `fewshot/plan.py`:
    
    - `build_fewshot_plan_excluding(adapter, split, balanced_indices, n_pos, n_neg, n_hard_neg, min_pixels, seed, ensure_disjoint_across_organs=False)`
        
    - Use `adapter.labels_to_presence_vector` and `adapter.sample_point_in_mask`.
        
    - Store **original** points for positives and hard negatives.
        
    - Ensure **no overlap** with `balanced_indices`.
        
    - JSON includes: `dataset_tag`, `version`, `split`, `K`, label names, and a hash of excluded indices.
        
- `fewshot/providers.py`:
    
    - `PlannedFewshotProvider(plan_json, adapter)` maps original → canvas coords using geometry module; builds:
        
        - positives (`present=1`, correct point),
            
        - easy negatives (`present=0`, `point_canvas=null`),
            
        - hard negatives (`present=1`, point purposely _not_ inside mask), annotated in text comment.
            

---

## Presence Cache (dataset-aware)

`data/presence.py`:

`def compute_presence_matrix_cached(     adapter, split: str, indices, *, min_pixels: int,     cache_dir: str | Path, force_recompute: bool = False ) -> tuple[np.ndarray, np.ndarray, Path, dict]:     """     Returns:       Y: [N, K] uint8, counts_per_image: [N] ints, cache_path, meta     """`

- Cache key must include: `adapter.dataset_tag`, `adapter.version`, `split`, `min_pixels`, **tuple(indices)** and **K**.
    
- Save `.npz` and `.meta.json`. Validate **indices hash** on load.
    

---

## Selection Algorithms (dataset-agnostic K)

- `selection/balance_greedy.py`: original multi-label L1 reduction.
    
- `selection/balance_caps.py`: target-aware with rare boost + caps.  
    Works for any **K**; computes pool prevalence from cached presence matrix.
    

---

## Evaluator & Summaries

- `eval/runner.py`: `PointingEvaluator(adapter, model, prompt_cfg)`
    
    - Builds 12 (or K) organ queries using `adapter.id2label`/`adapter.label_ids`.
        
    - Uses geometry for letterbox (adapter may expose `recommended_canvas`).
        
    - Writes per-example JSON: `schema_version="v2"`, include `dataset_tag`, `version`, `K`, config hash.
        
- `eval/summarize.py`: unchanged logic but **K-agnostic** (no hardcoded 12).
    

---

## Geometry & Parser (unchanged)

- `geometry/canvas.py`: letterbox and coordinate transforms.
    
- `eval/parser.py`: strict JSON first; fallback `[x,y]`; bounds checks use canvas meta.
    

---

## CLIs (now take `--dataset`)

Each CLI accepts `--dataset {cholecseg8k,endoscape,...}` and loads the adapter via the registry and dataset YAML:

- `build_presence_cache.py`
    
- `select_balanced.py`
    
- `build_fewshot_plan.py` (**supports** `--n-hard-neg`)
    
- `run_pointing.py`  
    Args: `--dataset`, `--exp-id`, `--prompt-name`, `--models`, `--indices-json`, `--fewshot-plan`, `--canvas 768 768` (default from adapter), `--min-pixels`.
    
- `summarize_pointing.py`
    
- `visualize_balance.py`
    

Results live under: `results/pointing/<dataset>/<exp_id>/<prompt>/<model>/...`

---

## Configs (examples)

`configs/dataset_cholecseg8k.yaml`

`name: cholecseg8k hf_name: minwoosun/CholecSeg8k split: train ignore_index: -1 recommended_canvas: [768, 768]`

`configs/dataset_endoscape.yaml`

`name: endoscape root: /path/to/endoscape split: train ignore_index: -1 recommended_canvas: [768, 768] label_map:  # optional remapping or aliasing if needed   1: "Esophagus"   2: "Stomach"   # ...`

---

## Tests

- `test_datasets.py`:
    
    - Build both adapters.
        
    - `example_to_tensors` shapes/types.
        
    - `labels_to_presence_vector` on synthetic masks, **K matches**.
        
    - `sample_point_in_mask` returns inside-mask point when present.
        
- Existing tests (canvas, parser, presence, metrics) must run irrespective of dataset.
    

---

## Acceptance Criteria

- `pytest -q` passes.
    
- `build_presence_cache.py` runs for both datasets; cache paths contain dataset tag/version/K.
    
- `select_balanced.py` produces balanced indices for both datasets with the same configs adapted to K.
    
- `build_fewshot_plan.py` saves plan with stored `point_original` (positives + hard negatives), **no overlap** with eval indices.
    
- `run_pointing.py` reproduces prior CholecSeg8k behavior and works unchanged on EndoScape.
    
- `summarize_pointing.py` prints per-organ tables for variable K.
    
- `visualize_balance.py` shows stacked bars + heatmaps for original vs selected on both datasets.
    

---

## Migration Steps (commit plan)

1. **chore(pkg):** scaffold `src/endopoint/*`, pyproject, ruff, mypy, pytest.
    
2. **feat(datasets):** implement `DatasetAdapter`, `register_dataset`, add `cholecseg8k.py`, `endoscape.py`.
    
3. **feat(data):** presence cache uses adapter; add tests.
    
4. **feat(selection):** greedy + caps (K-agnostic).
    
5. **feat(fewshot):** plan + provider; add hard negatives; multi-dataset fields in JSON.
    
6. **feat(prompts/models):** builders, registry, adapters.
    
7. **feat(eval):** evaluator v2 with config hash; results under `<dataset>/<exp_id>`.
    
8. **feat(cli):** dataset argument; config loading.
    
9. **feat(vis/summarize):** K-agnostic plots/tables.
    
10. **docs:** README with dataset plug-in guide (how to add a new adapter).
    

---

## How to add a new dataset later

1. Create `src/endopoint/datasets/<new>.py` implementing `DatasetAdapter`.
    
2. Register it:
    

`@register_dataset("newds") def build_newds(**cfg) -> DatasetAdapter:     return NewDSAdapter(**cfg)`

3. Add `configs/dataset_<new>.yaml`.
    
4. All CLIs work with `--dataset newds`—no other changes.
    

---

## Code Snippets to Reuse

### Config hashing

`import hashlib, json  def config_hash(obj: dict) -> str:     canon = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")     return hashlib.sha256(canon).hexdigest()[:12]`

### Safe JSON I/O

`from pathlib import Path import json  def load_json(p: Path): return json.loads(Path(p).read_text()) def save_json(p: Path, obj: dict):     p.parent.mkdir(parents=True, exist_ok=True)     p.write_text(json.dumps(obj, indent=2))`

---

## Notes (EndoScape specifics)

- Don’t assume the same organ set or class count as CholecSeg8k. Adapters expose `id2label`, `label_ids`, `ignore_index`, `recommended_canvas`.
    
- All algorithms/metrics must use **K from the adapter**.
    
- If a dataset’s labels differ semantically, keep selection/eval **per-dataset** (no cross-dataset unions needed for this paper). Optionally support `label_aliases` in dataset YAML if you want display harmonization.
    

---

Follow this spec exactly. If something is underspecified, choose the simplest option that preserves current behavior and document the decision in the PR.