"""
foodllm_bench.py — model-agnostic core for the bioactive-food LLM benchmark + ablation.

Runs against ANY OpenAI-compatible chat endpoint. Ollama exposes one at
http://localhost:11434/v1 , so this works with your local 32B model with no API keys.

Pure standard library + `requests`. No Claude / cloud dependencies.
"""
import json, re, time, os, urllib.request, urllib.error

# ---------------------------------------------------------------- vocab / data
def load_vocab(path="data/bioactivity_vocab.json"):
    with open(path) as f:
        return json.load(f)

STRATA = ["Western", "East Asian", "South Asian", "African"]
BIOAVAIL = ["low", "moderate", "high"]

# ---------------------------------------------------------------- prompts
def system_prompt(vocab):
    return (
        "You are an expert food chemist and phytochemistry researcher. "
        "You will be given a food bioactive compound. Predict four properties and "
        "return ONLY a single JSON object with exactly these keys: "
        '"bioactivity", "food_source", "cultural_context", "bioavailability". '
        "Rules:\n"
        f"- bioactivity MUST be exactly one of: {', '.join(vocab)}.\n"
        "- food_source: the main food(s) or plant(s) the compound is found in (short phrase).\n"
        f"- cultural_context MUST be exactly one of: {', '.join(STRATA)}.\n"
        f"- bioavailability MUST be exactly one of: {', '.join(BIOAVAIL)}.\n"
        "Return the JSON object only, no prose, no markdown fences."
    )

def make_prompt(row, condition="A", provenance_note=None):
    """Build the user prompt for a compound under a grounding condition.
    A = baseline (name+SMILES+formula)
    B = + structure descriptors + Murcko scaffold
    C = + provenance note
    D = B + C
    """
    p = [f"Compound name: {row['name']}",
         f"SMILES: {row['canonical_smiles']}",
         f"Molecular formula: {row['formula']}"]
    if condition in ("B", "D"):
        desc = (f"Molecular weight {row.get('mw','?')}, logP {row.get('logp','?')}, "
                f"H-bond donors {row.get('hbd','?')}, acceptors {row.get('hba','?')}, "
                f"TPSA {row.get('tpsa','?')}, rings {row.get('rings','?')}.")
        p.append("Computed structural descriptors: " + desc)
        if row.get("murcko_scaffold"):
            p.append(f"Bemis-Murcko scaffold (SMILES): {row['murcko_scaffold']}")
    if condition in ("C", "D") and provenance_note:
        p.append(f"Botanical/culinary provenance: {provenance_note}")
    p.append('\nReturn the JSON object with keys "bioactivity", "food_source", '
             '"cultural_context", "bioavailability".')
    return "\n".join(p)

# ---------------------------------------------------------------- OpenAI-compatible client
class ChatClient:
    """Minimal OpenAI-compatible /v1/chat/completions client (works with Ollama)."""
    def __init__(self, base_url="http://localhost:11434/v1", model="qwen2.5:32b",
                 api_key="ollama", timeout=180, temperature=0.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.temperature = temperature

    def chat(self, system, user, max_retries=3):
        payload = {
            "model": self.model,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": self.temperature,
            "stream": False,
        }
        data = json.dumps(payload).encode()
        last_err = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    self.base_url + "/chat/completions", data=data,
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {self.api_key}"})
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    out = json.loads(r.read().decode())
                return out["choices"][0]["message"]["content"]
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError,
                    json.JSONDecodeError, TimeoutError) as e:
                last_err = e
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"chat failed after {max_retries} retries: {last_err}")

# ---------------------------------------------------------------- tolerant JSON parse
_KEYS = ["bioactivity", "food_source", "cultural_context", "bioavailability"]

def parse_json(text):
    """Tolerant parse: strip fences, grab first {...}, salvage keys if needed."""
    if not text:
        return {}
    t = text.strip()
    t = re.sub(r"^```(?:json)?", "", t).strip()
    t = re.sub(r"```$", "", t).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return {k: obj.get(k) for k in _KEYS}
        except json.JSONDecodeError:
            pass
    # salvage: line-wise "key": value
    obj = {}
    for k in _KEYS:
        mm = re.search(rf'"?{k}"?\s*[:=]\s*"?([^"\n,}}]+)', t, flags=re.I)
        if mm:
            obj[k] = mm.group(1).strip().strip('"')
    return obj

# ---------------------------------------------------------------- deterministic grading
_STOP = {"and","or","the","of","a","an","from","in","various","many","other","spp","sp",
         "extract","seed","seeds","root","roots","leaf","leaves","fruit","fruits","plant",
         "plants","food","foods","source","sources","etc"}

def _toks(s):
    if not isinstance(s, str):
        return set()
    s = re.sub(r"\([^)]*\)", " ", s.lower())
    words = re.findall(r"[a-z]+", s)
    words = [w[:-1] if (w.endswith("s") and len(w) > 3) else w for w in words]
    return {w for w in words if w not in _STOP and len(w) > 2}

def food_match(pred, ref):
    p, r = _toks(pred), _toks(ref)
    return bool(p and r and (p & r))

def exact(a, b):
    return (str(a).strip().lower() == str(b).strip().lower()) if (a and b) else False

def grade_row(pred, gt):
    return {
        "correct_bioactivity": exact(pred.get("bioactivity"), gt["bioactivity_vocab"]),
        "correct_food": food_match(pred.get("food_source"), gt["food_source"]),
        "correct_cultural": exact(pred.get("cultural_context"), gt["stratum"]),
        "correct_bioavail": exact(pred.get("bioavailability"), gt["bioavailability"]),
    }
