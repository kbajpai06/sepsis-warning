"""Write scripts/sample_request.json: a synthetic septic patient's first N hours."""
import json
import sys

from sepsis.config import DYNAMIC, LABEL, PID, TIME
from sepsis.data.synthetic import make_synthetic

df = make_synthetic(400, seed=7)
pid = df[df[LABEL] == 1][PID].iloc[0]
first_label = int(df[(df[PID] == pid) & (df[LABEL] == 1)][TIME].min())
# default: 2 hours before sepsis onset (onset = first labelled hour + 6)
hours = int(sys.argv[1]) if len(sys.argv) > 1 else first_label + 4
p = df[(df[PID] == pid) & (df[TIME] <= hours)]
obs = [{"hour": int(r[TIME]), **{c: (None if r[c] != r[c] else round(float(r[c]), 2)) for c in DYNAMIC}}
       for _, r in p.iterrows()]
req = {"patient_id": f"demo-{pid}", "age": float(p["Age"].iloc[0]), "gender": int(p["Gender"].iloc[0]),
       "observations": obs}
json.dump(req, open("scripts/sample_request.json", "w"), indent=2)
print("wrote scripts/sample_request.json with", len(obs), "hours")
