"""The fit step with the lowest held-out loss (step 0 excluded) in a gains_fit.py log; prints "STEP NLL".
Large fits overfit after a few hundred steps: export this step, not the last one.
    python best_step.py FIT.log"""
import re
import sys

if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
    sys.exit(__doc__)
best = None
for m in re.finditer(r"step (\d+) heldout nll ([\d.]+)", open(sys.argv[1]).read()):
    s, v = int(m.group(1)), float(m.group(2))
    if s and (best is None or v < best[1]):
        best = (s, v)
print(f"{best[0]} {best[1]}" if best else "0 nan")
