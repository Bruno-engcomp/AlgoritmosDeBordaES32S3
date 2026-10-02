import csv
from pathlib import Path
import matplotlib.pyplot as plt

rows = []
for line in Path("resultados.csv").read_text().splitlines():
    if not line.startswith("CSV,"):
        continue
    _, alg, param, cycles, us, checksum = line.split(",")
    rows.append({
        "alg": alg,
        "param": int(param),
        "cycles": int(cycles),
        "us": float(us),
        "checksum": float(checksum),
    })

for alg in sorted(set(r["alg"] for r in rows)):
    data = [r for r in rows if r["alg"] == alg]
    x = [r["param"] for r in data]
    y = [r["cycles"] for r in data]

    plt.figure()
    plt.plot(x, y, marker="o")
    plt.xlabel("Parâmetro")
    plt.ylabel("Ciclos médios")
    plt.title(f"Benchmark ESP32-S3 — {alg}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{alg.lower()}_ciclos.png", dpi=160)
    plt.show()
