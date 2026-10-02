import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ciclo medio por amostra so faz sentido para os filtros que
# processam um fluxo; MFCC e Kalman sao medidos por chamada.
PER_SAMPLE = {"GOERTZEL", "MEDIAN", "MAVG"}

XLABEL = {
    "MFCC": "Tamanho da FFT (N)",
    "KALMAN": "Número de estados (n)",
    "GOERTZEL": "Amostras por bloco (N)",
    "MEDIAN": "Janela (k)",
    "MAVG": "Janela (N)",
}

YLABEL = {
    "MFCC": "Ciclos por janela de 30 ms",
    "KALMAN": "Ciclos por atualização",
    "GOERTZEL": "Ciclos por amostra",
    "MEDIAN": "Ciclos por amostra",
    "MAVG": "Ciclos por amostra",
}

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

if not rows:
    raise SystemExit("nenhuma linha CSV encontrada em resultados.csv")

with open("resultados_resumido.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["algoritmo", "parametro", "ciclos", "tempo_ns"])
    for r in rows:
        w.writerow([r["alg"], r["param"], r["cycles"], int(r["us"])])

for alg in sorted(set(r["alg"] for r in rows)):
    data = sorted(
        (r for r in rows if r["alg"] == alg),
        key=lambda r: r["param"],
    )
    x = [r["param"] for r in data]
    y = [r["cycles"] for r in data]

    plt.figure()
    plt.plot(x, y, marker="o")
    if alg in PER_SAMPLE:
        plt.xscale("log", base=2)
    plt.xlabel(XLABEL.get(alg, "Parâmetro"))
    plt.ylabel(YLABEL.get(alg, "Ciclos médios"))
    plt.title(f"Benchmark ESP32-S3 — {alg}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{alg.lower()}_ciclos.png", dpi=160)
    plt.close()

print(f"{len(rows)} pontos de dados em {len(set(r['alg'] for r in rows))} algoritmos")
print("graficos: *_ciclos.png | tabela: resultados_resumido.csv")
