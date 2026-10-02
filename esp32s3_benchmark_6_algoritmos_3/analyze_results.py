import csv
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ciclo medio so faz sentido para o LMS, que processa um fluxo e e
# normalizado por amostra. Os classificadores sao medidos por
# inferencia (microsegundos) e o DWT por transformada inteira.
PER_SAMPLE = {"LMS"}

# Varreduras em potencias de 2 (numero de taps, de vetores-suporte,
# de arvores, tamanho da base, raio da janela) ficam melhores em
# escala log; niveis do DWT e banda do DTW tambem, mas o DTW e
# mantido linear para deixar a RAM visivel como reta.
LOGX = {"LMS", "SVM", "SVMRBF", "RFOREST", "KNN", "DWT_HAAR", "DWT_DB4"}

XLABEL = {
    "LMS": "Número de taps (M)",
    "SVM": "Vetores-suporte (kernel linear)",
    "SVMRBF": "Vetores-suporte (kernel RBF)",
    "RFOREST": "Número de árvores (T)",
    "KNN": "Tamanho da base (N)",
    "DWT_HAAR": "Níveis do DWT (Haar)",
    "DWT_DB4": "Níveis do DWT (db4)",
    "DTW": "Raio da janela Sakoe-Chiba (r)",
}

YLABEL = {
    "LMS": "Ciclos por amostra",
    "SVM": "Ciclos por inferência",
    "SVMRBF": "Ciclos por inferência",
    "RFOREST": "Ciclos por inferência",
    "KNN": "Ciclos por inferência",
    "DWT_HAAR": "Ciclos por transformada",
    "DWT_DB4": "Ciclos por transformada",
    "DTW": "Ciclos por comparação",
}

TITLES = {
    "LMS": "LMS adaptativo (NLMS)",
    "SVM": "SVM kernel linear",
    "SVMRBF": "SVM kernel RBF",
    "RFOREST": "Random Forest",
    "KNN": "k-NN",
    "DWT_HAAR": "DWT Haar",
    "DWT_DB4": "DWT db4",
    "DTW": "DTW (janela Sakoe-Chiba)",
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
    if alg in LOGX:
        plt.xscale("log", base=2)
    if alg in PER_SAMPLE:
        plt.yscale("log", base=2)
    plt.xlabel(XLABEL.get(alg, "Parâmetro"))
    plt.ylabel(YLABEL.get(alg, "Ciclos médios"))
    plt.title(f"Benchmark ESP32-S3 — {TITLES.get(alg, alg)}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{alg.lower()}_ciclos.png", dpi=160)
    plt.close()

print(f"{len(rows)} pontos de dados em {len(set(r['alg'] for r in rows))} algoritmos")
print("graficos: *_ciclos.png | tabela: resultados_resumido.csv")