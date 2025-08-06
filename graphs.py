import pandas as pd
import matplotlib.pyplot as plt

# === 1. Carregar o CSV ===
df = pd.read_csv("rba-dataset.csv")  # <-- troca pro nome real se for outro

# === 2. Limpar e converter coluna booleana ===
df.columns = df.columns.str.strip()
if df["Is Attack IP"].dtype == object:
    df["Is Attack IP"] = df["Is Attack IP"].str.lower().map({"true": True, "false": False})

# === 3. Filtrar apenas ataques ===
df_attacks = df[df["Is Attack IP"] == True]
print(f"\n✅ Total de ataques: {len(df_attacks)}")

# === 4. Top 10 países com mais ataques ===
country_counts = df_attacks["Country"].value_counts().head(10)

# === 5. Top 15 navegadores com ataques ===
browser_counts = df_attacks["Browser Name and Version"].value_counts().head(15)

# === 6. RTT - histograma (removendo outliers extremos) ===
rtt_values = df_attacks["Round-Trip Time [ms]"]
rtt_filtered = rtt_values[rtt_values < rtt_values.quantile(0.95)]

# === Gráfico 1: Países ===
plt.figure(figsize=(8, 4))
country_counts.plot(kind="bar", color="skyblue")
plt.title("🌍 Top 10 Países com Mais Ataques")
plt.ylabel("Número de ataques")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Gráfico 2: Navegadores ===
plt.figure(figsize=(10, 4))
browser_counts.plot(kind="bar", color="salmon")
plt.title("🌐 Top 15 Navegadores em Tentativas de Ataque")
plt.ylabel("Número de ataques")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# === Gráfico 3: RTT ===
plt.figure(figsize=(8, 4))
plt.hist(rtt_filtered, bins=50, color="mediumseagreen")
plt.title("⚡ Distribuição de RTT em Ataques")
plt.xlabel("RTT (ms)")
plt.ylabel("Número de ataques")
plt.tight_layout()
plt.show()
