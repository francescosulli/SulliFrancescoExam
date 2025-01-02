import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde

# Caricamento del dataset
file_path = 'Euclid_ammassi.csv'  # Modifica il percorso se necessario
data = pd.read_csv(file_path)

# Controllo del dataset
print("Prime righe del dataset:")
print(data.head())

# Dividere i dati per bin di redshift
bins = data['z'].unique()
print(f"Bin di redshift disponibili: {bins}")

# 1. Print di tutti i dati in due grafici uno affianco all’altro
# Istogramma della massa (normale e logaritmica)
plt.figure(figsize=(12, 6))

# Istogramma della massa (normale)
plt.subplot(1, 2, 1)
plt.hist(data['mass'], bins=100, alpha=0.7, color='blue')
plt.title("Distribuzione delle Masse (Normale)")
plt.xlabel("Massa [M$_\odot$]")
plt.ylabel("Frequenza")
plt.grid(True, linestyle='--', alpha=0.5)

# Istogramma della massa (logaritmica)
plt.subplot(1, 2, 2)
plt.hist(np.log10(data['mass']), bins=100, alpha=0.7, color='green')
plt.title("Distribuzione delle Masse (Logaritmica)")
plt.xlabel("Log10(Massa) [M$_\odot$]")
plt.ylabel("Frequenza")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 2. Confronto tra massa e redshift e tra massa e volume (uno accanto all'altro)
plt.figure(figsize=(12, 6))

# Massa vs Redshift
plt.subplot(1, 2, 1)
plt.scatter(data['z'], data['mass'], alpha=0.5, color='orange')
plt.title("Massa vs Redshift")
plt.xlabel("Redshift (z)")
plt.ylabel("Massa [M$_\odot$]")
plt.grid(True, linestyle='--', alpha=0.5)

# Massa vs Volume
plt.subplot(1, 2, 2)
plt.scatter(data['vol'], data['mass'], alpha=0.5, color='purple')
plt.title("Massa vs Volume")
plt.xlabel("Volume")
plt.ylabel("Massa [M$_\odot$]")
plt.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# 3. Istogramma con i 3 bin divisi per colore con massa logaritmica
plt.figure(figsize=(10, 7))
colors = ['blue', 'green', 'red']
for i, z_bin in enumerate(bins):
    subset = data[data['z'] == z_bin]
    masses = subset['mass']
    plt.hist(np.log10(masses), bins=100, alpha=0.5, color=colors[i], label=f'Redshift {z_bin}')

plt.title("Distribuzione delle Masse - Tutti i Bin di Redshift (Logaritmica sulla Massa)")
plt.xlabel("Log10(Massa) [M$_\odot$]")
plt.ylabel("Frequenza")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

# Funzione per creare l'istogramma con stima dell'andamento (logaritmica sulla massa)
def plot_mass_distribution_with_kde(z_bin):
    subset = data[data['z'] == z_bin]
    masses = subset['mass']

    # Stima dell'andamento (distribuzione normale con KDE)
    kde = stats.gaussian_kde(np.log10(masses))  # Applicare log alla massa
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    plt.hist(np.log10(masses), bins=100, alpha=0.7, color='blue', label="Istogramma")
    plt.plot(x, kde(x), color='red', lw=2, label="Andamento stimato")

    # Statistiche
    mean_mass = np.mean(masses)
    median_mass = np.median(masses)
    std_mass = np.std(masses)
    total_data = len(masses)

    # Visualizzazione dei risultati
    plt.title(f"Distribuzione delle Masse - Redshift {z_bin}")
    plt.xlabel("Log10(Massa) [M$_\odot$]")
    plt.ylabel("Frequenza")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    print(f"Bin di Redshift {z_bin}:")
    print(f"Numero di dati: {total_data}")
    print(f"Media della massa: {mean_mass:.2f} M$_\odot$")
    print(f"Mediana della massa: {median_mass:.2f} M$_\odot$")
    print(f"Deviazione standard della massa: {std_mass:.2f} M$_\odot$")
    print("\n")

# 4, 5, 6, 7 Istogrammi singoli con stima dell'andamento per ciascun bin

# Funzione per creare l'istogramma con stima dell'andamento usando KDE
def plot_mass_distribution_with_kde(z_bin):
    subset = data[data['z'] == z_bin]
    masses = subset['mass']

    # Rimuoviamo eventuali valori zero o negativi (anche se sappiamo che non ci sono)
    masses = masses[masses > 0]

    # Calcoliamo l'istogramma logaritmico della massa
    log_masses = np.log10(masses)

    # Istogramma
    plt.hist(log_masses, bins=100, alpha=0.7, color='blue', density=True, label="Istogramma")

    # Calcoliamo la KDE (stima della densità)
    kde = gaussian_kde(log_masses, bw_method='scott')  # Stima della densità usando il metodo di Scott
    xmin, xmax = log_masses.min(), log_masses.max()
    x = np.linspace(xmin, xmax, 1000)  # Gamma di valori da stimare per la densità
    plt.plot(x, kde(x), color='red', lw=2, label="Densità stimata (KDE)")

    # Statistiche
    mean_mass = np.mean(masses)
    median_mass = np.median(masses)
    std_mass = np.std(masses)
    total_data = len(masses)

    # Visualizzazione dei risultati
    plt.title(f"Distribuzione delle Masse - Redshift {z_bin}")
    plt.xlabel("Log10(Massa) [M$_\odot$]")
    plt.ylabel("Frequenza (Normale)")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

    # Stampa delle statistiche
    print(f"Bin di Redshift {z_bin}:")
    print(f"Numero di dati: {total_data}")
    print(f"Media della massa: {mean_mass:.2f} M$_\odot$")
    print(f"Mediana della massa: {median_mass:.2f} M$_\odot$")
    print(f"Deviazione standard della massa: {std_mass:.2f} M$_\odot$")
    print("\n")

# 4, 5, 6. Istogrammi singoli con stima dell'andamento (KDE) per ciascun bin
for z_bin in bins:
    plot_mass_distribution_with_kde(z_bin)

# 9. Penultimo grafico: Confronto tra i 3 bin con densità stimate (KDE)
plt.figure(figsize=(10, 7))
colors = ['blue', 'green', 'red']

for i, z_bin in enumerate(bins):
    subset = data[data['z'] == z_bin]
    masses = subset['mass']

    # Rimuoviamo eventuali valori zero o negativi (se presenti)
    masses = masses[masses > 0]
    log_masses = np.log10(masses)

    # Calcoliamo la KDE
    kde = gaussian_kde(log_masses, bw_method='scott')  # Metodo di smoothing
    x = np.linspace(log_masses.min(), log_masses.max(), 1000)

    # Plot delle densità stimate
    plt.plot(x, kde(x), color=colors[i], lw=2, label=f"Redshift {z_bin}")

    # Istogramma
    plt.hist(log_masses, bins=100, alpha=0.2, color=colors[i], density=True)

plt.title("Confronto tra i Bin di Redshift con Densità Stimate (KDE)")
plt.xlabel("Log10(Massa) [M$_\odot$]")
plt.ylabel("Densità (Normale)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()


# 8. Confronto tra le medie con grafico a linee
means = []
std_devs = []
labels = []
for z_bin in bins:
    subset = data[data['z'] == z_bin]
    masses = subset['mass']
    mean_mass = np.mean(masses)
    std_dev = np.std(masses)
    means.append(mean_mass)
    std_devs.append(std_dev)
    labels.append(f"Redshift {z_bin}")

# Creazione del grafico a linee per il confronto tra le medie
plt.figure(figsize=(8, 6))
plt.plot(labels, means, marker='o', linestyle='-', color='orange', label="Media della Massa")
plt.fill_between(labels, np.array(means) - np.array(std_devs), np.array(means) + np.array(std_devs), color='orange', alpha=0.3, label="Deviazione Standard")
plt.title("Confronto tra le Medie delle Masse per Bin di Redshift")
plt.xlabel("Bin di Redshift")
plt.ylabel("Massa Media [M$_\odot$]")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()