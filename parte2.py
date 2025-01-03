import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import mass_function
from scipy import stats

# Funzione modificata per evitare divisione per zero nella KL-Divergence
def kullback_leibler_divergence(p, q, epsilon=1e-10):
    # Aggiungi un piccolo valore di regolarizzazione per evitare log(0)
    p = p / np.sum(p) + epsilon
    q = q / np.sum(q) + epsilon
    return np.sum(p * np.log(p / q))

# Funzione per calcolare le metriche
def calculate_metrics(dn_dlnM_observed, hmf_theoretical, sigma_observed):
    # Calcolare la Divergenza di Kullback-Leibler (con regolarizzazione per evitare zero)
    kl_divergence = kullback_leibler_divergence(dn_dlnM_observed, hmf_theoretical)
    return kl_divergence

# Configurazione della cosmologia
params = {
    'flat': True,
    'H0': 67.77,       # Costante di Hubble
    'Om0': 0.31,       # Densità di materia
    'Ob0': 0.049,      # Densità di barioni
    'sigma8': 0.81,    # Ampiezza delle fluttuazioni della materia
    'ns': 0.96         # Indice spettrale
}
cosmology.addCosmology('myCosmo', params)
cosmo = cosmology.setCosmology('myCosmo')

# Caricamento del dataset
file_path = 'Euclid_ammassi.csv'  # Modifica il percorso del file se necessario
data = pd.read_csv(file_path)

# Dividere i dati per bin di redshift
bins = data['z'].unique()

# Analisi della Halo Mass Function
for z_bin in bins:
    print(f"Analizzando il bin di redshift: {z_bin}")

    # Selezione dei dati per il bin corrente
    subset = data[data['z'] == z_bin]
    masses = subset['mass']
    volume = subset['vol'].iloc[0]  # Assumiamo che il volume sia costante per bin

    # Calcolo dell'istogramma delle masse osservate
    log_masses = np.log10(masses)
    hist, bin_edges = np.histogram(log_masses, bins=50, density=False)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # Normalizzazione in unità di densità (dN/dlnM)
    dlnM = np.log(10) * (bin_edges[1:] - bin_edges[:-1])  # Conversione dei bin
    dn_dlnM_observed = hist / (volume * dlnM)  # Normalizzazione con il volume

    # Calcolo della HMF teorica
    m_theory = 10**bin_centers
    hmf_theoretical = mass_function.massFunction(
        m_theory,
        z=z_bin,
        mdef='vir',
        model='despali16',
        q_out='dndlnM'
    )

    # Calcolo delle metriche
    sigma_observed = np.sqrt(hist) / (volume * dlnM)
    kl_divergence = calculate_metrics(dn_dlnM_observed, hmf_theoretical, sigma_observed)

    # Stampa dei risultati
    print(f"Redshift {z_bin} - Kullback-Leibler Divergence: {kl_divergence:.4e}")

    # Test di Kolmogorov-Smirnov (K-S)
    ks_stat, ks_p_value = stats.ks_2samp(dn_dlnM_observed, hmf_theoretical)
    print(f"Redshift {z_bin} - Statistiche K-S: {ks_stat:.2f}, p-value: {ks_p_value:.2f}")

    # Plot delle distribuzioni (normale + differenza)
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Grafico della HMF
    ax[0].plot(bin_centers, dn_dlnM_observed, label='Osservato', color='blue', marker='o', linestyle='')
    ax[0].plot(bin_centers, hmf_theoretical, label='Teorico', color='red', linestyle='--')
    ax[0].set_title(f"Halo Mass Function (HMF) - Redshift {z_bin}")
    ax[0].set_xlabel("Log10(Massa) [M$_\odot$]")
    ax[0].set_ylabel("dN/dlnM [h$^3$ Mpc$^{-3}$]")
    ax[0].legend()
    ax[0].grid(True, linestyle='--', alpha=0.5)

    # Grafico della differenza
    difference = dn_dlnM_observed - hmf_theoretical
    ax[1].plot(bin_centers, difference, label='Differenza (Osservato - Teorico)', color='green', marker='o', linestyle='')
    ax[1].axhline(0, color='black', linestyle='--')  # Linea orizzontale a y = 0
    ax[1].set_title(f"Differenza tra Osservato e Teorico - Redshift {z_bin}")
    ax[1].set_xlabel("Log10(Massa) [M$_\odot$]")
    ax[1].set_ylabel("Differenza")
    ax[1].legend()
    ax[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()  # Ottimizza la disposizione
    plt.show()

    # Grafico del rapporto Osservato/Teorico
    ratio = dn_dlnM_observed / hmf_theoretical
    sigma_ratio = sigma_observed / hmf_theoretical

    plt.figure(figsize=(10, 6))
    plt.errorbar(bin_centers, ratio, yerr=sigma_ratio, fmt='o', color='black', label='Rapporto Osservato/Teorico')
    plt.axhline(1, color='red', linestyle='--', label='Valore atteso = 1')
    plt.title(f"Confronto dati vs teoria - Redshift {z_bin}")
    plt.xlabel("Log10(Massa) [M$_\odot$]")
    plt.ylabel("Rapporto Osservato/Teorico")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()