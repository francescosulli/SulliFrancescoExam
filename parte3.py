import pandas as pd
import numpy as np
from scipy.stats import poisson
import emcee
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from colossus.lss import mass_function

# Caricamento del dataset
file_path = 'Euclid_ammassi.csv'  # Modifica il percorso se necessario
data = pd.read_csv(file_path)

# Rimuovere colonne non necessarie
if 'Unnamed: 0' in data.columns:
    data.drop(columns=['Unnamed: 0'], inplace=True)

# Dividere i dati per bin di redshift
bins = sorted(data['z'].unique())
print(f"Bin di redshift disponibili: {bins}")

# Configurazione cosmologica iniziale
cosmology.setCosmology('planck18', print_warnings=False, relspecies=False)

# Funzione per calcolare la log-likelihood
def log_likelihood(params, m_arr, observed_counts, z, volume_bin):
    sigma8, Om0 = params

    # Prior piatto per evitare regioni non fisiche
    if not (0.1 <= sigma8 <= 1.0 and 0.1 <= Om0 <= 1.0):  # Limitazione dei parametri
        return -np.inf

    # Disabilita contributi relativistici se necessario
    relspecies = False if Om0 >= 1.0 else True

    # Configura i parametri cosmologici
    params_cosmo = {
        'flat': True,
        'H0': 67.77,
        'Ob0': 0.049,
        'ns': 0.95,
        'sigma8': sigma8,
        'Om0': Om0,
        'relspecies': False
    }

    # Aggiorna la cosmologia
    try:
        cosmology.addCosmology('myCosmo', params_cosmo)
        cosmology.setCosmology('myCosmo')
    except Exception as e:
        print(f"Errore durante il calcolo della likelihood: {e}")
        return -np.inf

    # Calcola la HMF teorica
    mfunc = mass_function.massFunction(m_arr, z, mdef='vir', model='despali16', q_out='dndlnM')
    expected_counts = mfunc * volume_bin

    # Assicurati che observed_counts e expected_counts abbiano la stessa lunghezza
    min_len = min(len(observed_counts), len(expected_counts))
    observed_counts = observed_counts[:min_len]
    expected_counts = expected_counts[:min_len]

    # Calcolo della log-likelihood
    loglike = np.sum(poisson.logpmf(observed_counts, expected_counts))
    return loglike

# Funzione per la log-posterior
def log_posterior(params, m_arr, observed_counts, z, volume_bin):
    return log_likelihood(params, m_arr, observed_counts, z, volume_bin)

# Parametri iniziali e setup dell'MCMC
n_walkers = 10
n_steps = 250
initial_guess = [0.8, 0.3]  # Punto iniziale nell'MCMC

# Riduzione delle oscillazioni iniziali
perturbation = 0.005  # Ampiezza ridotta per oscillazioni limitate
results = []

# Funzione per calcolare la mediana e intervallo di confidenza
def get_summary(samples):
    median = np.median(samples, axis=0)
    lower = np.percentile(samples, 16, axis=0)
    upper = np.percentile(samples, 84, axis=0)
    return median, lower, upper

# Loop sui bin di redshift
for z_bin in bins:
    print(f"\nAnalisi per il bin di redshift z = {z_bin}")
    subset = data[data['z'] == z_bin]

    # Massa e conteggi osservati
    m_arr = np.logspace(13, 15, 20)  # Range di masse (10^13 - 10^15 M_sun)
    observed_counts, _ = np.histogram(subset['mass'], bins=np.log10(m_arr))
    volume_bin = subset['vol'].mean()  # Volume del bin di redshift

    # Inizializza l'MCMC
    pos = initial_guess + perturbation * np.random.randn(n_walkers, 2)
    sampler = emcee.EnsembleSampler(n_walkers, 2, log_posterior, args=(m_arr, observed_counts, z_bin, volume_bin))
    sampler.run_mcmc(pos, n_steps, progress=True)

    # Salva i risultati
    samples = sampler.get_chain(discard=100, thin=15, flat=True)
    results.append(samples)

    # Calcolo delle statistiche
    median, lower, upper = get_summary(samples)
    print(f"z = {z_bin}:")
    print(f"  σ8 = {median[0]:.3f} [{lower[0]:.3f}, {upper[0]:.3f}]")
    print(f"  Ωm = {median[1]:.3f} [{lower[1]:.3f}, {upper[1]:.3f}]")

    # Plot delle distribuzioni posteriori
    plt.figure(figsize=(10, 5))
    plt.hist(samples[:, 0], bins=30, color='blue', alpha=0.7, label='sigma8')
    plt.hist(samples[:, 1], bins=30, color='green', alpha=0.7, label='Om0')
    plt.title(f'Distribuzioni posteriori per z = {z_bin}')
    plt.xlabel('Parametro')
    plt.ylabel('Frequenza')
    plt.legend()
    plt.show()

    # Plot dell'avanzamento durante l'MCMC (step per walker)
    plt.figure(figsize=(10, 5))
    plt.plot(sampler.chain[:, :, 0].T, color='blue', alpha=0.7)  # Traccia sigma8
    plt.plot(sampler.chain[:, :, 1].T, color='green', alpha=0.7)  # Traccia Om0
    plt.title(f'Avanzamento dei parametri durante l\'MCMC per z = {z_bin}')
    plt.xlabel('Step')
    plt.ylabel('Parametro')
    plt.show()

# Analisi congiunta
print("\nAnalisi congiunta dei bin di redshift:")
all_samples = np.vstack(results)
median, lower, upper = get_summary(all_samples)
print(f"Congiunto:")
print(f"  σ8 = {median[0]:.3f} [{lower[0]:.3f}, {upper[0]:.3f}]")
print(f"  Ωm = {median[1]:.3f} [{lower[1]:.3f}, {upper[1]:.3f}]")

# Plot finale delle distribuzioni posteriori
plt.figure(figsize=(10, 5))
plt.hist(all_samples[:, 0], bins=30, color='blue', alpha=0.7, label='sigma8')
plt.hist(all_samples[:, 1], bins=30, color='green', alpha=0.7, label='Om0')
plt.title('Distribuzioni posteriori congiunte')
plt.xlabel('Parametro')
plt.ylabel('Frequenza')
plt.legend()
plt.show()

# Plot finale dell'avanzamento dei parametri (congiunto)
plt.figure(figsize=(10, 5))
plt.plot(sampler.chain[:, :, 0].T, color='blue', alpha=0.7)  # Traccia sigma8
plt.plot(sampler.chain[:, :, 1].T, color='green', alpha=0.7)  # Traccia Om0
plt.title('Avanzamento dei parametri durante l\'MCMC congiunto')
plt.xlabel('Step')
plt.ylabel('Parametro')
plt.show()