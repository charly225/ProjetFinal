import numpy as np
import time
import os
from typing import Callable, Dict, List
import matplotlib.pyplot as plt
from scipy.special import roots_legendre, roots_laguerre
from scipy import integrate

# ============================================================================
# PARTIE 1 : MÉTHODES D'INTÉGRATION (NETTOYÉES)
# ============================================================================

def simpson(f: Callable, a: float, b: float, n: int) -> float:
    z = np.linspace(a, b, 2*n + 1)
    h = (b - a) / (6 * n)
    somme = f(z[0]) + f(z[-1])
    for i in range(1, 2*n, 2): somme += 4 * f(z[i])
    for i in range(2, 2*n, 2): somme += 2 * f(z[i])
    return h * somme

def gauss_legendre(f: Callable, a: float, b: float, n: int = 12) -> float:
    yi, alpha_i = roots_legendre(n)
    somme = 0.0
    for i in range(n):
        t = (b - a) / 2 * yi[i] + (a + b) / 2
        somme += alpha_i[i] * f(t)
    return (b - a) / 2 * somme

def gauss_chebyshev(f: Callable, n: int = 12) -> float:
    """Calcule ∫_{-1}^{1} f(t)/√(1-t²) dt"""
    k = np.arange(1, n + 1)
    yi = np.cos((2*k - 1) * np.pi / (2*n))
    alpha_i = np.pi / n
    return np.sum(alpha_i * f(yi))

def gauss_laguerre_specific(f: Callable, n: int = 12) -> float:
    """Calcule ∫_{0}^{∞} f(t) * exp(-t) dt"""
    yi, alpha_i = roots_laguerre(n)
    return np.sum(alpha_i * f(yi))

def spline_quadratique(f: Callable, a: float, b: float, n: int) -> float:
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    z = np.zeros(n + 1)
    z[0] = 0 
    for i in range(n):
        z[i + 1] = 2 * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - z[i]
    integrale_totale = 0.0
    for i in range(n):
        h = x[i + 1] - x[i]
        a_i = (z[i + 1] - z[i]) / (2 * h)
        b_i = z[i]
        c_i = y[i]
        integrale_totale += a_i * h**3 / 3 + b_i * h**2 / 2 + c_i * h
    return integrale_totale

# ============================================================================
# PARTIE 2 : ANALYSE ET AFFICHAGE (TABLEAUX & GRAPHIQUES)
# ============================================================================

def calculer_integrale_exacte(f: Callable, a: float, b: float, famille: str) -> float:
    if famille == 'chebyshev':
        g = lambda x: f(x) / np.sqrt(1 - x**2 + 1e-15)
        res, _ = integrate.quad(g, -1, 1)
    elif famille == 'laguerre':
        g = lambda x: f(x) * np.exp(-x)
        res, _ = integrate.quad(g, 0, np.inf)
    else:
        res, _ = integrate.quad(f, a, b)
    return res

def afficher_tableau_console(nom_f: str, resultats: Dict, valeurs_n: List[int]):
    """Affiche un résumé des erreurs et du temps dans la console."""
    print(f"\n" + "="*115)
    print(f"RÉSUMÉ DES PERFORMANCES : {nom_f.upper()}")
    print("="*115)
    
    # Header
    header = f"{'n':<5} |"
    for methode in resultats.keys():
        if any(not np.isnan(e) for e in resultats[methode]['erreurs']):
            header += f" {methode:<25} |"
    print(header)
    print("-" * 115)

    # Lignes
    for i, n in enumerate(valeurs_n):
        row = f"{n:<5} |"
        for methode in resultats.keys():
            if any(not np.isnan(e) for e in resultats[methode]['erreurs']):
                err = resultats[methode]['erreurs'][i]
                tps = resultats[methode]['temps'][i]
                if np.isnan(err):
                    row += f" {'N/A':<25} |"
                else:
                    row += f" E:{err:8.2e} T:{tps:6.1f}µs |"
        print(row)
    print("="*115 + "\n")

def analyser_et_tracer(config: Dict, valeurs_n: List[int]):
    f = config['f']
    a, b = config['a'], config['b']
    famille = config['famille']
    nom_f = config['nom']
    exacte = calculer_integrale_exacte(f, a, b, famille)

    couleurs = {'Simpson': '#988ED5', 'Gauss-Legendre': '#E24A33', 'Spline': '#777777', 
                'Gauss-Chebyshev': '#348ABD', 'Gauss-Laguerre': '#467821'}
    markers = {'Simpson': 'v', 'Gauss-Legendre': 'o', 'Spline': 'x', 
               'Gauss-Chebyshev': 's', 'Gauss-Laguerre': 'D'}

    if famille == 'chebyshev':
        f_std = lambda x: f(x) / np.sqrt(1 - x**2 + 1e-15)
        nom_specifique = 'Gauss-Chebyshev'
    elif famille == 'laguerre':
        f_std = lambda x: f(x) * np.exp(-x)
        nom_specifique = 'Gauss-Laguerre'
    else:
        f_std = f
        nom_specifique = None

    resultats = {m: {'erreurs': [], 'temps': []} for m in couleurs.keys()}

    for n in valeurs_n:
        # 1. Simpson
        t0 = time.perf_counter(); res = simpson(f_std, a, b, n); t1 = time.perf_counter()
        resultats['Simpson']['erreurs'].append(abs(res - exacte))
        resultats['Simpson']['temps'].append((t1 - t0) * 1e6)

        # 2. Legendre
        t0 = time.perf_counter(); res = gauss_legendre(f_std, a, b, n); t1 = time.perf_counter()
        resultats['Gauss-Legendre']['erreurs'].append(abs(res - exacte))
        resultats['Gauss-Legendre']['temps'].append((t1 - t0) * 1e6)

        # 3. Spline
        t0 = time.perf_counter(); res = spline_quadratique(f_std, a, b, n); t1 = time.perf_counter()
        resultats['Spline']['erreurs'].append(abs(res - exacte))
        resultats['Spline']['temps'].append((t1 - t0) * 1e6)

        # 4. Spécifique (Gauss-Chebyshev ou Laguerre)
        for m_spec in ['Gauss-Chebyshev', 'Gauss-Laguerre']:
            if nom_specifique == m_spec:
                t0 = time.perf_counter()
                res = gauss_chebyshev(f, n) if famille == 'chebyshev' else gauss_laguerre_specific(f, n)
                t1 = time.perf_counter()
                resultats[m_spec]['erreurs'].append(abs(res - exacte))
                resultats[m_spec]['temps'].append((t1 - t0) * 1e6)
            else:
                resultats[m_spec]['erreurs'].append(np.nan)
                resultats[m_spec]['temps'].append(np.nan)

    # --- AFFICHAGE CONSOLE ---
    afficher_tableau_console(nom_f, resultats, valeurs_n)

    # --- TRACÉ GRAPHIQUE ---
    plt.style.use('ggplot')
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.2)

    # 1. Graphe Fonction
    ax1 = fig.add_subplot(gs[0, 0])
    x_limit = 25 if famille == 'laguerre' else b
    x_plot = np.linspace(a + (1e-5 if famille=='chebyshev' else 0), x_limit - (1e-5 if famille=='chebyshev' else 0), 500)
    y_plot = [f_std(x) for x in x_plot]
    ax1.plot(x_plot, y_plot, color='black', lw=2)
    ax1.fill_between(x_plot, 0, y_plot, color='gray', alpha=0.3)
    ax1.set_title(f"Fonction: {nom_f}", fontsize=14)
    ax1.set_ylabel("f(x) pondérée")

    # 2. Convergence
    ax2 = fig.add_subplot(gs[0, 1])
    for methode, data in resultats.items():
        if not all(np.isnan(data['erreurs'])):
            errs = [e if e > 1e-16 else 1e-16 for e in data['erreurs']]
            ax2.loglog(valeurs_n, errs, label=methode, color=couleurs[methode], 
                       marker=markers[methode], markersize=7, alpha=0.8)
    ax2.set_title("Convergence de l'erreur (Log-Log)", fontsize=14)
    ax2.set_ylabel("Erreur Absolue")
    ax2.set_xlabel("N (Nombre de points)")
    ax2.legend(facecolor='white', frameon=True)

    # 3. Temps
    ax3 = fig.add_subplot(gs[1, :])
    active_methods = [m for m in resultats if not all(np.isnan(resultats[m]['temps']))]
    n_meth = len(active_methods)
    width = 0.8 / n_meth
    indices = np.arange(len(valeurs_n))

    for i, m in enumerate(active_methods):
        pos = indices - 0.4 + (i + 0.5) * width
        ax3.bar(pos, resultats[m]['temps'], width, label=m, color=couleurs[m], alpha=0.8)

    ax3.set_title("Temps de calcul moyen (Échelle Log)", fontsize=14)
    ax3.set_ylabel("Temps (µs)")
    ax3.set_xlabel("N (Nombre de points)")
    ax3.set_xticks(indices)
    ax3.set_xticklabels(valeurs_n)
    ax3.set_yscale('log')
    ax3.legend(loc='upper left', facecolor='white', frameon=True)

    # SAUVEGARDE
    nom_fichier = f"Resultat_{nom_f.replace(' ', '_').replace('(', '').replace(')', '')}.png"
    plt.savefig(nom_fichier, dpi=300, bbox_inches='tight')
    print(f"Graphique sauvegardé : {nom_fichier}")
    plt.show()

# ============================================================================
# PARTIE 3 : MAIN
# ============================================================================

if __name__ == "__main__":
    # Liste de N points à tester
    valeurs_n = [5, 10, 20, 40, 60, 100, 150]

    tests = [
        {
            "nom": "Chebyshev (Singularité)",
            "f": lambda x: np.exp(x), 
            "a": -1, "b": 1, "famille": "chebyshev"
        },
        {
            "nom": "Laguerre (Infini)",
            "f": lambda x: x**2,
            "a": 0, "b": 25, "famille": "laguerre"
        },
        {
            "nom": "Mixte (Singularité locale)",
            "f": lambda x: 1 / np.sqrt(x + 0.001), 
            "a": 0, "b": 1, "famille": "standard" 
        },
        {
            "nom": "Neutre (Polynôme)",
            "f": lambda x: x**3 + 2*x**2 + 1,
            "a": -2, "b": 2, "famille": "standard"
        }
    ]

    for t in tests:
        analyser_et_tracer(t, valeurs_n)