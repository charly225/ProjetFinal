import numpy as np
import time
from typing import Callable, Tuple, Dict, List
import matplotlib.pyplot as plt
from scipy.special import roots_legendre, roots_chebyt, roots_laguerre

# ============================================================================
# PARTIE 1 : MÉTHODE DE SIMPSON
# ============================================================================

def simpson(f: Callable, a: float, b: float, n: int) -> float:
    """
    Méthode de Simpson pour l'intégration numérique.
    
    Formule: I_S = (b-a)/(6n) * [f(z_0) + 4f(z_1) + 2f(z_2) + 4f(z_3) + ... + f(z_2n)]
    
    Args:
        f: Fonction à intégrer
        a: Borne inférieure
        b: Borne supérieure
        n: Nombre de sous-intervalles (le nombre total de points sera 2n+1)
    
    Returns:
        Approximation de l'intégrale
    """
    # Création des 2n+1 points
    z = np.linspace(a, b, 2*n + 1)
    h = (b - a) / (6 * n)
    
    # Calcul de la somme avec les coefficients [1, 4, 2, 4, 2, ..., 4, 1]
    somme = f(z[0]) + f(z[-1])  # Premier et dernier terme (coef 1)
    
    # Termes pairs (indices impairs) : coefficient 4
    for i in range(1, 2*n, 2):
        somme += 4 * f(z[i])
    
    # Termes impairs (indices pairs, sauf 0 et 2n) : coefficient 2
    for i in range(2, 2*n, 2):
        somme += 2 * f(z[i])
    
    return h * somme


# ============================================================================
# PARTIE 2 : MÉTHODES DE GAUSS
# ============================================================================

# Tables de Gauss-Legendre pour n=12 points (document page 37)
GAUSS_LEGENDRE_12 = {
    'points': np.array([
        0.98156063424671925069,
        0.90411725637047485667,
        0.76990267419430468703,
        0.5873179542866174472,
        0.36783149899818019375,
        0.12523340851146891547,
        -0.12523340851146891547,
        -0.36783149899818019375,
        -0.58731795428661744729,
        -0.76990267419430468703,
        -0.90411725637047485667,
        -0.98156063424671925069
    ]),
    'poids': np.array([
        0.04717533638651182,
        0.10693932599531843,
        0.16007832854334622,
        0.20316742672306592,
        0.23349253653835480,
        0.24914704581340278,
        0.24914704581340278,
        0.23349253653835480,
        0.20316742672306592,
        0.16007832854334622,
        0.10693932599531843,
        0.04717533638651182
    ])
}

def gauss_legendre(f: Callable, a: float, b: float, n: int = 12) -> float:
    """
    Méthode de Gauss-Legendre pour l'intégration numérique.
    
    Intègre sur [a,b] en se ramenant à [-1,1] par changement de variable:
    t = (b-a)/2 * u + (a+b)/2
    
    Args:
        f: Fonction à intégrer
        a: Borne inférieure
        b: Borne supérieure
        n: Nombre de points (peut être variable)
    
    Returns:
        Approximation de l'intégrale
    """
    # Utiliser la table fixe si n=12, sinon calculer dynamiquement
    if n == 12:
        yi = GAUSS_LEGENDRE_12['points']
        alpha_i = GAUSS_LEGENDRE_12['poids']
    else:
        # Calcul dynamique des racines et poids de Legendre
        yi, alpha_i = roots_legendre(n)
    
    # Changement de variable pour ramener [a,b] vers [-1,1]
    somme = 0.0
    for i in range(n):
        t = (b - a) / 2 * yi[i] + (a + b) / 2
        somme += alpha_i[i] * f(t)
    
    return (b - a) / 2 * somme


# Tables de Chebyshev
def gauss_chebyshev(f: Callable, a: float, b: float, n: int = 12) -> float:
    """
    Méthode de Gauss-Chebyshev pour l'intégration numérique.
    
    Calcule: ∫_{-1}^{1} f(t)/√(1-t²) dt
    Puis adapte pour [a,b] si nécessaire.
    
    Args:
        f: Fonction à intégrer
        a: Borne inférieure (typiquement -1)
        b: Borne supérieure (typiquement 1)
        n: Nombre de points (variable)
    
    Returns:
        Approximation de l'intégrale pondérée
    """
    # Calcul des points de Chebyshev
    k = np.arange(1, n + 1)
    yi = np.cos((2*k - 1) * np.pi / (2*n))
    
    # Poids identiques
    alpha_i = np.pi / n
    
    # Calcul de la somme
    somme = 0.0
    for i in range(n):
        # Changement de variable si nécessaire
        if a != -1 or b != 1:
            t = (b - a) / 2 * yi[i] + (a + b) / 2
        else:
            t = yi[i]
        somme += alpha_i * f(t)
    
    # Facteur d'échelle si l'intervalle n'est pas [-1,1]
    if a != -1 or b != 1:
        return (b - a) / 2 * somme
    
    return somme


# Tables de Gauss-Laguerre pour n=12 points
GAUSS_LAGUERRE_12 = {
    'points': np.array([
        0.115722117358021,
        0.611757484515131,
        1.512610269776419,
        2.833751337743509,
        4.599227639418353,
        6.844525453115181,
        9.621316842456871,
        13.006054993306350,
        17.116855187462260,
        22.151090379396983,
        28.487967250983992,
        37.099121044466926
    ]),
    'poids': np.array([
        0.298952699587270,
        0.498697599716481,
        0.457884396861530,
        0.286062978970674,
        0.123352454587957,
        0.036808402146640,
        0.007385254653473,
        0.000976136448744,
        0.000079253593343,
        0.000003162912506,
        0.000000045351506,
        0.000000000138386
    ])
}

def gauss_laguerre(f: Callable, n: int = 12) -> float:
    """
    Méthode de Gauss-Laguerre pour l'intégration numérique.
    
    Calcule: ∫_{0}^{∞} exp(-t) * f(t) dt
    
    Args:
        f: Fonction à intégrer
        n: Nombre de points (variable)
    
    Returns:
        Approximation de l'intégrale pondérée
    """
    # Utiliser la table fixe si n=12, sinon calculer dynamiquement
    if n == 12:
        yi = GAUSS_LAGUERRE_12['points']
        alpha_i = GAUSS_LAGUERRE_12['poids']
    else:
        # Calcul dynamique des racines et poids de Laguerre
        yi, alpha_i = roots_laguerre(n)
    
    # Calcul de la somme
    somme = 0.0
    for i in range(n):
        somme += alpha_i[i] * f(yi[i])
    
    return somme


# ============================================================================
# PARTIE 3 : SPLINE QUADRATIQUE
# ============================================================================

def spline_quadratique(f: Callable, a: float, b: float, n: int) -> float:
    """
    Intégration par spline quadratique.
    
    Divise [a,b] en n intervalles et interpole par des polynômes de degré 2.
    
    Args:
        f: Fonction à intégrer
        a: Borne inférieure
        b: Borne supérieure
        n: Nombre d'intervalles
    
    Returns:
        Approximation de l'intégrale
    """
    # Création des points d'interpolation
    x = np.linspace(a, b, n + 1)
    y = np.array([f(xi) for xi in x])
    
    # Initialisation des dérivées (spline naturel)
    z = np.zeros(n + 1)
    z[0] = 0  # Condition au bord
    
    # Calcul des dérivées z[i]
    for i in range(n):
        z[i + 1] = 2 * (y[i + 1] - y[i]) / (x[i + 1] - x[i]) - z[i]
    
    # Intégration sur chaque intervalle
    integrale_totale = 0.0
    
    for i in range(n):
        h = x[i + 1] - x[i]
        
        # Coefficients du polynôme quadratique local
        a_i = (z[i + 1] - z[i]) / (2 * h)
        b_i = z[i]
        c_i = y[i]
        
        # Intégrale du polynôme: ∫[x_i, x_{i+1}] (a(t-x_i)² + b(t-x_i) + c) dt
        # = a*h³/3 + b*h²/2 + c*h
        integrale_locale = a_i * h**3 / 3 + b_i * h**2 / 2 + c_i * h
        integrale_totale += integrale_locale
    
    return integrale_totale


# ============================================================================
# PARTIE 4 : FONCTIONS DE TEST
# ============================================================================

def test_function_chebyshev(x):
    """Fonction adaptée à Gauss-Chebyshev: f(x)/√(1-x²)"""
    return 1 / np.sqrt(1 + np.cos(x))

def test_function_laguerre(x):
    """Fonction adaptée à Gauss-Laguerre: exp(-x)*f(x)"""
    return x**2

def test_function_combined(x):
    """Fonction combinant les deux familles"""
    return np.exp(-x/2) / np.sqrt(1 + x**2)

def test_function_neutral(x):
    """Fonction neutre (polynomiale)"""
    return x**2 + 2*x + 1


def calculer_integrale_exacte(f: Callable, a: float, b: float, 
                              methode: str = 'quad') -> float:
    """Calcule l'intégrale exacte (référence) avec scipy."""
    from scipy import integrate
    
    if methode == 'quad':
        result, _ = integrate.quad(f, a, b)
    elif methode == 'romberg':
        result = integrate.romberg(f, a, b)
    else:
        result, _ = integrate.quad(f, a, b)
    
    return result

# ============================================================================
# PARTIE 5 : ANALYSE COMPARATIVE ET VISUALISATION (STYLE PRO)
# ============================================================================

def analyser_convergence(f: Callable, a: float, b: float, exacte: float,
                        valeurs_n: List[int], nom_fonction: str = ""):
    """
    Analyse et génère un rapport graphique professionnel (Style ggplot).
    """
    # Configuration du style "ggplot" comme sur vos images exemples
    plt.style.use('ggplot')
    
    # Couleurs professionnelles (inspirées des images fournies)
    couleurs = {
        'Simpson': '#988ED5',          # Violet
        'Gauss-Legendre': '#E24A33',   # Rouge
        'Gauss-Chebyshev': '#348ABD',  # Bleu
        'Spline': '#777777'            # Gris
    }
    markers = {
        'Simpson': 'v', 
        'Gauss-Legendre': 'o', 
        'Gauss-Chebyshev': 's', 
        'Spline': 'x'
    }

    # Stockage des résultats
    resultats = {k: {'erreurs': [], 'temps': []} for k in couleurs.keys()}
    
    # --- CALCULS ---
    for n in valeurs_n:
        methodes = [
            ('Simpson', lambda: simpson(f, a, b, n)),
            ('Gauss-Legendre', lambda: gauss_legendre(f, a, b, n)),
            ('Spline', lambda: spline_quadratique(f, a, b, n))
        ]
        
        # Ajout conditionnel de Chebyshev et Laguerre
        if a == -1 and b == 1:
            methodes.append(('Gauss-Chebyshev', lambda: gauss_chebyshev(f, a, b, n)))
            
        # Exécution des méthodes
        for nom, func in methodes:
            try:
                # Utilisation de perf_counter pour une précision maximale (microsecondes)
                t0 = time.perf_counter()
                res = func()
                t1 = time.perf_counter()
                
                dt_micro = (t1 - t0) * 1e6  # Conversion en microsecondes
                err = abs(res - exacte)
                
                resultats[nom]['erreurs'].append(err)
                resultats[nom]['temps'].append(dt_micro)
            except Exception as e:
                resultats[nom]['erreurs'].append(np.nan)
                resultats[nom]['temps'].append(np.nan)
                
        # Remplir avec NaN pour les méthodes non applicables (ex: Chebyshev sur [0,5])
        for nom in resultats:
            if len(resultats[nom]['erreurs']) < len(valeurs_n) and resultats[nom]['erreurs']:
                 # Si la méthode a commencé à tourner mais pas cette itération (rare)
                 pass
            elif len(resultats[nom]['erreurs']) == 0:
                 # Si la méthode n'est pas applicable du tout
                 pass

    # --- VISUALISATION ---
    fig = plt.figure(figsize=(16, 10))
    # Grille : 2 lignes, 2 colonnes. La ligne du bas prendra toute la largeur.
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.3, wspace=0.15)

    # 1. GRAPHIQUE FONCTION (Haut Gauche)
    ax_func = fig.add_subplot(gs[0, 0])
    
    # Gestion des bornes infinies pour l'affichage
    plot_a, plot_b = a, b
    if b > 100: plot_b = 25  # Limite affichage pour Laguerre
    
    x_plot = np.linspace(plot_a, plot_b, 500)
    y_plot = [f(x) for x in x_plot]
    
    ax_func.plot(x_plot, y_plot, 'k-', linewidth=2, label='f(x)')
    ax_func.fill_between(x_plot, 0, y_plot, color='gray', alpha=0.3)
    ax_func.set_title(f'Fonction: {nom_fonction}', fontsize=14)
    ax_func.set_ylabel('f(x)')
    
    # 2. GRAPHIQUE CONVERGENCE (Haut Droite)
    ax_err = fig.add_subplot(gs[0, 1])
    
    for nom, data in resultats.items():
        if data['erreurs'] and not all(np.isnan(data['erreurs'])):
            # Filtrer les 0 pour le log
            errs = [e if e > 1e-16 else np.nan for e in data['erreurs']]
            ax_err.loglog(valeurs_n, errs, 
                          marker=markers[nom], 
                          color=couleurs[nom], 
                          label=nom, linewidth=1.5, markersize=6)
            
    ax_err.set_title("Convergence de l'erreur (échelle log)", fontsize=14)
    ax_err.set_ylabel('Erreur Absolue')
    ax_err.set_xlabel('N (points)')
    ax_err.legend(loc='best', frameon=True, facecolor='white', framealpha=0.8)
    ax_err.grid(True, which="both", ls="-", alpha=0.5)

    # 3. GRAPHIQUE TEMPS - BAR CHART GROUPÉ (Bas)
    ax_time = fig.add_subplot(gs[1, :]) # Prend toute la largeur
    
    bar_width = 0.15
    indices = np.arange(len(valeurs_n))
    active_methods = [m for m, d in resultats.items() if d['temps'] and not all(np.isnan(d['temps']))]
    
    # Calcul du décalage pour centrer les barres
    offset_start = - (len(active_methods) * bar_width) / 2
    
    for i, nom in enumerate(active_methods):
        temps = resultats[nom]['temps']
        pos = indices + offset_start + (i * bar_width) + (bar_width/2)
        
        ax_time.bar(pos, temps, bar_width, 
                   label=nom, color=couleurs[nom], alpha=0.9)

    ax_time.set_title('Temps de calcul moyen (micro-secondes)', fontsize=14)
    ax_time.set_ylabel('Temps (µs)')
    ax_time.set_xlabel('N (points)')
    ax_time.set_xticks(indices)
    ax_time.set_xticklabels(valeurs_n)
    ax_time.set_yscale('log') # Échelle log pour voir les petites et grandes valeurs
    ax_time.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.8)
    
    plt.tight_layout()
    nom_fichier = f'Resultat_{nom_fonction.split()[0]}.png'
    plt.savefig(nom_fichier, dpi=150)
    print(f"Graphique sauvegardé : {nom_fichier}")
    plt.show()


def afficher_tableau_comparatif(resultats: Dict, valeurs_n: List[int], nom_fonction: str):
    """
    Affiche un tableau comparatif détaillé dans le terminal.
    
    Args:
        resultats: Dictionnaire {méthode: {'erreurs': [...], 'temps': [...]}}
        valeurs_n: Liste des valeurs de n testées
        nom_fonction: Nom de la fonction testée
    """
    print(f"\n{'='*100}")
    print(f"TABLEAU COMPARATIF - {nom_fonction}")
    print(f"{'='*100}")
    
    # Extraction des méthodes actives (qui ont des données)
    methodes_actives = [m for m, d in resultats.items() 
                       if d['erreurs'] and not all(np.isnan(d['erreurs']))]
    
    # En-tête du tableau
    header = f"{'n':<8}"
    for methode in methodes_actives:
        header += f"{methode:<28}"
    print(header)
    print('-' * 100)
    
    # Lignes de données
    for i, n in enumerate(valeurs_n):
        row = f"{n:<8}"
        for methode in methodes_actives:
            if i < len(resultats[methode]['erreurs']):
                err = resultats[methode]['erreurs'][i]
                temps = resultats[methode]['temps'][i]
                
                if not np.isnan(err) and not np.isnan(temps):
                    row += f"E:{err:8.2e} T:{temps:6.0f}μs   "
                else:
                    row += f"{'N/A':<28}"
            else:
                row += f"{'N/A':<28}"
        print(row)
    
    print('=' * 100)


def generer_graphique_comparaison_globale(tous_resultats: Dict, valeurs_n: List[int]):
    """
    Génère un graphique 2×2 comparant les 4 fonctions.
    
    Args:
        tous_resultats: Dictionnaire {nom_fonction: {méthode: {'erreurs': [...], 'temps': [...]}}}
        valeurs_n: Liste des valeurs de n testées
    """
    plt.style.use('ggplot')
    
    # Couleurs cohérentes avec votre code
    couleurs = {
        'Simpson': '#988ED5',
        'Gauss-Legendre': '#E24A33',
        'Gauss-Chebyshev': '#348ABD',
        'Spline': '#777777'
    }
    markers = {
        'Simpson': 'v',
        'Gauss-Legendre': 'o',
        'Gauss-Chebyshev': 's',
        'Spline': 'x'
    }
    
    # Création de la figure 2×2
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.patch.set_facecolor('white')
    
    # Ordre des fonctions (ajustez selon vos noms exacts)
    fonctions_ordre = list(tous_resultats.keys())[:4]  # Prend les 4 premières
    
    for idx, nom_fonction in enumerate(fonctions_ordre):
        ax = axes[idx // 2, idx % 2]
        resultats = tous_resultats[nom_fonction]
        
        # Tracer les courbes de convergence
        for methode, data in resultats.items():
            if data['erreurs'] and not all(np.isnan(data['erreurs'])):
                # Filtrer les erreurs nulles pour le log
                errs = [e if e > 1e-16 else np.nan for e in data['erreurs']]
                
                ax.loglog(valeurs_n, errs,
                         marker=markers.get(methode, 'o'),
                         color=couleurs.get(methode, 'gray'),
                         label=methode,
                         linewidth=1.5,
                         markersize=6)
        
        # Mise en forme
        ax.set_title(nom_fonction, fontsize=13, fontweight='bold')
        ax.set_xlabel('N (points)', fontsize=11)
        ax.set_ylabel('Erreur Absolue', fontsize=11)
        ax.legend(loc='best', frameon=True, facecolor='white', framealpha=0.8, fontsize=9)
        ax.grid(True, which="both", ls="-", alpha=0.5)
    
    # Titre général
    fig.suptitle('Comparaison Globale - Convergence des Méthodes',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    plt.savefig('Comparaison_Globale.png', dpi=200, bbox_inches='tight')
    print(f"\n✅ Graphique global sauvegardé : Comparaison_Globale.png")
    plt.show()


def analyse_complete_avec_essentiels():
    """
    Version améliorée du main avec les ajouts essentiels.
    """
    print("\n" + "="*100)
    print("ANALYSE COMPLÈTE DES MÉTHODES D'INTÉGRATION NUMÉRIQUE")
    print("="*100)
    
    # valeurs_n = [5, 10, 15, 20, 30, 40, 60, 100]
   
    valeurs_n = [5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 120, 150]
    
    # Stockage global des résultats
    tous_resultats = {}
    
    # ========================================================================
    # TEST 1 : Fonction Chebyshev
    # ========================================================================
    print("\n[1/4] Analyse de la fonction Chebyshev...")
    
    f_cheb = test_function_chebyshev
    a_cheb, b_cheb = -1, 1
    exacte_cheb = calculer_integrale_exacte(f_cheb, a_cheb, b_cheb)
    
    # Analyse graphique
    analyser_convergence(
        f=f_cheb,
        a=a_cheb, b=b_cheb,
        exacte=exacte_cheb,
        valeurs_n=valeurs_n,
        nom_fonction="Singularité (Tchebychev)"
    )
    
    # Récupération des résultats pour le tableau et le graphique global
    resultats_cheb = calculer_resultats_pour_tableau(
        f_cheb, a_cheb, b_cheb, exacte_cheb, valeurs_n
    )
    tous_resultats["Singularité (Tchebychev)"] = resultats_cheb
    
    # Affichage du tableau
    afficher_tableau_comparatif(resultats_cheb, valeurs_n, "Singularité (Tchebychev)")
    
    # ========================================================================
    # TEST 2 : Fonction Laguerre
    # ========================================================================
    print("\n[2/4] Analyse de la fonction Laguerre...")
    
    f_lag = lambda x: np.exp(-x) * x**2
    a_lag, b_lag = 0, 25
    exacte_lag = 2.0
    
    analyser_convergence(
        f=f_lag,
        a=a_lag, b=b_lag,
        exacte=exacte_lag,
        valeurs_n=valeurs_n,
        nom_fonction="Infini (Laguerre)"
    )
    
    resultats_lag = calculer_resultats_pour_tableau(
        f_lag, a_lag, b_lag, exacte_lag, valeurs_n
    )
    tous_resultats["Infini (Laguerre)"] = resultats_lag
    afficher_tableau_comparatif(resultats_lag, valeurs_n, "Infini (Laguerre)")
    
    # ========================================================================
    # TEST 3 : Fonction Standard (Lorentzienne)
    # ========================================================================
    print("\n[3/4] Analyse de la fonction Standard...")
    
    f_std = lambda x: 1 / (1 + x**2)
    a_std, b_std = -1, 1
    exacte_std = np.pi / 2
    
    analyser_convergence(
        f=f_std,
        a=a_std, b=b_std,
        exacte=exacte_std,
        valeurs_n=valeurs_n,
        nom_fonction="Standard (Lorentzienne)"
    )
    
    resultats_std = calculer_resultats_pour_tableau(
        f_std, a_std, b_std, exacte_std, valeurs_n
    )
    tous_resultats["Standard (Lorentzienne)"] = resultats_std
    afficher_tableau_comparatif(resultats_std, valeurs_n, "Standard (Lorentzienne)")
    
    # ========================================================================
    # TEST 4 : Fonction Mixte
    # ========================================================================
    print("\n[4/4] Analyse de la fonction Mixte...")
    
    f_mix = test_function_combined
    a_mix, b_mix = 0, 3
    exacte_mix = calculer_integrale_exacte(f_mix, a_mix, b_mix)
    
    analyser_convergence(
        f=f_mix,
        a=a_mix, b=b_mix,
        exacte=exacte_mix,
        valeurs_n=valeurs_n,
        nom_fonction="Mixte (Exp + Singularité)"
    )
    
    resultats_mix = calculer_resultats_pour_tableau(
        f_mix, a_mix, b_mix, exacte_mix, valeurs_n
    )
    tous_resultats["Mixte (Exp + Singularité)"] = resultats_mix
    afficher_tableau_comparatif(resultats_mix, valeurs_n, "Mixte (Exp + Singularité)")
    
    # ========================================================================
    # GÉNÉRATION DU GRAPHIQUE COMPARATIF GLOBAL
    # ========================================================================
    print("\n" + "="*100)
    print("GÉNÉRATION DU GRAPHIQUE COMPARATIF GLOBAL...")
    print("="*100)
    
    generer_graphique_comparaison_globale(tous_resultats, valeurs_n)
    
    print("\n" + "="*100)
    print("✅ ANALYSE TERMINÉE - Tous les graphiques et tableaux ont été générés")
    print("="*100)
    print("\nFichiers générés :")
    print("  • Resultat_Singularité.png")
    print("  • Resultat_Infini.png")
    print("  • Resultat_Standard.png")
    print("  • Resultat_Mixte.png")
    print("  • Comparaison_Globale.png  ← NOUVEAU !")
    print("="*100 + "\n")


def calculer_resultats_pour_tableau(f: Callable, a: float, b: float, 
                                    exacte: float, valeurs_n: List[int]) -> Dict:
    """
    Calcule les résultats pour une fonction donnée (sans affichage graphique).
    Utile pour remplir le tableau comparatif et le graphique global.
    
    Returns:
        Dict avec {méthode: {'erreurs': [...], 'temps': [...]}}
    """
    resultats = {
        'Simpson': {'erreurs': [], 'temps': []},
        'Gauss-Legendre': {'erreurs': [], 'temps': []},
        'Gauss-Chebyshev': {'erreurs': [], 'temps': []},
        'Spline': {'erreurs': [], 'temps': []}
    }
    
    for n in valeurs_n:
        # Simpson
        try:
            t0 = time.perf_counter()
            res = simpson(f, a, b, n)
            t1 = time.perf_counter()
            resultats['Simpson']['erreurs'].append(abs(res - exacte))
            resultats['Simpson']['temps'].append((t1 - t0) * 1e6)
        except:
            resultats['Simpson']['erreurs'].append(np.nan)
            resultats['Simpson']['temps'].append(np.nan)
        
        # Gauss-Legendre
        try:
            t0 = time.perf_counter()
            res = gauss_legendre(f, a, b, n)
            t1 = time.perf_counter()
            resultats['Gauss-Legendre']['erreurs'].append(abs(res - exacte))
            resultats['Gauss-Legendre']['temps'].append((t1 - t0) * 1e6)
        except:
            resultats['Gauss-Legendre']['erreurs'].append(np.nan)
            resultats['Gauss-Legendre']['temps'].append(np.nan)
        
        # Gauss-Chebyshev (seulement si [-1, 1])
        if a == -1 and b == 1:
            try:
                t0 = time.perf_counter()
                res = gauss_chebyshev(f, a, b, n)
                t1 = time.perf_counter()
                resultats['Gauss-Chebyshev']['erreurs'].append(abs(res - exacte))
                resultats['Gauss-Chebyshev']['temps'].append((t1 - t0) * 1e6)
            except:
                resultats['Gauss-Chebyshev']['erreurs'].append(np.nan)
                resultats['Gauss-Chebyshev']['temps'].append(np.nan)
        else:
            resultats['Gauss-Chebyshev']['erreurs'].append(np.nan)
            resultats['Gauss-Chebyshev']['temps'].append(np.nan)
        
        # Spline
        try:
            t0 = time.perf_counter()
            res = spline_quadratique(f, a, b, n)
            t1 = time.perf_counter()
            resultats['Spline']['erreurs'].append(abs(res - exacte))
            resultats['Spline']['temps'].append((t1 - t0) * 1e6)
        except:
            resultats['Spline']['erreurs'].append(np.nan)
            resultats['Spline']['temps'].append(np.nan)
    
    return resultats

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    # Lancer la nouvelle version avec les essentiels
    analyse_complete_avec_essentiels()