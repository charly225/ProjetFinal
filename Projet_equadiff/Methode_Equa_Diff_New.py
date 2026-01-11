"""
Résolution d'équations différentielles
Polytech'Paris-UPMC
Implémentation complète avec gestion des erreurs et sauvegarde d'images
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# MÉTHODES D'INTÉGRATION À PAS SÉPARÉS
# ============================================================================

def methode_euler(f, x0, y0, h, n):
    """Méthode d'Euler (page 24) - Ordre 1"""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for k in range(n):
        y[k+1] = y[k] + h * f(x[k], y[k])
        x[k+1] = x[k] + h
    return x, y

def methode_heun(f, x0, y0, h, n):
    """Méthode de Heun (page 27) - Ordre 2"""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for k in range(n):
        phi = f(x[k] + h/2, y[k] + (h/2) * f(x[k], y[k]))
        y[k+1] = y[k] + h * phi
        x[k+1] = x[k] + h
    return x, y

def methode_runge_kutta_4(f, x0, y0, h, n):
    """Méthode de Runge-Kutta d'ordre 4 (page 38) - Ordre 4"""
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    x[0], y[0] = x0, y0
    for k in range(n):
        k1 = h * f(x[k], y[k])
        k2 = h * f(x[k] + h/2, y[k] + k1/2)
        k3 = h * f(x[k] + h/2, y[k] + k2/2)
        k4 = h * f(x[k] + h, y[k] + k3)
        y[k+1] = y[k] + (k1 + 2*k2 + 2*k3 + k4) / 6
        x[k+1] = x[k] + h
    return x, y

# ============================================================================
# EXEMPLES DU DOCUMENT AVEC GESTION DES ERREURS ET IMAGES
# ============================================================================

def exemple_1_compare(h):
    """Exemple 1: z'(x) = 0.1 * x * z(x), z(0) = 1"""
    f = lambda x, y: 0.1 * x * y
    x0, y0, b = 0.0, 1.0, 5.0
    n = int(b / h)
    
    # Calcul numérique
    x_euler, y_euler = methode_euler(f, x0, y0, h, n)
    x_heun, y_heun = methode_heun(f, x0, y0, h, n)
    x_rk4, y_rk4 = methode_runge_kutta_4(f, x0, y0, h, n)
    
    # Solution exacte
    x_exact = np.linspace(0, b, 1000)
    y_exact_courbe = np.exp(0.05 * x_exact**2)
    
    # Graphique
    plt.figure(figsize=(12, 7))
    plt.plot(x_exact, y_exact_courbe, 'b-', label='Solution exacte', linewidth=2.5, alpha=0.8)
    plt.plot(x_euler, y_euler, 'ro-', label='Euler', markersize=6, alpha=0.7)
    plt.plot(x_heun, y_heun, 'gs-', label='Heun', markersize=5, alpha=0.7)
    plt.plot(x_rk4, y_rk4, 'm^-', label='Runge-Kutta 4', markersize=5, alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('z(x)')
    plt.title(f"Exemple 1: h = {h}")
    plt.legend()
    plt.ylim([0.8, 4.0]) # Fixation de l'axe Y
    plt.tight_layout()
    
    # Sauvegarde de l'image
    nom_fichier = f"Exemple1_h{h}.png"
    plt.savefig(nom_fichier, dpi=200)
    plt.show()
    plt.close()
    
    # CALCUL DES ERREURS (Gestion des erreurs)
    err_e = np.max(np.abs(y_euler - np.exp(0.05 * x_euler**2)))
    err_h = np.max(np.abs(y_heun - np.exp(0.05 * x_heun**2)))
    err_r = np.max(np.abs(y_rk4 - np.exp(0.05 * x_rk4**2)))
    
    print(f"\n--- Erreurs Exemple 1 (h = {h}) ---")
    print(f"Euler:        Erreur max = {err_e:.6e}")
    print(f"Heun:         Erreur max = {err_h:.6e}")
    print(f"Runge-Kutta:  Erreur max = {err_r:.6e}")


def exemple_3_compare(h):
    """Exemple 3: z'(x) = π cos(πx) z(x), z(0) = 1"""
    f = lambda x, y: np.pi * np.cos(np.pi * x) * y
    x0, y0, b = 0.0, 1.0, 6.0
    n = int(b / h)
    
    # Calcul numérique
    x_euler, y_euler = methode_euler(f, x0, y0, h, n)
    x_heun, y_heun = methode_heun(f, x0, y0, h, n)
    x_rk4, y_rk4 = methode_runge_kutta_4(f, x0, y0, h, n)
    
    # Solution exacte
    x_exact = np.linspace(0, b, 1000)
    y_exact_courbe = np.exp(np.sin(np.pi * x_exact))
    
    # Graphique
    plt.figure(figsize=(12, 7))
    plt.plot(x_exact, y_exact_courbe, 'b-', label='Solution exacte', linewidth=2.5, alpha=0.8)
    plt.plot(x_euler, y_euler, 'ro-', label='Euler', markersize=6, alpha=0.7)
    plt.plot(x_heun, y_heun, 'gs-', label='Heun', markersize=5, alpha=0.7)
    plt.plot(x_rk4, y_rk4, 'm^-', label='Runge-Kutta 4', markersize=5, alpha=0.7)
    
    # FIXATION DE L'AXE Y (pour éviter l'écrasement de la solution exacte)
    plt.ylim([0, 4.0]) 
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('z(x)')
    plt.title(f"Exemple 3: h = {h}")
    plt.legend()
    plt.tight_layout()
    
    # Sauvegarde de l'image
    nom_fichier = f"Exemple3_h{h}.png"
    plt.savefig(nom_fichier, dpi=200)
    plt.show()
    plt.close()
    
    # CALCUL DES ERREURS (Gestion des erreurs)
    err_e = np.max(np.abs(y_euler - np.exp(np.sin(np.pi * x_euler))))
    err_h = np.max(np.abs(y_heun - np.exp(np.sin(np.pi * x_heun))))
    err_r = np.max(np.abs(y_rk4 - np.exp(np.sin(np.pi * x_rk4))))
    
    print(f"\n--- Erreurs Exemple 3 (h = {h}) ---")
    print(f"Euler:        Erreur max = {err_e:.6e}")
    print(f"Heun:         Erreur max = {err_h:.6e}")
    print(f"Runge-Kutta:  Erreur max = {err_r:.6e}")


# ============================================================================
# LANCEMENT DES SIMULATIONS
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Tests des exemples du document - Comparaison des méthodes")
    print("=" * 70)
    
    # Exemple 1 avec différents pas
    print("\n" + "="*70)
    print("EXEMPLE 1: z'(x) = 0.1×x×z(x), z(0) = 1")
    print("="*70)
    
    print("\n>>> Avec un pas h = 0.5")
    exemple_1_compare(0.5)
    
    print("\n>>> Avec un pas h = 0.3")
    exemple_1_compare(0.3)

    print("\n>>> Avec un pas h = 0.15")
    exemple_1_compare(0.15)
    
    # Exemple 3
    print("\n" + "="*70)
    print("EXEMPLE 3: z'(x) = π cos(πx) z(x), z(0) = 1")
    print("="*70)
    # Pas demandés : 0.5, 0.3, 0.15 et 0.06

    print("\n>>> Avec un pas h = 0.5")
    exemple_3_compare(0.5)

    print("\n>>> Avec un pas h = 0.3")
    exemple_3_compare(0.3)

    print("\n>>> Avec un pas h = 0.15")
    exemple_3_compare(0.15)

    print("\n>>> Avec un pas h = 0.06")
    exemple_3_compare(0.06)
    
    print("\n" + "="*50)
    print("Simulation terminée. Images PNG sauvegardées.")
    print("="*50)