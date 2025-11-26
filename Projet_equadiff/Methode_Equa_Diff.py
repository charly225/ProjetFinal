"""
Résolution d'équations différentielles
Polytech'Paris-UPMC
Implémentation fidèle au document du cours
"""

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# MÉTHODES D'INTÉGRATION À PAS SÉPARÉS
# ============================================================================

def methode_euler(f, x0, y0, h, n):
    """
    Méthode d'Euler (page 24) - Ordre 1
    
    y_{k+1} = y_k + h * f(x_k, y_k)
    
    Args:
        f: fonction f(x, y)
        x0: condition initiale x_0
        y0: condition initiale y_0
        h: pas d'intégration
        n: nombre de pas
    
    Returns:
        x: array des x_k
        y: array des y_k
    """
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    
    x[0] = x0
    y[0] = y0
    
    for k in range(n):
        y[k+1] = y[k] + h * f(x[k], y[k])
        x[k+1] = x[k] + h
    
    return x, y


def methode_heun(f, x0, y0, h, n):
    """
    Méthode de Heun (page 27) - Ordre 2
    
    φ(x, y, h) = f(x + h/2, y + h/2 * f(x, y))
    y_{k+1} = y_k + h * φ(x_k, y_k, h)
    
    Args:
        f: fonction f(x, y)
        x0: condition initiale x_0
        y0: condition initiale y_0
        h: pas d'intégration
        n: nombre de pas
    
    Returns:
        x: array des x_k
        y: array des y_k
    """
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    
    x[0] = x0
    y[0] = y0
    
    for k in range(n):
        phi = f(x[k] + h/2, y[k] + (h/2) * f(x[k], y[k]))
        y[k+1] = y[k] + h * phi
        x[k+1] = x[k] + h
    
    return x, y


def methode_euler_modifiee(f, x0, y0, h, n):
    """
    Méthode d'Euler modifiée (page 37) - Ordre 2
    
    φ(x, y, h) = f(x, y)/2 + 1/2 * f(x + h, y + h*f(x, y))
    y_{k+1} = y_k + h * φ(x_k, y_k, h)
    
    Args:
        f: fonction f(x, y)
        x0: condition initiale x_0
        y0: condition initiale y_0
        h: pas d'intégration
        n: nombre de pas
    
    Returns:
        x: array des x_k
        y: array des y_k
    """
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    
    x[0] = x0
    y[0] = y0
    
    for k in range(n):
        f1 = f(x[k], y[k])
        f2 = f(x[k] + h, y[k] + h * f1)
        phi = (f1 + f2) / 2
        y[k+1] = y[k] + h * phi
        x[k+1] = x[k] + h
    
    return x, y


def methode_runge_kutta_4(f, x0, y0, h, n):
    """
    Méthode de Runge-Kutta d'ordre 4 (page 38) - Ordre 4
    
    k₁ = h*f(xₙ, yₙ)
    k₂ = h*f(xₙ + h/2, yₙ + k₁/2)
    k₃ = h*f(xₙ + h/2, yₙ + k₂/2)
    k₄ = h*f(xₙ + h, yₙ + k₃)
    
    yₙ₊₁ = yₙ + 1/6(k₁ + 2k₂ + 2k₃ + k₄)
    
    Args:
        f: fonction f(x, y)
        x0: condition initiale x_0
        y0: condition initiale y_0
        h: pas d'intégration
        n: nombre de pas
    
    Returns:
        x: array des x_k
        y: array des y_k
    """
    x = np.zeros(n + 1)
    y = np.zeros(n + 1)
    
    x[0] = x0
    y[0] = y0
    
    for k in range(n):
        k1 = h * f(x[k], y[k])
        k2 = h * f(x[k] + h/2, y[k] + k1/2)
        k3 = h * f(x[k] + h/2, y[k] + k2/2)
        k4 = h * f(x[k] + h, y[k] + k3)
        
        y[k+1] = y[k] + (k1 + 2*k2 + 2*k3 + k4) / 6
        x[k+1] = x[k] + h
    
    return x, y


# ============================================================================
# EXEMPLES DU DOCUMENT
# ============================================================================

def exemple_1_compare(h):
    """
    Exemple page 5, 7, 26, 32, 39 - Comparaison des 3 méthodes
    
    Équation: z'(x) = 0.1 * x * z(x)
    Condition: z(0) = 1
    Solution exacte: z(x) = exp(0.05 * x²)
    """
    f = lambda x, y: 0.1 * x * y
    x0, y0 = 0.0, 1.0
    b = 5.0
    n = int(b / h)
    
    # Calcul avec les 3 méthodes
    x_euler, y_euler = methode_euler(f, x0, y0, h, n)
    x_heun, y_heun = methode_heun(f, x0, y0, h, n)
    x_rk4, y_rk4 = methode_runge_kutta_4(f, x0, y0, h, n)
    
    # Solution exacte
    x_exact = np.linspace(0, b, 1000)
    y_exact = np.exp(0.05 * x_exact**2)
    
    # Graphique comparatif
    plt.figure(figsize=(12, 7))
    plt.plot(x_exact, y_exact, 'b-', label='Solution exacte', linewidth=2.5, alpha=0.8)
    plt.plot(x_euler, y_euler, 'ro-', label='Euler', markersize=6, alpha=0.7)
    plt.plot(x_heun, y_heun, 'gs-', label='Heun', markersize=5, alpha=0.7)
    plt.plot(x_rk4, y_rk4, 'm^-', label='Runge-Kutta 4', markersize=5, alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('z(x)', fontsize=12)
    plt.title(f"Exemple 1: z'(x) = 0.1×x×z(x), z(0) = 1\nComparaison des méthodes avec h = {h}", fontsize=13)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # Calcul des erreurs
    y_exact_euler = np.exp(0.05 * x_euler**2)
    y_exact_heun = np.exp(0.05 * x_heun**2)
    y_exact_rk4 = np.exp(0.05 * x_rk4**2)
    
    erreur_euler = np.max(np.abs(y_euler - y_exact_euler))
    erreur_heun = np.max(np.abs(y_heun - y_exact_heun))
    erreur_rk4 = np.max(np.abs(y_rk4 - y_exact_rk4))
    
    print(f"\n--- Exemple 1 avec h = {h} ---")
    print(f"Euler:        Erreur max = {erreur_euler:.6e}")
    print(f"Heun:         Erreur max = {erreur_heun:.6e}")
    print(f"Runge-Kutta:  Erreur max = {erreur_rk4:.6e}")


def exemple_2_compare(h):
    """
    Exemple page 9 - Comparaison des 3 méthodes
    
    Équation: z'(x) = (1 - 30x²)/√x + 15z(x)
    Condition: z(1) = 1
    Solution exacte: z(x) = √x
    """
    f = lambda x, y: (1 - 30*x**2) / np.sqrt(x) + 15*y
    x0, y0 = 1.0, 1.0
    b = 5.0
    n = int((b - x0) / h)
    
    # Calcul avec les 3 méthodes
    x_euler, y_euler = methode_euler(f, x0, y0, h, n)
    x_heun, y_heun = methode_heun(f, x0, y0, h, n)
    x_rk4, y_rk4 = methode_runge_kutta_4(f, x0, y0, h, n)
    
    # Solution exacte
    x_exact = np.linspace(x0, b, 1000)
    y_exact = np.sqrt(x_exact)
    
    # Graphique comparatif
    plt.figure(figsize=(12, 7))
    plt.plot(x_exact, y_exact, 'b-', label='Solution exacte √x', linewidth=2.5, alpha=0.8)
    plt.plot(x_euler, y_euler, 'ro-', label='Euler', markersize=6, alpha=0.7)
    plt.plot(x_heun, y_heun, 'gs-', label='Heun', markersize=5, alpha=0.7)
    plt.plot(x_rk4, y_rk4, 'm^-', label='Runge-Kutta 4', markersize=5, alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('z(x)', fontsize=12)
    plt.title(f"Exemple 2: Équation instable (page 9)\nComparaison des méthodes avec h = {h}", fontsize=13)
    plt.legend(fontsize=11)
    plt.ylim([0, 4])
    plt.tight_layout()
    plt.show()
    
    print(f"\n--- Exemple 2 (instable) avec h = {h} ---")
    print("Note: Cette équation est instable et montre la divergence de la méthode d'Euler")


def exemple_3_compare(h):
    """
    Exemple page 26, 32, 39 - Comparaison des 3 méthodes
    
    Équation: z'(x) = π cos(πx) z(x)
    Condition: z(0) = 1
    Solution exacte: z(x) = exp(sin(πx))
    """
    f = lambda x, y: np.pi * np.cos(np.pi * x) * y
    x0, y0 = 0.0, 1.0
    b = 6.0
    n = int(b / h)
    
    # Calcul avec les 3 méthodes
    x_euler, y_euler = methode_euler(f, x0, y0, h, n)
    x_heun, y_heun = methode_heun(f, x0, y0, h, n)
    x_rk4, y_rk4 = methode_runge_kutta_4(f, x0, y0, h, n)
    
    # Solution exacte
    x_exact = np.linspace(0, b, 1000)
    y_exact = np.exp(np.sin(np.pi * x_exact))
    
    # Graphique comparatif
    plt.figure(figsize=(12, 7))
    plt.plot(x_exact, y_exact, 'b-', label='Solution exacte', linewidth=2.5, alpha=0.8)
    plt.plot(x_euler, y_euler, 'ro-', label='Euler', markersize=6, alpha=0.7)
    plt.plot(x_heun, y_heun, 'gs-', label='Heun', markersize=5, alpha=0.7)
    plt.plot(x_rk4, y_rk4, 'm^-', label='Runge-Kutta 4', markersize=5, alpha=0.7)
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('x', fontsize=12)
    plt.ylabel('z(x)', fontsize=12)
    plt.title(f"Exemple 3: z'(x) = π cos(πx) z(x), z(0) = 1\nComparaison des méthodes avec h = {h}", fontsize=13)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    # Calcul des erreurs
    y_exact_euler = np.exp(np.sin(np.pi * x_euler))
    y_exact_heun = np.exp(np.sin(np.pi * x_heun))
    y_exact_rk4 = np.exp(np.sin(np.pi * x_rk4))
    
    erreur_euler = np.max(np.abs(y_euler - y_exact_euler))
    erreur_heun = np.max(np.abs(y_heun - y_exact_heun))
    erreur_rk4 = np.max(np.abs(y_rk4 - y_exact_rk4))
    
    print(f"\n--- Exemple 3 avec h = {h} ---")
    print(f"Euler:        Erreur max = {erreur_euler:.6e}")
    print(f"Heun:         Erreur max = {erreur_heun:.6e}")
    print(f"Runge-Kutta:  Erreur max = {erreur_rk4:.6e}")


# ============================================================================
# TESTS
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
    
    # Exemple 2 (problème instable)
    print("\n" + "="*70)
    print("EXEMPLE 2: Équation instable (page 9)")
    print("="*70)
    exemple_2_compare(1.0)
    
    # Exemple 3
    print("\n" + "="*70)
    print("EXEMPLE 3: z'(x) = π cos(πx) z(x), z(0) = 1")
    print("="*70)
    
    print("\n>>> Avec un pas h = 0.5")
    exemple_3_compare(0.5)
    
    print("\n>>> Avec un pas h = 0.3")
    exemple_3_compare(0.3)
    
    print("\n>>> Avec un pas h = 0.15")
    exemple_3_compare(0.15)