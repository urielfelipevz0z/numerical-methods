#!/usr/bin/env python3
"""
Módulo de gráficas para métodos numéricos
Funciones para crear visualizaciones con matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import sympy as sp
from typing import Optional, List, Tuple, Union
import warnings
warnings.filterwarnings('ignore')

# Configuración global de matplotlib
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16

def configurar_grafica(titulo: str, xlabel: str = "x", ylabel: str = "f(x)"):
    """
    Configura una gráfica con estilo consistente
    
    Args:
        titulo: Título de la gráfica
        xlabel: Etiqueta del eje X
        ylabel: Etiqueta del eje Y
    """
    plt.title(titulo, fontweight='bold', pad=20)
    plt.xlabel(xlabel, fontweight='bold')
    plt.ylabel(ylabel, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
    plt.axvline(x=0, color='black', linewidth=0.8, alpha=0.7)

def graficar_funcion(
    expr: sp.Expr, 
    rango: Tuple[float, float], 
    titulo: str = "Función",
    puntos_especiales: Optional[List[Tuple[float, float, str, str]]] = None,
    mostrar: bool = True
):
    """
    Grafica una función simbólica
    
    Args:
        expr: Expresión simbólica de SymPy
        rango: Tupla (x_min, x_max) para el rango de graficación
        titulo: Título de la gráfica
        puntos_especiales: Lista de tuplas (x, y, label, color) para marcar puntos
        mostrar: Si True, muestra la gráfica inmediatamente
    """
    x_vals = np.linspace(rango[0], rango[1], 1000)
    x_sym = sp.Symbol('x')
    
    try:
        # Convertir a función numpy para evaluación rápida
        f_lambdified = sp.lambdify(x_sym, expr, 'numpy')
        y_vals = f_lambdified(x_vals)
        
        plt.figure(figsize=(12, 8))
        plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {expr}')
        
        # Marcar puntos especiales
        if puntos_especiales:
            for x, y, label, color in puntos_especiales:
                plt.plot(x, y, 'o', color=color, markersize=8, label=label)
                plt.annotate(f'({x:.4f}, {y:.4f})', 
                           xy=(x, y), xytext=(10, 10), 
                           textcoords='offset points',
                           bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.7))
        
        configurar_grafica(titulo)
        plt.legend()
        plt.tight_layout()
        
        if mostrar:
            plt.show()
            
    except Exception as e:
        print(f"Error al graficar la función: {e}")

def graficar_convergencia_biseccion(
    iteraciones: List[int],
    intervalos: List[Tuple[float, float]],
    errores: List[float],
    raiz_aproximada: float
):
    """
    Grafica la convergencia del método de bisección
    
    Args:
        iteraciones: Lista de números de iteración
        intervalos: Lista de tuplas (a, b) para cada iteración
        errores: Lista de errores absolutos
        raiz_aproximada: Valor de la raíz encontrada
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfica 1: Reducción del intervalo
    longitudes = [abs(b - a) for a, b in intervalos]
    ax1.semilogy(iteraciones, longitudes, 'bo-', linewidth=2, markersize=6)
    ax1.set_title('Reducción del Intervalo - Método de Bisección', fontweight='bold')
    ax1.set_xlabel('Iteración')
    ax1.set_ylabel('Longitud del Intervalo')
    ax1.grid(True, alpha=0.3)
    
    # Gráfica 2: Error absoluto
    ax2.semilogy(iteraciones, errores, 'ro-', linewidth=2, markersize=6)
    ax2.set_title('Convergencia del Error - Método de Bisección', fontweight='bold')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Error Absoluto')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def graficar_convergencia_newton(
    iteraciones: List[int],
    valores_x: List[float],
    errores: List[float],
    expr: sp.Expr,
    rango_grafica: Optional[Tuple[float, float]] = None
):
    """
    Grafica la convergencia del método de Newton-Raphson
    
    Args:
        iteraciones: Lista de números de iteración
        valores_x: Lista de aproximaciones de x
        errores: Lista de errores absolutos
        expr: Expresión de la función
        rango_grafica: Rango para graficar la función
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfica 1: Función y aproximaciones
    if rango_grafica:
        x_vals = np.linspace(rango_grafica[0], rango_grafica[1], 1000)
        x_sym = sp.Symbol('x')
        f_lambdified = sp.lambdify(x_sym, expr, 'numpy')
        y_vals = f_lambdified(x_vals)
        
        ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {expr}')
        ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
        
        # Marcar aproximaciones
        for i, x in enumerate(valores_x):
            y = float(expr.subs(sp.Symbol('x'), x))
            color = plt.cm.viridis(i / len(valores_x))
            ax1.plot(x, y, 'o', color=color, markersize=8, label=f'x_{i}')
            
        ax1.set_title('Función y Aproximaciones Sucesivas', fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
    
    # Gráfica 2: Convergencia de x
    ax2.plot(iteraciones, valores_x, 'go-', linewidth=2, markersize=6)
    ax2.set_title('Convergencia de las Aproximaciones', fontweight='bold')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Valor de x')
    ax2.grid(True, alpha=0.3)
    
    # Gráfica 3: Error absoluto
    ax3.semilogy(iteraciones, errores, 'ro-', linewidth=2, markersize=6)
    ax3.set_title('Convergencia del Error Absoluto', fontweight='bold')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Error Absoluto')
    ax3.grid(True, alpha=0.3)
    
    # Gráfica 4: Diferencias sucesivas
    if len(valores_x) > 1:
        diferencias = [abs(valores_x[i] - valores_x[i-1]) for i in range(1, len(valores_x))]
        ax4.semilogy(iteraciones[1:], diferencias, 'mo-', linewidth=2, markersize=6)
        ax4.set_title('Diferencias Sucesivas |x_n - x_{n-1}|', fontweight='bold')
        ax4.set_xlabel('Iteración')
        ax4.set_ylabel('|x_n - x_{n-1}|')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def graficar_punto_fijo(
    expr_g: sp.Expr,
    valores_x: List[float],
    rango: Tuple[float, float],
    punto_fijo: float
):
    """
    Grafica la convergencia del método de punto fijo
    
    Args:
        expr_g: Expresión de g(x)
        valores_x: Lista de aproximaciones
        rango: Rango para graficar
        punto_fijo: Punto fijo encontrado
    """
    x_vals = np.linspace(rango[0], rango[1], 1000)
    x_sym = sp.Symbol('x')
    g_lambdified = sp.lambdify(x_sym, expr_g, 'numpy')
    y_vals = g_lambdified(x_vals)
    
    plt.figure(figsize=(12, 8))
    
    # Graficar g(x) y y = x
    plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'g(x) = {expr_g}')
    plt.plot(x_vals, x_vals, 'r--', linewidth=2, label='y = x')
    
    # Dibujar el proceso iterativo (telaraña)
    if len(valores_x) > 1:
        for i in range(len(valores_x) - 1):
            x_curr = valores_x[i]
            x_next = valores_x[i + 1]
            
            # Línea vertical desde (x_i, x_i) hasta (x_i, g(x_i))
            plt.plot([x_curr, x_curr], [x_curr, x_next], 'g-', alpha=0.7, linewidth=1)
            
            # Línea horizontal desde (x_i, g(x_i)) hasta (g(x_i), g(x_i))
            plt.plot([x_curr, x_next], [x_next, x_next], 'g-', alpha=0.7, linewidth=1)
    
    # Marcar el punto fijo
    plt.plot(punto_fijo, punto_fijo, 'ro', markersize=10, label=f'Punto fijo: {punto_fijo:.6f}')
    
    configurar_grafica('Método de Punto Fijo - Proceso Iterativo')
    plt.legend()
    plt.tight_layout()
    plt.show()

def graficar_regresion(
    x_datos: np.ndarray,
    y_datos: np.ndarray,
    x_ajuste: np.ndarray,
    y_ajuste: np.ndarray,
    titulo: str = "Ajuste de Curva",
    etiqueta_datos: str = "Datos",
    etiqueta_ajuste: str = "Ajuste"
):
    """
    Grafica datos y curva de ajuste
    
    Args:
        x_datos: Datos X originales
        y_datos: Datos Y originales
        x_ajuste: Puntos X para la curva de ajuste
        y_ajuste: Puntos Y de la curva de ajuste
        titulo: Título de la gráfica
        etiqueta_datos: Etiqueta para los datos
        etiqueta_ajuste: Etiqueta para el ajuste
    """
    plt.figure(figsize=(12, 8))
    
    # Datos originales
    plt.scatter(x_datos, y_datos, color='red', s=50, alpha=0.7, label=etiqueta_datos)
    
    # Curva de ajuste
    plt.plot(x_ajuste, y_ajuste, 'b-', linewidth=2, label=etiqueta_ajuste)
    
    configurar_grafica(titulo)
    plt.legend()
    plt.tight_layout()
    plt.show()

def graficar_sistema_convergencia(
    iteraciones: List[int],
    valores_x: List[float],
    valores_y: List[float],
    errores: List[float]
):
    """
    Grafica la convergencia de un sistema de ecuaciones
    
    Args:
        iteraciones: Lista de números de iteración
        valores_x: Valores de x en cada iteración
        valores_y: Valores de y en cada iteración
        errores: Errores en cada iteración
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Trayectoria en el plano XY
    ax1.plot(valores_x, valores_y, 'bo-', linewidth=2, markersize=6, alpha=0.7)
    ax1.plot(valores_x[0], valores_y[0], 'go', markersize=10, label='Punto inicial')
    ax1.plot(valores_x[-1], valores_y[-1], 'ro', markersize=10, label='Solución')
    ax1.set_title('Trayectoria de Convergencia en el Plano XY', fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Convergencia de x
    ax2.plot(iteraciones, valores_x, 'bo-', linewidth=2, markersize=6)
    ax2.set_title('Convergencia de x', fontweight='bold')
    ax2.set_xlabel('Iteración')
    ax2.set_ylabel('Valor de x')
    ax2.grid(True, alpha=0.3)
    
    # Convergencia de y
    ax3.plot(iteraciones, valores_y, 'ro-', linewidth=2, markersize=6)
    ax3.set_title('Convergencia de y', fontweight='bold')
    ax3.set_xlabel('Iteración')
    ax3.set_ylabel('Valor de y')
    ax3.grid(True, alpha=0.3)
    
    # Error total
    ax4.semilogy(iteraciones, errores, 'go-', linewidth=2, markersize=6)
    ax4.set_title('Convergencia del Error Total', fontweight='bold')
    ax4.set_xlabel('Iteración')
    ax4.set_ylabel('Error Total')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def graficar_polinomio_deflacion(
    coeficientes_originales: List[float],
    raices_conocidas: List[Union[float, complex]],
    polinomios_deflactados: List[List[float]]
):
    """
    Grafica el proceso de deflación de polinomios
    
    Args:
        coeficientes_originales: Coeficientes del polinomio original
        raices_conocidas: Raíces utilizadas en la deflación
        polinomios_deflactados: Lista de polinomios después de cada deflación
    """
    x_sym = sp.Symbol('x')
    
    # Polinomio original
    poly_original = sum(coef * x_sym**i for i, coef in enumerate(reversed(coeficientes_originales)))
    
    # Determinar rango de graficación
    raices_reales = [float(r.real) if isinstance(r, complex) else r for r in raices_conocidas 
                    if isinstance(r, (int, float)) or abs(r.imag) < 1e-10]
    
    if raices_reales:
        x_min = min(raices_reales) - 2
        x_max = max(raices_reales) + 2
    else:
        x_min, x_max = -5, 5
    
    x_vals = np.linspace(x_min, x_max, 1000)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Gráfica 1: Polinomio original y raíces
    try:
        f_original = sp.lambdify(x_sym, poly_original, 'numpy')
        y_original = f_original(x_vals)
        
        ax1.plot(x_vals, y_original, 'b-', linewidth=2, label=f'P(x) original (grado {len(coeficientes_originales)-1})')
        ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
        
        # Marcar raíces reales
        for i, raiz in enumerate(raices_conocidas):
            if isinstance(raiz, (int, float)) or abs(raiz.imag) < 1e-10:
                x_raiz = float(raiz.real) if isinstance(raiz, complex) else raiz
                ax1.plot(x_raiz, 0, 'ro', markersize=8, label=f'Raíz {i+1}: {x_raiz:.4f}')
        
        ax1.set_title('Polinomio Original y Raíces Encontradas', fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('P(x)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error al graficar polinomio: {e}', 
                transform=ax1.transAxes, ha='center', va='center')
    
    # Gráfica 2: Comparación de grados
    etapas = list(range(len(polinomios_deflactados) + 1))
    grados = [len(coeficientes_originales) - 1] + [len(poly) - 1 for poly in polinomios_deflactados]
    
    ax2.plot(etapas, grados, 'go-', linewidth=2, markersize=8)
    ax2.set_title('Reducción del Grado durante la Deflación', fontweight='bold')
    ax2.set_xlabel('Etapa de Deflación')
    ax2.set_ylabel('Grado del Polinomio')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(etapas)
    
    plt.tight_layout()
    plt.show()

def graficar_division_polinomios(
    dividendo: List[float],
    divisor: List[float],
    cociente: List[float],
    residuo: List[float]
):
    """
    Grafica la división de polinomios
    
    Args:
        dividendo: Coeficientes del dividendo
        divisor: Coeficientes del divisor
        cociente: Coeficientes del cociente
        residuo: Coeficientes del residuo
    """
    x_sym = sp.Symbol('x')
    
    # Crear polinomios simbólicos
    def crear_polinomio(coefs):
        if not coefs:
            return 0
        return sum(coef * x_sym**i for i, coef in enumerate(reversed(coefs)))
    
    poly_dividendo = crear_polinomio(dividendo)
    poly_divisor = crear_polinomio(divisor)
    poly_cociente = crear_polinomio(cociente) if cociente else 0
    poly_residuo = crear_polinomio(residuo) if residuo else 0
    
    # Rango de graficación
    x_vals = np.linspace(-5, 5, 1000)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Funciones lambda para evaluación
    try:
        f_dividendo = sp.lambdify(x_sym, poly_dividendo, 'numpy')
        f_divisor = sp.lambdify(x_sym, poly_divisor, 'numpy')
        
        # Gráfica 1: Dividendo
        y_dividendo = f_dividendo(x_vals)
        ax1.plot(x_vals, y_dividendo, 'b-', linewidth=2, label=f'Dividendo: {poly_dividendo}')
        ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
        ax1.set_title('Polinomio Dividendo', fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('P(x)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Gráfica 2: Divisor
        y_divisor = f_divisor(x_vals)
        ax2.plot(x_vals, y_divisor, 'r-', linewidth=2, label=f'Divisor: {poly_divisor}')
        ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
        ax2.set_title('Polinomio Divisor', fontweight='bold')
        ax2.set_xlabel('x')
        ax2.set_ylabel('D(x)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Gráfica 3: Cociente
        if cociente:
            f_cociente = sp.lambdify(x_sym, poly_cociente, 'numpy')
            y_cociente = f_cociente(x_vals)
            ax3.plot(x_vals, y_cociente, 'g-', linewidth=2, label=f'Cociente: {poly_cociente}')
        else:
            ax3.axhline(y=0, color='g', linewidth=2, label='Cociente: 0')
        
        ax3.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
        ax3.set_title('Polinomio Cociente', fontweight='bold')
        ax3.set_xlabel('x')
        ax3.set_ylabel('Q(x)')
        ax3.grid(True, alpha=0.3)
        ax3.legend()
        
        # Gráfica 4: Verificación P(x) = D(x)*Q(x) + R(x)
        if cociente:
            verificacion = poly_divisor * poly_cociente + poly_residuo
            f_verificacion = sp.lambdify(x_sym, verificacion, 'numpy')
            y_verificacion = f_verificacion(x_vals)
            
            ax4.plot(x_vals, y_dividendo, 'b-', linewidth=2, label='P(x) original')
            ax4.plot(x_vals, y_verificacion, 'r--', linewidth=2, label='D(x)×Q(x) + R(x)')
            ax4.set_title('Verificación de la División', fontweight='bold')
            ax4.set_xlabel('x')
            ax4.set_ylabel('Valor')
            ax4.grid(True, alpha=0.3)
            ax4.legend()
        else:
            ax4.text(0.5, 0.5, 'No hay cociente para verificar', 
                    transform=ax4.transAxes, ha='center', va='center')
        
    except Exception as e:
        for ax in [ax1, ax2, ax3, ax4]:
            ax.text(0.5, 0.5, f'Error al graficar: {e}', 
                   transform=ax.transAxes, ha='center', va='center')
    
    plt.tight_layout()
    plt.show()

def graficar_cuadratica(
    a: float, 
    b: float, 
    c: float, 
    raices: List[Union[float, complex]], 
    vertice: Tuple[float, float]
):
    """
    Grafica una función cuadrática con análisis completo
    
    Args:
        a, b, c: Coeficientes de la ecuación ax² + bx + c = 0
        raices: Lista de raíces (reales o complejas)
        vertice: Coordenadas del vértice (h, k)
    """
    # Determinar rango de graficación
    h, k = vertice
    rango_x = max(5, abs(h) + 3)
    x_vals = np.linspace(h - rango_x, h + rango_x, 1000)
    y_vals = a * x_vals**2 + b * x_vals + c
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Gráfica 1: Parábola completa
    ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {a}x² + {b}x + {c}')
    ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
    ax1.axvline(x=0, color='black', linewidth=0.8, alpha=0.7)
    
    # Marcar vértice
    ax1.plot(h, k, 'ro', markersize=10, label=f'Vértice ({h:.3f}, {k:.3f})')
    
    # Marcar raíces reales
    for i, raiz in enumerate(raices):
        if isinstance(raiz, (int, float)) or abs(raiz.imag) < 1e-10:
            x_raiz = float(raiz.real) if isinstance(raiz, complex) else raiz
            ax1.plot(x_raiz, 0, 'go', markersize=8, label=f'Raíz {i+1}: {x_raiz:.3f}')
    
    # Intersección con eje Y
    ax1.plot(0, c, 'mo', markersize=8, label=f'Intersección Y: (0, {c})')
    
    ax1.set_title('Función Cuadrática', fontweight='bold')
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Gráfica 2: Zoom cerca del vértice
    zoom_range = max(1, rango_x / 3)
    x_zoom = np.linspace(h - zoom_range, h + zoom_range, 500)
    y_zoom = a * x_zoom**2 + b * x_zoom + c
    
    ax2.plot(x_zoom, y_zoom, 'b-', linewidth=2)
    ax2.plot(h, k, 'ro', markersize=10, label=f'Vértice ({h:.3f}, {k:.3f})')
    ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
    ax2.axvline(x=h, color='red', linewidth=1, linestyle='--', alpha=0.7, label=f'Eje simetría: x = {h:.3f}')
    
    ax2.set_title('Zoom en el Vértice', fontweight='bold')
    ax2.set_xlabel('x')
    ax2.set_ylabel('f(x)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Gráfica 3: Análisis del discriminante
    discriminante = b**2 - 4*a*c
    ax3.text(0.1, 0.9, f'Discriminante Δ = {discriminante:.6f}', transform=ax3.transAxes, fontsize=12, fontweight='bold')
    
    if discriminante > 0:
        ax3.text(0.1, 0.8, 'Δ > 0: Dos raíces reales distintas', transform=ax3.transAxes, fontsize=11, color='green')
        ax3.text(0.1, 0.7, 'La parábola corta el eje X en dos puntos', transform=ax3.transAxes, fontsize=10)
    elif discriminante == 0:
        ax3.text(0.1, 0.8, 'Δ = 0: Una raíz real doble', transform=ax3.transAxes, fontsize=11, color='orange')
        ax3.text(0.1, 0.7, 'La parábola es tangente al eje X', transform=ax3.transAxes, fontsize=10)
    else:
        ax3.text(0.1, 0.8, 'Δ < 0: Dos raíces complejas', transform=ax3.transAxes, fontsize=11, color='red')
        ax3.text(0.1, 0.7, 'La parábola no corta el eje X', transform=ax3.transAxes, fontsize=10)
    
    # Mostrar las raíces
    ax3.text(0.1, 0.5, 'Raíces:', transform=ax3.transAxes, fontsize=12, fontweight='bold')
    for i, raiz in enumerate(raices):
        if isinstance(raiz, complex):
            if abs(raiz.imag) < 1e-10:
                ax3.text(0.1, 0.4 - i*0.1, f'x_{i+1} = {raiz.real:.6f}', transform=ax3.transAxes, fontsize=10)
            else:
                ax3.text(0.1, 0.4 - i*0.1, f'x_{i+1} = {raiz.real:.3f} + {raiz.imag:.3f}i', transform=ax3.transAxes, fontsize=10)
        else:
            ax3.text(0.1, 0.4 - i*0.1, f'x_{i+1} = {raiz:.6f}', transform=ax3.transAxes, fontsize=10)
    
    ax3.set_title('Análisis de Raíces', fontweight='bold')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    
    # Gráfica 4: Propiedades de la parábola
    ax4.text(0.1, 0.9, 'Propiedades de la Parábola:', transform=ax4.transAxes, fontsize=12, fontweight='bold')
    ax4.text(0.1, 0.8, f'Coeficiente principal: a = {a}', transform=ax4.transAxes, fontsize=11)
    
    abertura = "hacia arriba" if a > 0 else "hacia abajo"
    tipo_vertice = "mínimo" if a > 0 else "máximo"
    
    ax4.text(0.1, 0.7, f'Abertura: {abertura}', transform=ax4.transAxes, fontsize=11)
    ax4.text(0.1, 0.6, f'El vértice es un {tipo_vertice}', transform=ax4.transAxes, fontsize=11)
    ax4.text(0.1, 0.5, f'Vértice: ({h:.3f}, {k:.3f})', transform=ax4.transAxes, fontsize=11)
    ax4.text(0.1, 0.4, f'Eje de simetría: x = {h:.3f}', transform=ax4.transAxes, fontsize=11)
    ax4.text(0.1, 0.3, f'Intersección con Y: (0, {c})', transform=ax4.transAxes, fontsize=11)
    
    ax4.set_title('Propiedades', fontweight='bold')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.show()

def graficar_bairstow(
    coeficientes: List[float],
    raices: List[Union[float, complex]],
    historial_iteraciones: List[List[dict]]
):
    """
    Grafica los resultados del método de Bairstow
    
    Args:
        coeficientes: Coeficientes del polinomio original
        raices: Todas las raíces encontradas
        historial_iteraciones: Historial de convergencia para cada factor
    """
    x_sym = sp.Symbol('x')
    poly_original = sum(coef * x_sym**i for i, coef in enumerate(reversed(coeficientes)))
    
    # Determinar número de subgráficas necesarias
    num_factores = len(historial_iteraciones)
    
    if num_factores == 0:
        return
    
    # Configurar subgráficas
    if num_factores == 1:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        axes_convergencia = [ax3]
    else:
        fig = plt.figure(figsize=(18, 12))
        ax1 = plt.subplot(2, 3, 1)
        ax2 = plt.subplot(2, 3, 2)
        axes_convergencia = [plt.subplot(2, 3, 3 + i) for i in range(min(num_factores, 3))]
    
    # Gráfica 1: Polinomio original y raíces
    raices_reales = [r for r in raices if isinstance(r, (int, float)) or abs(r.imag) < 1e-10]
    
    if raices_reales:
        x_min = min(float(r.real) if isinstance(r, complex) else r for r in raices_reales) - 2
        x_max = max(float(r.real) if isinstance(r, complex) else r for r in raices_reales) + 2
    else:
        x_min, x_max = -5, 5
    
    x_vals = np.linspace(x_min, x_max, 1000)
    
    try:
        f_original = sp.lambdify(x_sym, poly_original, 'numpy')
        y_original = f_original(x_vals)
        
        ax1.plot(x_vals, y_original, 'b-', linewidth=2, label=f'P(x) (grado {len(coeficientes)-1})')
        ax1.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
        
        # Marcar raíces reales
        for i, raiz in enumerate(raices):
            if isinstance(raiz, (int, float)) or abs(raiz.imag) < 1e-10:
                x_raiz = float(raiz.real) if isinstance(raiz, complex) else raiz
                ax1.plot(x_raiz, 0, 'ro', markersize=6, alpha=0.7)
        
        ax1.set_title('Polinomio Original y Raíces', fontweight='bold')
        ax1.set_xlabel('x')
        ax1.set_ylabel('P(x)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
    except Exception as e:
        ax1.text(0.5, 0.5, f'Error al graficar: {e}', transform=ax1.transAxes, ha='center')
    
    # Gráfica 2: Raíces en el plano complejo
    ax2.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
    ax2.axvline(x=0, color='black', linewidth=0.8, alpha=0.7)
    
    for i, raiz in enumerate(raices):
        if isinstance(raiz, complex):
            ax2.plot(raiz.real, raiz.imag, 'bo', markersize=8, alpha=0.7, label=f'Raíz {i+1}')
        else:
            ax2.plot(raiz, 0, 'ro', markersize=8, alpha=0.7, label=f'Raíz {i+1}')
    
    ax2.set_title('Raíces en el Plano Complejo', fontweight='bold')
    ax2.set_xlabel('Parte Real')
    ax2.set_ylabel('Parte Imaginaria')
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Gráficas de convergencia para cada factor
    for i, (historial, ax) in enumerate(zip(historial_iteraciones[:len(axes_convergencia)], axes_convergencia)):
        if historial:
            iteraciones = [info['iteracion'] for info in historial]
            errores = [info['error'] for info in historial]
            
            ax.semilogy(iteraciones, errores, 'o-', linewidth=2, markersize=4)
            ax.set_title(f'Convergencia Factor {i+1}', fontweight='bold')
            ax.set_xlabel('Iteración')
            ax.set_ylabel('Error')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def mostrar_grafica_interactiva():
    """
    Muestra todas las gráficas generadas durante la sesión
    """
    plt.show(block=False)
    input("\nPresione Enter para continuar...")
    plt.close('all')
