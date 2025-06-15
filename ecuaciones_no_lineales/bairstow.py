#!/usr/bin/env python3
"""
Método de Bairstow - Implementación con menús interactivos
Método para encontrar raíces complejas de polinomios mediante factorización cuadrática
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import sympy as sp
import os
from typing import List, Tuple, Optional, Union
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_flotante, validar_entero, validar_polinomio,
    crear_menu, limpiar_pantalla, mostrar_progreso,
    formatear_numero, formatear_tabla_resultados,
    graficar_bairstow
)

console = Console()

class MetodoBairstow:
    def __init__(self):
        self.coeficientes = []
        self.polinomio_original = None
        self.r = 0.0  # Parámetro inicial para x
        self.s = 0.0  # Parámetro inicial para término constante
        self.tolerancia = 1e-10
        self.max_iteraciones = 100
        self.factores_cuadraticos = []
        self.raices = []
        self.iteraciones_realizadas = 0
        self.historial_iteraciones = []

    def mostrar_menu_principal(self):
        """Muestra el menú principal del método"""
        opciones = [
            "Ingresar polinomio",
            "Configurar parámetros iniciales",
            "Configurar tolerancia e iteraciones",
            "Ejecutar método de Bairstow",
            "Ver resultados",
            "Mostrar convergencia",
            "Mostrar gráficos",
            "Ver ayuda",
            "Salir"
        ]
        return crear_menu("MÉTODO DE BAIRSTOW", opciones)

    def ingresar_polinomio(self):
        """Menú para ingreso del polinomio"""
        limpiar_pantalla()
        console.print(Panel.fit("INGRESO DE POLINOMIO", style="bold green"))
        
        while True:
            try:
                console.print("\n[bold]Opciones de ingreso:[/bold]")
                console.print("1. Ingresar coeficientes directamente")
                console.print("2. Ingresar polinomio simbólico")
                
                opcion = validar_entero("Seleccione opción (1-2): ", 1, 2)
                
                if opcion == 1:
                    self._ingresar_coeficientes()
                else:
                    self._ingresar_simbolico()
                
                # Verificar que el grado sea al menos 2
                if len(self.coeficientes) < 3:
                    console.print("[red]El método de Bairstow requiere polinomios de grado ≥ 2[/red]")
                    continue
                
                # Mostrar el polinomio ingresado
                self._mostrar_polinomio()
                
                if input("\n¿Confirmar polinomio? (s/n): ").lower() == 's':
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Operación cancelada[/yellow]")
                return

    def _ingresar_coeficientes(self):
        """Ingresa coeficientes del polinomio directamente"""
        console.print("\n[bold]Ingreso de coeficientes (de mayor a menor grado):[/bold]")
        
        grado = validar_entero("Grado del polinomio (≥ 2): ", 2, 20)
        self.coeficientes = []
        
        for i in range(grado + 1):
            exponente = grado - i
            coef = validar_flotante(f"Coeficiente de x^{exponente}: ")
            self.coeficientes.append(coef)
        
        # Crear polinomio simbólico para verificación
        x = sp.Symbol('x')
        self.polinomio_original = sum(
            coef * x**i for i, coef in enumerate(reversed(self.coeficientes))
        )

    def _ingresar_simbolico(self):
        """Ingresa polinomio en forma simbólica"""
        console.print("\n[bold]Ejemplos:[/bold] x^4 - 2*x^3 + x^2 - 1, x^3 + 2*x^2 - 5*x + 3")
        
        while True:
            expr_str = input("Ingrese el polinomio: ").strip()
            try:
                x = sp.Symbol('x')
                self.polinomio_original = sp.expand(sp.sympify(expr_str))
                
                # Extraer coeficientes
                poly = sp.Poly(self.polinomio_original, x)
                self.coeficientes = [float(coef) for coef in poly.all_coeffs()]
                break
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

    def _mostrar_polinomio(self):
        """Muestra el polinomio actual"""
        if self.polinomio_original:
            console.print(f"\n[bold cyan]Polinomio:[/bold cyan] {self.polinomio_original}")
            console.print(f"[bold]Grado:[/bold] {len(self.coeficientes) - 1}")

    def configurar_parametros_iniciales(self):
        """Configura los parámetros iniciales r y s"""
        limpiar_pantalla()
        console.print(Panel.fit("PARÁMETROS INICIALES", style="bold blue"))
        
        console.print("\n[bold]Parámetros del factor cuadrático x² - rx - s:[/bold]")
        console.print(f"[bold]Valores actuales:[/bold] r = {formatear_numero(self.r)}, s = {formatear_numero(self.s)}")
        
        console.print("\n[yellow]Recomendaciones para valores iniciales:[/yellow]")
        console.print("• r, s cerca de 0 (ej: 0.1, -0.1)")
        console.print("• Evitar r = s = 0 exactamente")
        console.print("• Probar diferentes valores si no converge")
        
        if input("\n¿Cambiar parámetros iniciales? (s/n): ").lower() == 's':
            console.print("\n[bold]Ingreso de nuevos valores:[/bold]")
            self.r = validar_flotante("Nuevo valor de r: ", puede_ser_cero=True)
            self.s = validar_flotante("Nuevo valor de s: ", puede_ser_cero=True)
            
            console.print(f"[green]Parámetros actualizados: r = {formatear_numero(self.r)}, s = {formatear_numero(self.s)}[/green]")
            input("Presione Enter para continuar...")

    def configurar_tolerancia(self):
        """Configura tolerancia y máximo de iteraciones"""
        limpiar_pantalla()
        console.print(Panel.fit("CONFIGURACIÓN DE CONVERGENCIA", style="bold magenta"))
        
        console.print(f"\n[bold]Parámetros actuales:[/bold]")
        console.print(f"Tolerancia: {self.tolerancia}")
        console.print(f"Máximo iteraciones: {self.max_iteraciones}")
        
        if input("\n¿Cambiar parámetros? (s/n): ").lower() == 's':
            self.tolerancia = validar_flotante(
                "Nueva tolerancia (1e-10): ", 1e-15, 1e-3, self.tolerancia
            )
            self.max_iteraciones = validar_entero(
                "Nuevas máximo iteraciones (100): ", 10, 1000, self.max_iteraciones
            )

    def ejecutar_bairstow(self):
        """Ejecuta el método de Bairstow"""
        if not self.coeficientes:
            console.print("[red]Primero debe ingresar un polinomio[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("EJECUTANDO MÉTODO DE BAIRSTOW", style="bold yellow"))
        
        self._mostrar_polinomio()
        console.print(f"\n[bold]Parámetros:[/bold] r₀ = {formatear_numero(self.r)}, s₀ = {formatear_numero(self.s)}")
        console.print(f"[bold]Tolerancia:[/bold] {self.tolerancia}")
        
        try:
            self.factores_cuadraticos = []
            self.raices = []
            self.historial_iteraciones = []
            
            # Trabajar con una copia de los coeficientes
            coefs_trabajo = self.coeficientes.copy()
            
            console.print(f"\n[cyan]Iniciando factorización...[/cyan]")
            
            factor_num = 1
            while len(coefs_trabajo) >= 3:
                console.print(f"\n[bold]--- Factor cuadrático {factor_num} ---[/bold]")
                
                # Ejecutar Bairstow para encontrar un factor cuadrático
                r_final, s_final, iteraciones = self._bairstow_iteracion(coefs_trabajo)
                
                if iteraciones >= self.max_iteraciones:
                    console.print(f"[yellow]⚠ No convergió en {self.max_iteraciones} iteraciones[/yellow]")
                    console.print(f"Últimos valores: r = {formatear_numero(r_final)}, s = {formatear_numero(s_final)}")
                    break
                
                # Guardar factor cuadrático
                factor_cuadratico = [1, -r_final, -s_final]  # x² - rx - s
                self.factores_cuadraticos.append(factor_cuadratico)
                
                console.print(f"[green]✓ Factor encontrado: x² - {formatear_numero(r_final)}x - {formatear_numero(s_final)}[/green]")
                console.print(f"[green]✓ Convergió en {iteraciones} iteraciones[/green]")
                
                # Encontrar raíces del factor cuadrático
                raices_factor = self._resolver_cuadratica(1, -r_final, -s_final)
                self.raices.extend(raices_factor)
                
                # Deflactar el polinomio
                coefs_trabajo = self._deflactar_cuadratico(coefs_trabajo, r_final, s_final)
                
                factor_num += 1
            
            # Manejar factores lineales restantes
            if len(coefs_trabajo) == 2:  # Factor lineal
                raiz_lineal = -coefs_trabajo[1] / coefs_trabajo[0]
                self.raices.append(raiz_lineal)
                console.print(f"[green]✓ Factor lineal: x - {formatear_numero(raiz_lineal)}[/green]")
            
            console.print(f"\n[bold green]Método completado. {len(self.factores_cuadraticos)} factores cuadráticos encontrados.[/bold green]")
            
        except Exception as e:
            console.print(f"[red]Error durante la ejecución: {e}[/red]")
        
        input("Presione Enter para continuar...")

    def _bairstow_iteracion(self, coeficientes: List[float]) -> Tuple[float, float, int]:
        """Ejecuta las iteraciones de Bairstow para un factor cuadrático"""
        n = len(coeficientes) - 1
        r, s = self.r, self.s
        
        iteracion_info = []
        
        for iteracion in range(self.max_iteraciones):
            # Calcular secuencias b y c mediante división sintética
            b = self._division_sintetica_b(coeficientes, r, s)
            c = self._division_sintetica_c(b, r, s)
            
            # Verificar convergencia
            if len(b) >= 2:
                error_b = abs(b[-1]) + abs(b[-2])
                if error_b < self.tolerancia:
                    return r, s, iteracion + 1
            
            # Calcular determinante del sistema
            if len(c) >= 2:
                det = c[-3]**2 - c[-2]*(c[-4] if len(c) > 3 else 0)
                
                if abs(det) < 1e-15:
                    # Cambiar valores iniciales si el determinante es muy pequeño
                    r += 0.1
                    s += 0.1
                    continue
                
                # Calcular correcciones usando el sistema lineal
                dr = (b[-2]*c[-2] - b[-1]*c[-3]) / det
                ds = (b[-1]*(c[-4] if len(c) > 3 else 0) - b[-2]*c[-3]) / det
                
                # Actualizar r y s
                r += dr
                s += ds
                
                # Guardar información de la iteración
                iteracion_info.append({
                    'iteracion': iteracion + 1,
                    'r': r,
                    's': s,
                    'dr': dr,
                    'ds': ds,
                    'error': error_b
                })
        
        # Guardar historial para esta factorización
        self.historial_iteraciones.append(iteracion_info)
        
        return r, s, self.max_iteraciones

    def _division_sintetica_b(self, coefs: List[float], r: float, s: float) -> List[float]:
        """Realiza división sintética para obtener la secuencia b"""
        n = len(coefs)
        b = [0] * n
        
        b[0] = coefs[0]
        if n > 1:
            b[1] = coefs[1] + r * b[0]
        
        for i in range(2, n):
            b[i] = coefs[i] + r * b[i-1] + s * b[i-2]
        
        return b

    def _division_sintetica_c(self, b: List[float], r: float, s: float) -> List[float]:
        """Realiza división sintética para obtener la secuencia c"""
        n = len(b)
        c = [0] * n
        
        c[0] = b[0]
        if n > 1:
            c[1] = b[1] + r * c[0]
        
        for i in range(2, n):
            c[i] = b[i] + r * c[i-1] + s * c[i-2]
        
        return c

    def _resolver_cuadratica(self, a: float, b: float, c: float) -> List[complex]:
        """Resuelve ecuación cuadrática y devuelve las raíces"""
        discriminante = b**2 - 4*a*c
        
        if discriminante >= 0:
            sqrt_disc = np.sqrt(discriminante)
            x1 = (-b + sqrt_disc) / (2*a)
            x2 = (-b - sqrt_disc) / (2*a)
        else:
            real_part = -b / (2*a)
            imag_part = np.sqrt(-discriminante) / (2*a)
            x1 = complex(real_part, imag_part)
            x2 = complex(real_part, -imag_part)
        
        return [x1, x2]

    def _deflactar_cuadratico(self, coefs: List[float], r: float, s: float) -> List[float]:
        """Deflacta el polinomio dividiendo por x² - rx - s"""
        b = self._division_sintetica_b(coefs, r, s)
        # El polinomio deflactado son los primeros n-2 términos de b
        return b[:-2] if len(b) > 2 else []

    def mostrar_resultados(self):
        """Muestra los resultados del método de Bairstow"""
        if not self.factores_cuadraticos and not self.raices:
            console.print("[red]No hay resultados para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("RESULTADOS DEL MÉTODO DE BAIRSTOW", style="bold green"))
        
        # Mostrar polinomio original
        self._mostrar_polinomio()
        
        # Mostrar factores cuadráticos
        if self.factores_cuadraticos:
            console.print(f"\n[bold]Factores cuadráticos encontrados:[/bold]")
            for i, factor in enumerate(self.factores_cuadraticos, 1):
                a, b, c = factor
                console.print(f"Factor {i}: x² - {formatear_numero(-b)}x - {formatear_numero(-c)}")
        
        # Mostrar todas las raíces
        if self.raices:
            tabla = Table(title="Raíces del Polinomio")
            tabla.add_column("Raíz", style="cyan")
            tabla.add_column("Tipo", style="magenta")
            tabla.add_column("Valor", style="yellow")
            
            for i, raiz in enumerate(self.raices, 1):
                if isinstance(raiz, complex):
                    if abs(raiz.imag) < 1e-10:
                        tipo = "Real"
                        valor = formatear_numero(raiz.real)
                    else:
                        tipo = "Compleja"
                        valor = formatear_numero(raiz)
                else:
                    tipo = "Real"
                    valor = formatear_numero(raiz)
                
                tabla.add_row(f"x₍{i}₎", tipo, valor)
            
            console.print(tabla)
        
        # Verificación
        console.print(f"\n[bold]Verificación:[/bold]")
        if self.raices:
            x = sp.Symbol('x')
            # Verificar algunas raíces evaluando el polinomio original
            for i, raiz in enumerate(self.raices[:3]):  # Solo verificar las primeras 3
                if isinstance(raiz, complex):
                    valor = complex(self.polinomio_original.subs(x, raiz))
                else:
                    valor = float(self.polinomio_original.subs(x, raiz))
                
                console.print(f"P(x₍{i+1}₎) = {formatear_numero(valor)}")
        
        input("\nPresione Enter para continuar...")

    def mostrar_convergencia(self):
        """Muestra el historial de convergencia"""
        if not self.historial_iteraciones:
            console.print("[red]No hay historial de convergencia para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("HISTORIAL DE CONVERGENCIA", style="bold cyan"))
        
        for factor_num, historial in enumerate(self.historial_iteraciones, 1):
            console.print(f"\n[bold]Factor cuadrático {factor_num}:[/bold]")
            
            tabla = Table(title=f"Convergencia Factor {factor_num}")
            tabla.add_column("Iteración", style="cyan")
            tabla.add_column("r", style="blue")
            tabla.add_column("s", style="blue")
            tabla.add_column("Δr", style="green")
            tabla.add_column("Δs", style="green")
            tabla.add_column("Error", style="red")
            
            for info in historial[-10:]:  # Mostrar últimas 10 iteraciones
                tabla.add_row(
                    str(info['iteracion']),
                    formatear_numero(info['r']),
                    formatear_numero(info['s']),
                    formatear_numero(info['dr']),
                    formatear_numero(info['ds']),
                    formatear_numero(info['error'])
                )
            
            console.print(tabla)
            
            if len(historial) > 10:
                console.print(f"[yellow]Mostrando últimas 10 de {len(historial)} iteraciones[/yellow]")
        
        input("\nPresione Enter para continuar...")

    def mostrar_graficos(self):
        """Muestra gráficos del método de Bairstow"""
        if not self.raices:
            console.print("[red]No hay resultados para graficar[/red]")
            input("Presione Enter para continuar...")
            return
        
        console.print("[cyan]Generando gráficos...[/cyan]")
        graficar_bairstow(
            self.coeficientes,
            self.raices,
            self.historial_iteraciones
        )

    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        limpiar_pantalla()
        ayuda_texto = """
[bold blue]MÉTODO DE BAIRSTOW[/bold blue]

[bold]¿Qué es?[/bold]
El método de Bairstow es un algoritmo numérico para encontrar todas las raíces
(reales y complejas) de un polinomio mediante factorización en productos de
factores cuadráticos reales.

[bold]¿Cómo funciona?[/bold]
1. Busca factores de la forma x² - rx - s
2. Usa iteración de Newton-Raphson en dos variables (r, s)
3. Realiza división sintética para evaluar residuos
4. Deflacta el polinomio con cada factor encontrado

[bold]Algoritmo:[/bold]
• División sintética: bᵢ = aᵢ + r·bᵢ₊₁ + s·bᵢ₊₂
• Sistema lineal para correcciones Δr, Δs
• Iteración: rₙ₊₁ = rₙ + Δr, sₙ₊₁ = sₙ + Δs

[bold]Ventajas:[/bold]
• Encuentra todas las raíces (reales y complejas)
• Trabaja solo con aritmética real
• Convergencia cuadrática cerca de la solución
• No requiere estimaciones de raíces individuales

[bold]Desventajas:[/bold]
• Sensible a valores iniciales de r y s
• Puede no converger con estimaciones pobres
• Acumulación de errores en deflación
• Más complejo que métodos para raíces individuales

[bold]Consejos de uso:[/bold]
• Usar valores iniciales pequeños no nulos
• Si no converge, probar diferentes r₀, s₀
• Verificar resultados evaluando P(x) en las raíces
• Para polinomios mal condicionados, usar mayor precisión

[bold]Aplicaciones:[/bold]
• Análisis de sistemas de control
• Procesamiento de señales (filtros)
• Análisis de estabilidad
• Factorización completa de polinomios
        """
        
        console.print(Panel(ayuda_texto, title="AYUDA", border_style="blue"))
        input("\nPresione Enter para continuar...")

    def main(self):
        """Función principal con el bucle del menú"""
        while True:
            try:
                limpiar_pantalla()
                opcion = self.mostrar_menu_principal()
                
                if opcion == 1:
                    self.ingresar_polinomio()
                elif opcion == 2:
                    self.configurar_parametros_iniciales()
                elif opcion == 3:
                    self.configurar_tolerancia()
                elif opcion == 4:
                    self.ejecutar_bairstow()
                elif opcion == 5:
                    self.mostrar_resultados()
                elif opcion == 6:
                    self.mostrar_convergencia()
                elif opcion == 7:
                    self.mostrar_graficos()
                elif opcion == 8:
                    self.mostrar_ayuda()
                elif opcion == 9:
                    console.print("[bold green]¡Hasta luego![/bold green]")
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[bold red]Programa interrumpido[/bold red]")
                break
            except Exception as e:
                console.print(f"[bold red]Error inesperado: {e}[/bold red]")
                input("Presione Enter para continuar...")

if __name__ == "__main__":
    bairstow = MetodoBairstow()
    bairstow.main()
