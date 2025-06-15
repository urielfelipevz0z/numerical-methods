#!/usr/bin/env python3
"""
Polinomios de Orden 2 - Implementación con menús interactivos
Métodos especializados para resolver ecuaciones cuadráticas
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import sympy as sp
import os
import cmath
from typing import List, Tuple, Optional, Union
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_flotante, validar_entero, validar_polinomio,
    crear_menu, limpiar_pantalla, mostrar_progreso,
    formatear_numero, formatear_tabla_resultados,
    graficar_cuadratica
)

console = Console()

class PolinomioOrden2:
    def __init__(self):
        self.a = 0.0  # Coeficiente de x²
        self.b = 0.0  # Coeficiente de x
        self.c = 0.0  # Término independiente
        self.raices = []
        self.discriminante = 0.0
        self.vertice = (0.0, 0.0)
        self.metodo = "formula_general"  # "formula_general", "completar_cuadrado", "factorizacion"
        self.pasos_solucion = []

    def mostrar_menu_principal(self):
        """Muestra el menú principal del método"""
        opciones = [
            "Ingresar coeficientes",
            "Seleccionar método de solución",
            "Resolver ecuación",
            "Analizar discriminante",
            "Encontrar vértice y eje de simetría",
            "Ver resultados completos",
            "Mostrar gráfico",
            "Ver ayuda",
            "Salir"
        ]
        return crear_menu("ECUACIONES CUADRÁTICAS", opciones)

    def ingresar_coeficientes(self):
        """Menú para ingreso de coeficientes"""
        limpiar_pantalla()
        console.print(Panel.fit("INGRESO DE COEFICIENTES", style="bold green"))
        
        console.print("\n[bold]Ecuación cuadrática:[/bold] ax² + bx + c = 0")
        console.print("[yellow]Nota: 'a' debe ser diferente de cero[/yellow]")
        
        while True:
            try:
                self.a = validar_flotante("Coeficiente 'a' (x²): ")
                
                if abs(self.a) < 1e-12:
                    console.print("[red]Error: 'a' no puede ser cero (no sería ecuación cuadrática)[/red]")
                    continue
                
                self.b = validar_flotante("Coeficiente 'b' (x): ", puede_ser_cero=True)
                self.c = validar_flotante("Coeficiente 'c' (constante): ", puede_ser_cero=True)
                
                # Mostrar la ecuación
                self._mostrar_ecuacion()
                
                if input("\n¿Confirmar coeficientes? (s/n): ").lower() == 's':
                    # Calcular discriminante inmediatamente
                    self.discriminante = self.b**2 - 4*self.a*self.c
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Operación cancelada[/yellow]")
                return

    def _mostrar_ecuacion(self):
        """Muestra la ecuación cuadrática actual"""
        # Formatear la ecuación con signos apropiados
        terminos = []
        
        # Término x²
        if self.a == 1:
            terminos.append("x²")
        elif self.a == -1:
            terminos.append("-x²")
        else:
            terminos.append(f"{formatear_numero(self.a)}x²")
        
        # Término x
        if self.b != 0:
            if self.b > 0 and terminos:
                if self.b == 1:
                    terminos.append("+ x")
                else:
                    terminos.append(f"+ {formatear_numero(self.b)}x")
            elif self.b < 0:
                if self.b == -1:
                    terminos.append("- x")
                else:
                    terminos.append(f"- {formatear_numero(abs(self.b))}x")
            else:  # b > 0 y es el primer término (a = 0, pero ya verificamos que a ≠ 0)
                if self.b == 1:
                    terminos.append("x")
                else:
                    terminos.append(f"{formatear_numero(self.b)}x")
        
        # Término constante
        if self.c != 0:
            if self.c > 0 and terminos:
                terminos.append(f"+ {formatear_numero(self.c)}")
            elif self.c < 0:
                terminos.append(f"- {formatear_numero(abs(self.c))}")
            else:
                terminos.append(f"{formatear_numero(self.c)}")
        
        ecuacion = " ".join(terminos) + " = 0"
        console.print(f"\n[bold cyan]Ecuación:[/bold cyan] {ecuacion}")
        console.print(f"[bold]a = {formatear_numero(self.a)}, b = {formatear_numero(self.b)}, c = {formatear_numero(self.c)}[/bold]")

    def seleccionar_metodo(self):
        """Selecciona el método de solución"""
        limpiar_pantalla()
        console.print(Panel.fit("SELECCIÓN DE MÉTODO", style="bold blue"))
        
        console.print("\n[bold]Métodos disponibles:[/bold]")
        console.print("1. Fórmula general (cuadrática)")
        console.print("2. Completar el cuadrado")
        console.print("3. Factorización (si es posible)")
        
        metodos = ["formula_general", "completar_cuadrado", "factorizacion"]
        nombres = ["Fórmula general", "Completar cuadrado", "Factorización"]
        
        console.print(f"\n[bold]Método actual:[/bold] {nombres[metodos.index(self.metodo)]}")
        
        if input("\n¿Cambiar método? (s/n): ").lower() == 's':
            opcion = validar_entero("Seleccione método (1-3): ", 1, 3)
            self.metodo = metodos[opcion - 1]
            
            console.print(f"[green]Método cambiado a: {nombres[opcion - 1]}[/green]")
            input("Presione Enter para continuar...")

    def resolver_ecuacion(self):
        """Resuelve la ecuación cuadrática"""
        if self.a == 0:
            console.print("[red]Primero debe ingresar coeficientes válidos[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("RESOLVIENDO ECUACIÓN", style="bold yellow"))
        
        self._mostrar_ecuacion()
        
        try:
            if self.metodo == "formula_general":
                self._resolver_formula_general()
            elif self.metodo == "completar_cuadrado":
                self._resolver_completar_cuadrado()
            elif self.metodo == "factorizacion":
                self._resolver_factorizacion()
            
            console.print(f"\n[bold green]Resolución completada[/bold green]")
            
        except Exception as e:
            console.print(f"[red]Error durante la resolución: {e}[/red]")
        
        input("Presione Enter para continuar...")

    def _resolver_formula_general(self):
        """Resuelve usando la fórmula general"""
        console.print(f"\n[bold cyan]Método: Fórmula General[/bold cyan]")
        
        self.pasos_solucion = []
        self.pasos_solucion.append("Fórmula general: x = (-b ± √(b² - 4ac)) / (2a)")
        
        # Calcular discriminante
        self.discriminante = self.b**2 - 4*self.a*self.c
        self.pasos_solucion.append(
            f"Discriminante: Δ = b² - 4ac = ({formatear_numero(self.b)})² - 4({formatear_numero(self.a)})({formatear_numero(self.c)})"
        )
        self.pasos_solucion.append(f"Δ = {formatear_numero(self.discriminante)}")
        
        # Resolver según el discriminante
        if self.discriminante > 0:
            # Dos raíces reales distintas
            sqrt_discriminante = np.sqrt(self.discriminante)
            x1 = (-self.b + sqrt_discriminante) / (2 * self.a)
            x2 = (-self.b - sqrt_discriminante) / (2 * self.a)
            
            self.raices = [x1, x2]
            self.pasos_solucion.append("Como Δ > 0, existen dos raíces reales distintas:")
            self.pasos_solucion.append(f"x₁ = (-{formatear_numero(self.b)} + √{formatear_numero(self.discriminante)}) / (2·{formatear_numero(self.a)}) = {formatear_numero(x1)}")
            self.pasos_solucion.append(f"x₂ = (-{formatear_numero(self.b)} - √{formatear_numero(self.discriminante)}) / (2·{formatear_numero(self.a)}) = {formatear_numero(x2)}")
            
        elif self.discriminante == 0:
            # Una raíz real (raíz doble)
            x = -self.b / (2 * self.a)
            self.raices = [x]
            self.pasos_solucion.append("Como Δ = 0, existe una raíz real doble:")
            self.pasos_solucion.append(f"x = -b / (2a) = -{formatear_numero(self.b)} / (2·{formatear_numero(self.a)}) = {formatear_numero(x)}")
            
        else:
            # Dos raíces complejas conjugadas
            sqrt_discriminante = cmath.sqrt(self.discriminante)
            x1 = (-self.b + sqrt_discriminante) / (2 * self.a)
            x2 = (-self.b - sqrt_discriminante) / (2 * self.a)
            
            self.raices = [x1, x2]
            self.pasos_solucion.append("Como Δ < 0, existen dos raíces complejas conjugadas:")
            self.pasos_solucion.append(f"x₁ = {formatear_numero(x1)}")
            self.pasos_solucion.append(f"x₂ = {formatear_numero(x2)}")

    def _resolver_completar_cuadrado(self):
        """Resuelve completando el cuadrado"""
        console.print(f"\n[bold cyan]Método: Completar el Cuadrado[/bold cyan]")
        
        self.pasos_solucion = []
        
        # Paso 1: Dividir por 'a' si es necesario
        if abs(self.a - 1.0) > 1e-12:
            a_norm, b_norm, c_norm = 1.0, self.b/self.a, self.c/self.a
            self.pasos_solucion.append(f"Dividir por a = {formatear_numero(self.a)}:")
            self.pasos_solucion.append(f"x² + {formatear_numero(b_norm)}x + {formatear_numero(c_norm)} = 0")
        else:
            a_norm, b_norm, c_norm = self.a, self.b, self.c
            self.pasos_solucion.append(f"Ecuación: x² + {formatear_numero(b_norm)}x + {formatear_numero(c_norm)} = 0")
        
        # Paso 2: Mover el término constante
        self.pasos_solucion.append(f"Mover término constante: x² + {formatear_numero(b_norm)}x = {formatear_numero(-c_norm)}")
        
        # Paso 3: Completar el cuadrado
        h = b_norm / 2
        k = h**2
        self.pasos_solucion.append(f"Completar el cuadrado: agregar y restar ({formatear_numero(b_norm)}/2)² = {formatear_numero(k)}")
        self.pasos_solucion.append(f"x² + {formatear_numero(b_norm)}x + {formatear_numero(k)} = {formatear_numero(-c_norm + k)}")
        
        # Paso 4: Factorizar
        self.pasos_solucion.append(f"Factorizar: (x + {formatear_numero(h)})² = {formatear_numero(-c_norm + k)}")
        
        # Paso 5: Resolver
        discriminante_mod = -c_norm + k
        
        if discriminante_mod > 0:
            sqrt_disc = np.sqrt(discriminante_mod)
            x1 = -h + sqrt_disc
            x2 = -h - sqrt_disc
            self.raices = [x1, x2]
            self.pasos_solucion.append(f"x + {formatear_numero(h)} = ±√{formatear_numero(discriminante_mod)} = ±{formatear_numero(sqrt_disc)}")
            self.pasos_solucion.append(f"x₁ = {formatear_numero(x1)}, x₂ = {formatear_numero(x2)}")
            
        elif discriminante_mod == 0:
            x = -h
            self.raices = [x]
            self.pasos_solucion.append(f"x + {formatear_numero(h)} = 0")
            self.pasos_solucion.append(f"x = {formatear_numero(x)} (raíz doble)")
            
        else:
            sqrt_disc = cmath.sqrt(discriminante_mod)
            x1 = -h + sqrt_disc
            x2 = -h - sqrt_disc
            self.raices = [x1, x2]
            self.pasos_solucion.append(f"x + {formatear_numero(h)} = ±√{formatear_numero(discriminante_mod)} = ±{formatear_numero(sqrt_disc)}")
            self.pasos_solucion.append(f"x₁ = {formatear_numero(x1)}, x₂ = {formatear_numero(x2)} (complejas)")

    def _resolver_factorizacion(self):
        """Intenta resolver por factorización"""
        console.print(f"\n[bold cyan]Método: Factorización[/bold cyan]")
        
        self.pasos_solucion = []
        
        # Intentar factorización para casos especiales
        if self._intentar_factorizar():
            return
        
        # Si no se puede factorizar fácilmente, usar fórmula general
        console.print("[yellow]No se encontró factorización simple. Usando fórmula general.[/yellow]")
        self._resolver_formula_general()

    def _intentar_factorizar(self) -> bool:
        """Intenta factorizar la ecuación cuadrática"""
        # Caso 1: c = 0 (factor común x)
        if abs(self.c) < 1e-12:
            self.pasos_solucion.append("Factor común x:")
            self.pasos_solucion.append(f"x({formatear_numero(self.a)}x + {formatear_numero(self.b)}) = 0")
            
            x1 = 0
            x2 = -self.b / self.a
            self.raices = [x1, x2]
            
            self.pasos_solucion.append(f"x₁ = 0")
            self.pasos_solucion.append(f"x₂ = -{formatear_numero(self.b)}/{formatear_numero(self.a)} = {formatear_numero(x2)}")
            return True
        
        # Caso 2: Diferencia de cuadrados (b = 0)
        if abs(self.b) < 1e-12:
            if self.a * self.c < 0:  # ax² + c = 0 con ac < 0
                self.pasos_solucion.append("Diferencia de cuadrados:")
                sqrt_ratio = np.sqrt(-self.c / self.a)
                self.pasos_solucion.append(f"{formatear_numero(self.a)}x² + {formatear_numero(self.c)} = 0")
                self.pasos_solucion.append(f"x² = {formatear_numero(-self.c/self.a)}")
                
                x1 = sqrt_ratio
                x2 = -sqrt_ratio
                self.raices = [x1, x2]
                
                self.pasos_solucion.append(f"x = ±√{formatear_numero(-self.c/self.a)} = ±{formatear_numero(sqrt_ratio)}")
                return True
        
        # Caso 3: Trinomio cuadrado perfecto
        discriminante = self.b**2 - 4*self.a*self.c
        if abs(discriminante) < 1e-12:
            self.pasos_solucion.append("Trinomio cuadrado perfecto:")
            x = -self.b / (2 * self.a)
            self.raices = [x]
            self.pasos_solucion.append(f"(x + {formatear_numero(self.b/(2*self.a))})² = 0")
            self.pasos_solucion.append(f"x = {formatear_numero(x)} (raíz doble)")
            return True
        
        # Caso 4: Intentar factorización por tanteo (para coeficientes enteros pequeños)
        if (abs(self.a - round(self.a)) < 1e-10 and 
            abs(self.b - round(self.b)) < 1e-10 and 
            abs(self.c - round(self.c)) < 1e-10):
            
            return self._factorizar_enteros()
        
        return False

    def _factorizar_enteros(self) -> bool:
        """Intenta factorizar cuando los coeficientes son enteros"""
        a, b, c = int(round(self.a)), int(round(self.b)), int(round(self.c))
        
        # Buscar factores de a*c que sumen b
        ac = a * c
        
        for i in range(1, abs(ac) + 1):
            if ac % i == 0:
                j = ac // i
                
                # Probar diferentes combinaciones de signos
                for p, q in [(i, j), (-i, -j), (i, -j), (-i, j)]:
                    if p + q == b:
                        # Encontramos factorización
                        self.pasos_solucion.append(f"Buscar dos números que multiplicados den {ac} y sumados den {b}")
                        self.pasos_solucion.append(f"Los números son {p} y {q}")
                        
                        # Calcular raíces
                        # ax² + bx + c = 0 se factoriza como (px + r)(qx + s) = 0
                        # donde p*q = a, r*s = c, ps + qr = b
                        
                        # Para simplificar, usar la fórmula general pero mostrar factorización
                        discriminante = b**2 - 4*a*c
                        sqrt_disc = np.sqrt(discriminante)
                        x1 = (-b + sqrt_disc) / (2*a)
                        x2 = (-b - sqrt_disc) / (2*a)
                        
                        self.raices = [x1, x2]
                        self.pasos_solucion.append(f"Factorización encontrada, raíces: x₁ = {formatear_numero(x1)}, x₂ = {formatear_numero(x2)}")
                        return True
        
        return False

    def analizar_discriminante(self):
        """Analiza el discriminante y sus implicaciones"""
        if self.a == 0:
            console.print("[red]Primero debe ingresar coeficientes válidos[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("ANÁLISIS DEL DISCRIMINANTE", style="bold magenta"))
        
        self._mostrar_ecuacion()
        
        # Calcular discriminante
        self.discriminante = self.b**2 - 4*self.a*self.c
        
        console.print(f"\n[bold]Discriminante:[/bold] Δ = b² - 4ac")
        console.print(f"Δ = ({formatear_numero(self.b)})² - 4({formatear_numero(self.a)})({formatear_numero(self.c)})")
        console.print(f"Δ = {formatear_numero(self.discriminante)}")
        
        # Análisis según el valor
        tabla = Table(title="Análisis del Discriminante")
        tabla.add_column("Condición", style="cyan")
        tabla.add_column("Interpretación", style="yellow")
        tabla.add_column("Naturaleza de las raíces", style="green")
        
        if self.discriminante > 0:
            tabla.add_row(
                "Δ > 0",
                "Positivo",
                "Dos raíces reales distintas"
            )
            console.print(tabla)
            console.print(f"\n[bold green]✓ La ecuación tiene dos soluciones reales diferentes[/bold green]")
            
        elif self.discriminante == 0:
            tabla.add_row(
                "Δ = 0",
                "Cero",
                "Una raíz real doble"
            )
            console.print(tabla)
            console.print(f"\n[bold yellow]• La ecuación tiene una solución real repetida[/bold yellow]")
            console.print(f"• La parábola es tangente al eje x")
            
        else:
            tabla.add_row(
                "Δ < 0",
                "Negativo",
                "Dos raíces complejas conjugadas"
            )
            console.print(tabla)
            console.print(f"\n[bold red]⚠ La ecuación no tiene soluciones reales[/bold red]")
            console.print(f"• Tiene dos soluciones complejas conjugadas")
            console.print(f"• La parábola no corta el eje x")
        
        input("\nPresione Enter para continuar...")

    def encontrar_vertice(self):
        """Encuentra el vértice y eje de simetría"""
        if self.a == 0:
            console.print("[red]Primero debe ingresar coeficientes válidos[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("VÉRTICE Y EJE DE SIMETRÍA", style="bold cyan"))
        
        self._mostrar_ecuacion()
        
        # Calcular vértice
        x_vertice = -self.b / (2 * self.a)
        y_vertice = self.a * x_vertice**2 + self.b * x_vertice + self.c
        self.vertice = (x_vertice, y_vertice)
        
        console.print("\n[bold]Cálculo del vértice:[/bold]")
        console.print(f"x del vértice: x = -b/(2a) = -{formatear_numero(self.b)}/(2·{formatear_numero(self.a)}) = {formatear_numero(x_vertice)}")
        console.print(f"y del vértice: y = f({formatear_numero(x_vertice)}) = {formatear_numero(y_vertice)}")
        
        # Información adicional
        tabla = Table(title="Información de la Parábola")
        tabla.add_column("Propiedad", style="cyan")
        tabla.add_column("Valor", style="yellow")
        tabla.add_column("Significado", style="green")
        
        tabla.add_row("Vértice", f"({formatear_numero(x_vertice)}, {formatear_numero(y_vertice)})", "Punto máximo/mínimo")
        tabla.add_row("Eje de simetría", f"x = {formatear_numero(x_vertice)}", "Línea vertical de simetría")
        
        abertura = "hacia arriba" if self.a > 0 else "hacia abajo"
        tipo_vertice = "mínimo" if self.a > 0 else "máximo"
        
        tabla.add_row("Abertura", abertura, f"Vértice es {tipo_vertice}")
        tabla.add_row("Coef. principal", formatear_numero(self.a), "Determina abertura y amplitud")
        
        console.print(tabla)
        
        # Información sobre intersecciones
        console.print(f"\n[bold]Intersección con eje y:[/bold] (0, {formatear_numero(self.c)})")
        
        if hasattr(self, 'raices') and self.raices:
            console.print(f"[bold]Intersecciones con eje x:[/bold]")
            for i, raiz in enumerate(self.raices):
                if isinstance(raiz, complex):
                    console.print(f"  x₍{i+1}₎ = {formatear_numero(raiz)} (compleja)")
                else:
                    console.print(f"  x₍{i+1}₎ = {formatear_numero(raiz)}")
        
        input("\nPresione Enter para continuar...")

    def mostrar_resultados_completos(self):
        """Muestra todos los resultados del análisis"""
        if self.a == 0:
            console.print("[red]Primero debe ingresar coeficientes válidos[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("RESULTADOS COMPLETOS", style="bold green"))
        
        # Ecuación
        self._mostrar_ecuacion()
        
        # Discriminante
        console.print(f"\n[bold]Discriminante:[/bold] Δ = {formatear_numero(self.discriminante)}")
        
        # Raíces
        if self.raices:
            console.print(f"\n[bold]Raíces:[/bold]")
            for i, raiz in enumerate(self.raices):
                if isinstance(raiz, complex):
                    console.print(f"  x₍{i+1}₎ = {formatear_numero(raiz)}")
                else:
                    console.print(f"  x₍{i+1}₎ = {formatear_numero(raiz)}")
        
        # Vértice
        if hasattr(self, 'vertice'):
            console.print(f"\n[bold]Vértice:[/bold] ({formatear_numero(self.vertice[0])}, {formatear_numero(self.vertice[1])})")
        
        # Pasos de solución
        if self.pasos_solucion:
            console.print(f"\n[bold]Pasos de solución ({self.metodo.replace('_', ' ').title()}):[/bold]")
            for i, paso in enumerate(self.pasos_solucion, 1):
                console.print(f"{i}. {paso}")
        
        input("\nPresione Enter para continuar...")

    def mostrar_grafico(self):
        """Muestra el gráfico de la función cuadrática"""
        if self.a == 0:
            console.print("[red]Primero debe ingresar coeficientes válidos[/red]")
            input("Presione Enter para continuar...")
            return
        
        console.print("[cyan]Generando gráfico...[/cyan]")
        
        # Asegurar que tenemos el vértice calculado
        if not hasattr(self, 'vertice') or self.vertice == (0.0, 0.0):
            x_vertice = -self.b / (2 * self.a)
            y_vertice = self.a * x_vertice**2 + self.b * x_vertice + self.c
            self.vertice = (x_vertice, y_vertice)
        
        # Asegurar que tenemos las raíces calculadas
        if not self.raices:
            self._resolver_formula_general()
        
        graficar_cuadratica(self.a, self.b, self.c, self.raices, self.vertice)

    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre ecuaciones cuadráticas"""
        limpiar_pantalla()
        ayuda_texto = """
[bold blue]ECUACIONES CUADRÁTICAS[/bold blue]

[bold]Forma general:[/bold] ax² + bx + c = 0 (donde a ≠ 0)

[bold]Métodos de solución implementados:[/bold]

[bold cyan]1. Fórmula General:[/bold cyan]
x = (-b ± √(b² - 4ac)) / (2a)
• Funciona para cualquier ecuación cuadrática
• Método más directo y confiable

[bold cyan]2. Completar el Cuadrado:[/bold cyan]
• Transforma la ecuación a la forma (x - h)² = k
• Útil para encontrar el vértice
• Método geométricamente intuitivo

[bold cyan]3. Factorización:[/bold cyan]
• Busca expresar como producto de binomios
• Más rápido cuando es posible
• Limitado a casos especiales

[bold]Análisis del Discriminante (Δ = b² - 4ac):[/bold]
• Δ > 0: Dos raíces reales distintas
• Δ = 0: Una raíz real doble (vértice en eje x)
• Δ < 0: Dos raíces complejas conjugadas

[bold]Propiedades de la Parábola:[/bold]
• Vértice: (-b/2a, f(-b/2a))
• Eje de simetría: x = -b/2a
• Si a > 0: abre hacia arriba (mínimo)
• Si a < 0: abre hacia abajo (máximo)

[bold]Aplicaciones:[/bold]
• Movimiento parabólico
• Optimización de funciones cuadráticas
• Intersecciones y puntos críticos
• Modelado de fenómenos cuadráticos
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
                    self.ingresar_coeficientes()
                elif opcion == 2:
                    self.seleccionar_metodo()
                elif opcion == 3:
                    self.resolver_ecuacion()
                elif opcion == 4:
                    self.analizar_discriminante()
                elif opcion == 5:
                    self.encontrar_vertice()
                elif opcion == 6:
                    self.mostrar_resultados_completos()
                elif opcion == 7:
                    self.mostrar_grafico()
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
    cuadratica = PolinomioOrden2()
    cuadratica.main()
