#!/usr/bin/env python3
"""
Interpolación de Lagrange - Implementación con menús interactivos
Interpolación polinomial usando los polinomios base de Lagrange
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.progress import track
import os
import time
from typing import Tuple, List, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utilidades import *

console = Console()

class LagrangeInterpolator:
    def __init__(self):
        self.x_datos = None
        self.y_datos = None
        self.n_puntos = 0
        self.polinomio_coefs = None
        self.resultados_evaluacion = {}
        self.puntos_interpolacion = None
        self.valores_interpolados = None
        
    def ingresar_datos(self):
        """Menú para ingresar los puntos de datos"""
        limpiar_pantalla()
        mostrar_titulo("INGRESO DE DATOS PARA INTERPOLACIÓN")
        
        console.print("\n[bold blue]Opciones de ingreso:[/bold blue]")
        console.print("1. Ingresar puntos manualmente")
        console.print("2. Usar función predefinida")
        console.print("3. Cargar desde archivo")
        console.print("4. Volver al menú principal")
        
        try:
            opcion = input("\nSeleccione una opción (1-4): ").strip()
            
            if opcion == "1":
                return self._ingresar_puntos_manual()
            elif opcion == "2":
                return self._usar_funcion_predefinida()
            elif opcion == "3":
                return self._cargar_desde_archivo()
            elif opcion == "4":
                return False
            else:
                console.print("[red]Opción inválida.[/red]")
                time.sleep(1)
                return self.ingresar_datos()
                
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            input("\nPresione Enter para continuar...")
            return False
    
    def _ingresar_puntos_manual(self):
        """Ingresa puntos manualmente"""
        try:
            while True:
                try:
                    self.n_puntos = int(input("\nNúmero de puntos de datos: "))
                    if self.n_puntos >= 2:
                        break
                    else:
                        console.print("[red]Se necesitan al menos 2 puntos.[/red]")
                except ValueError:
                    console.print("[red]Por favor ingrese un número válido.[/red]")
            
            self.x_datos = np.zeros(self.n_puntos)
            self.y_datos = np.zeros(self.n_puntos)
            
            console.print(f"\n[bold blue]Ingrese los {self.n_puntos} puntos (x, y):[/bold blue]")
            
            for i in range(self.n_puntos):
                while True:
                    try:
                        console.print(f"\n[yellow]Punto {i+1}:[/yellow]")
                        x = float(input(f"  x{i+1}: "))
                        
                        # Verificar que x no se repita
                        if i > 0 and x in self.x_datos[:i]:
                            console.print("[red]El valor de x ya existe. Use un valor diferente.[/red]")
                            continue
                            
                        y = float(input(f"  y{i+1}: "))
                        
                        self.x_datos[i] = x
                        self.y_datos[i] = y
                        break
                        
                    except ValueError:
                        console.print("[red]Por favor ingrese números válidos.[/red]")
            
            # Ordenar puntos por x
            indices = np.argsort(self.x_datos)
            self.x_datos = self.x_datos[indices]
            self.y_datos = self.y_datos[indices]
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error al ingresar datos: {str(e)}[/red]")
            return False
    
    def _usar_funcion_predefinida(self):
        """Genera puntos usando una función predefinida"""
        try:
            console.print("\n[bold blue]Funciones disponibles:[/bold blue]")
            console.print("1. f(x) = x²")
            console.print("2. f(x) = sin(x)")
            console.print("3. f(x) = e^x")
            console.print("4. f(x) = ln(x)")
            console.print("5. f(x) = 1/(1+x²) (Función de Runge)")
            
            opcion_func = input("\nSeleccione función (1-5): ").strip()
            
            # Definir función
            funciones = {
                "1": (lambda x: x**2, "x²", (-3, 3)),
                "2": (lambda x: np.sin(x), "sin(x)", (0, 2*np.pi)),
                "3": (lambda x: np.exp(x), "e^x", (-1, 1)),
                "4": (lambda x: np.log(x), "ln(x)", (0.1, 3)),
                "5": (lambda x: 1/(1 + x**2), "1/(1+x²)", (-5, 5))
            }
            
            if opcion_func not in funciones:
                console.print("[red]Opción inválida.[/red]")
                return False
            
            func, nombre_func, (x_min, x_max) = funciones[opcion_func]
            
            # Número de puntos
            while True:
                try:
                    self.n_puntos = int(input(f"\nNúmero de puntos a generar: "))
                    if self.n_puntos >= 2:
                        break
                    else:
                        console.print("[red]Se necesitan al menos 2 puntos.[/red]")
                except ValueError:
                    console.print("[red]Por favor ingrese un número válido.[/red]")
            
            # Generar puntos
            self.x_datos = np.linspace(x_min, x_max, self.n_puntos)
            self.y_datos = func(self.x_datos)
            
            console.print(f"[green]✓ Generados {self.n_puntos} puntos de f(x) = {nombre_func}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error al generar función: {str(e)}[/red]")
            return False
    
    def _cargar_desde_archivo(self):
        """Carga datos desde archivo CSV"""
        try:
            filename = input("\nNombre del archivo CSV: ").strip()
            
            if not filename.endswith('.csv'):
                filename += '.csv'
            
            if not os.path.exists(filename):
                console.print(f"[red]Archivo {filename} no encontrado.[/red]")
                return False
            
            datos = np.loadtxt(filename, delimiter=',', skiprows=1)
            
            if datos.shape[1] < 2:
                console.print("[red]El archivo debe tener al menos 2 columnas (x, y).[/red]")
                return False
            
            self.x_datos = datos[:, 0]
            self.y_datos = datos[:, 1]
            self.n_puntos = len(self.x_datos)
            
            # Verificar puntos únicos
            if len(np.unique(self.x_datos)) != len(self.x_datos):
                console.print("[red]Hay valores x duplicados en los datos.[/red]")
                return False
            
            # Ordenar por x
            indices = np.argsort(self.x_datos)
            self.x_datos = self.x_datos[indices]
            self.y_datos = self.y_datos[indices]
            
            console.print(f"[green]✓ Cargados {self.n_puntos} puntos desde {filename}[/green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error al cargar archivo: {str(e)}[/red]")
            return False
    
    def mostrar_datos(self):
        """Muestra los datos ingresados"""
        if self.x_datos is None:
            console.print("[red]No hay datos ingresados.[/red]")
            return
        
        limpiar_pantalla()
        mostrar_titulo("DATOS PARA INTERPOLACIÓN")
        
        # Tabla de datos
        table = Table(title=f"Puntos de Datos ({self.n_puntos} puntos)", show_header=True, header_style="bold blue")
        table.add_column("i", justify="center", style="cyan")
        table.add_column("x_i", justify="center", style="green")
        table.add_column("y_i", justify="center", style="yellow")
        
        for i in range(self.n_puntos):
            table.add_row(str(i), f"{self.x_datos[i]:.6f}", f"{self.y_datos[i]:.6f}")
        
        console.print(table)
        
        # Estadísticas básicas
        stats_table = Table(title="Estadísticas", show_header=True, header_style="bold blue")
        stats_table.add_column("Variable", style="cyan")
        stats_table.add_column("Mínimo", justify="right", style="green")
        stats_table.add_column("Máximo", justify="right", style="green")
        stats_table.add_column("Rango", justify="right", style="yellow")
        
        x_min, x_max = np.min(self.x_datos), np.max(self.x_datos)
        y_min, y_max = np.min(self.y_datos), np.max(self.y_datos)
        
        stats_table.add_row("x", f"{x_min:.6f}", f"{x_max:.6f}", f"{x_max - x_min:.6f}")
        stats_table.add_row("y", f"{y_min:.6f}", f"{y_max:.6f}", f"{y_max - y_min:.6f}")
        
        console.print(stats_table)
        input("\nPresione Enter para continuar...")
    
    def polinomio_lagrange_base(self, i: int, x: float) -> float:
        """Calcula el i-ésimo polinomio base de Lagrange en el punto x"""
        resultado = 1.0
        for j in range(self.n_puntos):
            if i != j:
                resultado *= (x - self.x_datos[j]) / (self.x_datos[i] - self.x_datos[j])
        return resultado
    
    def evaluar_interpolacion(self, x: float) -> float:
        """Evalúa el polinomio interpolador de Lagrange en x"""
        if self.x_datos is None:
            raise ValueError("No hay datos disponibles")
        
        resultado = 0.0
        for i in range(self.n_puntos):
            Li = self.polinomio_lagrange_base(i, x)
            resultado += self.y_datos[i] * Li
        
        return resultado
    
    def calcular_polinomio_explicito(self):
        """Calcula los coeficientes del polinomio interpolador"""
        try:
            # Usar diferencias divididas para obtener coeficientes
            # Alternativamente, expandir la forma de Lagrange
            self.polinomio_coefs = np.zeros(self.n_puntos)
            
            # Para cada término del polinomio de Lagrange
            for i in range(self.n_puntos):
                # Contribución del término y_i * L_i(x)
                coef_temp = np.zeros(self.n_puntos)
                coef_temp[0] = self.y_datos[i]
                
                # Expandir el producto L_i(x)
                for j in range(self.n_puntos):
                    if i != j:
                        # Multiplicar por (x - x_j) / (x_i - x_j)
                        factor = 1.0 / (self.x_datos[i] - self.x_datos[j])
                        
                        # Multiplicar coef_temp por (x - x_j)
                        new_coef = np.zeros(self.n_puntos)
                        for k in range(self.n_puntos - 1):
                            new_coef[k+1] += coef_temp[k] * factor
                            new_coef[k] -= coef_temp[k] * self.x_datos[j] * factor
                        
                        coef_temp = new_coef
                
                self.polinomio_coefs += coef_temp
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error al calcular coeficientes: {str(e)}[/red]")
            return False
    
    def evaluar_puntos_especificos(self):
        """Permite evaluar el interpolador en puntos específicos"""
        if self.x_datos is None:
            console.print("[red]Primero debe ingresar datos.[/red]")
            return
        
        limpiar_pantalla()
        mostrar_titulo("EVALUACIÓN DEL INTERPOLADOR")
        
        console.print("\n[bold blue]Opciones de evaluación:[/bold blue]")
        console.print("1. Evaluar en puntos específicos")
        console.print("2. Evaluar en rango con paso fijo")
        console.print("3. Volver al menú principal")
        
        try:
            opcion = input("\nSeleccione una opción (1-3): ").strip()
            
            if opcion == "1":
                self._evaluar_puntos_individuales()
            elif opcion == "2":
                self._evaluar_rango()
            elif opcion == "3":
                return
            else:
                console.print("[red]Opción inválida.[/red]")
                time.sleep(1)
                
        except Exception as e:
            console.print(f"[red]Error: {str(e)}[/red]")
            input("\nPresione Enter para continuar...")
    
    def _evaluar_puntos_individuales(self):
        """Evalúa en puntos específicos ingresados por el usuario"""
        try:
            puntos_eval = []
            
            console.print("\n[bold blue]Ingrese puntos para evaluar (termine con 'fin'):[/bold blue]")
            
            while True:
                entrada = input(f"Punto {len(puntos_eval) + 1} (x): ").strip().lower()
                
                if entrada == 'fin':
                    break
                
                try:
                    x = float(entrada)
                    y = self.evaluar_interpolacion(x)
                    puntos_eval.append((x, y))
                    console.print(f"  f({x:.6f}) = {y:.6f}")
                except ValueError:
                    console.print("[red]Valor inválido. Ingrese un número o 'fin'.[/red]")
            
            if puntos_eval:
                self.resultados_evaluacion['puntos_individuales'] = puntos_eval
                console.print(f"[green]✓ Evaluados {len(puntos_eval)} puntos.[/green]")
            
        except Exception as e:
            console.print(f"[red]Error en evaluación: {str(e)}[/red]")
    
    def _evaluar_rango(self):
        """Evalúa en un rango con paso fijo"""
        try:
            x_min = float(input("\nX mínimo: "))
            x_max = float(input("X máximo: "))
            n_puntos = int(input("Número de puntos: "))
            
            if n_puntos < 2:
                console.print("[red]Se necesitan al menos 2 puntos.[/red]")
                return
            
            self.puntos_interpolacion = np.linspace(x_min, x_max, n_puntos)
            self.valores_interpolados = np.array([
                self.evaluar_interpolacion(x) for x in track(
                    self.puntos_interpolacion, 
                    description="Evaluando interpolador..."
                )
            ])
            
            self.resultados_evaluacion['rango'] = {
                'x': self.puntos_interpolacion,
                'y': self.valores_interpolados,
                'x_min': x_min,
                'x_max': x_max,
                'n_puntos': n_puntos
            }
            
            console.print(f"[green]✓ Evaluado en {n_puntos} puntos del rango [{x_min}, {x_max}].[/green]")
            
        except Exception as e:
            console.print(f"[red]Error en evaluación: {str(e)}[/red]")
    
    def mostrar_resultados(self):
        """Muestra los resultados de la interpolación"""
        if self.x_datos is None:
            console.print("[red]No hay datos disponibles.[/red]")
            return
        
        limpiar_pantalla()
        mostrar_titulo("RESULTADOS DE LA INTERPOLACIÓN")
        
        # Información general
        info_panel = Panel(
            f"[bold blue]Interpolación de Lagrange[/bold blue]\n"
            f"Puntos de datos: {self.n_puntos}\n"
            f"Grado del polinomio: {self.n_puntos - 1}",
            title="Información General",
            border_style="blue"
        )
        console.print(info_panel)
        
        # Mostrar polinomios base de Lagrange
        console.print(f"\n[bold blue]Polinomios Base de Lagrange:[/bold blue]")
        for i in range(min(self.n_puntos, 5)):  # Mostrar máximo 5
            denominador_str = " × ".join([
                f"({self.x_datos[i]:.3f} - {self.x_datos[j]:.3f})" 
                for j in range(self.n_puntos) if i != j
            ])
            numerador_str = " × ".join([
                f"(x - {self.x_datos[j]:.3f})" 
                for j in range(self.n_puntos) if i != j
            ])
            
            console.print(f"L_{i}(x) = [{numerador_str}] / [{denominador_str}]")
        
        if self.n_puntos > 5:
            console.print(f"... y {self.n_puntos - 5} polinomios más")
        
        # Mostrar resultados de evaluación si existen
        if 'puntos_individuales' in self.resultados_evaluacion:
            console.print(f"\n[bold blue]Evaluaciones Individuales:[/bold blue]")
            eval_table = Table(show_header=True, header_style="bold blue")
            eval_table.add_column("x", justify="center", style="cyan")
            eval_table.add_column("P(x)", justify="center", style="green")
            
            for x, y in self.resultados_evaluacion['puntos_individuales']:
                eval_table.add_row(f"{x:.6f}", f"{y:.6f}")
            
            console.print(eval_table)
        
        # Error en puntos de datos (debe ser cero)
        console.print(f"\n[bold blue]Verificación en Puntos de Datos:[/bold blue]")
        error_table = Table(show_header=True, header_style="bold blue")
        error_table.add_column("i", justify="center", style="cyan")
        error_table.add_column("x_i", justify="center", style="yellow")
        error_table.add_column("y_i", justify="center", style="green")
        error_table.add_column("P(x_i)", justify="center", style="green")
        error_table.add_column("Error", justify="center", style="red")
        
        for i in range(self.n_puntos):
            valor_interpolado = self.evaluar_interpolacion(self.x_datos[i])
            error = abs(valor_interpolado - self.y_datos[i])
            
            error_table.add_row(
                str(i),
                f"{self.x_datos[i]:.6f}",
                f"{self.y_datos[i]:.6f}",
                f"{valor_interpolado:.6f}",
                f"{error:.2e}"
            )
        
        console.print(error_table)
        input("\nPresione Enter para continuar...")
    
    def graficar_interpolacion(self):
        """Genera gráfica de la interpolación"""
        if self.x_datos is None:
            console.print("[red]No hay datos disponibles.[/red]")
            return
        
        try:
            # Generar puntos para la curva suave
            x_min = np.min(self.x_datos) - 0.5 * (np.max(self.x_datos) - np.min(self.x_datos))
            x_max = np.max(self.x_datos) + 0.5 * (np.max(self.x_datos) - np.min(self.x_datos))
            x_curva = np.linspace(x_min, x_max, 1000)
            
            console.print("[cyan]Generando gráfica...[/cyan]")
            y_curva = np.array([
                self.evaluar_interpolacion(x) for x in track(
                    x_curva,
                    description="Calculando interpolación..."
                )
            ])
            
            plt.figure(figsize=(12, 8))
            
            # Puntos de datos originales
            plt.plot(self.x_datos, self.y_datos, 'ro', markersize=8, label='Datos originales', zorder=3)
            
            # Curva interpolada
            plt.plot(x_curva, y_curva, 'b-', linewidth=2, label='Interpolación de Lagrange', zorder=2)
            
            # Puntos evaluados si existen
            if 'rango' in self.resultados_evaluacion:
                puntos_eval = self.resultados_evaluacion['rango']
                plt.plot(puntos_eval['x'], puntos_eval['y'], 'g+', markersize=6, 
                        label='Puntos evaluados', zorder=3)
            
            # Anotaciones en puntos de datos
            for i in range(self.n_puntos):
                plt.annotate(f'({self.x_datos[i]:.2f}, {self.y_datos[i]:.2f})',
                           (self.x_datos[i], self.y_datos[i]),
                           xytext=(5, 5), textcoords='offset points',
                           fontsize=8, alpha=0.7)
            
            plt.grid(True, alpha=0.3)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title(f'Interpolación de Lagrange - Grado {self.n_puntos - 1}')
            plt.legend()
            
            # Información en la gráfica
            info_text = f'Puntos: {self.n_puntos}\nGrado: {self.n_puntos - 1}'
            plt.text(0.02, 0.98, info_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            console.print(f"[red]Error al generar gráfica: {str(e)}[/red]")

def mostrar_menu_principal():
    """Muestra el menú principal"""
    limpiar_pantalla()
    
    panel = Panel(
        "[bold blue]INTERPOLACIÓN DE LAGRANGE[/bold blue]\n\n"
        "Construye un polinomio interpolador que pasa exactamente\n"
        "por todos los puntos de datos dados usando los polinomios\n"
        "base de Lagrange",
        title="📈 Interpolación Polinomial",
        border_style="blue"
    )
    console.print(panel)
    
    console.print("\n[bold green]Opciones disponibles:[/bold green]")
    console.print("1. 📝 Ingresar datos")
    console.print("2. 👁️  Ver datos actuales")
    console.print("3. 🔍 Evaluar interpolador")
    console.print("4. 📊 Ver resultados")
    console.print("5. 📈 Gráfica de interpolación")
    console.print("6. 💾 Exportar resultados")
    console.print("7. ❓ Ayuda y ejemplos")
    console.print("8. 🚪 Salir")

def mostrar_ayuda():
    """Muestra información de ayuda"""
    limpiar_pantalla()
    mostrar_titulo("AYUDA - INTERPOLACIÓN DE LAGRANGE")
    
    ayuda_text = """
[bold blue]¿Qué es la Interpolación de Lagrange?[/bold blue]
La interpolación de Lagrange construye un polinomio único de grado n-1 
que pasa exactamente por n puntos dados. Usa polinomios base especiales
llamados polinomios de Lagrange.

[bold blue]Fórmula:[/bold blue]
P(x) = Σ(i=0 to n-1) y_i × L_i(x)

donde L_i(x) = Π(j≠i) (x - x_j)/(x_i - x_j)

[bold blue]Características:[/bold blue]
• Pasa exactamente por todos los puntos dados
• El grado del polinomio es n-1 (n = número de puntos)
• No requiere resolver sistemas de ecuaciones
• Puede tener problemas de oscilación con muchos puntos

[bold blue]Ventajas:[/bold blue]
• Formulación directa y elegante
• Fácil de implementar
• Interpolación exacta

[bold blue]Desventajas:[/bold blue]
• Fenómeno de Runge con puntos equiespaciados
• Costoso computacionalmente para muchos puntos
• Puede ser numéricamente inestable

[bold blue]Recomendaciones:[/bold blue]
• Use pocos puntos (< 10) para evitar oscilaciones
• Considere interpolación por segmentos para muchos puntos
• Prefiera puntos no equiespaciados (puntos de Chebyshev)
"""
    
    panel = Panel(ayuda_text, title="Información", border_style="green")
    console.print(panel)
    input("\nPresione Enter para continuar...")

def ejemplo_predefinido():
    """Carga un ejemplo predefinido para demostración"""
    interpolador = LagrangeInterpolator()
    
    console.print("[bold blue]Cargando ejemplo predefinido...[/bold blue]")
    console.print("Función: f(x) = 1/(1 + x²) en [-5, 5] con 7 puntos")
    
    # Generar puntos de la función de Runge
    interpolador.n_puntos = 7
    interpolador.x_datos = np.linspace(-5, 5, 7)
    interpolador.y_datos = 1 / (1 + interpolador.x_datos**2)
    
    console.print(f"[green]✓ Ejemplo cargado con {interpolador.n_puntos} puntos.[/green]")
    return interpolador

def main():
    """Función principal"""
    interpolador = LagrangeInterpolator()
    
    while True:
        mostrar_menu_principal()
        
        try:
            opcion = input("\nSeleccione una opción (1-8): ").strip()
            
            if opcion == "1":
                if interpolador.ingresar_datos():
                    console.print("[green]✓ Datos ingresados correctamente.[/green]")
                    time.sleep(1)
                
            elif opcion == "2":
                interpolador.mostrar_datos()
                
            elif opcion == "3":
                interpolador.evaluar_puntos_especificos()
                
            elif opcion == "4":
                interpolador.mostrar_resultados()
                
            elif opcion == "5":
                if interpolador.x_datos is not None:
                    interpolador.graficar_interpolacion()
                else:
                    console.print("[red]Primero debe ingresar datos (opción 1).[/red]")
                input("\nPresione Enter para continuar...")
                
            elif opcion == "6":
                if interpolador.x_datos is not None:
                    exportar_resultados_lagrange(interpolador)
                else:
                    console.print("[red]Primero debe ingresar datos (opción 1).[/red]")
                input("\nPresione Enter para continuar...")
                
            elif opcion == "7":
                mostrar_ayuda()
                
                # Opción de cargar ejemplo
                if preguntar_si_no("\n¿Desea cargar un ejemplo predefinido?"):
                    interpolador = ejemplo_predefinido()
                    console.print("[green]✓ Ejemplo cargado. Use opción 2 para verlo.[/green]")
                    time.sleep(2)
                
            elif opcion == "8":
                if confirmar_salida():
                    console.print("\n[bold blue]¡Gracias por usar la Interpolación de Lagrange![/bold blue]")
                    break
                
            else:
                console.print("[red]Opción inválida. Seleccione un número del 1 al 8.[/red]")
                time.sleep(1)
                
        except KeyboardInterrupt:
            if confirmar_salida("\n\n¿Desea salir del programa?"):
                console.print("\n[bold blue]¡Hasta luego![/bold blue]")
                break
        except Exception as e:
            console.print(f"[red]Error inesperado: {str(e)}[/red]")
            input("\nPresione Enter para continuar...")

def exportar_resultados_lagrange(interpolador):
    """Exporta los resultados a archivo"""
    try:
        timestamp = obtener_timestamp()
        filename = f"lagrange_resultados_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DE INTERPOLACIÓN DE LAGRANGE\n")
            f.write("=" * 50 + "\n\n")
            
            # Datos originales
            f.write("DATOS ORIGINALES:\n")
            f.write(f"Número de puntos: {interpolador.n_puntos}\n")
            f.write(f"Grado del polinomio: {interpolador.n_puntos - 1}\n\n")
            
            f.write("Puntos de datos:\n")
            for i in range(interpolador.n_puntos):
                f.write(f"  ({interpolador.x_datos[i]:.6f}, {interpolador.y_datos[i]:.6f})\n")
            f.write("\n")
            
            # Verificación
            f.write("VERIFICACIÓN EN PUNTOS DE DATOS:\n")
            for i in range(interpolador.n_puntos):
                valor_interpolado = interpolador.evaluar_interpolacion(interpolador.x_datos[i])
                error = abs(valor_interpolado - interpolador.y_datos[i])
                f.write(f"  P({interpolador.x_datos[i]:.6f}) = {valor_interpolado:.6f}, "
                       f"Error = {error:.2e}\n")
            f.write("\n")
            
            # Evaluaciones si existen
            if 'puntos_individuales' in interpolador.resultados_evaluacion:
                f.write("EVALUACIONES INDIVIDUALES:\n")
                for x, y in interpolador.resultados_evaluacion['puntos_individuales']:
                    f.write(f"  P({x:.6f}) = {y:.6f}\n")
                f.write("\n")
            
            if 'rango' in interpolador.resultados_evaluacion:
                datos_rango = interpolador.resultados_evaluacion['rango']
                f.write(f"EVALUACIÓN EN RANGO [{datos_rango['x_min']}, {datos_rango['x_max']}]:\n")
                f.write(f"Número de puntos: {datos_rango['n_puntos']}\n")
                for i in range(len(datos_rango['x'])):
                    f.write(f"  P({datos_rango['x'][i]:.6f}) = {datos_rango['y'][i]:.6f}\n")
        
        console.print(f"[green]✓ Resultados exportados a: {filename}[/green]")
        
        # Opción de exportar datos para gráfica
        if preguntar_si_no("¿Desea exportar también datos para gráfica (CSV)?"):
            csv_filename = f"lagrange_datos_{timestamp}.csv"
            with open(csv_filename, 'w') as f:
                f.write("x,y_original,y_interpolado\n")
                
                # Exportar puntos originales y evaluación densa
                x_eval = np.linspace(np.min(interpolador.x_datos), np.max(interpolador.x_datos), 200)
                for x in x_eval:
                    y_interp = interpolador.evaluar_interpolacion(x)
                    # Marcar si es punto original
                    y_orig = ""
                    if any(abs(x - x_orig) < 1e-10 for x_orig in interpolador.x_datos):
                        idx = np.argmin(abs(interpolador.x_datos - x))
                        y_orig = interpolador.y_datos[idx]
                    
                    f.write(f"{x:.6f},{y_orig},{y_interp:.6f}\n")
            
            console.print(f"[green]✓ Datos CSV exportados a: {csv_filename}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error al exportar: {str(e)}[/red]")

if __name__ == "__main__":
    main()
