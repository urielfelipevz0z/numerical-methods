#!/usr/bin/env python3
"""
Método Simplex - Implementación con menús interactivos
Optimización lineal para problemas de programación lineal con restricciones
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

class SimplexSolver:
    def __init__(self):
        self.c = None  # Coeficientes de la función objetivo
        self.A = None  # Matriz de restricciones
        self.b = None  # Vector de términos independientes
        self.tableau = None  # Tableau simplex
        self.solucion = None
        self.valor_optimo = None
        self.historial = []
        self.es_maximizar = True
        
    def configurar_problema(self):
        """Configura el problema de programación lineal"""
        limpiar_pantalla()
        mostrar_titulo("CONFIGURACIÓN DEL PROBLEMA DE PROGRAMACIÓN LINEAL")
        
        try:
            # Tipo de optimización
            console.print("\n[bold blue]Tipo de optimización:[/bold blue]")
            console.print("1. Maximizar")
            console.print("2. Minimizar")
            
            while True:
                try:
                    opcion = int(input("\nSeleccione el tipo (1-2): "))
                    if opcion in [1, 2]:
                        self.es_maximizar = (opcion == 1)
                        break
                    else:
                        console.print("[red]Opción inválida. Seleccione 1 o 2.[/red]")
                except ValueError:
                    console.print("[red]Por favor ingrese un número válido.[/red]")
            
            # Número de variables
            while True:
                try:
                    n_vars = int(input("\nNúmero de variables de decisión: "))
                    if n_vars > 0:
                        break
                    else:
                        console.print("[red]El número de variables debe ser positivo.[/red]")
                except ValueError:
                    console.print("[red]Por favor ingrese un número válido.[/red]")
            
            # Función objetivo
            console.print(f"\n[bold blue]Coeficientes de la función objetivo (x1, x2, ..., x{n_vars}):[/bold blue]")
            self.c = np.zeros(n_vars)
            for i in range(n_vars):
                while True:
                    try:
                        self.c[i] = float(input(f"Coeficiente de x{i+1}: "))
                        break
                    except ValueError:
                        console.print("[red]Por favor ingrese un número válido.[/red]")
            
            # Si es minimización, convertir a maximización
            if not self.es_maximizar:
                self.c = -self.c
            
            # Número de restricciones
            while True:
                try:
                    n_rest = int(input("\nNúmero de restricciones (≤): "))
                    if n_rest > 0:
                        break
                    else:
                        console.print("[red]El número de restricciones debe ser positivo.[/red]")
                except ValueError:
                    console.print("[red]Por favor ingrese un número válido.[/red]")
            
            # Restricciones
            console.print(f"\n[bold blue]Restricciones (formato: a1*x1 + a2*x2 + ... ≤ b):[/bold blue]")
            self.A = np.zeros((n_rest, n_vars))
            self.b = np.zeros(n_rest)
            
            for i in range(n_rest):
                console.print(f"\n[yellow]Restricción {i+1}:[/yellow]")
                for j in range(n_vars):
                    while True:
                        try:
                            self.A[i, j] = float(input(f"  Coeficiente de x{j+1}: "))
                            break
                        except ValueError:
                            console.print("[red]Por favor ingrese un número válido.[/red]")
                
                while True:
                    try:
                        self.b[i] = float(input(f"  Término independiente: "))
                        if self.b[i] >= 0:
                            break
                        else:
                            console.print("[red]El término independiente debe ser no negativo.[/red]")
                    except ValueError:
                        console.print("[red]Por favor ingrese un número válido.[/red]")
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error al configurar el problema: {str(e)}[/red]")
            input("\nPresione Enter para continuar...")
            return False
    
    def mostrar_problema(self):
        """Muestra el problema configurado"""
        if self.c is None:
            console.print("[red]No hay problema configurado.[/red]")
            return
        
        limpiar_pantalla()
        mostrar_titulo("PROBLEMA DE PROGRAMACIÓN LINEAL CONFIGURADO")
        
        # Función objetivo
        objetivo = "Maximizar" if self.es_maximizar else "Minimizar"
        c_mostrar = self.c if self.es_maximizar else -self.c
        
        console.print(f"\n[bold blue]{objetivo}:[/bold blue]")
        funcion_obj = " + ".join([f"{c_mostrar[i]:.3f}*x{i+1}" for i in range(len(c_mostrar))])
        console.print(f"  z = {funcion_obj}")
        
        # Restricciones
        console.print(f"\n[bold blue]Sujeto a:[/bold blue]")
        for i in range(self.A.shape[0]):
            restriccion = " + ".join([f"{self.A[i,j]:.3f}*x{j+1}" for j in range(self.A.shape[1])])
            console.print(f"  {restriccion} ≤ {self.b[i]:.3f}")
        
        # No negatividad
        vars_str = ", ".join([f"x{i+1}" for i in range(len(self.c))])
        console.print(f"  {vars_str} ≥ 0")
        
        input("\nPresione Enter para continuar...")
    
    def crear_tableau_inicial(self):
        """Crea el tableau inicial del método simplex"""
        try:
            m, n = self.A.shape
            
            # Crear tableau ampliado
            # [A | I | b]
            # [c | 0 | 0]
            tableau_size = (m + 1, n + m + 1)
            self.tableau = np.zeros(tableau_size)
            
            # Matriz de restricciones con variables de holgura
            self.tableau[:m, :n] = self.A
            self.tableau[:m, n:n+m] = np.eye(m)
            self.tableau[:m, -1] = self.b
            
            # Función objetivo (cambiar signo para maximización)
            self.tableau[-1, :n] = -self.c
            
            # Variables básicas iniciales (variables de holgura)
            self.variables_basicas = list(range(n, n + m))
            
            return True
            
        except Exception as e:
            console.print(f"[red]Error al crear tableau inicial: {str(e)}[/red]")
            return False
    
    def es_optimo(self) -> bool:
        """Verifica si la solución actual es óptima"""
        # Solución óptima si todos los coeficientes de la función objetivo son ≥ 0
        return np.all(self.tableau[-1, :-1] >= -1e-10)
    
    def encontrar_columna_pivote(self) -> int:
        """Encuentra la columna pivote (más negativa en fila objetivo)"""
        fila_objetivo = self.tableau[-1, :-1]
        return np.argmin(fila_objetivo)
    
    def encontrar_fila_pivote(self, col_pivote: int) -> Optional[int]:
        """Encuentra la fila pivote usando la regla del cociente mínimo"""
        columna = self.tableau[:-1, col_pivote]
        rhs = self.tableau[:-1, -1]
        
        # Calcular cocientes solo para elementos positivos
        cocientes = []
        filas_validas = []
        
        for i in range(len(columna)):
            if columna[i] > 1e-10:  # Evitar división por cero y números muy pequeños
                cocientes.append(rhs[i] / columna[i])
                filas_validas.append(i)
        
        if not cocientes:
            return None  # Problema no acotado
        
        # Encontrar el cociente mínimo
        min_idx = np.argmin(cocientes)
        return filas_validas[min_idx]
    
    def pivotear(self, fila_pivote: int, col_pivote: int):
        """Realiza la operación de pivoteo"""
        # Normalizar fila pivote
        pivote = self.tableau[fila_pivote, col_pivote]
        self.tableau[fila_pivote, :] /= pivote
        
        # Eliminar en otras filas
        for i in range(self.tableau.shape[0]):
            if i != fila_pivote:
                factor = self.tableau[i, col_pivote]
                self.tableau[i, :] -= factor * self.tableau[fila_pivote, :]
        
        # Actualizar variable básica
        self.variables_basicas[fila_pivote] = col_pivote
    
    def resolver_simplex(self):
        """Ejecuta el algoritmo simplex"""
        try:
            if not self.crear_tableau_inicial():
                return False
            
            limpiar_pantalla()
            mostrar_titulo("EJECUCIÓN DEL MÉTODO SIMPLEX")
            
            iteracion = 0
            max_iteraciones = 100
            
            # Guardar tableau inicial
            self.historial = [self.tableau.copy()]
            
            console.print(f"[bold blue]Tableau inicial:[/bold blue]")
            self.mostrar_tableau(self.tableau)
            
            with console.status("[bold green]Ejecutando algoritmo simplex...") as status:
                while not self.es_optimo() and iteracion < max_iteraciones:
                    iteracion += 1
                    
                    # Encontrar columna pivote
                    col_pivote = self.encontrar_columna_pivote()
                    
                    # Encontrar fila pivote
                    fila_pivote = self.encontrar_fila_pivote(col_pivote)
                    
                    if fila_pivote is None:
                        console.print("\n[red]Problema no acotado - no existe solución finita.[/red]")
                        return False
                    
                    # Mostrar información de la iteración
                    console.print(f"\n[yellow]Iteración {iteracion}:[/yellow]")
                    console.print(f"Columna pivote: {col_pivote + 1}")
                    console.print(f"Fila pivote: {fila_pivote + 1}")
                    console.print(f"Elemento pivote: {self.tableau[fila_pivote, col_pivote]:.6f}")
                    
                    # Realizar pivoteo
                    self.pivotear(fila_pivote, col_pivote)
                    
                    # Guardar en historial
                    self.historial.append(self.tableau.copy())
                    
                    # Mostrar tableau actualizado
                    console.print(f"\n[bold blue]Tableau después de la iteración {iteracion}:[/bold blue]")
                    self.mostrar_tableau(self.tableau)
                    
                    time.sleep(1)  # Pausa para visualización
            
            if iteracion >= max_iteraciones:
                console.print(f"\n[red]Se alcanzó el máximo de iteraciones ({max_iteraciones}).[/red]")
                return False
            
            # Extraer solución
            self.extraer_solucion()
            
            console.print(f"\n[bold green]¡Solución óptima encontrada en {iteracion} iteraciones![/bold green]")
            return True
            
        except Exception as e:
            console.print(f"[red]Error durante la ejecución del simplex: {str(e)}[/red]")
            return False
    
    def mostrar_tableau(self, tableau):
        """Muestra el tableau de forma formateada"""
        table = Table(show_header=True, header_style="bold blue")
        
        # Headers
        n_vars = self.A.shape[1]
        headers = [f"x{i+1}" for i in range(n_vars)]
        headers.extend([f"s{i+1}" for i in range(self.A.shape[0])])
        headers.append("RHS")
        
        for header in headers:
            table.add_column(header, justify="center", style="cyan")
        
        # Filas de restricciones
        for i in range(tableau.shape[0] - 1):
            row = [f"{tableau[i, j]:.6f}" for j in range(tableau.shape[1])]
            table.add_row(*row)
        
        # Fila objetivo
        obj_row = [f"{tableau[-1, j]:.6f}" for j in range(tableau.shape[1])]
        table.add_row(*obj_row, style="bold yellow")
        
        console.print(table)
    
    def extraer_solucion(self):
        """Extrae la solución del tableau final"""
        n_vars = self.A.shape[1]
        self.solucion = np.zeros(n_vars)
        
        # Encontrar variables básicas
        for i, var_basica in enumerate(self.variables_basicas):
            if var_basica < n_vars:  # Es una variable de decisión
                self.solucion[var_basica] = self.tableau[i, -1]
        
        # Valor óptimo
        self.valor_optimo = self.tableau[-1, -1]
        if not self.es_maximizar:
            self.valor_optimo = -self.valor_optimo
    
    def mostrar_resultados(self):
        """Muestra los resultados de la optimización"""
        if self.solucion is None:
            console.print("[red]No hay solución disponible.[/red]")
            return
        
        limpiar_pantalla()
        mostrar_titulo("RESULTADOS DE LA OPTIMIZACIÓN")
        
        # Tabla de resultados
        table = Table(title="Solución Óptima", show_header=True, header_style="bold blue")
        table.add_column("Variable", style="cyan")
        table.add_column("Valor", justify="right", style="green")
        
        for i, valor in enumerate(self.solucion):
            table.add_row(f"x{i+1}", f"{valor:.6f}")
        
        console.print(table)
        
        # Valor óptimo
        tipo = "Máximo" if self.es_maximizar else "Mínimo"
        panel = Panel(
            f"[bold green]{tipo}: {self.valor_optimo:.6f}[/bold green]",
            title="Valor Óptimo",
            border_style="green"
        )
        console.print(panel)
        
        # Verificar restricciones
        console.print(f"\n[bold blue]Verificación de restricciones:[/bold blue]")
        verificacion_table = Table(show_header=True, header_style="bold blue")
        verificacion_table.add_column("Restricción", style="cyan")
        verificacion_table.add_column("Valor", justify="right")
        verificacion_table.add_column("Límite", justify="right")
        verificacion_table.add_column("Estado", justify="center")
        
        for i in range(self.A.shape[0]):
            valor = np.dot(self.A[i], self.solucion)
            limite = self.b[i]
            estado = "✓" if valor <= limite + 1e-6 else "✗"
            color = "green" if estado == "✓" else "red"
            
            verificacion_table.add_row(
                f"Restricción {i+1}",
                f"{valor:.6f}",
                f"{limite:.6f}",
                f"[{color}]{estado}[/{color}]"
            )
        
        console.print(verificacion_table)
    
    def graficar_region_factible(self):
        """Gráfica la región factible (solo para 2 variables)"""
        if self.A.shape[1] != 2:
            console.print("[yellow]La gráfica de región factible solo está disponible para 2 variables.[/yellow]")
            return
        
        try:
            plt.figure(figsize=(10, 8))
            
            # Crear grid de puntos
            x_max = max(self.b) * 1.5
            x = np.linspace(0, x_max, 400)
            y = np.linspace(0, x_max, 400)
            X, Y = np.meshgrid(x, y)
            
            # Evaluar restricciones
            factible = np.ones_like(X, dtype=bool)
            factible &= (X >= 0) & (Y >= 0)  # No negatividad
            
            # Aplicar restricciones
            for i in range(self.A.shape[0]):
                restriccion = self.A[i, 0] * X + self.A[i, 1] * Y <= self.b[i]
                factible &= restriccion
                
                # Dibujar línea de restricción
                if self.A[i, 1] != 0:
                    y_line = (self.b[i] - self.A[i, 0] * x) / self.A[i, 1]
                    mask = (y_line >= 0) & (y_line <= x_max)
                    plt.plot(x[mask], y_line[mask], 'b-', alpha=0.7, label=f'Restricción {i+1}')
            
            # Región factible
            plt.contourf(X, Y, factible.astype(int), levels=[0.5, 1.5], colors=['lightblue'], alpha=0.5)
            
            # Función objetivo
            if self.solucion is not None:
                c_mostrar = self.c if self.es_maximizar else -self.c
                if c_mostrar[1] != 0:
                    # Líneas de nivel de la función objetivo
                    for k in [0.5, 1.0, 1.5]:
                        z_val = k * self.valor_optimo
                        y_obj = (z_val - c_mostrar[0] * x) / c_mostrar[1]
                        mask = (y_obj >= 0) & (y_obj <= x_max)
                        alpha = 0.3 + 0.4 * k
                        plt.plot(x[mask], y_obj[mask], 'r--', alpha=alpha)
                
                # Punto óptimo
                plt.plot(self.solucion[0], self.solucion[1], 'ro', markersize=10, 
                        label=f'Óptimo: ({self.solucion[0]:.3f}, {self.solucion[1]:.3f})')
            
            plt.xlim(0, x_max)
            plt.ylim(0, x_max)
            plt.xlabel('x₁')
            plt.ylabel('x₂')
            plt.title('Región Factible y Solución Óptima')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.show()
            
        except Exception as e:
            console.print(f"[red]Error al generar gráfica: {str(e)}[/red]")

def mostrar_menu_principal():
    """Muestra el menú principal"""
    limpiar_pantalla()
    
    panel = Panel(
        "[bold blue]MÉTODO SIMPLEX[/bold blue]\n\n"
        "Optimización lineal para problemas de programación lineal\n"
        "con restricciones de desigualdad (≤) y variables no negativas",
        title="🔍 Optimización Lineal",
        border_style="blue"
    )
    console.print(panel)
    
    console.print("\n[bold green]Opciones disponibles:[/bold green]")
    console.print("1. 📝 Configurar nuevo problema")
    console.print("2. 👁️  Ver problema actual")
    console.print("3. ▶️  Resolver con Simplex")
    console.print("4. 📊 Ver resultados")
    console.print("5. 📈 Gráfica (2 variables)")
    console.print("6. 💾 Exportar resultados")
    console.print("7. ❓ Ayuda y ejemplos")
    console.print("8. 🚪 Salir")

def mostrar_ayuda():
    """Muestra información de ayuda"""
    limpiar_pantalla()
    mostrar_titulo("AYUDA - MÉTODO SIMPLEX")
    
    ayuda_text = """
[bold blue]¿Qué es el Método Simplex?[/bold blue]
El método Simplex es un algoritmo para resolver problemas de programación lineal,
encontrando la solución óptima moviéndose por los vértices de la región factible.

[bold blue]Formato del problema:[/bold blue]
Maximizar/Minimizar: c₁x₁ + c₂x₂ + ... + cₙxₙ
Sujeto a:
  a₁₁x₁ + a₁₂x₂ + ... + a₁ₙxₙ ≤ b₁
  a₂₁x₁ + a₂₂x₂ + ... + a₂ₙxₙ ≤ b₂
  ...
  x₁, x₂, ..., xₙ ≥ 0

[bold blue]Ejemplo:[/bold blue]
Maximizar: 3x₁ + 2x₂
Sujeto a:
  x₁ + x₂ ≤ 4
  2x₁ + x₂ ≤ 6
  x₁, x₂ ≥ 0

[bold blue]Pasos para usar el programa:[/bold blue]
1. Configurar el problema (función objetivo y restricciones)
2. Resolver usando el algoritmo Simplex
3. Ver resultados y gráficas
4. Exportar si es necesario

[bold yellow]Nota:[/bold yellow] Todas las restricciones deben ser de tipo ≤ 
y las variables deben ser no negativas.
"""
    
    panel = Panel(ayuda_text, title="Información", border_style="green")
    console.print(panel)
    input("\nPresione Enter para continuar...")

def ejemplo_predefinido():
    """Carga un ejemplo predefinido para demostración"""
    solver = SimplexSolver()
    
    console.print("[bold blue]Cargando ejemplo predefinido...[/bold blue]")
    console.print("Problema: Maximizar 3x₁ + 2x₂")
    console.print("Sujeto a: x₁ + x₂ ≤ 4, 2x₁ + x₂ ≤ 6, x₁,x₂ ≥ 0")
    
    # Configurar ejemplo
    solver.es_maximizar = True
    solver.c = np.array([3.0, 2.0])
    solver.A = np.array([[1.0, 1.0], [2.0, 1.0]])
    solver.b = np.array([4.0, 6.0])
    
    return solver

def main():
    """Función principal"""
    solver = SimplexSolver()
    
    while True:
        mostrar_menu_principal()
        
        try:
            opcion = input("\nSeleccione una opción (1-8): ").strip()
            
            if opcion == "1":
                if solver.configurar_problema():
                    console.print("[green]✓ Problema configurado correctamente.[/green]")
                    time.sleep(1)
                
            elif opcion == "2":
                solver.mostrar_problema()
                
            elif opcion == "3":
                if solver.c is None:
                    console.print("[red]Primero debe configurar un problema (opción 1).[/red]")
                    input("\nPresione Enter para continuar...")
                else:
                    if solver.resolver_simplex():
                        console.print("[green]✓ Optimización completada.[/green]")
                    input("\nPresione Enter para continuar...")
                
            elif opcion == "4":
                solver.mostrar_resultados()
                input("\nPresione Enter para continuar...")
                
            elif opcion == "5":
                if solver.solucion is not None:
                    solver.graficar_region_factible()
                else:
                    console.print("[red]Primero debe resolver el problema (opción 3).[/red]")
                input("\nPresione Enter para continuar...")
                
            elif opcion == "6":
                if solver.solucion is not None:
                    exportar_resultados_simplex(solver)
                else:
                    console.print("[red]Primero debe resolver el problema (opción 3).[/red]")
                input("\nPresione Enter para continuar...")
                
            elif opcion == "7":
                mostrar_ayuda()
                
                # Opción de cargar ejemplo
                if preguntar_si_no("\n¿Desea cargar un ejemplo predefinido?"):
                    solver = ejemplo_predefinido()
                    console.print("[green]✓ Ejemplo cargado. Use opción 2 para verlo.[/green]")
                    time.sleep(2)
                
            elif opcion == "8":
                if confirmar_salida():
                    console.print("\n[bold blue]¡Gracias por usar el Método Simplex![/bold blue]")
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

def exportar_resultados_simplex(solver):
    """Exporta los resultados a archivo"""
    try:
        timestamp = obtener_timestamp()
        filename = f"simplex_resultados_{timestamp}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DEL MÉTODO SIMPLEX\n")
            f.write("=" * 50 + "\n\n")
            
            # Problema
            f.write("PROBLEMA CONFIGURADO:\n")
            objetivo = "Maximizar" if solver.es_maximizar else "Minimizar"
            c_mostrar = solver.c if solver.es_maximizar else -solver.c
            f.write(f"{objetivo}: ")
            f.write(" + ".join([f"{c_mostrar[i]:.6f}*x{i+1}" for i in range(len(c_mostrar))]) + "\n\n")
            
            f.write("Sujeto a:\n")
            for i in range(solver.A.shape[0]):
                restriccion = " + ".join([f"{solver.A[i,j]:.6f}*x{j+1}" for j in range(solver.A.shape[1])])
                f.write(f"  {restriccion} <= {solver.b[i]:.6f}\n")
            
            vars_str = ", ".join([f"x{i+1}" for i in range(len(solver.c))])
            f.write(f"  {vars_str} >= 0\n\n")
            
            # Solución
            f.write("SOLUCIÓN ÓPTIMA:\n")
            for i, valor in enumerate(solver.solucion):
                f.write(f"x{i+1} = {valor:.6f}\n")
            
            tipo = "Máximo" if solver.es_maximizar else "Mínimo"
            f.write(f"\n{tipo}: {solver.valor_optimo:.6f}\n\n")
            
            # Verificación
            f.write("VERIFICACIÓN DE RESTRICCIONES:\n")
            for i in range(solver.A.shape[0]):
                valor = np.dot(solver.A[i], solver.solucion)
                limite = solver.b[i]
                estado = "CUMPLE" if valor <= limite + 1e-6 else "VIOLA"
                f.write(f"Restricción {i+1}: {valor:.6f} <= {limite:.6f} [{estado}]\n")
        
        console.print(f"[green]✓ Resultados exportados a: {filename}[/green]")
        
    except Exception as e:
        console.print(f"[red]Error al exportar: {str(e)}[/red]")

if __name__ == "__main__":
    main()
