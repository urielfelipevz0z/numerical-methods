#!/usr/bin/env python3
"""
Newton-Raphson para Sistemas - Implementación con menús interactivos
Método de Newton para resolver sistemas de ecuaciones no lineales
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import sympy as sp
import os
from typing import List, Tuple, Optional, Callable
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_flotante, validar_entero, validar_polinomio,
    crear_menu, limpiar_pantalla, mostrar_progreso,
    formatear_numero, formatear_tabla_resultados,
    graficar_sistema_convergencia
)

console = Console()

class NewtonRaphsonSistemas:
    def __init__(self):
        self.ecuaciones = []
        self.variables = []
        self.valores_iniciales = []
        self.tolerancia = 1e-10
        self.max_iteraciones = 100
        self.solucion = []
        self.iteraciones_realizadas = 0
        self.historial = []
        self.errores = []
        self.jacobiana_simbolica = None
        self.usar_diferencias_finitas = False

    def mostrar_menu_principal(self):
        """Muestra el menú principal del método"""
        opciones = [
            "Ingresar sistema de ecuaciones",
            "Configurar valores iniciales",
            "Configurar parámetros de convergencia",
            "Seleccionar método de derivación",
            "Ejecutar método de Newton",
            "Ver resultados",
            "Mostrar convergencia",
            "Analizar jacobiana",
            "Ver ayuda",
            "Salir"
        ]
        return crear_menu("NEWTON-RAPHSON PARA SISTEMAS", opciones)

    def ingresar_sistema(self):
        """Menú para ingreso del sistema de ecuaciones"""
        limpiar_pantalla()
        console.print(Panel.fit("INGRESO DEL SISTEMA", style="bold green"))
        
        console.print("\n[bold]Configuración del sistema:[/bold]")
        
        # Determinar número de variables
        num_vars = validar_entero("Número de variables (2-4): ", 2, 4)
        
        if num_vars == 2:
            self.variables = ['x', 'y']
            console.print("\n[yellow]Sistema de 2 ecuaciones con 2 incógnitas (x, y)[/yellow]")
        elif num_vars == 3:
            self.variables = ['x', 'y', 'z']
            console.print("\n[yellow]Sistema de 3 ecuaciones con 3 incógnitas (x, y, z)[/yellow]")
        else:
            self.variables = ['x', 'y', 'z', 'w']
            console.print("\n[yellow]Sistema de 4 ecuaciones con 4 incógnitas (x, y, z, w)[/yellow]")
        
        # Ingresar ecuaciones
        self.ecuaciones = []
        console.print(f"\n[bold]Ingrese las {num_vars} ecuaciones (f₁ = 0, f₂ = 0, ...):[/bold]")
        console.print("[cyan]Ejemplos:[/cyan]")
        console.print("  x**2 + y**2 - 4")
        console.print("  x - y + 1")
        console.print("  sin(x) + cos(y) - 1")
        console.print("  exp(x) + log(y) - 2")
        
        for i in range(num_vars):
            while True:
                try:
                    ecuacion_str = input(f"Ecuación f₍{i+1}₎ = 0: ").strip()
                    
                    # Crear símbolos
                    simbolos = [sp.Symbol(var) for var in self.variables]
                    
                    # Parsear la ecuación
                    ecuacion = sp.sympify(ecuacion_str)
                    
                    # Verificar que use las variables correctas
                    vars_ecuacion = ecuacion.free_symbols
                    vars_permitidas = set(simbolos)
                    
                    if not vars_ecuacion.issubset(vars_permitidas):
                        console.print(f"[red]Error: Use solo las variables {', '.join(self.variables)}[/red]")
                        continue
                    
                    self.ecuaciones.append(ecuacion)
                    console.print(f"[green]✓ f₍{i+1}₎({','.join(self.variables)}) = {ecuacion}[/green]")
                    break
                    
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    continue
        
        # Calcular jacobiana simbólica
        self._calcular_jacobiana_simbolica()
        
        console.print(f"\n[bold cyan]Sistema ingresado exitosamente:[/bold cyan]")
        for i, eq in enumerate(self.ecuaciones):
            console.print(f"f₍{i+1}₎({','.join(self.variables)}) = {eq} = 0")
        
        input("\nPresione Enter para continuar...")

    def _calcular_jacobiana_simbolica(self):
        """Calcula la matriz jacobiana simbólica"""
        try:
            simbolos = [sp.Symbol(var) for var in self.variables]
            self.jacobiana_simbolica = []
            
            for ecuacion in self.ecuaciones:
                fila = []
                for variable in simbolos:
                    derivada = sp.diff(ecuacion, variable)
                    fila.append(derivada)
                self.jacobiana_simbolica.append(fila)
            
            console.print("[green]✓ Jacobiana simbólica calculada[/green]")
            
        except Exception as e:
            console.print(f"[yellow]⚠ Error calculando jacobiana: {e}[/yellow]")
            self.jacobiana_simbolica = None

    def configurar_valores_iniciales(self):
        """Configura los valores iniciales"""
        if not self.ecuaciones:
            console.print("[red]Primero debe ingresar el sistema de ecuaciones[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("VALORES INICIALES", style="bold magenta"))
        
        console.print(f"\n[bold]Ingrese los valores iniciales:[/bold]")
        console.print("[yellow]Importante: Los valores iniciales deben estar cerca de la solución[/yellow]")
        
        self.valores_iniciales = []
        
        for i, var in enumerate(self.variables):
            valor = validar_flotante(f"Valor inicial para {var}₀: ", puede_ser_cero=True)
            self.valores_iniciales.append(valor)
        
        # Mostrar resumen
        console.print(f"\n[bold cyan]Valores iniciales:[/bold cyan]")
        for i, (var, val) in enumerate(zip(self.variables, self.valores_iniciales)):
            console.print(f"{var}₀ = {formatear_numero(val)}")
        
        # Evaluar funciones en valores iniciales
        console.print(f"\n[bold]Evaluación inicial:[/bold]")
        simbolos = [sp.Symbol(var) for var in self.variables]
        
        for i, ecuacion in enumerate(self.ecuaciones):
            try:
                valor = float(ecuacion.subs(dict(zip(simbolos, self.valores_iniciales))))
                console.print(f"f₍{i+1}₎({', '.join([formatear_numero(v) for v in self.valores_iniciales])}) = {formatear_numero(valor)}")
            except Exception as e:
                console.print(f"f₍{i+1}₎: Error en evaluación - {e}")
        
        input("\nPresione Enter para continuar...")

    def configurar_parametros(self):
        """Configura parámetros de convergencia"""
        limpiar_pantalla()
        console.print(Panel.fit("PARÁMETROS DE CONVERGENCIA", style="bold cyan"))
        
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

    def seleccionar_metodo_derivacion(self):
        """Selecciona el método para calcular derivadas"""
        limpiar_pantalla()
        console.print(Panel.fit("MÉTODO DE DERIVACIÓN", style="bold blue"))
        
        console.print("\n[bold]Métodos disponibles:[/bold]")
        console.print("1. Derivación analítica (simbólica)")
        console.print("2. Diferencias finitas numéricas")
        
        metodo_actual = "Diferencias finitas" if self.usar_diferencias_finitas else "Analítica"
        console.print(f"\n[bold]Método actual:[/bold] {metodo_actual}")
        
        if self.jacobiana_simbolica is None:
            console.print("[yellow]⚠ Jacobiana simbólica no disponible, usando diferencias finitas[/yellow]")
            self.usar_diferencias_finitas = True
            input("Presione Enter para continuar...")
            return
        
        if input("\n¿Cambiar método? (s/n): ").lower() == 's':
            opcion = validar_entero("Seleccione método (1-2): ", 1, 2)
            self.usar_diferencias_finitas = (opcion == 2)
            
            metodo_nuevo = "Diferencias finitas" if self.usar_diferencias_finitas else "Analítica"
            console.print(f"[green]Método cambiado a: {metodo_nuevo}[/green]")
            input("Presione Enter para continuar...")

    def ejecutar_newton(self):
        """Ejecuta el método de Newton-Raphson para sistemas"""
        if not self.ecuaciones or not self.valores_iniciales:
            console.print("[red]Debe completar la configuración del sistema[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("EJECUTANDO NEWTON-RAPHSON", style="bold yellow"))
        
        # Mostrar configuración
        console.print(f"\n[bold]Sistema de ecuaciones:[/bold]")
        for i, eq in enumerate(self.ecuaciones):
            console.print(f"f₍{i+1}₎ = {eq} = 0")
        
        console.print(f"\n[bold]Valores iniciales:[/bold] {[formatear_numero(v) for v in self.valores_iniciales]}")
        console.print(f"[bold]Tolerancia:[/bold] {self.tolerancia}")
        metodo = "Diferencias finitas" if self.usar_diferencias_finitas else "Analítica"
        console.print(f"[bold]Método jacobiana:[/bold] {metodo}")
        
        try:
            # Convertir funciones a callables
            simbolos = [sp.Symbol(var) for var in self.variables]
            f_funcs = []
            for ecuacion in self.ecuaciones:
                f_funcs.append(sp.lambdify(simbolos, ecuacion, 'numpy'))
            
            # Preparar jacobiana
            if not self.usar_diferencias_finitas and self.jacobiana_simbolica:
                j_funcs = []
                for fila in self.jacobiana_simbolica:
                    fila_funcs = []
                    for derivada in fila:
                        fila_funcs.append(sp.lambdify(simbolos, derivada, 'numpy'))
                    j_funcs.append(fila_funcs)
            
            # Inicializar
            self.historial = []
            self.errores = []
            x_actual = np.array(self.valores_iniciales, dtype=float)
            
            console.print(f"\n[cyan]Iniciando iteraciones...[/cyan]")
            
            for iteracion in track(range(self.max_iteraciones), description="Iterando..."):
                # Guardar estado actual
                self.historial.append(x_actual.copy())
                
                try:
                    # Evaluar funciones F(x)
                    F = np.array([float(f(*x_actual)) for f in f_funcs])
                    
                    # Calcular norma del vector F (criterio de parada)
                    norma_F = np.linalg.norm(F)
                    self.errores.append(norma_F)
                    
                    # Verificar convergencia
                    if norma_F < self.tolerancia:
                        self.solucion = x_actual
                        self.iteraciones_realizadas = iteracion + 1
                        console.print(f"\n[bold green]✓ Convergió en {self.iteraciones_realizadas} iteraciones[/bold green]")
                        console.print(f"[bold]||F(x)||:[/bold] {formatear_numero(norma_F)}")
                        break
                    
                    # Calcular jacobiana
                    if self.usar_diferencias_finitas or not self.jacobiana_simbolica:
                        J = self._calcular_jacobiana_numerica(f_funcs, x_actual)
                    else:
                        J = np.array([[float(j_func(*x_actual)) for j_func in fila] for fila in j_funcs])
                    
                    # Verificar singularidad
                    det_J = np.linalg.det(J)
                    if abs(det_J) < 1e-15:
                        console.print(f"\n[bold red]⚠ Jacobiana singular en iteración {iteracion}[/bold red]")
                        console.print(f"Determinante: {formatear_numero(det_J)}")
                        break
                    
                    # Resolver sistema lineal J * Δx = -F
                    try:
                        delta_x = np.linalg.solve(J, -F)
                    except np.linalg.LinAlgError:
                        console.print(f"\n[bold red]⚠ Error resolviendo sistema lineal en iteración {iteracion}[/bold red]")
                        break
                    
                    # Actualizar solución
                    x_nuevo = x_actual + delta_x
                    
                    # Verificar divergencia
                    if np.any(np.abs(x_nuevo) > 1e10):
                        console.print(f"\n[bold red]⚠ Divergencia detectada en iteración {iteracion}[/bold red]")
                        break
                    
                    x_actual = x_nuevo
                    
                except Exception as e:
                    console.print(f"\n[red]Error en iteración {iteracion}: {e}[/red]")
                    break
            
            else:
                console.print(f"\n[bold yellow]⚠ No convergió en {self.max_iteraciones} iteraciones[/bold yellow]")
                self.solucion = x_actual
                self.iteraciones_realizadas = self.max_iteraciones
            
            # Verificar solución final
            if len(self.solucion) > 0:
                self._verificar_solucion()
                
        except Exception as e:
            console.print(f"[red]Error durante la ejecución: {e}[/red]")
        
        input("Presione Enter para continuar...")

    def _calcular_jacobiana_numerica(self, f_funcs: List[Callable], x: np.ndarray, h: float = 1e-8) -> np.ndarray:
        """Calcula la jacobiana usando diferencias finitas"""
        n = len(x)
        J = np.zeros((n, n))
        
        for i, f_func in enumerate(f_funcs):
            f_x = float(f_func(*x))
            
            for j in range(n):
                x_pert = x.copy()
                x_pert[j] += h
                f_x_pert = float(f_func(*x_pert))
                
                J[i, j] = (f_x_pert - f_x) / h
        
        return J

    def _verificar_solucion(self):
        """Verifica la calidad de la solución encontrada"""
        console.print(f"\n[bold]Verificación de la solución:[/bold]")
        
        simbolos = [sp.Symbol(var) for var in self.variables]
        
        normas_f = []
        for i, ecuacion in enumerate(self.ecuaciones):
            try:
                f_evaluada = float(ecuacion.subs(dict(zip(simbolos, self.solucion))))
                normas_f.append(abs(f_evaluada))
                console.print(f"f₍{i+1}₎({', '.join([formatear_numero(x) for x in self.solucion])}) = {formatear_numero(f_evaluada)}")
            except Exception as e:
                console.print(f"f₍{i+1}₎: Error en evaluación - {e}")
        
        if normas_f:
            norma_total = np.linalg.norm(normas_f)
            console.print(f"[bold]||F(x*)||:[/bold] {formatear_numero(norma_total)}")

    def mostrar_resultados(self):
        """Muestra los resultados del método"""
        if not self.solucion:
            console.print("[red]No hay resultados para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("RESULTADOS", style="bold green"))
        
        # Mostrar solución
        tabla = Table(title="Solución del Sistema")
        tabla.add_column("Variable", style="cyan")
        tabla.add_column("Valor", style="yellow")
        
        for var, val in zip(self.variables, self.solucion):
            tabla.add_row(var, formatear_numero(val))
        
        console.print(tabla)
        
        # Información de convergencia
        console.print(f"\n[bold]Información de convergencia:[/bold]")
        console.print(f"Iteraciones realizadas: {self.iteraciones_realizadas}")
        console.print(f"||F(x*)|| final: {formatear_numero(self.errores[-1]) if self.errores else 'N/A'}")
        console.print(f"Tolerancia: {self.tolerancia}")
        metodo = "Diferencias finitas" if self.usar_diferencias_finitas else "Analítica"
        console.print(f"Método jacobiana: {metodo}")
        
        # Mostrar últimas iteraciones
        if len(self.historial) > 1:
            console.print(f"\n[bold]Últimas iteraciones:[/bold]")
            inicio = max(0, len(self.historial) - 5)
            
            tabla_iter = Table()
            tabla_iter.add_column("Iter", style="cyan")
            for var in self.variables:
                tabla_iter.add_column(var, style="yellow")
            tabla_iter.add_column("||F(x)||", style="red")
            
            for i in range(inicio, len(self.historial)):
                fila = [str(i)]
                for val in self.historial[i]:
                    fila.append(formatear_numero(val))
                if i < len(self.errores):
                    fila.append(formatear_numero(self.errores[i]))
                else:
                    fila.append("N/A")
                tabla_iter.add_row(*fila)
            
            console.print(tabla_iter)
        
        input("\nPresione Enter para continuar...")

    def mostrar_convergencia(self):
        """Muestra gráficos de convergencia"""
        if not self.historial or len(self.variables) > 2:
            console.print("[red]No hay datos de convergencia para graficar o sistema > 2D[/red]")
            input("Presione Enter para continuar...")
            return
        
        console.print("[cyan]Generando gráficos de convergencia...[/cyan]")
        
        # Preparar datos
        iteraciones = list(range(len(self.historial)))
        valores_x = [h[0] for h in self.historial]
        valores_y = [h[1] for h in self.historial]
        
        graficar_sistema_convergencia(iteraciones, valores_x, valores_y, self.errores)

    def analizar_jacobiana(self):
        """Analiza la matriz jacobiana en la solución"""
        if not self.solucion or not self.jacobiana_simbolica:
            console.print("[red]Primero debe encontrar una solución[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("ANÁLISIS DE LA JACOBIANA", style="bold purple"))
        
        try:
            simbolos = [sp.Symbol(var) for var in self.variables]
            
            # Calcular jacobiana en la solución
            J_solucion = np.zeros((len(self.variables), len(self.variables)))
            
            for i, fila in enumerate(self.jacobiana_simbolica):
                for j, derivada in enumerate(fila):
                    valor = float(derivada.subs(dict(zip(simbolos, self.solucion))))
                    J_solucion[i, j] = valor
            
            console.print(f"\n[bold]Jacobiana en la solución:[/bold]")
            console.print("J = ")
            for fila in J_solucion:
                console.print("  [" + ", ".join([f"{val:10.6f}" for val in fila]) + "]")
            
            # Propiedades de la matriz
            det_J = np.linalg.det(J_solucion)
            console.print(f"\n[bold]Propiedades:[/bold]")
            console.print(f"Determinante: {formatear_numero(det_J)}")
            
            if abs(det_J) > 1e-10:
                console.print("[green]✓ Matriz no singular[/green]")
                
                # Número de condición
                cond_J = np.linalg.cond(J_solucion)
                console.print(f"Número de condición: {formatear_numero(cond_J)}")
                
                if cond_J < 100:
                    console.print("[green]✓ Bien condicionada[/green]")
                elif cond_J < 1e6:
                    console.print("[yellow]⚠ Moderadamente condicionada[/yellow]")
                else:
                    console.print("[red]⚠ Mal condicionada[/red]")
                
                # Valores propios
                eigenvalues = np.linalg.eigvals(J_solucion)
                console.print(f"\n[bold]Valores propios:[/bold]")
                for i, eigenval in enumerate(eigenvalues):
                    console.print(f"λ₍{i+1}₎ = {formatear_numero(eigenval)}")
                
            else:
                console.print("[red]⚠ Matriz singular o casi singular[/red]")
                console.print("El método puede tener problemas de convergencia")
            
        except Exception as e:
            console.print(f"[red]Error en análisis: {e}[/red]")
        
        input("\nPresione Enter para continuar...")

    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        limpiar_pantalla()
        ayuda_texto = """
[bold blue]MÉTODO DE NEWTON-RAPHSON PARA SISTEMAS[/bold blue]

[bold]¿Qué es?[/bold]
Método iterativo para resolver sistemas de ecuaciones no lineales
F(x) = 0 usando linearización local y la matriz jacobiana.

[bold]Algoritmo:[/bold]
1. Partir de una estimación inicial x⁽⁰⁾
2. Calcular F(x⁽ᵏ⁾) y J(x⁽ᵏ⁾) (jacobiana)
3. Resolver J(x⁽ᵏ⁾) · Δx = -F(x⁽ᵏ⁾)
4. Actualizar: x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + Δx
5. Repetir hasta convergencia

[bold]Matriz Jacobiana:[/bold]
J[i,j] = ∂fᵢ/∂xⱼ

[bold]Condiciones de convergencia:[/bold]
• Estimación inicial cercana a la solución
• Jacobiana no singular en la solución
• Sistema bien condicionado

[bold]Ventajas:[/bold]
• Convergencia cuadrática (muy rápida)
• Robusto para sistemas bien condicionados
• Método estándar en software científico

[bold]Desventajas:[/bold]
• Requiere calcular jacobiana (costoso)
• Muy sensible a valores iniciales
• Puede fallar si jacobiana es singular
• No garantiza convergencia global

[bold]Métodos de derivación:[/bold]
• Analítica: Usa derivadas simbólicas exactas
• Diferencias finitas: Aproxima derivadas numéricamente

[bold]Consejos de uso:[/bold]
• Usar estimaciones iniciales de calidad
• Verificar que la jacobiana no sea singular
• Analizar el número de condición
• Probar diferentes valores iniciales si no converge

[bold]Aplicaciones:[/bold]
• Sistemas de ecuaciones no lineales
• Optimización no lineal
• Análisis de circuitos
• Problemas de ingeniería
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
                    self.ingresar_sistema()
                elif opcion == 2:
                    self.configurar_valores_iniciales()
                elif opcion == 3:
                    self.configurar_parametros()
                elif opcion == 4:
                    self.seleccionar_metodo_derivacion()
                elif opcion == 5:
                    self.ejecutar_newton()
                elif opcion == 6:
                    self.mostrar_resultados()
                elif opcion == 7:
                    self.mostrar_convergencia()
                elif opcion == 8:
                    self.analizar_jacobiana()
                elif opcion == 9:
                    self.mostrar_ayuda()
                elif opcion == 10:
                    console.print("[bold green]¡Hasta luego![/bold green]")
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[bold red]Programa interrumpido[/bold red]")
                break
            except Exception as e:
                console.print(f"[bold red]Error inesperado: {e}[/bold red]")
                input("Presione Enter para continuar...")

if __name__ == "__main__":
    newton = NewtonRaphsonSistemas()
    newton.main()
