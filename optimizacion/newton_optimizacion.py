#!/usr/bin/env python3
"""
Newton para Optimización - Implementación con menús interactivos
Método de Newton para optimización no restringida de funciones multivariables
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
    formatear_numero, formatear_tabla_resultados
)

console = Console()

class NewtonOptimizacion:
    def __init__(self):
        self.funcion = None
        self.variables = []
        self.num_variables = 1
        self.punto_inicial = []
        self.tolerancia = 1e-8
        self.max_iteraciones = 100
        self.alpha = 1.0  # Factor de amortiguamiento
        self.usar_busqueda_lineal = True
        self.solucion = []
        self.iteraciones_realizadas = 0
        self.historial = []
        self.valores_funcion = []
        self.gradientes = []
        self.hessiana_simbolica = None
        self.gradiente_simbolico = None

    def mostrar_menu_principal(self):
        """Muestra el menú principal del método"""
        opciones = [
            "Ingresar función objetivo",
            "Configurar punto inicial",
            "Configurar parámetros de optimización",
            "Ejecutar método de Newton",
            "Ver resultados",
            "Mostrar convergencia",
            "Analizar punto crítico",
            "Mostrar gráficos",
            "Ver ayuda",
            "Salir"
        ]
        return crear_menu("NEWTON PARA OPTIMIZACIÓN", opciones)

    def ingresar_funcion(self):
        """Menú para ingreso de la función objetivo"""
        limpiar_pantalla()
        console.print(Panel.fit("FUNCIÓN OBJETIVO", style="bold green"))
        
        # Seleccionar número de variables
        console.print("\n[bold]Configuración de la función:[/bold]")
        self.num_variables = validar_entero("Número de variables (1-3): ", 1, 3)
        
        if self.num_variables == 1:
            self.variables = ['x']
            console.print("\n[yellow]Función de una variable: f(x)[/yellow]")
        elif self.num_variables == 2:
            self.variables = ['x', 'y']
            console.print("\n[yellow]Función de dos variables: f(x, y)[/yellow]")
        else:
            self.variables = ['x', 'y', 'z']
            console.print("\n[yellow]Función de tres variables: f(x, y, z)[/yellow]")
        
        # Ingresar función
        console.print(f"\n[bold]Ingrese la función objetivo:[/bold]")
        console.print("[cyan]Ejemplos:[/cyan]")
        if self.num_variables == 1:
            console.print("  x**2 - 4*x + 3")
            console.print("  sin(x) + x**2")
        elif self.num_variables == 2:
            console.print("  x**2 + y**2 - 4*x - 6*y + 13")
            console.print("  x**2 + 2*y**2 + x*y - 8*x - 10*y")
        else:
            console.print("  x**2 + y**2 + z**2 - 2*x - 4*y - 6*z")
        
        while True:
            try:
                funcion_str = input("f(x) = ").strip()
                
                # Crear símbolos
                simbolos = [sp.Symbol(var) for var in self.variables]
                
                # Parsear la función
                self.funcion = sp.sympify(funcion_str)
                
                # Verificar variables
                vars_funcion = self.funcion.free_symbols
                vars_permitidas = set(simbolos)
                
                if not vars_funcion.issubset(vars_permitidas):
                    console.print(f"[red]Error: Use solo las variables {', '.join(self.variables)}[/red]")
                    continue
                
                # Calcular gradiente y hessiana
                self._calcular_derivadas()
                
                console.print(f"[green]✓ Función ingresada: f({','.join(self.variables)}) = {self.funcion}[/green]")
                break
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue
        
        input("\nPresione Enter para continuar...")

    def _calcular_derivadas(self):
        """Calcula gradiente y hessiana simbólicos"""
        try:
            simbolos = [sp.Symbol(var) for var in self.variables]
            
            # Calcular gradiente
            self.gradiente_simbolico = []
            for var in simbolos:
                derivada = sp.diff(self.funcion, var)
                self.gradiente_simbolico.append(derivada)
            
            console.print("[green]✓ Gradiente calculado[/green]")
            
            # Calcular hessiana
            self.hessiana_simbolica = []
            for i, var_i in enumerate(simbolos):
                fila = []
                for j, var_j in enumerate(simbolos):
                    derivada_segunda = sp.diff(self.gradiente_simbolico[i], var_j)
                    fila.append(derivada_segunda)
                self.hessiana_simbolica.append(fila)
            
            console.print("[green]✓ Hessiana calculada[/green]")
            
            # Mostrar derivadas
            console.print(f"\n[bold cyan]Gradiente:[/bold cyan]")
            for i, grad in enumerate(self.gradiente_simbolico):
                console.print(f"∂f/∂{self.variables[i]} = {grad}")
            
        except Exception as e:
            console.print(f"[red]Error calculando derivadas: {e}[/red]")
            self.gradiente_simbolico = None
            self.hessiana_simbolica = None

    def configurar_punto_inicial(self):
        """Configura el punto inicial"""
        if self.funcion is None:
            console.print("[red]Primero debe ingresar una función[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("PUNTO INICIAL", style="bold magenta"))
        
        console.print(f"\n[bold]Ingrese el punto inicial para optimización:[/bold]")
        
        self.punto_inicial = []
        for var in self.variables:
            valor = validar_flotante(f"Valor inicial para {var}₀: ", puede_ser_cero=True)
            self.punto_inicial.append(valor)
        
        # Evaluar función en punto inicial
        simbolos = [sp.Symbol(var) for var in self.variables]
        try:
            valor_inicial = float(self.funcion.subs(dict(zip(simbolos, self.punto_inicial))))
            
            console.print(f"\n[bold cyan]Punto inicial:[/bold cyan]")
            for var, val in zip(self.variables, self.punto_inicial):
                console.print(f"{var}₀ = {formatear_numero(val)}")
            
            console.print(f"[bold]f({', '.join([formatear_numero(v) for v in self.punto_inicial])}) = {formatear_numero(valor_inicial)}[/bold]")
            
            # Evaluar gradiente en punto inicial
            if self.gradiente_simbolico:
                console.print(f"\n[bold]Gradiente en punto inicial:[/bold]")
                for i, grad_expr in enumerate(self.gradiente_simbolico):
                    grad_val = float(grad_expr.subs(dict(zip(simbolos, self.punto_inicial))))
                    console.print(f"∂f/∂{self.variables[i]} = {formatear_numero(grad_val)}")
            
        except Exception as e:
            console.print(f"[red]Error evaluando función: {e}[/red]")
        
        input("\nPresione Enter para continuar...")

    def configurar_parametros(self):
        """Configura parámetros de optimización"""
        limpiar_pantalla()
        console.print(Panel.fit("PARÁMETROS DE OPTIMIZACIÓN", style="bold cyan"))
        
        console.print(f"\n[bold]Parámetros actuales:[/bold]")
        console.print(f"Tolerancia: {self.tolerancia}")
        console.print(f"Máximo iteraciones: {self.max_iteraciones}")
        console.print(f"Factor de amortiguamiento: {self.alpha}")
        console.print(f"Búsqueda lineal: {'Activada' if self.usar_busqueda_lineal else 'Desactivada'}")
        
        if input("\n¿Cambiar parámetros? (s/n): ").lower() == 's':
            self.tolerancia = validar_flotante(
                "Nueva tolerancia (1e-8): ", 1e-15, 1e-3, self.tolerancia
            )
            self.max_iteraciones = validar_entero(
                "Nuevas máximo iteraciones (100): ", 10, 1000, self.max_iteraciones
            )
            self.alpha = validar_flotante(
                "Factor de amortiguamiento (1.0): ", 0.1, 2.0, self.alpha
            )
            self.usar_busqueda_lineal = input("¿Usar búsqueda lineal? (s/n): ").lower() == 's'

    def ejecutar_newton(self):
        """Ejecuta el método de Newton para optimización"""
        if self.funcion is None or not self.punto_inicial:
            console.print("[red]Debe completar la configuración[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("EJECUTANDO NEWTON", style="bold yellow"))
        
        # Mostrar configuración
        console.print(f"\n[bold]Función:[/bold] f({','.join(self.variables)}) = {self.funcion}")
        console.print(f"[bold]Punto inicial:[/bold] {[formatear_numero(v) for v in self.punto_inicial]}")
        console.print(f"[bold]Tolerancia:[/bold] {self.tolerancia}")
        
        try:
            simbolos = [sp.Symbol(var) for var in self.variables]
            
            # Convertir a funciones evaluables
            f_func = sp.lambdify(simbolos, self.funcion, 'numpy')
            
            grad_funcs = []
            for grad_expr in self.gradiente_simbolico:
                grad_funcs.append(sp.lambdify(simbolos, grad_expr, 'numpy'))
            
            hess_funcs = []
            for fila in self.hessiana_simbolica:
                fila_funcs = []
                for hess_expr in fila:
                    fila_funcs.append(sp.lambdify(simbolos, hess_expr, 'numpy'))
                hess_funcs.append(fila_funcs)
            
            # Inicializar
            self.historial = []
            self.valores_funcion = []
            self.gradientes = []
            x_actual = np.array(self.punto_inicial, dtype=float)
            
            console.print(f"\n[cyan]Iniciando optimización...[/cyan]")
            
            for iteracion in track(range(self.max_iteraciones), description="Optimizando..."):
                # Guardar estado actual
                self.historial.append(x_actual.copy())
                
                try:
                    # Evaluar función
                    f_val = float(f_func(*x_actual))
                    self.valores_funcion.append(f_val)
                    
                    # Evaluar gradiente
                    gradiente = np.array([float(g_func(*x_actual)) for g_func in grad_funcs])
                    self.gradientes.append(gradiente.copy())
                    
                    # Verificar convergencia
                    norma_grad = np.linalg.norm(gradiente)
                    if norma_grad < self.tolerancia:
                        self.solucion = x_actual
                        self.iteraciones_realizadas = iteracion + 1
                        console.print(f"\n[bold green]✓ Convergió en {self.iteraciones_realizadas} iteraciones[/bold green]")
                        console.print(f"[bold]||∇f||:[/bold] {formatear_numero(norma_grad)}")
                        break
                    
                    # Evaluar hessiana
                    hessiana = np.array([[float(h_func(*x_actual)) for h_func in fila] 
                                       for fila in hess_funcs])
                    
                    # Verificar si la hessiana es definida positiva
                    eigenvalues = np.linalg.eigvals(hessiana)
                    
                    if np.all(eigenvalues > 1e-10):
                        # Hessiana definida positiva - usar Newton completo
                        try:
                            direccion = -np.linalg.solve(hessiana, gradiente)
                        except np.linalg.LinAlgError:
                            # Si falla, usar gradiente descendente
                            direccion = -gradiente
                            console.print(f"[yellow]⚠ Usando gradiente descendente en iteración {iteracion}[/yellow]")
                    else:
                        # Modificar hessiana o usar gradiente descendente
                        direccion = -gradiente
                        console.print(f"[yellow]⚠ Hessiana no definida positiva en iteración {iteracion}[/yellow]")
                    
                    # Búsqueda lineal o paso fijo
                    if self.usar_busqueda_lineal:
                        alpha_optimo = self._busqueda_lineal(f_func, x_actual, direccion)
                    else:
                        alpha_optimo = self.alpha
                    
                    # Actualizar punto
                    x_nuevo = x_actual + alpha_optimo * direccion
                    
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
            
            # Análisis final
            if len(self.solucion) > 0:
                self._analizar_solucion()
                
        except Exception as e:
            console.print(f"[red]Error durante la optimización: {e}[/red]")
        
        input("Presione Enter para continuar...")

    def _busqueda_lineal(self, f_func: Callable, x: np.ndarray, direccion: np.ndarray, 
                         alpha_inicial: float = 1.0) -> float:
        """Búsqueda lineal simple usando retroceso"""
        alpha = alpha_inicial
        beta = 0.5  # Factor de reducción
        c1 = 1e-4   # Parámetro de Armijo
        
        f_x = float(f_func(*x))
        grad_f_d = np.dot(self.gradientes[-1], direccion)
        
        for _ in range(20):  # Máximo 20 retrocesos
            x_nuevo = x + alpha * direccion
            try:
                f_nuevo = float(f_func(*x_nuevo))
                
                # Condición de Armijo
                if f_nuevo <= f_x + c1 * alpha * grad_f_d:
                    return alpha
                
                alpha *= beta
                
            except:
                alpha *= beta
        
        return alpha

    def _analizar_solucion(self):
        """Analiza la naturaleza del punto crítico encontrado"""
        console.print(f"\n[bold]Análisis del punto crítico:[/bold]")
        
        try:
            simbolos = [sp.Symbol(var) for var in self.variables]
            
            # Evaluar hessiana en la solución
            hessiana_final = np.array([[float(h_expr.subs(dict(zip(simbolos, self.solucion))))
                                      for h_expr in fila] for fila in self.hessiana_simbolica])
            
            # Valores propios de la hessiana
            eigenvalues = np.linalg.eigvals(hessiana_final)
            
            console.print(f"Valores propios de la hessiana: {[formatear_numero(ev) for ev in eigenvalues]}")
            
            if np.all(eigenvalues > 1e-10):
                console.print("[green]✓ Mínimo local (hessiana definida positiva)[/green]")
            elif np.all(eigenvalues < -1e-10):
                console.print("[red]✓ Máximo local (hessiana definida negativa)[/red]")
            else:
                console.print("[yellow]⚠ Punto silla (hessiana indefinida)[/yellow]")
            
        except Exception as e:
            console.print(f"[red]Error en análisis: {e}[/red]")

    def mostrar_resultados(self):
        """Muestra los resultados de la optimización"""
        if not self.solucion:
            console.print("[red]No hay resultados para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("RESULTADOS DE OPTIMIZACIÓN", style="bold green"))
        
        # Punto óptimo
        tabla = Table(title="Punto Óptimo Encontrado")
        tabla.add_column("Variable", style="cyan")
        tabla.add_column("Valor", style="yellow")
        
        for var, val in zip(self.variables, self.solucion):
            tabla.add_row(var, formatear_numero(val))
        
        console.print(tabla)
        
        # Valor de la función
        simbolos = [sp.Symbol(var) for var in self.variables]
        f_optimo = float(self.funcion.subs(dict(zip(simbolos, self.solucion))))
        
        console.print(f"\n[bold]Valor óptimo de la función:[/bold]")
        console.print(f"f({', '.join([formatear_numero(v) for v in self.solucion])}) = {formatear_numero(f_optimo)}")
        
        # Información de convergencia
        console.print(f"\n[bold]Información de convergencia:[/bold]")
        console.print(f"Iteraciones realizadas: {self.iteraciones_realizadas}")
        
        if self.gradientes:
            norma_grad_final = np.linalg.norm(self.gradientes[-1])
            console.print(f"||∇f|| final: {formatear_numero(norma_grad_final)}")
        
        console.print(f"Tolerancia: {self.tolerancia}")
        console.print(f"Búsqueda lineal: {'Activada' if self.usar_busqueda_lineal else 'Desactivada'}")
        
        # Gradiente final
        if self.gradientes:
            console.print(f"\n[bold]Gradiente en el punto óptimo:[/bold]")
            for i, grad_val in enumerate(self.gradientes[-1]):
                console.print(f"∂f/∂{self.variables[i]} = {formatear_numero(grad_val)}")
        
        input("\nPresione Enter para continuar...")

    def mostrar_convergencia(self):
        """Muestra gráficos de convergencia"""
        if not self.historial:
            console.print("[red]No hay datos de convergencia para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
        
        console.print("[cyan]Generando gráficos de convergencia...[/cyan]")
        
        try:
            if self.num_variables == 1:
                self._graficar_convergencia_1d()
            elif self.num_variables == 2:
                self._graficar_convergencia_2d()
            else:
                self._graficar_convergencia_nd()
                
        except Exception as e:
            console.print(f"[red]Error generando gráficos: {e}[/red]")

    def _graficar_convergencia_1d(self):
        """Gráficos para funciones de una variable"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Función y trayectoria
        x_vals = np.array(self.historial).flatten()
        x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
        x_plot = np.linspace(x_min, x_max, 1000)
        
        x_sym = sp.Symbol('x')
        f_lambdified = sp.lambdify(x_sym, self.funcion, 'numpy')
        y_plot = f_lambdified(x_plot)
        
        ax1.plot(x_plot, y_plot, 'b-', linewidth=2, label='f(x)')
        ax1.plot(x_vals, self.valores_funcion, 'ro-', markersize=6, alpha=0.7, label='Iteraciones')
        ax1.plot(self.solucion[0], self.valores_funcion[-1], 'go', markersize=10, label='Óptimo')
        ax1.set_title('Función y Convergencia')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Convergencia de x
        iteraciones = list(range(len(self.historial)))
        ax2.plot(iteraciones, x_vals, 'bo-', linewidth=2, markersize=4)
        ax2.axhline(y=self.solucion[0], color='red', linestyle='--', label='x*')
        ax2.set_title('Convergencia de x')
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('x')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Valor de la función
        ax3.plot(iteraciones, self.valores_funcion, 'go-', linewidth=2, markersize=4)
        ax3.set_title('Valor de la Función')
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('f(x)')
        ax3.grid(True, alpha=0.3)
        
        # Norma del gradiente
        if self.gradientes:
            normas_grad = [np.linalg.norm(g) for g in self.gradientes]
            ax4.semilogy(iteraciones, normas_grad, 'ro-', linewidth=2, markersize=4)
            ax4.axhline(y=self.tolerancia, color='green', linestyle='--', label='Tolerancia')
            ax4.set_title('Norma del Gradiente')
            ax4.set_xlabel('Iteración')
            ax4.set_ylabel('||∇f||')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _graficar_convergencia_2d(self):
        """Gráficos para funciones de dos variables"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Contornos y trayectoria
        x_vals = [h[0] for h in self.historial]
        y_vals = [h[1] for h in self.historial]
        
        x_min, x_max = min(x_vals) - 1, max(x_vals) + 1
        y_min, y_max = min(y_vals) - 1, max(y_vals) + 1
        
        x_grid = np.linspace(x_min, x_max, 100)
        y_grid = np.linspace(y_min, y_max, 100)
        X, Y = np.meshgrid(x_grid, y_grid)
        
        x_sym, y_sym = sp.symbols('x y')
        f_lambdified = sp.lambdify([x_sym, y_sym], self.funcion, 'numpy')
        Z = f_lambdified(X, Y)
        
        contour = ax1.contour(X, Y, Z, levels=20, alpha=0.6)
        ax1.clabel(contour, inline=True, fontsize=8)
        ax1.plot(x_vals, y_vals, 'ro-', linewidth=2, markersize=6, alpha=0.7, label='Trayectoria')
        ax1.plot(self.solucion[0], self.solucion[1], 'go', markersize=10, label='Óptimo')
        ax1.set_title('Contornos y Trayectoria de Optimización')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Convergencia de variables
        iteraciones = list(range(len(self.historial)))
        ax2.plot(iteraciones, x_vals, 'bo-', linewidth=2, markersize=4, label='x')
        ax2.plot(iteraciones, y_vals, 'ro-', linewidth=2, markersize=4, label='y')
        ax2.set_title('Convergencia de Variables')
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Valor')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Valor de la función
        ax3.plot(iteraciones, self.valores_funcion, 'go-', linewidth=2, markersize=4)
        ax3.set_title('Valor de la Función')
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('f(x,y)')
        ax3.grid(True, alpha=0.3)
        
        # Norma del gradiente
        if self.gradientes:
            normas_grad = [np.linalg.norm(g) for g in self.gradientes]
            ax4.semilogy(iteraciones, normas_grad, 'ro-', linewidth=2, markersize=4)
            ax4.axhline(y=self.tolerancia, color='green', linestyle='--', label='Tolerancia')
            ax4.set_title('Norma del Gradiente')
            ax4.set_xlabel('Iteración')
            ax4.set_ylabel('||∇f||')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def _graficar_convergencia_nd(self):
        """Gráficos para funciones de n variables"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        iteraciones = list(range(len(self.historial)))
        
        # Convergencia de todas las variables
        for i, var in enumerate(self.variables):
            vals = [h[i] for h in self.historial]
            ax1.plot(iteraciones, vals, 'o-', linewidth=2, markersize=4, label=f'{var}')
        
        ax1.set_title('Convergencia de Variables')
        ax1.set_xlabel('Iteración')
        ax1.set_ylabel('Valor')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Valor de la función
        ax2.plot(iteraciones, self.valores_funcion, 'go-', linewidth=2, markersize=4)
        ax2.set_title('Valor de la Función')
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('f(x)')
        ax2.grid(True, alpha=0.3)
        
        # Norma del gradiente
        if self.gradientes:
            normas_grad = [np.linalg.norm(g) for g in self.gradientes]
            ax3.semilogy(iteraciones, normas_grad, 'ro-', linewidth=2, markersize=4)
            ax3.axhline(y=self.tolerancia, color='green', linestyle='--', label='Tolerancia')
            ax3.set_title('Norma del Gradiente')
            ax3.set_xlabel('Iteración')
            ax3.set_ylabel('||∇f||')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Distancia al óptimo
        if len(self.solucion) > 0:
            distancias = [np.linalg.norm(np.array(h) - self.solucion) for h in self.historial]
            ax4.semilogy(iteraciones, distancias, 'mo-', linewidth=2, markersize=4)
            ax4.set_title('Distancia al Punto Óptimo')
            ax4.set_xlabel('Iteración')
            ax4.set_ylabel('||x - x*||')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def analizar_punto_critico(self):
        """Análisis detallado del punto crítico"""
        if not self.solucion:
            console.print("[red]Primero debe encontrar un punto crítico[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("ANÁLISIS DEL PUNTO CRÍTICO", style="bold purple"))
        
        try:
            simbolos = [sp.Symbol(var) for var in self.variables]
            
            # Evaluar hessiana en la solución
            console.print(f"\n[bold]Matriz Hessiana en el punto crítico:[/bold]")
            hessiana = np.array([[float(h_expr.subs(dict(zip(simbolos, self.solucion))))
                                for h_expr in fila] for fila in self.hessiana_simbolica])
            
            # Mostrar matriz
            for i, fila in enumerate(hessiana):
                fila_str = "  [" + ", ".join([f"{val:8.4f}" for val in fila]) + "]"
                console.print(fila_str)
            
            # Análisis de valores propios
            eigenvalues = np.linalg.eigvals(hessiana)
            console.print(f"\n[bold]Valores propios:[/bold]")
            for i, ev in enumerate(eigenvalues):
                console.print(f"λ₍{i+1}₎ = {formatear_numero(ev)}")
            
            # Clasificación del punto crítico
            console.print(f"\n[bold]Clasificación:[/bold]")
            if np.all(eigenvalues > 1e-10):
                console.print("[green]✓ MÍNIMO LOCAL[/green]")
                console.print("Todos los valores propios son positivos (hessiana definida positiva)")
            elif np.all(eigenvalues < -1e-10):
                console.print("[red]✓ MÁXIMO LOCAL[/red]")
                console.print("Todos los valores propios son negativos (hessiana definida negativa)")
            elif np.any(eigenvalues > 1e-10) and np.any(eigenvalues < -1e-10):
                console.print("[yellow]⚠ PUNTO SILLA[/yellow]")
                console.print("Hay valores propios positivos y negativos (hessiana indefinida)")
            else:
                console.print("[orange]? CASO DEGENERADO[/orange]")
                console.print("Al menos un valor propio es cero (hessiana semidefinida)")
            
            # Información adicional
            det_hess = np.linalg.det(hessiana)
            traza_hess = np.trace(hessiana)
            
            console.print(f"\n[bold]Propiedades adicionales:[/bold]")
            console.print(f"Determinante: {formatear_numero(det_hess)}")
            console.print(f"Traza: {formatear_numero(traza_hess)}")
            
            if self.num_variables == 2:
                # Criterio del determinante para 2D
                console.print(f"\n[bold]Criterio del determinante (2D):[/bold]")
                if det_hess > 0:
                    if hessiana[0, 0] > 0:
                        console.print("det(H) > 0 y f_xx > 0 → Mínimo local")
                    else:
                        console.print("det(H) > 0 y f_xx < 0 → Máximo local")
                elif det_hess < 0:
                    console.print("det(H) < 0 → Punto silla")
                else:
                    console.print("det(H) = 0 → Caso degenerado")
            
        except Exception as e:
            console.print(f"[red]Error en análisis: {e}[/red]")
        
        input("\nPresione Enter para continuar...")

    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        limpiar_pantalla()
        ayuda_texto = """
[bold blue]MÉTODO DE NEWTON PARA OPTIMIZACIÓN[/bold blue]

[bold]¿Qué es?[/bold]
Método iterativo para encontrar puntos críticos (máximos, mínimos, puntos silla)
de funciones multivariables usando información de segundo orden (hessiana).

[bold]Algoritmo:[/bold]
1. Partir de un punto inicial x⁽⁰⁾
2. Calcular gradiente ∇f(x⁽ᵏ⁾) y hessiana H(x⁽ᵏ⁾)
3. Resolver H(x⁽ᵏ⁾) · d = -∇f(x⁽ᵏ⁾)
4. Actualizar: x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ + α · d
5. Repetir hasta ||∇f|| < tolerancia

[bold]Condiciones necesarias de optimalidad:[/bold]
• Primer orden: ∇f(x*) = 0
• Segundo orden: H(x*) debe ser definida positiva (mínimo)
                 o definida negativa (máximo)

[bold]Ventajas:[/bold]
• Convergencia cuadrática cerca de la solución
• Encuentra el tipo de punto crítico
• Eficiente para funciones cuadráticas

[bold]Desventajas:[/bold]
• Requiere calcular hessiana (costoso)
• Puede converger a puntos silla
• Sensible al punto inicial
• Puede fallar si hessiana es singular

[bold]Modificaciones implementadas:[/bold]
• Amortiguamiento: Usar α < 1 para mejorar convergencia
• Búsqueda lineal: Encontrar α óptimo en cada iteración
• Regularización: Si hessiana no es def. positiva, usar grad. descendente

[bold]Clasificación de puntos críticos:[/bold]
• Mínimo local: Todos los valores propios > 0
• Máximo local: Todos los valores propios < 0
• Punto silla: Valores propios mixtos (+ y -)

[bold]Aplicaciones:[/bold]
• Optimización de funciones suaves
• Ajuste de modelos no lineales
• Diseño de ingeniería
• Machine learning (entrenamiento de redes)
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
                    self.ingresar_funcion()
                elif opcion == 2:
                    self.configurar_punto_inicial()
                elif opcion == 3:
                    self.configurar_parametros()
                elif opcion == 4:
                    self.ejecutar_newton()
                elif opcion == 5:
                    self.mostrar_resultados()
                elif opcion == 6:
                    self.mostrar_convergencia()
                elif opcion == 7:
                    self.analizar_punto_critico()
                elif opcion == 8:
                    if self.num_variables <= 2:
                        self.mostrar_convergencia()
                    else:
                        console.print("[yellow]Gráficos solo disponibles para 1-2 variables[/yellow]")
                        input("Presione Enter para continuar...")
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
    newton_opt = NewtonOptimizacion()
    newton_opt.main()
