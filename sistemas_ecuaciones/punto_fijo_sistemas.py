#!/usr/bin/env python3
"""
Punto Fijo para Sistemas - Implementación con menús interactivos
Método iterativo para resolver sistemas de ecuaciones no lineales
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

class PuntoFijoSistemas:
    def __init__(self):
        self.ecuaciones = []
        self.funciones_g = []
        self.variables = []
        self.valores_iniciales = []
        self.tolerancia = 1e-10
        self.max_iteraciones = 100
        self.solucion = []
        self.iteraciones_realizadas = 0
        self.historial = []
        self.errores = []

    def mostrar_menu_principal(self):
        """Muestra el menú principal del método"""
        opciones = [
            "Ingresar sistema de ecuaciones",
            "Definir funciones de iteración g(x,y)",
            "Configurar valores iniciales",
            "Configurar parámetros de convergencia",
            "Ejecutar método de punto fijo",
            "Ver resultados",
            "Mostrar convergencia",
            "Analizar estabilidad",
            "Ver ayuda",
            "Salir"
        ]
        return crear_menu("PUNTO FIJO PARA SISTEMAS", opciones)

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
        console.print("  exp(x) + y - 2")
        
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
                    console.print(f"[green]✓ f₍{i+1}₎(x,y) = {ecuacion}[/green]")
                    break
                    
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    continue
        
        console.print(f"\n[bold cyan]Sistema ingresado exitosamente:[/bold cyan]")
        for i, eq in enumerate(self.ecuaciones):
            console.print(f"f₍{i+1}₎({','.join(self.variables)}) = {eq} = 0")
        
        input("\nPresione Enter para continuar...")

    def definir_funciones_g(self):
        """Define las funciones de iteración g(x,y)"""
        if not self.ecuaciones:
            console.print("[red]Primero debe ingresar el sistema de ecuaciones[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("FUNCIONES DE ITERACIÓN", style="bold blue"))
        
        console.print("\n[bold]Sistema actual:[/bold]")
        for i, eq in enumerate(self.ecuaciones):
            console.print(f"f₍{i+1}₎ = {eq} = 0")
        
        console.print(f"\n[bold]Defina las funciones de iteración g₍ᵢ₎:[/bold]")
        console.print(f"[yellow]El método usará: {self.variables[0]}ₙ₊₁ = g₁, {self.variables[1]}ₙ₊₁ = g₂, ...[/yellow]")
        
        console.print("\n[bold]Opciones:[/bold]")
        console.print("1. Despejar automáticamente (recomendado)")
        console.print("2. Ingresar funciones g manualmente")
        
        opcion = validar_entero("Seleccione opción (1-2): ", 1, 2)
        
        if opcion == 1:
            self._despejar_automatico()
        else:
            self._ingresar_g_manual()

    def _despejar_automatico(self):
        """Intenta despejar automáticamente las funciones g"""
        console.print("\n[cyan]Intentando despeje automático...[/cyan]")
        
        self.funciones_g = []
        simbolos = [sp.Symbol(var) for var in self.variables]
        
        for i, ecuacion in enumerate(self.ecuaciones):
            var_objetivo = simbolos[i]
            
            try:
                # Intentar resolver para la variable objetivo
                soluciones = sp.solve(ecuacion, var_objetivo)
                
                if soluciones:
                    # Tomar la primera solución
                    g_func = soluciones[0]
                    self.funciones_g.append(g_func)
                    console.print(f"[green]✓ g₍{i+1}₎ = {g_func}[/green]")
                else:
                    # Si no se puede despejar, usar transformación simple
                    g_func = var_objetivo - ecuacion
                    self.funciones_g.append(g_func)
                    console.print(f"[yellow]⚠ g₍{i+1}₎ = {g_func} (transformación simple)[/yellow]")
                    
            except Exception as e:
                # Transformación por defecto
                g_func = simbolos[i] - ecuacion
                self.funciones_g.append(g_func)
                console.print(f"[yellow]⚠ g₍{i+1}₎ = {g_func} (por defecto)[/yellow]")
        
        # Mostrar sistema de iteración resultante
        console.print(f"\n[bold cyan]Sistema de iteración:[/bold cyan]")
        for i, g in enumerate(self.funciones_g):
            console.print(f"{self.variables[i]}ₙ₊₁ = {g}")

    def _ingresar_g_manual(self):
        """Permite ingresar las funciones g manualmente"""
        console.print(f"\n[bold]Ingrese las funciones g₍ᵢ₎:[/bold]")
        
        self.funciones_g = []
        simbolos = [sp.Symbol(var) for var in self.variables]
        
        for i in range(len(self.variables)):
            while True:
                try:
                    console.print(f"\nPara {self.variables[i]}ₙ₊₁ = g₍{i+1}₎({','.join(self.variables)})")
                    g_str = input(f"g₍{i+1}₎ = ").strip()
                    
                    g_func = sp.sympify(g_str)
                    
                    # Verificar variables
                    vars_g = g_func.free_symbols
                    vars_permitidas = set(simbolos)
                    
                    if not vars_g.issubset(vars_permitidas):
                        console.print(f"[red]Error: Use solo las variables {', '.join(self.variables)}[/red]")
                        continue
                    
                    self.funciones_g.append(g_func)
                    console.print(f"[green]✓ g₍{i+1}₎ = {g_func}[/green]")
                    break
                    
                except Exception as e:
                    console.print(f"[red]Error: {e}[/red]")
                    continue

    def configurar_valores_iniciales(self):
        """Configura los valores iniciales"""
        if not self.funciones_g:
            console.print("[red]Primero debe definir las funciones de iteración[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("VALORES INICIALES", style="bold magenta"))
        
        console.print(f"\n[bold]Ingrese los valores iniciales:[/bold]")
        self.valores_iniciales = []
        
        for i, var in enumerate(self.variables):
            valor = validar_flotante(f"Valor inicial para {var}₀: ", puede_ser_cero=True)
            self.valores_iniciales.append(valor)
        
        # Mostrar resumen
        console.print(f"\n[bold cyan]Valores iniciales:[/bold cyan]")
        for i, (var, val) in enumerate(zip(self.variables, self.valores_iniciales)):
            console.print(f"{var}₀ = {formatear_numero(val)}")
        
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

    def ejecutar_punto_fijo(self):
        """Ejecuta el método de punto fijo para sistemas"""
        if not self.funciones_g or not self.valores_iniciales:
            console.print("[red]Debe completar la configuración del sistema[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("EJECUTANDO PUNTO FIJO", style="bold yellow"))
        
        # Mostrar configuración
        console.print(f"\n[bold]Sistema de iteración:[/bold]")
        for i, g in enumerate(self.funciones_g):
            console.print(f"{self.variables[i]}ₙ₊₁ = {g}")
        
        console.print(f"\n[bold]Valores iniciales:[/bold] {[formatear_numero(v) for v in self.valores_iniciales]}")
        console.print(f"[bold]Tolerancia:[/bold] {self.tolerancia}")
        
        try:
            # Convertir funciones a callables
            simbolos = [sp.Symbol(var) for var in self.variables]
            g_funcs = []
            for g in self.funciones_g:
                g_funcs.append(sp.lambdify(simbolos, g, 'numpy'))
            
            # Inicializar
            self.historial = []
            self.errores = []
            valores_actuales = np.array(self.valores_iniciales, dtype=float)
            
            console.print(f"\n[cyan]Iniciando iteraciones...[/cyan]")
            
            for iteracion in track(range(self.max_iteraciones), description="Iterando..."):
                # Guardar valores actuales
                self.historial.append(valores_actuales.copy())
                
                # Calcular nuevos valores
                try:
                    nuevos_valores = np.array([
                        float(g_func(*valores_actuales)) for g_func in g_funcs
                    ])
                except Exception as e:
                    console.print(f"[red]Error en iteración {iteracion}: {e}[/red]")
                    break
                
                # Calcular error
                error = np.linalg.norm(nuevos_valores - valores_actuales)
                self.errores.append(error)
                
                # Verificar convergencia
                if error < self.tolerancia:
                    self.solucion = nuevos_valores
                    self.iteraciones_realizadas = iteracion + 1
                    console.print(f"\n[bold green]✓ Convergió en {self.iteraciones_realizadas} iteraciones[/bold green]")
                    console.print(f"[bold]Error final:[/bold] {formatear_numero(error)}")
                    break
                
                # Actualizar valores
                valores_actuales = nuevos_valores
                
                # Verificar divergencia
                if np.any(np.abs(valores_actuales) > 1e10):
                    console.print(f"\n[bold red]⚠ Divergencia detectada en iteración {iteracion}[/bold red]")
                    break
            
            else:
                console.print(f"\n[bold yellow]⚠ No convergió en {self.max_iteraciones} iteraciones[/bold yellow]")
                self.solucion = valores_actuales
                self.iteraciones_realizadas = self.max_iteraciones
            
            # Verificar solución
            if len(self.solucion) > 0:
                self._verificar_solucion()
                
        except Exception as e:
            console.print(f"[red]Error durante la ejecución: {e}[/red]")
        
        input("Presione Enter para continuar...")

    def _verificar_solucion(self):
        """Verifica la calidad de la solución encontrada"""
        console.print(f"\n[bold]Verificación de la solución:[/bold]")
        
        simbolos = [sp.Symbol(var) for var in self.variables]
        
        for i, ecuacion in enumerate(self.ecuaciones):
            try:
                # Evaluar ecuación en la solución
                f_evaluada = float(ecuacion.subs(dict(zip(simbolos, self.solucion))))
                console.print(f"f₍{i+1}₎({', '.join([formatear_numero(x) for x in self.solucion])}) = {formatear_numero(f_evaluada)}")
            except Exception as e:
                console.print(f"f₍{i+1}₎: Error en evaluación - {e}")

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
        console.print(f"Error final: {formatear_numero(self.errores[-1]) if self.errores else 'N/A'}")
        console.print(f"Tolerancia: {self.tolerancia}")
        
        # Verificación
        console.print(f"\n[bold]Verificación (evaluando f₍ᵢ₎ en la solución):[/bold]")
        simbolos = [sp.Symbol(var) for var in self.variables]
        
        for i, ecuacion in enumerate(self.ecuaciones):
            try:
                f_evaluada = float(ecuacion.subs(dict(zip(simbolos, self.solucion))))
                console.print(f"f₍{i+1}₎ = {formatear_numero(f_evaluada)}")
            except Exception as e:
                console.print(f"f₍{i+1}₎: Error - {e}")
        
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

    def analizar_estabilidad(self):
        """Analiza la estabilidad del punto fijo"""
        if not self.solucion or not self.funciones_g:
            console.print("[red]Primero debe encontrar una solución[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("ANÁLISIS DE ESTABILIDAD", style="bold purple"))
        
        try:
            # Calcular matriz jacobiana de G en el punto fijo
            simbolos = [sp.Symbol(var) for var in self.variables]
            
            console.print(f"\n[bold]Matriz Jacobiana de G en el punto fijo:[/bold]")
            
            jacobiana = []
            for i, g in enumerate(self.funciones_g):
                fila = []
                for j, var in enumerate(simbolos):
                    derivada = sp.diff(g, var)
                    valor = float(derivada.subs(dict(zip(simbolos, self.solucion))))
                    fila.append(valor)
                jacobiana.append(fila)
            
            # Mostrar matriz
            J = np.array(jacobiana)
            console.print("J = ")
            for fila in J:
                console.print("  [" + ", ".join([f"{val:8.4f}" for val in fila]) + "]")
            
            # Calcular valores propios
            eigenvalues = np.linalg.eigvals(J)
            console.print(f"\n[bold]Valores propios de J:[/bold]")
            
            estable = True
            for i, eigenval in enumerate(eigenvalues):
                modulo = abs(eigenval)
                console.print(f"λ₍{i+1}₎ = {formatear_numero(eigenval)}, |λ₍{i+1}₎| = {formatear_numero(modulo)}")
                
                if modulo >= 1:
                    estable = False
            
            # Conclusión sobre estabilidad
            console.print(f"\n[bold]Análisis de estabilidad:[/bold]")
            if estable:
                console.print("[green]✓ El punto fijo es localmente estable[/green]")
                console.print("Todos los valores propios tienen módulo < 1")
            else:
                console.print("[red]⚠ El punto fijo es inestable[/red]")
                console.print("Al menos un valor propio tiene módulo ≥ 1")
            
        except Exception as e:
            console.print(f"[red]Error en análisis de estabilidad: {e}[/red]")
        
        input("\nPresione Enter para continuar...")

    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        limpiar_pantalla()
        ayuda_texto = """
[bold blue]MÉTODO DE PUNTO FIJO PARA SISTEMAS[/bold blue]

[bold]¿Qué es?[/bold]
Método iterativo para resolver sistemas de ecuaciones no lineales
transformando el sistema F(x) = 0 en x = G(x).

[bold]Algoritmo:[/bold]
1. Transformar fᵢ(x₁,...,xₙ) = 0 en xᵢ = gᵢ(x₁,...,xₙ)
2. Partir de valores iniciales x⁽⁰⁾
3. Iterar: x⁽ᵏ⁺¹⁾ = G(x⁽ᵏ⁾)
4. Repetir hasta convergencia

[bold]Condiciones de convergencia:[/bold]
• ||G'(x*)|| < 1 en el punto fijo x*
• Todos los valores propios de la matriz jacobiana < 1 en módulo
• Valores iniciales suficientemente cerca de la solución

[bold]Ventajas:[/bold]
• Simple de implementar
• No requiere cálculo de derivadas del usuario
• Converge para sistemas bien condicionados

[bold]Desventajas:[/bold]
• Convergencia lenta (lineal)
• Muy sensible a la transformación G elegida
• Puede diverger con valores iniciales pobres
• Difícil encontrar buenas transformaciones

[bold]Consejos de uso:[/bold]
• Probar diferentes formas de despejar las variables
• Usar valores iniciales cercanos a la solución esperada
• Verificar la estabilidad del punto fijo encontrado
• Si no converge, probar otras transformaciones

[bold]Aplicaciones:[/bold]
• Sistemas de ecuaciones algebraicas no lineales
• Puntos de equilibrio en sistemas dinámicos
• Problemas de optimización con restricciones
• Análisis de circuitos no lineales
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
                    self.definir_funciones_g()
                elif opcion == 3:
                    self.configurar_valores_iniciales()
                elif opcion == 4:
                    self.configurar_parametros()
                elif opcion == 5:
                    self.ejecutar_punto_fijo()
                elif opcion == 6:
                    self.mostrar_resultados()
                elif opcion == 7:
                    self.mostrar_convergencia()
                elif opcion == 8:
                    self.analizar_estabilidad()
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
    punto_fijo = PuntoFijoSistemas()
    punto_fijo.main()
