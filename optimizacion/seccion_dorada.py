#!/usr/bin/env python3
"""
Sección Dorada - Implementación con menús interactivos
Método de optimización unidimensional usando la razón áurea
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import sympy as sp
import os
from typing import List, Tuple, Optional, Callable, Union
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_flotante, validar_entero, validar_funcion,
    crear_menu, limpiar_pantalla, mostrar_progreso,
    formatear_numero, formatear_tabla_resultados,
    confirmar_accion, mostrar_mensaje_exito, mostrar_mensaje_error,
    esperar_enter, mostrar_titulo_principal, mostrar_ayuda_metodo
)

console = Console()

class SeccionDorada:
    def __init__(self):
        self.funcion = None
        self.a = None  # Límite inferior
        self.b = None  # Límite superior
        self.tolerancia = 1e-6
        self.max_iteraciones = 100
        self.phi = (1 + np.sqrt(5)) / 2  # Razón áurea
        self.resphi = 2 - self.phi  # 1/φ
        
        # Resultados
        self.solucion = None
        self.valor_optimo = None
        self.iteraciones_realizadas = 0
        self.historial_intervalos = []
        self.historial_puntos = []
        self.historial_valores = []
        self.convergencia = False
        
    def ingresar_funcion(self):
        """Menú para ingreso de la función"""
        limpiar_pantalla()
        console.print(Panel.fit("FUNCIÓN OBJETIVO", style="bold green"))
        
        console.print("\n[bold]Ingrese la función a optimizar:[/bold]")
        console.print("[cyan]Ejemplos:[/cyan]")
        console.print("  x**2 - 4*x + 3")
        console.print("  sin(x) + x**2")
        console.print("  (x-2)**4 + (x-2)**2")
        console.print("  exp(-x**2) * cos(x)")
        
        while True:
            try:
                funcion_str = input("\nf(x) = ").strip()
                
                # Parsear la función
                x = sp.Symbol('x')
                self.funcion = sp.sympify(funcion_str)
                
                # Verificar que solo use la variable x
                if not self.funcion.free_symbols.issubset({x}):
                    console.print("[red]Error: Use solo la variable x[/red]")
                    continue
                
                # Prueba de evaluación
                test_val = float(self.funcion.subs(x, 1.0))
                
                console.print(f"[green]✓ Función ingresada: f(x) = {self.funcion}[/green]")
                console.print(f"[dim]Prueba: f(1) = {formatear_numero(test_val)}[/dim]")
                break
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue
                
        input("\nPresione Enter para continuar...")
    
    def ingresar_intervalo(self):
        """Menú para ingresar el intervalo de búsqueda"""
        if self.funcion is None:
            console.print("[red]Primero debe ingresar una función[/red]")
            input("Presione Enter para continuar...")
            return
            
        limpiar_pantalla()
        console.print(Panel.fit("INTERVALO DE BÚSQUEDA", style="bold blue"))
        
        console.print("\n[bold]Ingrese el intervalo [a,b] para la búsqueda:[/bold]")
        console.print("[yellow]Nota: El método encuentra un mínimo en este intervalo[/yellow]")
        
        while True:
            try:
                self.a = validar_flotante("Límite inferior a: ")
                self.b = validar_flotante("Límite superior b: ")
                
                if self.a >= self.b:
                    console.print("[red]Error: Debe cumplirse a < b[/red]")
                    continue
                
                # Evaluar función en los extremos
                x = sp.Symbol('x')
                fa = float(self.funcion.subs(x, self.a))
                fb = float(self.funcion.subs(x, self.b))
                
                console.print(f"\n[bold cyan]Intervalo configurado:[/bold cyan]")
                console.print(f"[a, b] = [{formatear_numero(self.a)}, {formatear_numero(self.b)}]")
                console.print(f"f(a) = {formatear_numero(fa)}")
                console.print(f"f(b) = {formatear_numero(fb)}")
                
                # Mostrar longitud inicial
                longitud = self.b - self.a
                console.print(f"Longitud inicial: {formatear_numero(longitud)}")
                
                # Pregunta por gráfica del intervalo
                if confirmar_accion("¿Desea ver la gráfica de la función en el intervalo?"):
                    self._graficar_funcion_intervalo()
                
                break
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                
        input("\nPresione Enter para continuar...")
    
    def _graficar_funcion_intervalo(self):
        """Grafica la función en el intervalo especificado"""
        try:
            margen = (self.b - self.a) * 0.1
            x_vals = np.linspace(self.a - margen, self.b + margen, 1000)
            
            x_sym = sp.Symbol('x')
            f_lambdified = sp.lambdify(x_sym, self.funcion, 'numpy')
            y_vals = f_lambdified(x_vals)
            
            plt.figure(figsize=(10, 6))
            plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {self.funcion}')
            
            # Marcar el intervalo
            plt.axvline(x=self.a, color='red', linestyle='--', alpha=0.7, label=f'a = {self.a}')
            plt.axvline(x=self.b, color='red', linestyle='--', alpha=0.7, label=f'b = {self.b}')
            
            # Sombrear el intervalo de búsqueda
            y_min, y_max = plt.ylim()
            plt.fill_betweenx([y_min, y_max], self.a, self.b, alpha=0.2, color='yellow', 
                             label='Intervalo de búsqueda')
            
            plt.title('Función Objetivo e Intervalo de Búsqueda')
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            console.print(f"[red]Error generando gráfica: {e}[/red]")
    
    def configurar_parametros(self):
        """Configura parámetros del método"""
        limpiar_pantalla()
        console.print(Panel.fit("PARÁMETROS DEL MÉTODO", style="bold cyan"))
        
        console.print(f"\n[bold]Parámetros actuales:[/bold]")
        console.print(f"Tolerancia: {self.tolerancia}")
        console.print(f"Máximo iteraciones: {self.max_iteraciones}")
        console.print(f"Razón áurea φ: {formatear_numero(self.phi)}")
        
        if input("\n¿Cambiar parámetros? (s/n): ").lower() == 's':
            self.tolerancia = validar_flotante(
                "Nueva tolerancia (1e-6): ", 1e-15, 1e-3, self.tolerancia
            )
            self.max_iteraciones = validar_entero(
                "Nuevas máximo iteraciones (100): ", 10, 1000, self.max_iteraciones
            )
            
        input("Presione Enter para continuar...")
    
    def ejecutar_seccion_dorada(self):
        """Ejecuta el método de sección dorada"""
        if self.funcion is None or self.a is None or self.b is None:
            console.print("[red]Debe completar la configuración[/red]")
            input("Presione Enter para continuar...")
            return
            
        limpiar_pantalla()
        console.print(Panel.fit("EJECUTANDO SECCIÓN DORADA", style="bold yellow"))
        
        # Mostrar configuración
        console.print(f"\n[bold]Función:[/bold] f(x) = {self.funcion}")
        console.print(f"[bold]Intervalo:[/bold] [{formatear_numero(self.a)}, {formatear_numero(self.b)}]")
        console.print(f"[bold]Tolerancia:[/bold] {self.tolerancia}")
        
        try:
            x = sp.Symbol('x')
            f_func = sp.lambdify(x, self.funcion, 'numpy')
            
            # Inicializar
            self.historial_intervalos = []
            self.historial_puntos = []
            self.historial_valores = []
            
            a, b = self.a, self.b
            
            # Calcular puntos iniciales
            c = a + self.resphi * (b - a)
            d = a + (1 - self.resphi) * (b - a)
            
            fc = float(f_func(c))
            fd = float(f_func(d))
            
            console.print(f"\n[cyan]Iniciando optimización...[/cyan]")
            console.print(f"Punto inicial c = {formatear_numero(c)}, f(c) = {formatear_numero(fc)}")
            console.print(f"Punto inicial d = {formatear_numero(d)}, f(d) = {formatear_numero(fd)}")
            
            for iteracion in track(range(self.max_iteraciones), description="Optimizando..."):
                # Guardar estado actual
                self.historial_intervalos.append((a, b))
                self.historial_puntos.append((c, d))
                self.historial_valores.append((fc, fd))
                
                # Verificar convergencia
                if abs(b - a) < self.tolerancia:
                    self.solucion = (a + b) / 2
                    self.valor_optimo = float(f_func(self.solucion))
                    self.iteraciones_realizadas = iteracion + 1
                    self.convergencia = True
                    console.print(f"\n[bold green]✓ Convergió en {self.iteraciones_realizadas} iteraciones[/bold green]")
                    console.print(f"[bold]Solución:[/bold] x* = {formatear_numero(self.solucion)}")
                    console.print(f"[bold]Valor óptimo:[/bold] f(x*) = {formatear_numero(self.valor_optimo)}")
                    break
                
                # Aplicar regla de sección dorada
                if fc < fd:
                    # El mínimo está en [a, d]
                    b = d
                    d = c
                    fd = fc
                    c = a + self.resphi * (b - a)
                    fc = float(f_func(c))
                else:
                    # El mínimo está en [c, b]
                    a = c
                    c = d
                    fc = fd
                    d = a + (1 - self.resphi) * (b - a)
                    fd = float(f_func(d))
                
                # Mostrar progreso cada 10 iteraciones
                if (iteracion + 1) % 10 == 0:
                    longitud_actual = b - a
                    console.print(f"Iteración {iteracion + 1}: Intervalo = [{formatear_numero(a)}, {formatear_numero(b)}], Longitud = {formatear_numero(longitud_actual)}")
            
            else:
                # No convergió
                self.solucion = (a + b) / 2
                self.valor_optimo = float(f_func(self.solucion))
                self.iteraciones_realizadas = self.max_iteraciones
                self.convergencia = False
                console.print(f"\n[bold yellow]⚠ No convergió en {self.max_iteraciones} iteraciones[/bold yellow]")
                console.print(f"[bold]Mejor aproximación:[/bold] x* ≈ {formatear_numero(self.solucion)}")
                console.print(f"[bold]Valor:[/bold] f(x*) ≈ {formatear_numero(self.valor_optimo)}")
            
            # Análisis final
            self._analizar_solucion()
            
        except Exception as e:
            console.print(f"[red]Error durante la optimización: {e}[/red]")
        
        input("Presione Enter para continuar...")
    
    def _analizar_solucion(self):
        """Analiza la solución encontrada"""
        if self.solucion is None:
            return
            
        try:
            x = sp.Symbol('x')
            
            # Calcular derivadas para análisis
            primera_derivada = sp.diff(self.funcion, x)
            segunda_derivada = sp.diff(primera_derivada, x)
            
            # Evaluar en la solución
            f_primera = float(primera_derivada.subs(x, self.solucion))
            f_segunda = float(segunda_derivada.subs(x, self.solucion))
            
            console.print(f"\n[bold cyan]Análisis de la solución:[/bold cyan]")
            console.print(f"f'(x*) = {formatear_numero(f_primera)}")
            console.print(f"f''(x*) = {formatear_numero(f_segunda)}")
            
            # Clasificar el punto crítico
            if abs(f_primera) < 1e-6:
                if f_segunda > 1e-6:
                    tipo = "[green]MÍNIMO LOCAL[/green]"
                elif f_segunda < -1e-6:
                    tipo = "[red]MÁXIMO LOCAL[/red]"
                else:
                    tipo = "[yellow]PUNTO DE INFLEXIÓN[/yellow]"
            else:
                tipo = "[yellow]NO ES PUNTO CRÍTICO[/yellow]"
            
            console.print(f"Tipo: {tipo}")
            
            # Información de convergencia
            if self.historial_intervalos:
                reduccion_total = (self.historial_intervalos[0][1] - self.historial_intervalos[0][0]) / (self.historial_intervalos[-1][1] - self.historial_intervalos[-1][0])
                console.print(f"Reducción del intervalo: {formatear_numero(reduccion_total)}x")
            
        except Exception as e:
            console.print(f"[yellow]No se pudo completar el análisis: {e}[/yellow]")
    
    def mostrar_resultados(self):
        """Muestra los resultados detallados"""
        if self.solucion is None:
            console.print("[red]No hay resultados para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
            
        limpiar_pantalla()
        console.print(Panel.fit("RESULTADOS SECCIÓN DORADA", style="bold green"))
        
        # Resultado principal
        console.print(f"\n[bold cyan]Solución encontrada:[/bold cyan]")
        console.print(f"x* = {formatear_numero(self.solucion)}")
        console.print(f"f(x*) = {formatear_numero(self.valor_optimo)}")
        
        # Información de convergencia
        console.print(f"\n[bold]Información de convergencia:[/bold]")
        console.print(f"Convergió: {'Sí' if self.convergencia else 'No'}")
        console.print(f"Iteraciones: {self.iteraciones_realizadas}")
        console.print(f"Tolerancia: {self.tolerancia}")
        
        # Intervalo final
        if self.historial_intervalos:
            intervalo_final = self.historial_intervalos[-1]
            longitud_final = intervalo_final[1] - intervalo_final[0]
            console.print(f"Intervalo final: [{formatear_numero(intervalo_final[0])}, {formatear_numero(intervalo_final[1])}]")
            console.print(f"Longitud final: {formatear_numero(longitud_final)}")
        
        # Tabla de últimas iteraciones
        if len(self.historial_intervalos) > 0:
            console.print(f"\n[bold]Últimas iteraciones:[/bold]")
            tabla = Table()
            tabla.add_column("Iter", style="cyan")
            tabla.add_column("Intervalo [a,b]", style="white")
            tabla.add_column("Longitud", style="yellow")
            tabla.add_column("Puntos c,d", style="green")
            tabla.add_column("f(c), f(d)", style="magenta")
            
            # Mostrar últimas 5 iteraciones
            inicio = max(0, len(self.historial_intervalos) - 5)
            for i in range(inicio, len(self.historial_intervalos)):
                intervalo = self.historial_intervalos[i]
                longitud = intervalo[1] - intervalo[0]
                puntos = self.historial_puntos[i]
                valores = self.historial_valores[i]
                
                tabla.add_row(
                    str(i + 1),
                    f"[{formatear_numero(intervalo[0])}, {formatear_numero(intervalo[1])}]",
                    formatear_numero(longitud),
                    f"({formatear_numero(puntos[0])}, {formatear_numero(puntos[1])})",
                    f"({formatear_numero(valores[0])}, {formatear_numero(valores[1])})"
                )
            
            console.print(tabla)
        
        input("\nPresione Enter para continuar...")
    
    def mostrar_convergencia(self):
        """Muestra gráficos de convergencia"""
        if not self.historial_intervalos:
            console.print("[red]No hay datos de convergencia para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
            
        console.print("[cyan]Generando gráficos de convergencia...[/cyan]")
        
        try:
            self._graficar_convergencia()
        except Exception as e:
            console.print(f"[red]Error generando gráficos: {e}[/red]")
            
        input("Presione Enter para continuar...")
    
    def _graficar_convergencia(self):
        """Genera gráficos de convergencia"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        iteraciones = list(range(1, len(self.historial_intervalos) + 1))
        
        # 1. Función y convergencia del intervalo
        margen = (self.b - self.a) * 0.2
        x_vals = np.linspace(self.a - margen, self.b + margen, 1000)
        x_sym = sp.Symbol('x')
        f_lambdified = sp.lambdify(x_sym, self.funcion, 'numpy')
        y_vals = f_lambdified(x_vals)
        
        ax1.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {self.funcion}')
        
        # Mostrar evolución del intervalo
        for i, (a_i, b_i) in enumerate(self.historial_intervalos[::max(1, len(self.historial_intervalos)//5)]):
            alpha = 0.3 + 0.7 * i / len(self.historial_intervalos)
            ax1.axvspan(a_i, b_i, alpha=alpha, color='red', label=f'Iter {i+1}' if i < 3 else "")
        
        if self.solucion:
            ax1.axvline(x=self.solucion, color='green', linestyle='--', linewidth=3, label=f'x* = {formatear_numero(self.solucion)}')
            ax1.plot(self.solucion, self.valor_optimo, 'go', markersize=10, label='Óptimo')
        
        ax1.set_title('Función y Convergencia del Intervalo')
        ax1.set_xlabel('x')
        ax1.set_ylabel('f(x)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Reducción del intervalo
        longitudes = [b_i - a_i for a_i, b_i in self.historial_intervalos]
        ax2.semilogy(iteraciones, longitudes, 'ro-', linewidth=2, markersize=4)
        ax2.axhline(y=self.tolerancia, color='green', linestyle='--', label='Tolerancia')
        ax2.set_title('Reducción del Intervalo')
        ax2.set_xlabel('Iteración')
        ax2.set_ylabel('Longitud del Intervalo')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Convergencia de los puntos c y d
        c_vals = [puntos[0] for puntos in self.historial_puntos]
        d_vals = [puntos[1] for puntos in self.historial_puntos]
        
        ax3.plot(iteraciones, c_vals, 'bo-', linewidth=2, markersize=4, label='c')
        ax3.plot(iteraciones, d_vals, 'ro-', linewidth=2, markersize=4, label='d')
        
        if self.solucion:
            ax3.axhline(y=self.solucion, color='green', linestyle='--', label=f'x* = {formatear_numero(self.solucion)}')
        
        ax3.set_title('Convergencia de Puntos Internos')
        ax3.set_xlabel('Iteración')
        ax3.set_ylabel('Valor')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Valores de la función en c y d
        fc_vals = [valores[0] for valores in self.historial_valores]
        fd_vals = [valores[1] for valores in self.historial_valores]
        
        ax4.plot(iteraciones, fc_vals, 'bo-', linewidth=2, markersize=4, label='f(c)')
        ax4.plot(iteraciones, fd_vals, 'ro-', linewidth=2, markersize=4, label='f(d)')
        
        if self.valor_optimo:
            ax4.axhline(y=self.valor_optimo, color='green', linestyle='--', label=f'f(x*) = {formatear_numero(self.valor_optimo)}')
        
        ax4.set_title('Valores de la Función')
        ax4.set_xlabel('Iteración')
        ax4.set_ylabel('f(x)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        mostrar_ayuda_metodo(
            "Método de Sección Dorada",
            "Algoritmo de optimización unidimensional que usa la razón áurea (φ = 1.618...) "
            "para reducir sistemáticamente el intervalo de búsqueda. En cada iteración, "
            "evalúa la función en dos puntos internos y elimina la porción del intervalo "
            "que no puede contener el óptimo.",
            [
                "Optimización de funciones unimodales (un solo mínimo/máximo)",
                "Problemas donde las derivadas son difíciles de calcular",
                "Búsqueda de línea en algoritmos de optimización multidimensional",
                "Optimización robusta sin requisitos de diferenciabilidad"
            ],
            [
                "No requiere derivadas (método de orden cero)",
                "Convergencia garantizada para funciones unimodales",
                "Reducción óptima del intervalo en cada iteración",
                "Robustez numérica y simplicidad de implementación",
                "Aplicable a funciones discontinuas"
            ],
            [
                "Solo para optimización unidimensional",
                "Convergencia lineal (más lenta que Newton)",
                "Requiere función unimodal en el intervalo",
                "Necesita intervalo inicial que contenga el óptimo"
            ],
            [
                "• Ratio de reducción: 1/φ ≈ 0.618 por iteración",
                "• Error después de n iteraciones: ε₀ * (1/φ)ⁿ",
                "• Número de evaluaciones: n + 1",
                "• Óptimo para minimizar evaluaciones de función"
            ]
        )
    
    def main(self):
        """Función principal con el bucle del menú"""
        while True:
            limpiar_pantalla()
            mostrar_titulo_principal("Sección Dorada", "Optimización Unidimensional")
            
            opciones = [
                "🎯 Ingresar función objetivo",
                "📏 Configurar intervalo de búsqueda",
                "⚙️  Configurar parámetros",
                "🚀 Ejecutar sección dorada",
                "📊 Ver resultados",
                "📈 Ver convergencia",
                "❓ Ayuda",
                "🚪 Salir"
            ]
            
            # Mostrar estado actual
            estado = []
            if self.funcion:
                estado.append(f"[green]✓[/green] Función: f(x) = {self.funcion}")
            else:
                estado.append("[red]✗[/red] Función no definida")
                
            if self.a is not None and self.b is not None:
                estado.append(f"[green]✓[/green] Intervalo: [{formatear_numero(self.a)}, {formatear_numero(self.b)}]")
            else:
                estado.append("[red]✗[/red] Intervalo no definido")
                
            estado.append(f"Tolerancia: {self.tolerancia}")
            
            if self.solucion is not None:
                estado.append(f"[green]✓[/green] Solución: x* = {formatear_numero(self.solucion)}")
            
            console.print(Panel("\n".join(estado), title="📋 Estado Actual", border_style="blue"))
            
            opcion = crear_menu("Seleccione una opción:", opciones)
            
            if opcion == 1:
                self.ingresar_funcion()
            elif opcion == 2:
                self.ingresar_intervalo()
            elif opcion == 3:
                self.configurar_parametros()
            elif opcion == 4:
                self.ejecutar_seccion_dorada()
            elif opcion == 5:
                self.mostrar_resultados()
            elif opcion == 6:
                self.mostrar_convergencia()
            elif opcion == 7:
                self.mostrar_ayuda()
            elif opcion == 8:
                console.print("\n[bold green]¡Hasta luego![/bold green]")
                break

if __name__ == "__main__":
    seccion = SeccionDorada()
    seccion.main()
