#!/usr/bin/env python3
"""
Método de Punto Fijo - Implementación con menús interactivos
Encuentra raíces transformando f(x)=0 a x=g(x) y usando iteración x_{n+1}=g(x_n)
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from tqdm import tqdm

# Agregar directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_numero, validar_opcion_menu, validar_funcion, validar_tolerancia,
    validar_max_iteraciones, confirmar_accion, limpiar_pantalla, 
    mostrar_titulo_principal, mostrar_menu_opciones, mostrar_banner_metodo,
    mostrar_estado_configuracion, mostrar_progreso_ejecucion,
    mostrar_resultado_final, mostrar_mensaje_error, mostrar_mensaje_exito,
    mostrar_ayuda_metodo, esperar_enter, crear_tabla_iteraciones, formatear_numero,
    graficar_funcion, graficar_punto_fijo
)

console = Console()

class MetodoPuntoFijo:
    def __init__(self):
        self.funcion_g = None
        self.funcion_g_str = ""
        self.derivada_g = None
        self.x0 = None
        self.tolerancia = 1e-6
        self.max_iteraciones = 100
        self.resultados = None
        
    def configuracion_completa(self) -> bool:
        """Verifica si la configuración está completa"""
        return all([
            self.funcion_g is not None,
            self.x0 is not None,
            self.tolerancia is not None,
            self.max_iteraciones is not None
        ])
    
    def mostrar_configuracion(self):
        """Muestra el estado actual de la configuración"""
        config = {
            "Función g(x)": self.funcion_g_str if self.funcion_g_str else None,
            "Valor inicial x₀": self.x0,
            "Tolerancia": self.tolerancia,
            "Máx. Iteraciones": self.max_iteraciones
        }
        mostrar_estado_configuracion(config)
    
    def ingresar_funcion(self):
        """Menú para ingresar la función g(x)"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Punto Fijo", "Ingreso de Función g(x)")
        
        console.print(Panel(
            "[bold cyan]Transformación de f(x) = 0 a x = g(x)[/bold cyan]\n\n"
            "[yellow]Formas comunes de transformación:[/yellow]\n"
            "• f(x) = x² - a → g(x) = √a (para x > 0)\n"
            "• f(x) = x - cos(x) → g(x) = cos(x)\n"
            "• f(x) = x² + x - 1 → g(x) = 1 - x² o g(x) = (1-x²)/x\n\n"
            "[green]Ejemplos válidos de g(x):[/green]\n"
            "• cos(x)\n"
            "• (x**2 + 1)/3\n"
            "• sqrt(x + 1)\n"
            "• exp(-x)\n\n"
            "[red]⚠️ Para convergencia: |g'(x)| < 1 cerca del punto fijo[/red]",
            title="💡 Transformación a Punto Fijo",
            border_style="blue"
        ))
        
        # Menú de opciones para ingresar g(x)
        opciones = [
            "Ingresar g(x) manualmente",
            "Ver ejemplos de transformaciones",
            "Ayuda para transformar f(x) = 0",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Opciones de ingreso", False)
        opcion = validar_opcion_menu([1, 2, 3, 4])
        
        if opcion == 1:
            self._ingresar_g_manual()
        elif opcion == 2:
            self._mostrar_ejemplos_transformacion()
        elif opcion == 3:
            self._mostrar_ayuda_transformacion()
        elif opcion == 4:
            return
    
    def _ingresar_g_manual(self):
        """Ingresa g(x) manualmente"""
        while True:
            funcion_input = input("\nIngrese g(x): ").strip()
            
            if not funcion_input:
                console.print("[red]❌ La función no puede estar vacía[/red]")
                continue
            
            es_valida, expr, error = validar_funcion(funcion_input)
            
            if es_valida:
                self.funcion_g = expr
                self.funcion_g_str = funcion_input
                
                # Calcular derivada para análisis de convergencia
                try:
                    x = sp.Symbol('x')
                    self.derivada_g = sp.diff(self.funcion_g, x)
                    
                    mostrar_mensaje_exito(f"Función g(x) = {funcion_input}")
                    console.print(f"[cyan]g'(x) = {self.derivada_g}[/cyan]")
                    
                    # Analizar convergencia en algunos puntos
                    self._analizar_convergencia_inicial()
                    
                    # Mostrar gráfica de g(x) y y=x
                    if confirmar_accion("¿Desea ver la gráfica de g(x) y y=x?"):
                        try:
                            self._graficar_g_y_identidad()
                        except Exception as e:
                            mostrar_mensaje_error(f"Error al graficar: {e}")
                    
                    esperar_enter()
                    break
                    
                except Exception as e:
                    mostrar_mensaje_error(f"Error al calcular la derivada: {e}")
                    self.derivada_g = None
                    break
            else:
                mostrar_mensaje_error(f"Error en la función: {error}")
    
    def _mostrar_ejemplos_transformacion(self):
        """Muestra ejemplos de transformaciones comunes"""
        limpiar_pantalla()
        mostrar_titulo_principal("Punto Fijo", "Ejemplos de Transformaciones")
        
        ejemplos = [
            ("x² - 2 = 0", "g(x) = √2", "Raíz cuadrada positiva"),
            ("x² - 2 = 0", "g(x) = 2/x", "Dividir por x"),
            ("x - cos(x) = 0", "g(x) = cos(x)", "Despejar x"),
            ("e^x - 2x - 1 = 0", "g(x) = (e^x - 1)/2", "Despejar x linealmente"),
            ("x² + x - 1 = 0", "g(x) = 1 - x²", "Despejar x"),
            ("x³ - x - 1 = 0", "g(x) = x + 1", "Agregar x a ambos lados"),
        ]
        
        table = Table(title="Ejemplos de Transformaciones f(x) = 0 → x = g(x)", border_style="green")
        table.add_column("f(x) = 0", style="cyan")
        table.add_column("x = g(x)", style="yellow")
        table.add_column("Método", style="white")
        
        for fx, gx, metodo in ejemplos:
            table.add_row(fx, gx, metodo)
        
        console.print(table)
        console.print()
        
        console.print(Panel(
            "[bold yellow]Criterios para elegir g(x):[/bold yellow]\n\n"
            "1. |g'(x)| < 1 en el intervalo de interés\n"
            "2. g(x) debe ser continua\n"
            "3. El punto fijo debe estar en el dominio\n"
            "4. Experimentar con diferentes transformaciones\n\n"
            "[green]💡 Si una transformación no converge, pruebe otra[/green]",
            border_style="blue"
        ))
        
        esperar_enter()
    
    def _mostrar_ayuda_transformacion(self):
        """Muestra ayuda para transformar ecuaciones"""
        limpiar_pantalla()
        mostrar_titulo_principal("Punto Fijo", "Ayuda para Transformaciones")
        
        console.print(Panel(
            "[bold cyan]Métodos para transformar f(x) = 0 a x = g(x):[/bold cyan]\n\n"
            "[yellow]1. Despeje directo:[/yellow]\n"
            "   x - cos(x) = 0 → x = cos(x)\n\n"
            "[yellow]2. Suma/resta de x:[/yellow]\n"
            "   x² - x - 1 = 0 → x = x² - 1 (suma x)\n\n"
            "[yellow]3. División:[/yellow]\n"
            "   x² - a = 0 → x = a/x\n\n"
            "[yellow]4. Combinación lineal:[/yellow]\n"
            "   f(x) = 0 → x = x + αf(x)\n"
            "   (α pequeño para convergencia)\n\n"
            "[yellow]5. Función inversa:[/yellow]\n"
            "   e^x - 2 = 0 → x = ln(2)\n\n"
            "[red]⚠️ No todas las transformaciones convergen[/red]",
            title="🔧 Guía de Transformación",
            border_style="green"
        ))
        
        esperar_enter()
    
    def _analizar_convergencia_inicial(self):
        """Analiza convergencia en algunos puntos de prueba"""
        if self.derivada_g is None:
            return
        
        console.print("\n[yellow]Análisis de convergencia en puntos de prueba:[/yellow]")
        
        puntos_prueba = [-2, -1, 0, 1, 2]
        x = sp.Symbol('x')
        
        for punto in puntos_prueba:
            try:
                derivada_val = float(self.derivada_g.subs(x, punto))
                if abs(derivada_val) < 1:
                    estado = f"[green]✓ Converge (|g'| = {abs(derivada_val):.3f})[/green]"
                else:
                    estado = f"[red]✗ Diverge (|g'| = {abs(derivada_val):.3f})[/red]"
                
                console.print(f"  x = {punto}: {estado}")
            except:
                console.print(f"  x = {punto}: [dim]No evaluable[/dim]")
    
    def _graficar_g_y_identidad(self):
        """Grafica g(x) y la función identidad y=x"""
        try:
            x_vals = np.linspace(-10, 10, 1000)
            x_sym = sp.Symbol('x')
            
            # Evaluar g(x)
            g_lambdified = sp.lambdify(x_sym, self.funcion_g, 'numpy')
            
            # Filtrar valores válidos
            g_vals = []
            x_validos = []
            
            for x_val in x_vals:
                try:
                    g_val = g_lambdified(x_val)
                    if np.isfinite(g_val) and not np.isnan(g_val):
                        g_vals.append(g_val)
                        x_validos.append(x_val)
                except:
                    pass
            
            if len(g_vals) > 0:
                plt.figure(figsize=(12, 8))
                plt.plot(x_validos, g_vals, 'b-', linewidth=2, label=f'g(x) = {self.funcion_g_str}')
                plt.plot(x_vals, x_vals, 'r--', linewidth=2, label='y = x')
                
                # Encontrar intersecciones aproximadas
                intersecciones = []
                for i in range(len(x_validos)-1):
                    if len(g_vals) > i+1:
                        if (g_vals[i] - x_validos[i]) * (g_vals[i+1] - x_validos[i+1]) < 0:
                            intersecciones.append(x_validos[i])
                
                # Marcar puntos fijos aproximados
                for inter in intersecciones[:5]:  # Máximo 5 puntos
                    g_inter = g_lambdified(inter)
                    plt.plot(inter, g_inter, 'go', markersize=8, 
                            label=f'Punto fijo ≈ {inter:.2f}')
                
                plt.xlabel('x')
                plt.ylabel('y')
                plt.title('g(x) vs y = x - Búsqueda de Puntos Fijos')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.xlim(-10, 10)
                plt.ylim(-10, 10)
                plt.show()
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al graficar: {e}")
    
    def ingresar_valor_inicial(self):
        """Menú para ingresar el valor inicial x₀"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Punto Fijo", "Valor Inicial x₀")
        
        if self.funcion_g is None:
            mostrar_mensaje_error("Primero debe ingresar la función g(x)")
            esperar_enter()
            return
        
        console.print(Panel(
            "[bold cyan]Ingrese el valor inicial x₀[/bold cyan]\n\n"
            "[yellow]Consideraciones importantes:[/yellow]\n"
            "• x₀ debe estar cerca del punto fijo esperado\n"
            "• |g'(x₀)| < 1 para garantizar convergencia\n"
            "• Pruebe diferentes valores si no converge\n"
            "• El dominio de g(x) debe incluir x₀",
            title="📍 Selección de Valor Inicial",
            border_style="yellow"
        ))
        
        # Mostrar gráfica para ayudar a elegir x₀
        if confirmar_accion("¿Desea ver la gráfica para elegir un buen x₀?"):
            try:
                self._graficar_g_y_identidad()
            except Exception as e:
                mostrar_mensaje_error(f"Error al graficar: {e}")
        
        while True:
            self.x0 = validar_numero("Ingrese el valor inicial x₀", "float")
            
            # Evaluar g(x₀) y g'(x₀)
            try:
                x = sp.Symbol('x')
                gx0 = float(self.funcion_g.subs(x, self.x0))
                
                console.print(f"\n[cyan]g(x₀) = g({self.x0}) = {formatear_numero(gx0)}[/cyan]")
                
                if self.derivada_g:
                    try:
                        gpx0 = float(self.derivada_g.subs(x, self.x0))
                        console.print(f"[cyan]g'(x₀) = g'({self.x0}) = {formatear_numero(gpx0)}[/cyan]")
                        
                        if abs(gpx0) >= 1:
                            mostrar_mensaje_error(
                                f"⚠️ |g'(x₀)| = {abs(gpx0):.3f} ≥ 1\n"
                                "El método puede no converger desde este punto."
                            )
                            if not confirmar_accion("¿Desea continuar de todas formas?"):
                                continue
                        else:
                            console.print(f"[green]✓ |g'(x₀)| = {abs(gpx0):.3f} < 1 - Buena convergencia esperada[/green]")
                    except:
                        console.print("[yellow]No se pudo evaluar g'(x₀)[/yellow]")
                
                # Mostrar primera iteración
                console.print(f"[green]Primera iteración: x₁ = g(x₀) = {formatear_numero(gx0)}[/green]")
                
                mostrar_mensaje_exito(f"Valor inicial configurado: x₀ = {self.x0}")
                esperar_enter()
                break
                
            except Exception as e:
                mostrar_mensaje_error(f"Error al evaluar g(x₀): {e}")
                if not confirmar_accion("¿Desea intentar con otro valor?"):
                    break
    
    def configurar_parametros(self):
        """Menú para configurar tolerancia y máximo de iteraciones"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Punto Fijo", "Configuración de Parámetros")
        
        console.print(Panel(
            f"[bold cyan]Configuración actual:[/bold cyan]\n\n"
            f"Tolerancia: {self.tolerancia}\n"
            f"Máximo de iteraciones: {self.max_iteraciones}",
            title="⚙️ Parámetros Actuales",
            border_style="cyan"
        ))
        
        opciones = [
            "Cambiar tolerancia",
            "Cambiar máximo de iteraciones",
            "Restaurar valores por defecto",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Configurar parámetros", False)
        opcion = validar_opcion_menu([1, 2, 3, 4])
        
        if opcion == 1:
            console.print("\n[yellow]Tolerancia para |x_{n+1} - x_n|[/yellow]")
            nueva_tol = validar_numero(
                f"Nueva tolerancia (actual: {self.tolerancia})", 
                "float", 
                min_val=1e-15, 
                max_val=1.0,
                excluir_cero=True
            )
            
            es_valida, error = validar_tolerancia(nueva_tol)
            if es_valida:
                self.tolerancia = nueva_tol
                mostrar_mensaje_exito(f"Tolerancia actualizada: {nueva_tol}")
            else:
                mostrar_mensaje_error(error)
                
        elif opcion == 2:
            console.print("\n[yellow]Máximo número de iteraciones antes de parar[/yellow]")
            nuevo_max = validar_numero(
                f"Nuevo máximo (actual: {self.max_iteraciones})", 
                "int", 
                min_val=1, 
                max_val=10000
            )
            
            es_valido, error = validar_max_iteraciones(nuevo_max)
            if es_valido:
                self.max_iteraciones = nuevo_max
                mostrar_mensaje_exito(f"Máximo de iteraciones actualizado: {nuevo_max}")
            else:
                mostrar_mensaje_error(error)
                
        elif opcion == 3:
            self.tolerancia = 1e-6
            self.max_iteraciones = 100
            mostrar_mensaje_exito("Parámetros restaurados a valores por defecto")
            
        if opcion != 4:
            esperar_enter()
    
    def ejecutar_metodo(self):
        """Ejecuta el método de punto fijo"""
        if not self.configuracion_completa():
            mostrar_mensaje_error("La configuración no está completa")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Punto Fijo", "Ejecutando Algoritmo")
        
        self.mostrar_configuracion()
        
        if not confirmar_accion("¿Desea ejecutar el método con esta configuración?"):
            return
        
        mostrar_progreso_ejecucion("Ejecutando método de punto fijo...")
        
        # Inicializar variables
        x = sp.Symbol('x')
        x_actual = float(self.x0)
        iteraciones = []
        valores_x = []
        valores_gx = []
        errores_abs = []
        errores_rel = []
        
        tiempo_inicio = time.time()
        
        # Barra de progreso
        with tqdm(total=self.max_iteraciones, desc="Iteraciones", unit="iter") as pbar:
            for i in range(self.max_iteraciones):
                try:
                    # Evaluar g(x_n)
                    gx = float(self.funcion_g.subs(x, x_actual))
                    
                    # Verificar que g(x) sea finito
                    if not np.isfinite(gx):
                        console.print(f"\n[red]❌ g(x) no es finito en iteración {i+1}[/red]")
                        break
                    
                    # Calcular errores
                    if i > 0:
                        error_abs = abs(gx - x_actual)
                        error_rel = error_abs / abs(gx) if abs(gx) > 1e-15 else error_abs
                    else:
                        error_abs = abs(gx - x_actual)
                        error_rel = error_abs
                    
                    # Guardar datos de la iteración
                    iteraciones.append(i + 1)
                    valores_x.append(x_actual)
                    valores_gx.append(gx)
                    errores_abs.append(error_abs)
                    errores_rel.append(error_rel)
                    
                    # Verificar convergencia
                    if error_abs < self.tolerancia:
                        # Agregar último punto
                        valores_x.append(gx)
                        break
                    
                    # Verificar divergencia
                    if abs(gx) > 1e10:
                        console.print(f"\n[red]❌ El método está divergiendo[/red]")
                        break
                    
                    # Verificar oscilación
                    if i > 10 and len(set(round(val, 6) for val in valores_x[-5:])) < 3:
                        console.print(f"\n[yellow]⚠️ Posible ciclo detectado[/yellow]")
                    
                    x_actual = gx
                    pbar.update(1)
                    
                except Exception as e:
                    console.print(f"\n[red]❌ Error numérico en iteración {i+1}: {e}[/red]")
                    break
        
        tiempo_ejecucion = time.time() - tiempo_inicio
        
        # Verificar si se encontró solución
        if len(valores_x) > len(iteraciones):
            # Se agregó el último punto (convergencia)
            punto_fijo = valores_x[-1]
            convergencia = True
        else:
            # No hubo convergencia
            punto_fijo = valores_x[-1] if valores_x else self.x0
            convergencia = False
        
        # Calcular g(punto_fijo) para verificar
        try:
            g_punto_fijo = float(self.funcion_g.subs(x, punto_fijo))
        except:
            g_punto_fijo = float('nan')
        
        # Preparar resultados
        self.resultados = {
            "punto_fijo": punto_fijo,
            "g_punto_fijo": g_punto_fijo,
            "error_punto_fijo": abs(punto_fijo - g_punto_fijo),
            "iteraciones": len(iteraciones),
            "error_final": errores_abs[-1] if errores_abs else float('inf'),
            "convergencia": convergencia,
            "tolerancia": self.tolerancia,
            "valor_inicial": self.x0,
            "tiempo_ejecucion": tiempo_ejecucion,
            "iteraciones_datos": {
                "numeros": iteraciones,
                "x_vals": valores_x,
                "gx_vals": valores_gx,
                "errores_abs": errores_abs,
                "errores_rel": errores_rel
            }
        }
        
        if convergencia:
            mostrar_mensaje_exito("¡Método convergió exitosamente!")
        else:
            mostrar_mensaje_error("El método no convergió. Pruebe otro valor inicial o transformación.")
        
        esperar_enter()
    
    def mostrar_resultados(self):
        """Muestra los resultados del método"""
        if self.resultados is None:
            mostrar_mensaje_error("No hay resultados disponibles. Ejecute el método primero.")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Punto Fijo", "Resultados")
        
        # Mostrar resultado principal
        resultado_display = {
            "punto_fijo": self.resultados["punto_fijo"],
            "g_punto_fijo": self.resultados["g_punto_fijo"],
            "error_punto_fijo": self.resultados["error_punto_fijo"],
            "iteraciones": self.resultados["iteraciones"],
            "error_final": self.resultados["error_final"],
            "convergencia": self.resultados["convergencia"],
            "tolerancia_usada": self.resultados["tolerancia"],
            "valor_inicial": self.resultados["valor_inicial"]
        }
        
        mostrar_resultado_final("Punto Fijo", resultado_display, self.resultados["tiempo_ejecucion"])
        
        # Menú de opciones para resultados
        opciones = [
            "Ver tabla de iteraciones",
            "Ver gráfica de convergencia (telaraña)",
            "Ver gráfica de g(x) con punto fijo",
            "Analizar velocidad de convergencia",
            "Exportar resultados",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Opciones de resultados", False)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6])
        
        if opcion == 1:
            self.mostrar_tabla_iteraciones()
        elif opcion == 2:
            self.mostrar_grafica_telarana()
        elif opcion == 3:
            self.mostrar_grafica_punto_fijo()
        elif opcion == 4:
            self.analizar_convergencia()
        elif opcion == 5:
            self.exportar_resultados()
        elif opcion == 6:
            return
    
    def mostrar_tabla_iteraciones(self):
        """Muestra la tabla detallada de iteraciones"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Punto Fijo", "Tabla de Iteraciones")
        
        datos = self.resultados["iteraciones_datos"]
        
        # Crear datos para la tabla
        valores_tabla = []
        for i in range(len(datos["numeros"])):
            valores_tabla.append({
                "x_n": datos["x_vals"][i],
                "g(x_n)": datos["gx_vals"][i],
                "|x_{n+1} - x_n|": datos["errores_abs"][i],
                "Error Relativo": datos["errores_rel"][i]
            })
        
        tabla = crear_tabla_iteraciones(
            datos["numeros"], 
            valores_tabla,
            "Iteraciones del Método de Punto Fijo"
        )
        
        console.print(tabla)
        esperar_enter()
    
    def mostrar_grafica_telarana(self):
        """Muestra la gráfica de telaraña del proceso iterativo"""
        console.print("[yellow]Generando gráfica de telaraña...[/yellow]")
        
        datos = self.resultados["iteraciones_datos"]
        
        try:
            # Determinar rango apropiado
            x_vals = datos["x_vals"]
            if len(x_vals) > 1:
                x_min, x_max = min(x_vals), max(x_vals)
                margen = max(abs(x_max - x_min), 2.0)
                x_centro = (x_min + x_max) / 2
                rango = (x_centro - margen, x_centro + margen)
            else:
                rango = (self.x0 - 5, self.x0 + 5)
            
            graficar_punto_fijo(
                self.funcion_g,
                datos["x_vals"][:min(10, len(datos["x_vals"]))],  # Máximo 10 iteraciones para claridad
                rango,
                self.resultados["punto_fijo"]
            )
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {e}")
        
        esperar_enter()
    
    def mostrar_grafica_punto_fijo(self):
        """Muestra g(x) con el punto fijo encontrado"""
        console.print("[yellow]Generando gráfica de g(x) con punto fijo...[/yellow]")
        
        try:
            punto_fijo = self.resultados["punto_fijo"]
            x = sp.Symbol('x')
            g_punto_fijo = float(self.funcion_g.subs(x, punto_fijo))
            
            # Rango de graficación alrededor del punto fijo
            margen = max(abs(punto_fijo - self.x0), 2.0)
            rango = (punto_fijo - margen, punto_fijo + margen)
            
            puntos_especiales = [
                (self.x0, float(self.funcion_g.subs(x, self.x0)), f"x₀ = {formatear_numero(self.x0)}", "blue"),
                (punto_fijo, g_punto_fijo, f"Punto fijo: {formatear_numero(punto_fijo)}", "red")
            ]
            
            graficar_funcion(
                self.funcion_g,
                rango,
                f"g(x) = {self.funcion_g_str} - Punto Fijo",
                puntos_especiales
            )
            
            # También mostrar y = x
            x_vals = np.linspace(rango[0], rango[1], 100)
            plt.plot(x_vals, x_vals, 'g--', linewidth=1, alpha=0.7, label='y = x')
            plt.legend()
            plt.show()
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {e}")
        
        esperar_enter()
    
    def analizar_convergencia(self):
        """Analiza la velocidad de convergencia del método"""
        limpiar_pantalla()
        mostrar_titulo_principal("Punto Fijo", "Análisis de Convergencia")
        
        datos = self.resultados["iteraciones_datos"]
        
        if len(datos["errores_abs"]) < 3:
            mostrar_mensaje_error("Se necesitan al menos 3 iteraciones para el análisis")
            esperar_enter()
            return
        
        # Calcular razón de convergencia
        errores = datos["errores_abs"]
        razones = []
        
        for i in range(1, len(errores)):
            if errores[i-1] > 0:
                razon = errores[i] / errores[i-1]
                razones.append(razon)
        
        # Estimar derivada de g en el punto fijo
        derivada_punto_fijo = None
        if self.derivada_g:
            try:
                x = sp.Symbol('x')
                derivada_punto_fijo = float(self.derivada_g.subs(x, self.resultados["punto_fijo"]))
            except:
                pass
        
        # Mostrar análisis
        console.print(Panel(
            f"[bold cyan]Análisis de Convergencia[/bold cyan]\n\n"
            f"Número de iteraciones: {len(datos['numeros'])}\n"
            f"Error final: {formatear_numero(self.resultados['error_final'])}\n"
            f"Convergencia: {'Sí' if self.resultados['convergencia'] else 'No'}\n\n"
            f"[yellow]Razón de convergencia promedio:[/yellow]\n"
            f"r ≈ {np.mean(razones):.4f}" + 
            (f" (teórico: |g'(α)| = {abs(derivada_punto_fijo):.4f})" if derivada_punto_fijo else "") +
            f"\n\n[yellow]Interpretación:[/yellow]\n" +
            ("Convergencia lineal rápida" if np.mean(razones) < 0.1 else
             "Convergencia lineal moderada" if np.mean(razones) < 0.5 else
             "Convergencia lenta" if np.mean(razones) < 1.0 else
             "No converge o diverge") if razones else "No calculable",
            title="📈 Análisis Numérico",
            border_style="green"
        ))
        
        # Mostrar tabla de razones
        if razones:
            table = Table(title="Razón de Convergencia por Iteración", border_style="blue")
            table.add_column("Iteración", style="cyan")
            table.add_column("Error", style="white")
            table.add_column("Razón r_n", style="yellow")
            
            for i, razon in enumerate(razones):
                table.add_row(
                    str(i + 2), 
                    formatear_numero(errores[i+1]), 
                    f"{razon:.4f}"
                )
            
            console.print(table)
        
        esperar_enter()
    
    def exportar_resultados(self):
        """Exporta los resultados a un archivo de texto"""
        try:
            nombre_archivo = f"punto_fijo_resultados_{int(time.time())}.txt"
            
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                f.write("RESULTADOS DEL MÉTODO DE PUNTO FIJO\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Función: g(x) = {self.funcion_g_str}\n")
                f.write(f"Valor inicial: x₀ = {self.x0}\n")
                f.write(f"Tolerancia: {self.tolerancia}\n")
                f.write(f"Máximo de iteraciones: {self.max_iteraciones}\n\n")
                
                f.write("RESULTADO:\n")
                f.write(f"Punto fijo: {formatear_numero(self.resultados['punto_fijo'])}\n")
                f.write(f"g(punto fijo): {formatear_numero(self.resultados['g_punto_fijo'])}\n")
                f.write(f"Error |x - g(x)|: {formatear_numero(self.resultados['error_punto_fijo'])}\n")
                f.write(f"Iteraciones: {self.resultados['iteraciones']}\n")
                f.write(f"Error final: {formatear_numero(self.resultados['error_final'])}\n")
                f.write(f"Convergencia: {'Sí' if self.resultados['convergencia'] else 'No'}\n")
                f.write(f"Tiempo de ejecución: {self.resultados['tiempo_ejecucion']:.4f} segundos\n\n")
                
                f.write("TABLA DE ITERACIONES:\n")
                f.write("-" * 80 + "\n")
                f.write(f"{'Iter':<6} {'x_n':<15} {'g(x_n)':<15} {'|x_{n+1}-x_n|':<15} {'Error Rel':<15}\n")
                f.write("-" * 80 + "\n")
                
                datos = self.resultados["iteraciones_datos"]
                for i in range(len(datos["numeros"])):
                    f.write(f"{datos['numeros'][i]:<6} "
                           f"{formatear_numero(datos['x_vals'][i]):<15} "
                           f"{formatear_numero(datos['gx_vals'][i]):<15} "
                           f"{formatear_numero(datos['errores_abs'][i]):<15} "
                           f"{formatear_numero(datos['errores_rel'][i]):<15}\n")
            
            mostrar_mensaje_exito(f"Resultados exportados a: {nombre_archivo}")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {e}")
        
        esperar_enter()
    
    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        mostrar_ayuda_metodo(
            "Método de Punto Fijo",
            "El método de punto fijo resuelve ecuaciones transformándolas a la forma x = g(x) "
            "y usando iteración x_{n+1} = g(x_n). Converge cuando |g'(x)| < 1 cerca del punto fijo.",
            [
                "Encontrar raíces de ecuaciones no lineales",
                "Resolver ecuaciones trascendentales",
                "Sistemas de ecuaciones (forma vectorial)",
                "Problemas de punto fijo en matemáticas aplicadas"
            ],
            [
                "No requiere derivadas de la función original",
                "Fácil de programar e implementar",
                "Funciona bien cuando converge",
                "Permite múltiples transformaciones"
            ],
            [
                "Convergencia depende de la transformación elegida",
                "Puede no converger o converger lentamente",
                "Requiere transformar f(x)=0 a x=g(x)",
                "Sensible al valor inicial"
            ]
        )

def main():
    """Función principal del programa"""
    metodo = MetodoPuntoFijo()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "MÉTODO DE PUNTO FIJO",
            "Iteración x_{n+1} = g(x_n) para encontrar puntos fijos"
        )
        
        mostrar_banner_metodo(
            "Método de Punto Fijo",
            "Transforma f(x) = 0 a x = g(x) y usa iteración para encontrar puntos fijos"
        )
        
        metodo.mostrar_configuracion()
        
        opciones = [
            "Ingresar función g(x)",
            "Configurar valor inicial x₀",
            "Configurar parámetros (tolerancia, iteraciones)",
            "Ejecutar método de punto fijo",
            "Ver resultados y análisis",
            "Ver ayuda del método",
            "Salir"
        ]
        
        mostrar_menu_opciones(opciones)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6, 7])
        
        if opcion == 1:
            metodo.ingresar_funcion()
        elif opcion == 2:
            metodo.ingresar_valor_inicial()
        elif opcion == 3:
            metodo.configurar_parametros()
        elif opcion == 4:
            metodo.ejecutar_metodo()
        elif opcion == 5:
            metodo.mostrar_resultados()
        elif opcion == 6:
            metodo.mostrar_ayuda()
        elif opcion == 7:
            console.print("\n[green]¡Gracias por usar el método de punto fijo![/green]")
            break

if __name__ == "__main__":
    main()
