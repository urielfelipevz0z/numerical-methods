#!/usr/bin/env python3
"""
Método de Newton-Raphson - Implementación con menús interactivos
Encuentra raíces de ecuaciones no lineales usando tangentes sucesivas
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
    graficar_funcion, graficar_convergencia_newton
)

console = Console()

class MetodoNewtonRaphson:
    def __init__(self):
        self.funcion = None
        self.derivada = None
        self.funcion_str = ""
        self.derivada_str = ""
        self.x0 = None
        self.tolerancia = 1e-8
        self.max_iteraciones = 50
        self.derivada_automatica = True
        self.resultados = None
        
    def configuracion_completa(self) -> bool:
        """Verifica si la configuración está completa"""
        return all([
            self.funcion is not None,
            self.derivada is not None,
            self.x0 is not None,
            self.tolerancia is not None,
            self.max_iteraciones is not None
        ])
    
    def mostrar_configuracion(self):
        """Muestra el estado actual de la configuración"""
        config = {
            "Función": self.funcion_str if self.funcion_str else None,
            "Derivada": self.derivada_str if self.derivada_str else None,
            "Valor inicial x₀": self.x0,
            "Tolerancia": self.tolerancia,
            "Máx. Iteraciones": self.max_iteraciones,
            "Derivada automática": "Sí" if self.derivada_automatica else "No"
        }
        mostrar_estado_configuracion(config)
    
    def ingresar_funcion(self):
        """Menú para ingresar la función"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Newton-Raphson", "Ingreso de Función")
        
        console.print(Panel(
            "[bold cyan]Ingrese la función f(x) = 0[/bold cyan]\n\n"
            "Ejemplos válidos:\n"
            "• x**2 - 4\n"
            "• cos(x) - x\n"
            "• exp(x) - 2*x - 1\n"
            "• x**3 - 2*x - 5\n\n"
            "[yellow]Funciones disponibles:[/yellow]\n"
            "sin, cos, tan, exp, log, ln, sqrt, abs\n\n"
            "[green]💡 La derivada se calculará automáticamente[/green]",
            title="💡 Ayuda para Funciones",
            border_style="blue"
        ))
        
        while True:
            funcion_input = input("\nIngrese f(x): ").strip()
            
            if not funcion_input:
                console.print("[red]❌ La función no puede estar vacía[/red]")
                continue
            
            es_valida, expr, error = validar_funcion(funcion_input)
            
            if es_valida:
                self.funcion = expr
                self.funcion_str = funcion_input
                
                # Calcular derivada automáticamente
                try:
                    x = sp.Symbol('x')
                    self.derivada = sp.diff(self.funcion, x)
                    self.derivada_str = str(self.derivada)
                    self.derivada_automatica = True
                    
                    mostrar_mensaje_exito(f"Función: f(x) = {funcion_input}")
                    console.print(f"[green]Derivada: f'(x) = {self.derivada_str}[/green]")
                    
                    # Preguntar si desea ingresar derivada manualmente
                    if confirmar_accion("¿Desea ingresar la derivada manualmente?"):
                        self.ingresar_derivada_manual()
                    
                    # Mostrar gráfica de la función
                    if confirmar_accion("¿Desea ver la gráfica de la función?"):
                        try:
                            graficar_funcion(self.funcion, (-10, 10), f"f(x) = {self.funcion_str}")
                        except Exception as e:
                            mostrar_mensaje_error(f"Error al graficar: {e}")
                    
                    esperar_enter()
                    break
                    
                except Exception as e:
                    mostrar_mensaje_error(f"Error al calcular la derivada: {e}")
                    if confirmar_accion("¿Desea ingresar la derivada manualmente?"):
                        self.ingresar_derivada_manual()
                        if self.derivada is not None:
                            break
            else:
                mostrar_mensaje_error(f"Error en la función: {error}")
    
    def ingresar_derivada_manual(self):
        """Permite ingresar la derivada manualmente"""
        console.print("\n[yellow]Ingrese la derivada f'(x) manualmente:[/yellow]")
        
        while True:
            derivada_input = input("Ingrese f'(x): ").strip()
            
            if not derivada_input:
                console.print("[red]❌ La derivada no puede estar vacía[/red]")
                continue
            
            es_valida, expr, error = validar_funcion(derivada_input)
            
            if es_valida:
                self.derivada = expr
                self.derivada_str = derivada_input
                self.derivada_automatica = False
                mostrar_mensaje_exito(f"Derivada ingresada: f'(x) = {derivada_input}")
                break
            else:
                mostrar_mensaje_error(f"Error en la derivada: {error}")
    
    def ingresar_valor_inicial(self):
        """Menú para ingresar el valor inicial x₀"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Newton-Raphson", "Valor Inicial x₀")
        
        if self.funcion is None or self.derivada is None:
            mostrar_mensaje_error("Primero debe ingresar la función y su derivada")
            esperar_enter()
            return
        
        console.print(Panel(
            "[bold cyan]Ingrese el valor inicial x₀[/bold cyan]\n\n"
            "[yellow]Consideraciones importantes:[/yellow]\n"
            "• El valor inicial debe estar cerca de la raíz\n"
            "• f'(x₀) ≠ 0 (derivada no debe ser cero)\n"
            "• Evite puntos de inflexión o mínimos/máximos locales\n"
            "• Un buen x₀ garantiza convergencia rápida",
            title="📍 Selección de Valor Inicial",
            border_style="yellow"
        ))
        
        # Mostrar gráfica para ayudar a elegir x₀
        if confirmar_accion("¿Desea ver la gráfica para elegir un buen x₀?"):
            try:
                graficar_funcion(self.funcion, (-10, 10), f"f(x) = {self.funcion_str}")
            except Exception as e:
                mostrar_mensaje_error(f"Error al graficar: {e}")
        
        while True:
            self.x0 = validar_numero("Ingrese el valor inicial x₀", "float")
            
            # Evaluar función y derivada en x₀
            try:
                x = sp.Symbol('x')
                fx0 = float(self.funcion.subs(x, self.x0))
                fpx0 = float(self.derivada.subs(x, self.x0))
                
                console.print(f"\n[cyan]f(x₀) = f({self.x0}) = {formatear_numero(fx0)}[/cyan]")
                console.print(f"[cyan]f'(x₀) = f'({self.x0}) = {formatear_numero(fpx0)}[/cyan]")
                
                # Verificar que la derivada no sea cero
                if abs(fpx0) < 1e-12:
                    mostrar_mensaje_error(
                        "⚠️ La derivada es muy pequeña o cero en este punto.\n"
                        "Esto puede causar problemas de convergencia."
                    )
                    if not confirmar_accion("¿Desea continuar de todas formas?"):
                        continue
                
                # Estimar próxima iteración
                if abs(fpx0) > 1e-12:
                    x1 = self.x0 - fx0 / fpx0
                    console.print(f"[green]Primera iteración: x₁ ≈ {formatear_numero(x1)}[/green]")
                
                mostrar_mensaje_exito(f"Valor inicial configurado: x₀ = {self.x0}")
                esperar_enter()
                break
                
            except Exception as e:
                mostrar_mensaje_error(f"Error al evaluar en x₀: {e}")
    
    def configurar_parametros(self):
        """Menú para configurar tolerancia y máximo de iteraciones"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Newton-Raphson", "Configuración de Parámetros")
        
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
            "Configurar derivada",
            "Restaurar valores por defecto",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Configurar parámetros", False)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5])
        
        if opcion == 1:
            console.print("\n[yellow]Tolerancia para |x_{n+1} - x_n| o |f(x_n)|[/yellow]")
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
                max_val=1000
            )
            
            es_valido, error = validar_max_iteraciones(nuevo_max)
            if es_valido:
                self.max_iteraciones = nuevo_max
                mostrar_mensaje_exito(f"Máximo de iteraciones actualizado: {nuevo_max}")
            else:
                mostrar_mensaje_error(error)
                
        elif opcion == 3:
            if self.funcion is not None:
                self.ingresar_derivada_manual()
            else:
                mostrar_mensaje_error("Primero debe ingresar una función")
                
        elif opcion == 4:
            self.tolerancia = 1e-8
            self.max_iteraciones = 50
            mostrar_mensaje_exito("Parámetros restaurados a valores por defecto")
            
        if opcion != 5:
            esperar_enter()
    
    def ejecutar_metodo(self):
        """Ejecuta el método de Newton-Raphson"""
        if not self.configuracion_completa():
            mostrar_mensaje_error("La configuración no está completa")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Newton-Raphson", "Ejecutando Algoritmo")
        
        self.mostrar_configuracion()
        
        if not confirmar_accion("¿Desea ejecutar el método con esta configuración?"):
            return
        
        mostrar_progreso_ejecucion("Ejecutando método de Newton-Raphson...")
        
        # Inicializar variables
        x = sp.Symbol('x')
        x_actual = float(self.x0)
        iteraciones = []
        valores_x = []
        valores_fx = []
        valores_fpx = []
        errores_abs = []
        errores_rel = []
        
        tiempo_inicio = time.time()
        
        # Barra de progreso
        with tqdm(total=self.max_iteraciones, desc="Iteraciones", unit="iter") as pbar:
            for i in range(self.max_iteraciones):
                # Evaluar función y derivada
                try:
                    fx = float(self.funcion.subs(x, x_actual))
                    fpx = float(self.derivada.subs(x, x_actual))
                    
                    # Verificar derivada no nula
                    if abs(fpx) < 1e-15:
                        console.print(f"\n[red]❌ Derivada nula en iteración {i+1}[/red]")
                        break
                    
                    # Calcular nueva aproximación
                    x_nuevo = x_actual - fx / fpx
                    
                    # Calcular errores
                    error_abs = abs(x_nuevo - x_actual) if i > 0 else abs(fx)
                    error_rel = error_abs / abs(x_nuevo) if abs(x_nuevo) > 1e-15 else error_abs
                    
                    # Guardar datos de la iteración
                    iteraciones.append(i + 1)
                    valores_x.append(x_actual)
                    valores_fx.append(fx)
                    valores_fpx.append(fpx)
                    errores_abs.append(error_abs)
                    errores_rel.append(error_rel)
                    
                    # Verificar convergencia
                    if error_abs < self.tolerancia or abs(fx) < self.tolerancia:
                        # Agregar última iteración
                        valores_x.append(x_nuevo)
                        break
                    
                    # Verificar divergencia
                    if abs(x_nuevo) > 1e10:
                        console.print(f"\n[red]❌ El método está divergiendo[/red]")
                        break
                    
                    x_actual = x_nuevo
                    pbar.update(1)
                    
                except Exception as e:
                    console.print(f"\n[red]❌ Error numérico en iteración {i+1}: {e}[/red]")
                    break
        
        tiempo_ejecucion = time.time() - tiempo_inicio
        
        # Verificar si se encontró solución
        if len(valores_x) > len(iteraciones):
            # Se agregó la última iteración (convergencia)
            x_final = valores_x[-1]
            fx_final = float(self.funcion.subs(x, x_final))
            convergencia = True
        else:
            # No hubo convergencia
            x_final = valores_x[-1] if valores_x else self.x0
            fx_final = valores_fx[-1] if valores_fx else float(self.funcion.subs(x, self.x0))
            convergencia = False
        
        # Preparar resultados
        self.resultados = {
            "raiz": x_final,
            "valor_funcion": fx_final,
            "iteraciones": len(iteraciones),
            "error_final": errores_abs[-1] if errores_abs else float('inf'),
            "convergencia": convergencia,
            "tolerancia": self.tolerancia,
            "valor_inicial": self.x0,
            "tiempo_ejecucion": tiempo_ejecucion,
            "iteraciones_datos": {
                "numeros": iteraciones,
                "x_vals": valores_x,
                "fx_vals": valores_fx,
                "fpx_vals": valores_fpx,
                "errores_abs": errores_abs,
                "errores_rel": errores_rel
            }
        }
        
        if convergencia:
            mostrar_mensaje_exito("¡Método convergió exitosamente!")
        else:
            mostrar_mensaje_error("El método no convergió. Revise el valor inicial.")
        
        esperar_enter()
    
    def mostrar_resultados(self):
        """Muestra los resultados del método"""
        if self.resultados is None:
            mostrar_mensaje_error("No hay resultados disponibles. Ejecute el método primero.")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Newton-Raphson", "Resultados")
        
        # Mostrar resultado principal
        resultado_display = {
            "raiz": self.resultados["raiz"],
            "valor_funcion": self.resultados["valor_funcion"],
            "iteraciones": self.resultados["iteraciones"],
            "error_final": self.resultados["error_final"],
            "convergencia": self.resultados["convergencia"],
            "tolerancia_usada": self.resultados["tolerancia"],
            "valor_inicial": self.resultados["valor_inicial"]
        }
        
        mostrar_resultado_final("Newton-Raphson", resultado_display, self.resultados["tiempo_ejecucion"])
        
        # Menú de opciones para resultados
        opciones = [
            "Ver tabla de iteraciones",
            "Ver gráfica de convergencia",
            "Ver gráfica de la función con la raíz",
            "Analizar velocidad de convergencia",
            "Exportar resultados",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Opciones de resultados", False)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6])
        
        if opcion == 1:
            self.mostrar_tabla_iteraciones()
        elif opcion == 2:
            self.mostrar_grafica_convergencia()
        elif opcion == 3:
            self.mostrar_grafica_funcion_raiz()
        elif opcion == 4:
            self.analizar_convergencia()
        elif opcion == 5:
            self.exportar_resultados()
        elif opcion == 6:
            return
    
    def mostrar_tabla_iteraciones(self):
        """Muestra la tabla detallada de iteraciones"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Newton-Raphson", "Tabla de Iteraciones")
        
        datos = self.resultados["iteraciones_datos"]
        
        # Crear datos para la tabla
        valores_tabla = []
        for i in range(len(datos["numeros"])):
            valores_tabla.append({
                "x_n": datos["x_vals"][i],
                "f(x_n)": datos["fx_vals"][i],
                "f'(x_n)": datos["fpx_vals"][i],
                "Error Abs": datos["errores_abs"][i],
                "Error Rel": datos["errores_rel"][i]
            })
        
        tabla = crear_tabla_iteraciones(
            datos["numeros"], 
            valores_tabla,
            "Iteraciones del Método de Newton-Raphson"
        )
        
        console.print(tabla)
        esperar_enter()
    
    def mostrar_grafica_convergencia(self):
        """Muestra gráficas de convergencia"""
        console.print("[yellow]Generando gráficas de convergencia...[/yellow]")
        
        datos = self.resultados["iteraciones_datos"]
        
        try:
            # Estimar rango para la gráfica de la función
            x_vals = datos["x_vals"]
            if len(x_vals) > 1:
                x_min, x_max = min(x_vals), max(x_vals)
                margen = max(abs(x_max - x_min), 2.0)
                x_centro = (x_min + x_max) / 2
                rango = (x_centro - margen, x_centro + margen)
            else:
                rango = (self.x0 - 5, self.x0 + 5)
            
            graficar_convergencia_newton(
                datos["numeros"],
                datos["x_vals"],
                datos["errores_abs"],
                self.funcion,
                rango
            )
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráficas: {e}")
        
        esperar_enter()
    
    def mostrar_grafica_funcion_raiz(self):
        """Muestra la función con la raíz encontrada"""
        console.print("[yellow]Generando gráfica de la función con la raíz...[/yellow]")
        
        try:
            raiz = self.resultados["raiz"]
            x = sp.Symbol('x')
            f_raiz = float(self.funcion.subs(x, raiz))
            
            # Rango de graficación alrededor de la raíz
            margen = max(abs(raiz - self.x0), 2.0)
            rango = (raiz - margen, raiz + margen)
            
            puntos_especiales = [
                (self.x0, float(self.funcion.subs(x, self.x0)), f"x₀ = {formatear_numero(self.x0)}", "blue"),
                (raiz, f_raiz, f"Raíz: {formatear_numero(raiz)}", "red")
            ]
            
            graficar_funcion(
                self.funcion,
                rango,
                f"f(x) = {self.funcion_str} - Convergencia Newton-Raphson",
                puntos_especiales
            )
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {e}")
        
        esperar_enter()
    
    def analizar_convergencia(self):
        """Analiza la velocidad de convergencia del método"""
        limpiar_pantalla()
        mostrar_titulo_principal("Newton-Raphson", "Análisis de Convergencia")
        
        datos = self.resultados["iteraciones_datos"]
        
        if len(datos["errores_abs"]) < 3:
            mostrar_mensaje_error("Se necesitan al menos 3 iteraciones para el análisis")
            esperar_enter()
            return
        
        # Calcular orden de convergencia
        errores = datos["errores_abs"]
        ordenes = []
        
        for i in range(2, len(errores)):
            if errores[i-1] > 0 and errores[i-2] > 0:
                try:
                    orden = np.log(errores[i] / errores[i-1]) / np.log(errores[i-1] / errores[i-2])
                    ordenes.append(orden)
                except:
                    pass
        
        # Mostrar análisis
        console.print(Panel(
            f"[bold cyan]Análisis de Convergencia[/bold cyan]\n\n"
            f"Número de iteraciones: {len(datos['numeros'])}\n"
            f"Error final: {formatear_numero(self.resultados['error_final'])}\n"
            f"Convergencia: {'Sí' if self.resultados['convergencia'] else 'No'}\n\n"
            f"[yellow]Orden de convergencia estimado:[/yellow]\n"
            f"Promedio: {np.mean(ordenes):.2f} (teórico: 2.0)\n"
            f"Rango: [{min(ordenes):.2f}, {max(ordenes):.2f}]" if ordenes else "No calculable",
            title="📈 Análisis Numérico",
            border_style="green"
        ))
        
        # Mostrar tabla de órdenes
        if ordenes:
            table = Table(title="Orden de Convergencia por Iteración", border_style="blue")
            table.add_column("Iteración", style="cyan")
            table.add_column("Orden", style="white")
            table.add_column("Interpretación", style="yellow")
            
            for i, orden in enumerate(ordenes):
                if orden < 1.5:
                    interpretacion = "Sublineal"
                elif orden < 1.8:
                    interpretacion = "Casi lineal"
                elif orden < 2.2:
                    interpretacion = "Cuadrática"
                else:
                    interpretacion = "Supercuadrática"
                
                table.add_row(str(i + 3), f"{orden:.3f}", interpretacion)
            
            console.print(table)
        
        esperar_enter()
    
    def exportar_resultados(self):
        """Exporta los resultados a un archivo de texto"""
        try:
            nombre_archivo = f"newton_raphson_resultados_{int(time.time())}.txt"
            
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                f.write("RESULTADOS DEL MÉTODO DE NEWTON-RAPHSON\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Función: f(x) = {self.funcion_str}\n")
                f.write(f"Derivada: f'(x) = {self.derivada_str}\n")
                f.write(f"Valor inicial: x₀ = {self.x0}\n")
                f.write(f"Tolerancia: {self.tolerancia}\n")
                f.write(f"Máximo de iteraciones: {self.max_iteraciones}\n\n")
                
                f.write("RESULTADO:\n")
                f.write(f"Raíz encontrada: {formatear_numero(self.resultados['raiz'])}\n")
                f.write(f"f(raíz): {formatear_numero(self.resultados['valor_funcion'])}\n")
                f.write(f"Iteraciones: {self.resultados['iteraciones']}\n")
                f.write(f"Error final: {formatear_numero(self.resultados['error_final'])}\n")
                f.write(f"Convergencia: {'Sí' if self.resultados['convergencia'] else 'No'}\n")
                f.write(f"Tiempo de ejecución: {self.resultados['tiempo_ejecucion']:.4f} segundos\n\n")
                
                f.write("TABLA DE ITERACIONES:\n")
                f.write("-" * 90 + "\n")
                f.write(f"{'Iter':<6} {'x_n':<15} {'f(x_n)':<15} {'f\\'(x_n)':<15} {'Error Abs':<15} {'Error Rel':<15}\n")
                f.write("-" * 90 + "\n")
                
                datos = self.resultados["iteraciones_datos"]
                for i in range(len(datos["numeros"])):
                    f.write(f"{datos['numeros'][i]:<6} "
                           f"{formatear_numero(datos['x_vals'][i]):<15} "
                           f"{formatear_numero(datos['fx_vals'][i]):<15} "
                           f"{formatear_numero(datos['fpx_vals'][i]):<15} "
                           f"{formatear_numero(datos['errores_abs'][i]):<15} "
                           f"{formatear_numero(datos['errores_rel'][i]):<15}\n")
            
            mostrar_mensaje_exito(f"Resultados exportados a: {nombre_archivo}")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {e}")
        
        esperar_enter()
    
    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        mostrar_ayuda_metodo(
            "Método de Newton-Raphson",
            "El método de Newton-Raphson utiliza la tangente a la curva para aproximar "
            "sucesivamente la raíz. Tiene convergencia cuadrática cuando converge, "
            "pero requiere que la derivada no sea cero y un buen valor inicial.",
            [
                "Encontrar raíces de ecuaciones no lineales",
                "Resolver ecuaciones trascendentales complejas",
                "Optimización (encontrar puntos críticos)",
                "Sistemas no lineales (versión multivariable)"
            ],
            [
                "Convergencia cuadrática muy rápida",
                "Pocas iteraciones necesarias",
                "Funciona bien con funciones suaves",
                "Fácil de implementar para una variable"
            ],
            [
                "Requiere calcular la derivada",
                "Sensible al valor inicial",
                "Puede diverger o ciclar",
                "Problemas cuando f'(x) = 0"
            ]
        )

def main():
    """Función principal del programa"""
    metodo = MetodoNewtonRaphson()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "MÉTODO DE NEWTON-RAPHSON",
            "Búsqueda de raíces usando tangentes sucesivas"
        )
        
        mostrar_banner_metodo(
            "Método de Newton-Raphson",
            "Encuentra raíces usando la fórmula: x_{n+1} = x_n - f(x_n)/f'(x_n)"
        )
        
        metodo.mostrar_configuracion()
        
        opciones = [
            "Ingresar función f(x) y derivada",
            "Configurar valor inicial x₀",
            "Configurar parámetros (tolerancia, iteraciones)",
            "Ejecutar método de Newton-Raphson",
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
            console.print("\n[green]¡Gracias por usar el método de Newton-Raphson![/green]")
            break

if __name__ == "__main__":
    main()
