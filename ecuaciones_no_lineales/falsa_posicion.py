#!/usr/bin/env python3
"""
Método de Falsa Posición (Regula Falsi) - Implementación con menús interactivos
Encuentra raíces usando interpolación lineal entre puntos con signos opuestos
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
    validar_numero, validar_opcion_menu, validar_funcion, validar_intervalo,
    validar_tolerancia, validar_max_iteraciones, confirmar_accion,
    limpiar_pantalla, mostrar_titulo_principal, mostrar_menu_opciones,
    mostrar_banner_metodo, mostrar_estado_configuracion, mostrar_progreso_ejecucion,
    mostrar_resultado_final, mostrar_mensaje_error, mostrar_mensaje_exito,
    mostrar_ayuda_metodo, esperar_enter, crear_tabla_iteraciones, formatear_numero,
    graficar_funcion
)

console = Console()

class MetodoFalsaPosicion:
    def __init__(self):
        self.funcion = None
        self.funcion_str = ""
        self.a = None
        self.b = None
        self.tolerancia = 1e-6
        self.max_iteraciones = 100
        self.metodo_modificado = False  # Para Illinois/Anderson-Björck
        self.resultados = None
        
    def configuracion_completa(self) -> bool:
        """Verifica si la configuración está completa"""
        return all([
            self.funcion is not None,
            self.a is not None,
            self.b is not None,
            self.tolerancia is not None,
            self.max_iteraciones is not None
        ])
    
    def mostrar_configuracion(self):
        """Muestra el estado actual de la configuración"""
        config = {
            "Función": self.funcion_str if self.funcion_str else None,
            "Intervalo [a,b]": f"[{self.a}, {self.b}]" if self.a is not None and self.b is not None else None,
            "Tolerancia": self.tolerancia,
            "Máx. Iteraciones": self.max_iteraciones,
            "Método modificado": "Sí (Illinois)" if self.metodo_modificado else "No (Clásico)"
        }
        mostrar_estado_configuracion(config)
    
    def ingresar_funcion(self):
        """Menú para ingresar la función"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Falsa Posición", "Ingreso de Función")
        
        console.print(Panel(
            "[bold cyan]Ingrese la función f(x) = 0[/bold cyan]\n\n"
            "Ejemplos válidos:\n"
            "• x**3 - 2*x - 5\n"
            "• sin(x) - 0.5\n"
            "• exp(x) - 3*x\n"
            "• x**2 - cos(x)\n\n"
            "[yellow]Funciones disponibles:[/yellow]\n"
            "sin, cos, tan, exp, log, ln, sqrt, abs\n\n"
            "[green]💡 Se requiere un intervalo [a,b] donde f(a)·f(b) < 0[/green]",
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
                mostrar_mensaje_exito(f"Función ingresada: f(x) = {funcion_input}")
                
                # Mostrar gráfica de la función
                if confirmar_accion("¿Desea ver la gráfica de la función?"):
                    try:
                        # Estimar rango automáticamente o usar uno por defecto
                        rango = (-10, 10)
                        if self.a is not None and self.b is not None:
                            margen = abs(self.b - self.a)
                            rango = (self.a - margen, self.b + margen)
                        
                        graficar_funcion(self.funcion, rango, f"f(x) = {self.funcion_str}")
                    except Exception as e:
                        mostrar_mensaje_error(f"Error al graficar: {e}")
                
                esperar_enter()
                break
            else:
                mostrar_mensaje_error(f"Error en la función: {error}")
    
    def ingresar_intervalo(self):
        """Menú para ingresar el intervalo [a,b]"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Falsa Posición", "Configuración de Intervalo")
        
        if self.funcion is None:
            mostrar_mensaje_error("Primero debe ingresar una función")
            esperar_enter()
            return
        
        console.print(Panel(
            "[bold cyan]Ingrese el intervalo [a,b] donde buscar la raíz[/bold cyan]\n\n"
            "[yellow]Requisitos:[/yellow]\n"
            "• a < b\n"
            "• f(a) y f(b) deben tener signos opuestos\n"
            "• La función debe ser continua en [a,b]\n\n"
            "[green]💡 Falsa posición converge más rápido que bisección[/green]",
            title="📏 Configuración de Intervalo",
            border_style="yellow"
        ))
        
        while True:
            self.a = validar_numero("Ingrese el extremo izquierdo (a)", "float")
            self.b = validar_numero("Ingrese el extremo derecho (b)", "float")
            
            # Validar intervalo
            es_valido, error = validar_intervalo(self.a, self.b)
            if not es_valido:
                mostrar_mensaje_error(error)
                continue
            
            # Evaluar función en los extremos
            try:
                x = sp.Symbol('x')
                fa = float(self.funcion.subs(x, self.a))
                fb = float(self.funcion.subs(x, self.b))
                
                console.print(f"\n[cyan]f({self.a}) = {formatear_numero(fa)}[/cyan]")
                console.print(f"[cyan]f({self.b}) = {formatear_numero(fb)}[/cyan]")
                
                # Verificar cambio de signo
                if fa * fb > 0:
                    mostrar_mensaje_error(
                        "f(a) y f(b) tienen el mismo signo.\n"
                        "No se garantiza que exista una raíz en este intervalo."
                    )
                    if not confirmar_accion("¿Desea continuar de todas formas?"):
                        continue
                else:
                    mostrar_mensaje_exito("✅ Se detectó cambio de signo - hay al menos una raíz")
                
                # Calcular primera aproximación
                if abs(fb - fa) > 1e-15:
                    c_primera = self.a - fa * (self.b - self.a) / (fb - fa)
                    fc_primera = float(self.funcion.subs(x, c_primera))
                    console.print(f"[green]Primera aproximación: c₁ = {formatear_numero(c_primera)}[/green]")
                    console.print(f"[green]f(c₁) = {formatear_numero(fc_primera)}[/green]")
                
                # Mostrar gráfica del intervalo
                if confirmar_accion("¿Desea ver la gráfica en el intervalo?"):
                    try:
                        margen = abs(self.b - self.a) * 0.2
                        rango = (self.a - margen, self.b + margen)
                        puntos_especiales = [
                            (self.a, fa, f"a={self.a}", "red"),
                            (self.b, fb, f"b={self.b}", "green")
                        ]
                        
                        # Agregar primera aproximación si existe
                        if 'c_primera' in locals():
                            puntos_especiales.append((c_primera, fc_primera, f"c₁={formatear_numero(c_primera)}", "orange"))
                        
                        graficar_funcion(
                            self.funcion, 
                            rango, 
                            f"f(x) = {self.funcion_str} en [{self.a}, {self.b}]",
                            puntos_especiales
                        )
                    except Exception as e:
                        mostrar_mensaje_error(f"Error al graficar: {e}")
                
                esperar_enter()
                break
                
            except Exception as e:
                mostrar_mensaje_error(f"Error al evaluar la función: {e}")
    
    def configurar_parametros(self):
        """Menú para configurar tolerancia y parámetros del método"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Falsa Posición", "Configuración de Parámetros")
        
        console.print(Panel(
            f"[bold cyan]Configuración actual:[/bold cyan]\n\n"
            f"Tolerancia: {self.tolerancia}\n"
            f"Máximo de iteraciones: {self.max_iteraciones}\n"
            f"Método modificado: {'Sí (Illinois)' if self.metodo_modificado else 'No (Clásico)'}",
            title="⚙️ Parámetros Actuales",
            border_style="cyan"
        ))
        
        opciones = [
            "Cambiar tolerancia",
            "Cambiar máximo de iteraciones",
            "Alternar método modificado (Illinois)",
            "Restaurar valores por defecto",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Configurar parámetros", False)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5])
        
        if opcion == 1:
            console.print("\n[yellow]La tolerancia determina cuándo parar las iteraciones[/yellow]")
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
            self.metodo_modificado = not self.metodo_modificado
            estado = "Illinois (modificado)" if self.metodo_modificado else "Clásico"
            mostrar_mensaje_exito(f"Método cambiado a: {estado}")
            
            console.print(Panel(
                "[yellow]Método Illinois:[/yellow]\n"
                "Modifica los valores de función para evitar convergencia lenta\n"
                "cuando uno de los extremos se mantiene fijo.\n\n"
                "[cyan]Ventajas:[/cyan] Converge más rápido en casos problemáticos\n"
                "[cyan]Desventajas:[/cyan] Ligeramente más complejo",
                title="ℹ️ Información del Método",
                border_style="blue"
            ))
                
        elif opcion == 4:
            self.tolerancia = 1e-6
            self.max_iteraciones = 100
            self.metodo_modificado = False
            mostrar_mensaje_exito("Parámetros restaurados a valores por defecto")
            
        if opcion != 5:
            esperar_enter()
    
    def ejecutar_metodo(self):
        """Ejecuta el método de falsa posición"""
        if not self.configuracion_completa():
            mostrar_mensaje_error("La configuración no está completa")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Falsa Posición", "Ejecutando Algoritmo")
        
        self.mostrar_configuracion()
        
        if not confirmar_accion("¿Desea ejecutar el método con esta configuración?"):
            return
        
        nombre_metodo = "Illinois" if self.metodo_modificado else "Falsa Posición Clásica"
        mostrar_progreso_ejecucion(f"Ejecutando método de {nombre_metodo.lower()}...")
        
        # Inicializar variables
        x = sp.Symbol('x')
        a, b = float(self.a), float(self.b)
        iteraciones = []
        valores_a = []
        valores_b = []
        valores_c = []
        valores_fa = []
        valores_fb = []
        valores_fc = []
        errores = []
        
        tiempo_inicio = time.time()
        
        # Evaluar función en extremos iniciales
        fa = float(self.funcion.subs(x, a))
        fb = float(self.funcion.subs(x, b))
        
        # Variables para método Illinois
        fa_original = fa
        fb_original = fb
        lado_anterior = 0  # -1: lado izquierdo, +1: lado derecho, 0: inicial
        
        # Barra de progreso
        with tqdm(total=self.max_iteraciones, desc="Iteraciones", unit="iter") as pbar:
            for i in range(self.max_iteraciones):
                # Calcular nueva aproximación usando interpolación lineal
                if abs(fb - fa) < 1e-15:
                    console.print(f"\n[red]❌ Denominador muy pequeño en iteración {i+1}[/red]")
                    break
                
                c = a - fa * (b - a) / (fb - fa)
                fc = float(self.funcion.subs(x, c))
                
                # Calcular error
                if i > 0:
                    error = abs(c - valores_c[-1])
                else:
                    error = abs(b - a) / 2.0
                
                # Guardar datos de la iteración
                iteraciones.append(i + 1)
                valores_a.append(a)
                valores_b.append(b)
                valores_c.append(c)
                valores_fa.append(fa)
                valores_fb.append(fb)
                valores_fc.append(fc)
                errores.append(error)
                
                # Verificar convergencia
                if error < self.tolerancia or abs(fc) < self.tolerancia:
                    break
                
                # Actualizar intervalo
                lado_actual = 0
                if fa * fc < 0:
                    # La raíz está entre a y c
                    b = c
                    fb_original = fc
                    lado_actual = -1  # Se movió el lado derecho
                    
                    # Método Illinois: modificar fa si el mismo lado se mantiene
                    if self.metodo_modificado and lado_anterior == lado_actual:
                        fa = fa / 2.0
                    else:
                        fa = fa_original
                else:
                    # La raíz está entre c y b
                    a = c
                    fa_original = fc
                    lado_actual = 1  # Se movió el lado izquierdo
                    
                    # Método Illinois: modificar fb si el mismo lado se mantiene
                    if self.metodo_modificado and lado_anterior == lado_actual:
                        fb = fb / 2.0
                    else:
                        fb = fb_original
                
                lado_anterior = lado_actual
                pbar.update(1)
        
        tiempo_ejecucion = time.time() - tiempo_inicio
        
        # Preparar resultados
        self.resultados = {
            "raiz": c if 'c' in locals() else (a + b) / 2,
            "valor_funcion": fc if 'fc' in locals() else 0,
            "iteraciones": len(iteraciones),
            "error_final": error if 'error' in locals() else float('inf'),
            "convergencia": error < self.tolerancia if 'error' in locals() else False,
            "tolerancia": self.tolerancia,
            "intervalo_final": [a, b],
            "metodo_usado": nombre_metodo,
            "tiempo_ejecucion": tiempo_ejecucion,
            "iteraciones_datos": {
                "numeros": iteraciones,
                "a_vals": valores_a,
                "b_vals": valores_b,
                "c_vals": valores_c,
                "fa_vals": valores_fa,
                "fb_vals": valores_fb,
                "fc_vals": valores_fc,
                "errores": errores
            }
        }
        
        mostrar_mensaje_exito("¡Método ejecutado exitosamente!")
        esperar_enter()
    
    def mostrar_resultados(self):
        """Muestra los resultados del método"""
        if self.resultados is None:
            mostrar_mensaje_error("No hay resultados disponibles. Ejecute el método primero.")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Falsa Posición", "Resultados")
        
        # Mostrar resultado principal
        resultado_display = {
            "raiz": self.resultados["raiz"],
            "valor_funcion": self.resultados["valor_funcion"],
            "iteraciones": self.resultados["iteraciones"],
            "error_final": self.resultados["error_final"],
            "convergencia": self.resultados["convergencia"],
            "metodo_usado": self.resultados["metodo_usado"],
            "tolerancia_usada": self.resultados["tolerancia"]
        }
        
        mostrar_resultado_final("Falsa Posición", resultado_display, self.resultados["tiempo_ejecucion"])
        
        # Menú de opciones para resultados
        opciones = [
            "Ver tabla de iteraciones",
            "Ver gráfica de convergencia",
            "Ver gráfica de la función con la raíz",
            "Comparar con bisección",
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
            self.comparar_con_biseccion()
        elif opcion == 5:
            self.exportar_resultados()
        elif opcion == 6:
            return
    
    def mostrar_tabla_iteraciones(self):
        """Muestra la tabla detallada de iteraciones"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Falsa Posición", "Tabla de Iteraciones")
        
        datos = self.resultados["iteraciones_datos"]
        
        # Crear datos para la tabla
        valores_tabla = []
        for i in range(len(datos["numeros"])):
            valores_tabla.append({
                "a": datos["a_vals"][i],
                "b": datos["b_vals"][i],
                "c": datos["c_vals"][i],
                "f(a)": datos["fa_vals"][i],
                "f(b)": datos["fb_vals"][i],
                "f(c)": datos["fc_vals"][i],
                "Error": datos["errores"][i]
            })
        
        tabla = crear_tabla_iteraciones(
            datos["numeros"], 
            valores_tabla,
            f"Iteraciones del Método de {self.resultados['metodo_usado']}"
        )
        
        console.print(tabla)
        esperar_enter()
    
    def mostrar_grafica_convergencia(self):
        """Muestra gráficas de convergencia"""
        console.print("[yellow]Generando gráficas de convergencia...[/yellow]")
        
        datos = self.resultados["iteraciones_datos"]
        
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
            
            # Gráfica 1: Evolución de las aproximaciones
            ax1.plot(datos["numeros"], datos["c_vals"], 'bo-', linewidth=2, markersize=6, label='Aproximaciones c_n')
            ax1.axhline(y=self.resultados["raiz"], color='red', linestyle='--', 
                       label=f'Raíz = {formatear_numero(self.resultados["raiz"])}')
            ax1.set_title(f'Convergencia de las Aproximaciones - {self.resultados["metodo_usado"]}', fontweight='bold')
            ax1.set_xlabel('Iteración')
            ax1.set_ylabel('Valor de c_n')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            
            # Gráfica 2: Error absoluto
            ax2.semilogy(datos["numeros"], datos["errores"], 'ro-', linewidth=2, markersize=6)
            ax2.set_title('Convergencia del Error Absoluto', fontweight='bold')
            ax2.set_xlabel('Iteración')
            ax2.set_ylabel('Error Absoluto')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráficas: {e}")
        
        esperar_enter()
    
    def mostrar_grafica_funcion_raiz(self):
        """Muestra la función con la raíz encontrada y el proceso iterativo"""
        console.print("[yellow]Generando gráfica de la función con raíz y proceso...[/yellow]")
        
        try:
            # Rango de graficación
            raiz = self.resultados["raiz"]
            rango_original = abs(self.b - self.a)
            margen = max(rango_original * 0.3, 1.0)
            rango = (raiz - margen, raiz + margen)
            
            # Evaluar función en la raíz
            x = sp.Symbol('x')
            f_raiz = float(self.funcion.subs(x, raiz))
            
            # Crear gráfica
            x_vals = np.linspace(rango[0], rango[1], 1000)
            f_lambdified = sp.lambdify(x, self.funcion, 'numpy')
            y_vals = f_lambdified(x_vals)
            
            plt.figure(figsize=(12, 8))
            plt.plot(x_vals, y_vals, 'b-', linewidth=2, label=f'f(x) = {self.funcion_str}')
            plt.axhline(y=0, color='black', linewidth=0.8, alpha=0.7)
            
            # Marcar puntos del proceso iterativo
            datos = self.resultados["iteraciones_datos"]
            if len(datos["c_vals"]) > 0:
                # Mostrar algunas iteraciones clave
                indices_mostrar = [0, len(datos["c_vals"])//2, -1] if len(datos["c_vals"]) > 2 else range(len(datos["c_vals"]))
                colores = ['orange', 'purple', 'red']
                
                for i, idx in enumerate(indices_mostrar):
                    if idx < len(datos["c_vals"]):
                        c_val = datos["c_vals"][idx]
                        fc_val = datos["fc_vals"][idx]
                        color = colores[i % len(colores)]
                        plt.plot(c_val, fc_val, 'o', color=color, markersize=8, 
                               label=f'c_{datos["numeros"][idx]} = {formatear_numero(c_val)}')
            
            # Marcar la raíz final
            plt.plot(raiz, f_raiz, 'ro', markersize=10, label=f'Raíz: {formatear_numero(raiz)}')
            
            plt.xlabel('x')
            plt.ylabel('f(x)')
            plt.title(f'f(x) = {self.funcion_str} - Proceso de {self.resultados["metodo_usado"]}')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {e}")
        
        esperar_enter()
    
    def comparar_con_biseccion(self):
        """Compara la velocidad de convergencia con bisección"""
        limpiar_pantalla()
        mostrar_titulo_principal("Falsa Posición", "Comparación con Bisección")
        
        console.print("[yellow]Simulando bisección para comparación...[/yellow]")
        
        # Simular bisección con los mismos datos iniciales
        x = sp.Symbol('x')
        a_bis, b_bis = float(self.a), float(self.b)
        iteraciones_bis = []
        errores_bis = []
        
        for i in range(min(self.max_iteraciones, len(self.resultados["iteraciones_datos"]["numeros"]))):
            c_bis = (a_bis + b_bis) / 2.0
            fc_bis = float(self.funcion.subs(x, c_bis))
            fa_bis = float(self.funcion.subs(x, a_bis))
            
            error_bis = abs(b_bis - a_bis) / 2.0
            iteraciones_bis.append(i + 1)
            errores_bis.append(error_bis)
            
            if error_bis < self.tolerancia:
                break
            
            if fa_bis * fc_bis < 0:
                b_bis = c_bis
            else:
                a_bis = c_bis
        
        # Mostrar comparación
        datos_fp = self.resultados["iteraciones_datos"]
        
        console.print(Panel(
            f"[bold cyan]Comparación de Convergencia[/bold cyan]\n\n"
            f"[yellow]{self.resultados['metodo_usado']}:[/yellow]\n"
            f"• Iteraciones: {self.resultados['iteraciones']}\n"
            f"• Error final: {formatear_numero(self.resultados['error_final'])}\n"
            f"• Raíz: {formatear_numero(self.resultados['raiz'])}\n\n"
            f"[yellow]Bisección:[/yellow]\n"
            f"• Iteraciones: {len(iteraciones_bis)}\n"
            f"• Error final: {formatear_numero(errores_bis[-1]) if errores_bis else 'N/A'}\n"
            f"• Raíz: {formatear_numero((a_bis + b_bis) / 2) if errores_bis else 'N/A'}\n\n"
            f"[green]Ventaja en iteraciones:[/green] "
            f"{max(0, len(iteraciones_bis) - self.resultados['iteraciones'])} menos",
            title="📊 Análisis Comparativo",
            border_style="green"
        ))
        
        # Gráfica comparativa
        if confirmar_accion("¿Desea ver la gráfica comparativa de convergencia?"):
            try:
                plt.figure(figsize=(12, 8))
                
                # Completar datos para que tengan la misma longitud
                max_iter = max(len(datos_fp["errores"]), len(errores_bis))
                fp_errores = datos_fp["errores"] + [datos_fp["errores"][-1]] * (max_iter - len(datos_fp["errores"]))
                bis_errores = errores_bis + [errores_bis[-1]] * (max_iter - len(errores_bis))
                
                plt.semilogy(range(1, len(fp_errores) + 1), fp_errores, 'b-o', 
                           linewidth=2, markersize=4, label=self.resultados["metodo_usado"])
                plt.semilogy(range(1, len(bis_errores) + 1), bis_errores, 'r-s', 
                           linewidth=2, markersize=4, label='Bisección')
                
                plt.axhline(y=self.tolerancia, color='gray', linestyle='--', alpha=0.7, label='Tolerancia')
                plt.xlabel('Iteración')
                plt.ylabel('Error Absoluto')
                plt.title('Comparación de Convergencia: Falsa Posición vs Bisección')
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                mostrar_mensaje_error(f"Error al generar gráfica: {e}")
        
        esperar_enter()
    
    def exportar_resultados(self):
        """Exporta los resultados a un archivo de texto"""
        try:
            nombre_archivo = f"falsa_posicion_resultados_{int(time.time())}.txt"
            
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                f.write("RESULTADOS DEL MÉTODO DE FALSA POSICIÓN\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Función: f(x) = {self.funcion_str}\n")
                f.write(f"Intervalo inicial: [{self.a}, {self.b}]\n")
                f.write(f"Método usado: {self.resultados['metodo_usado']}\n")
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
                f.write("-" * 100 + "\n")
                f.write(f"{'Iter':<6} {'a':<12} {'b':<12} {'c':<12} {'f(a)':<12} {'f(b)':<12} {'f(c)':<12} {'Error':<12}\n")
                f.write("-" * 100 + "\n")
                
                datos = self.resultados["iteraciones_datos"]
                for i in range(len(datos["numeros"])):
                    f.write(f"{datos['numeros'][i]:<6} "
                           f"{formatear_numero(datos['a_vals'][i]):<12} "
                           f"{formatear_numero(datos['b_vals'][i]):<12} "
                           f"{formatear_numero(datos['c_vals'][i]):<12} "
                           f"{formatear_numero(datos['fa_vals'][i]):<12} "
                           f"{formatear_numero(datos['fb_vals'][i]):<12} "
                           f"{formatear_numero(datos['fc_vals'][i]):<12} "
                           f"{formatear_numero(datos['errores'][i]):<12}\n")
            
            mostrar_mensaje_exito(f"Resultados exportados a: {nombre_archivo}")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {e}")
        
        esperar_enter()
    
    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        mostrar_ayuda_metodo(
            "Método de Falsa Posición (Regula Falsi)",
            "El método de falsa posición mejora la bisección usando interpolación lineal "
            "entre los puntos extremos para obtener mejores aproximaciones. Conecta los puntos "
            "(a,f(a)) y (b,f(b)) con una recta y toma su intersección con el eje X.",
            [
                "Encontrar raíces de ecuaciones no lineales",
                "Mejorar la velocidad de convergencia respecto a bisección",
                "Resolver ecuaciones donde se conoce un intervalo con cambio de signo",
                "Alternativa robusta cuando Newton-Raphson falla"
            ],
            [
                "Más rápido que bisección (superlineal)",
                "Siempre converge si hay raíz en el intervalo",
                "No requiere derivadas",
                "Numéricamente estable",
                "Método Illinois evita convergencia lenta"
            ],
            [
                "Puede converger lentamente en algunos casos",
                "Requiere intervalo con cambio de signo",
                "Solo encuentra una raíz por ejecución",
                "Puede tender a un extremo del intervalo"
            ]
        )

def main():
    """Función principal del programa"""
    metodo = MetodoFalsaPosicion()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "MÉTODO DE FALSA POSICIÓN",
            "Interpolación lineal para búsqueda de raíces"
        )
        
        mostrar_banner_metodo(
            "Método de Falsa Posición (Regula Falsi)",
            "Usa interpolación lineal entre extremos para aproximar raíces más rápido que bisección"
        )
        
        metodo.mostrar_configuracion()
        
        opciones = [
            "Ingresar función f(x)",
            "Configurar intervalo [a,b]",
            "Configurar parámetros (tolerancia, método)",
            "Ejecutar método de falsa posición",
            "Ver resultados y análisis",
            "Ver ayuda del método",
            "Salir"
        ]
        
        mostrar_menu_opciones(opciones)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6, 7])
        
        if opcion == 1:
            metodo.ingresar_funcion()
        elif opcion == 2:
            metodo.ingresar_intervalo()
        elif opcion == 3:
            metodo.configurar_parametros()
        elif opcion == 4:
            metodo.ejecutar_metodo()
        elif opcion == 5:
            metodo.mostrar_resultados()
        elif opcion == 6:
            metodo.mostrar_ayuda()
        elif opcion == 7:
            console.print("\n[green]¡Gracias por usar el método de falsa posición![/green]")
            break

if __name__ == "__main__":
    main()
