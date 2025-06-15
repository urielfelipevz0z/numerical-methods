#!/usr/bin/env python3
"""
Método de Bisección - Implementación con menús interactivos
Encuentra raíces de ecuaciones no lineales mediante búsqueda binaria en un intervalo
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
    mostrar_ayuda_metodo, esperar_enter,
    crear_tabla_iteraciones, formatear_numero,
    graficar_funcion, graficar_convergencia_biseccion
)

console = Console()

class MetodoBiseccion:
    def __init__(self):
        self.funcion = None
        self.funcion_str = ""
        self.a = None
        self.b = None
        self.tolerancia = 1e-6
        self.max_iteraciones = 100
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
            "Máx. Iteraciones": self.max_iteraciones
        }
        mostrar_estado_configuracion(config)
    
    def ingresar_funcion(self):
        """Menú para ingresar la función"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Bisección", "Ingreso de Función")
        
        console.print(Panel(
            "[bold cyan]Ingrese la función f(x) = 0[/bold cyan]\n\n"
            "Ejemplos válidos:\n"
            "• x**2 - 4\n"
            "• sin(x) - 0.5\n"
            "• exp(x) - 2*x\n"
            "• x**3 - 2*x - 5\n\n"
            "[yellow]Funciones disponibles:[/yellow]\n"
            "sin, cos, tan, exp, log, ln, sqrt, abs",
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
        mostrar_titulo_principal("Método de Bisección", "Configuración de Intervalo")
        
        if self.funcion is None:
            mostrar_mensaje_error("Primero debe ingresar una función")
            esperar_enter()
            return
        
        console.print(Panel(
            "[bold cyan]Ingrese el intervalo [a,b] donde buscar la raíz[/bold cyan]\n\n"
            "[yellow]Requisitos:[/yellow]\n"
            "• a < b\n"
            "• f(a) y f(b) deben tener signos opuestos\n"
            "• La función debe ser continua en [a,b]",
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
                
                # Mostrar gráfica del intervalo
                if confirmar_accion("¿Desea ver la gráfica en el intervalo?"):
                    try:
                        margen = abs(self.b - self.a) * 0.2
                        rango = (self.a - margen, self.b + margen)
                        puntos_especiales = [
                            (self.a, fa, f"a={self.a}", "red"),
                            (self.b, fb, f"b={self.b}", "green")
                        ]
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
        """Menú para configurar tolerancia y máximo de iteraciones"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Bisección", "Configuración de Parámetros")
        
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
            self.tolerancia = 1e-6
            self.max_iteraciones = 100
            mostrar_mensaje_exito("Parámetros restaurados a valores por defecto")
            
        if opcion != 4:
            esperar_enter()
    
    def ejecutar_metodo(self):
        """Ejecuta el método de bisección"""
        if not self.configuracion_completa():
            mostrar_mensaje_error("La configuración no está completa")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Bisección", "Ejecutando Algoritmo")
        
        self.mostrar_configuracion()
        
        if not confirmar_accion("¿Desea ejecutar el método con esta configuración?"):
            return
        
        mostrar_progreso_ejecucion("Ejecutando método de bisección...")
        
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
        intervalos = []
        
        tiempo_inicio = time.time()
        
        # Evaluar función en extremos iniciales
        fa = float(self.funcion.subs(x, a))
        fb = float(self.funcion.subs(x, b))
        
        # Barra de progreso
        with tqdm(total=self.max_iteraciones, desc="Iteraciones", unit="iter") as pbar:
            for i in range(self.max_iteraciones):
                # Calcular punto medio
                c = (a + b) / 2.0
                fc = float(self.funcion.subs(x, c))
                
                # Calcular error
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
                intervalos.append((a, b))
                
                # Verificar convergencia
                if error < self.tolerancia or abs(fc) < self.tolerancia:
                    break
                
                # Actualizar intervalo
                if fa * fc < 0:
                    b = c
                    fb = fc
                else:
                    a = c
                    fa = fc
                
                pbar.update(1)
        
        tiempo_ejecucion = time.time() - tiempo_inicio
        
        # Preparar resultados
        self.resultados = {
            "raiz": c,
            "valor_funcion": fc,
            "iteraciones": len(iteraciones),
            "error_final": error,
            "convergencia": error < self.tolerancia,
            "tolerancia": self.tolerancia,
            "intervalo_final": [a, b],
            "tiempo_ejecucion": tiempo_ejecucion,
            "iteraciones_datos": {
                "numeros": iteraciones,
                "a_vals": valores_a,
                "b_vals": valores_b,
                "c_vals": valores_c,
                "fa_vals": valores_fa,
                "fb_vals": valores_fb,
                "fc_vals": valores_fc,
                "errores": errores,
                "intervalos": intervalos
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
        mostrar_titulo_principal("Método de Bisección", "Resultados")
        
        # Mostrar resultado principal
        resultado_display = {
            "raiz": self.resultados["raiz"],
            "valor_funcion": self.resultados["valor_funcion"],
            "iteraciones": self.resultados["iteraciones"],
            "error_final": self.resultados["error_final"],
            "convergencia": self.resultados["convergencia"],
            "tolerancia_usada": self.resultados["tolerancia"]
        }
        
        mostrar_resultado_final("Bisección", resultado_display, self.resultados["tiempo_ejecucion"])
        
        # Menú de opciones para resultados
        opciones = [
            "Ver tabla de iteraciones",
            "Ver gráfica de convergencia",
            "Ver gráfica de la función con la raíz",
            "Exportar resultados",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Opciones de resultados", False)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5])
        
        if opcion == 1:
            self.mostrar_tabla_iteraciones()
        elif opcion == 2:
            self.mostrar_grafica_convergencia()
        elif opcion == 3:
            self.mostrar_grafica_funcion_raiz()
        elif opcion == 4:
            self.exportar_resultados()
        elif opcion == 5:
            return
    
    def mostrar_tabla_iteraciones(self):
        """Muestra la tabla detallada de iteraciones"""
        limpiar_pantalla()
        mostrar_titulo_principal("Método de Bisección", "Tabla de Iteraciones")
        
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
            "Iteraciones del Método de Bisección"
        )
        
        console.print(tabla)
        esperar_enter()
    
    def mostrar_grafica_convergencia(self):
        """Muestra gráficas de convergencia"""
        console.print("[yellow]Generando gráficas de convergencia...[/yellow]")
        
        datos = self.resultados["iteraciones_datos"]
        
        try:
            graficar_convergencia_biseccion(
                datos["numeros"],
                datos["intervalos"],
                datos["errores"],
                self.resultados["raiz"]
            )
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráficas: {e}")
        
        esperar_enter()
    
    def mostrar_grafica_funcion_raiz(self):
        """Muestra la función con la raíz encontrada"""
        console.print("[yellow]Generando gráfica de la función con la raíz...[/yellow]")
        
        try:
            # Rango de graficación alrededor de la raíz
            raiz = self.resultados["raiz"]
            rango_original = abs(self.b - self.a)
            margen = max(rango_original, 2.0)
            rango = (raiz - margen, raiz + margen)
            
            # Evaluar función en la raíz
            x = sp.Symbol('x')
            f_raiz = float(self.funcion.subs(x, raiz))
            
            puntos_especiales = [
                (raiz, f_raiz, f"Raíz: {formatear_numero(raiz)}", "red")
            ]
            
            graficar_funcion(
                self.funcion,
                rango,
                f"f(x) = {self.funcion_str} - Raíz encontrada",
                puntos_especiales
            )
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {e}")
        
        esperar_enter()
    
    def exportar_resultados(self):
        """Exporta los resultados a un archivo de texto"""
        try:
            nombre_archivo = f"biseccion_resultados_{int(time.time())}.txt"
            
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                f.write("RESULTADOS DEL MÉTODO DE BISECCIÓN\n")
                f.write("=" * 50 + "\n\n")
                
                f.write(f"Función: f(x) = {self.funcion_str}\n")
                f.write(f"Intervalo inicial: [{self.a}, {self.b}]\n")
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
                f.write("-" * 80 + "\n")
                f.write(f"{'Iter':<6} {'a':<12} {'b':<12} {'c':<12} {'f(c)':<12} {'Error':<12}\n")
                f.write("-" * 80 + "\n")
                
                datos = self.resultados["iteraciones_datos"]
                for i in range(len(datos["numeros"])):
                    f.write(f"{datos['numeros'][i]:<6} "
                           f"{formatear_numero(datos['a_vals'][i]):<12} "
                           f"{formatear_numero(datos['b_vals'][i]):<12} "
                           f"{formatear_numero(datos['c_vals'][i]):<12} "
                           f"{formatear_numero(datos['fc_vals'][i]):<12} "
                           f"{formatear_numero(datos['errores'][i]):<12}\n")
            
            mostrar_mensaje_exito(f"Resultados exportados a: {nombre_archivo}")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {e}")
        
        esperar_enter()
    
    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        mostrar_ayuda_metodo(
            "Método de Bisección",
            "El método de bisección es un algoritmo de búsqueda de raíces que funciona "
            "dividiendo repetidamente un intervalo por la mitad y seleccionando el subintervalo "
            "donde la función cambia de signo. Es robusto y siempre converge para funciones continuas.",
            [
                "Encontrar raíces de ecuaciones no lineales",
                "Resolver ecuaciones trascendentales",
                "Problemas de ingeniería donde se garantiza convergencia",
                "Como método de respaldo cuando otros fallan"
            ],
            [
                "Siempre converge si hay una raíz en el intervalo",
                "No requiere calcular derivadas",
                "Fácil de implementar y entender",
                "Numéricamente estable"
            ],
            [
                "Convergencia relativamente lenta (lineal)",
                "Requiere conocer un intervalo con cambio de signo",
                "Solo encuentra una raíz por ejecución",
                "No funciona con raíces de multiplicidad par"
            ]
        )

def main():
    """Función principal del programa"""
    metodo = MetodoBiseccion()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "MÉTODO DE BISECCIÓN",
            "Búsqueda de raíces por división de intervalos"
        )
        
        mostrar_banner_metodo(
            "Método de Bisección",
            "Encuentra raíces de ecuaciones no lineales dividiendo intervalos sucesivamente"
        )
        
        metodo.mostrar_configuracion()
        
        opciones = [
            "Ingresar función f(x)",
            "Configurar intervalo [a,b]",
            "Configurar parámetros (tolerancia, iteraciones)",
            "Ejecutar método de bisección",
            "Ver resultados y gráficas",
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
            console.print("\n[green]¡Gracias por usar el método de bisección![/green]")
            break

if __name__ == "__main__":
    main()
