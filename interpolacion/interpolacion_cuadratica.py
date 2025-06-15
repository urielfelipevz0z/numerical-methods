#!/usr/bin/env python3
"""
Interpolación Cuadrática - Implementación con menús interactivos
Interpolación cuadrática por tramos usando tres puntos consecutivos
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Agregar directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_numero, validar_opcion_menu, confirmar_accion, limpiar_pantalla,
    mostrar_titulo_principal, mostrar_menu_opciones, mostrar_banner_metodo,
    mostrar_estado_configuracion, mostrar_progreso_ejecucion,
    mostrar_resultado_final, mostrar_mensaje_error, mostrar_mensaje_exito,
    mostrar_ayuda_metodo, esperar_enter, formatear_numero,
    mostrar_tabla_datos, mostrar_estadisticas_datos
)

console = Console()

class InterpolacionCuadratica:
    def __init__(self):
        self.x_datos = None
        self.y_datos = None
        self.n_datos = 0
        self.puntos_interpolacion = None
        self.valores_interpolados = None
        self.metodo_ingreso = None
        self.polinomios_segmentos = None
        self.resultados = None
        
    def configuracion_completa(self) -> bool:
        """Verifica si la configuración está completa"""
        return (self.x_datos is not None and 
                self.y_datos is not None and 
                len(self.x_datos) >= 3)  # Mínimo 3 puntos para interpolación cuadrática
    
    def mostrar_configuracion(self):
        """Muestra el estado actual de la configuración"""
        tabla = Table(title="Estado de la Configuración")
        tabla.add_column("Parámetro", style="cyan")
        tabla.add_column("Valor", style="yellow")
        tabla.add_column("Estado", style="green")
        
        # Datos
        if self.x_datos is not None:
            tabla.add_row(
                "Datos base",
                f"{len(self.x_datos)} puntos cargados",
                "✓ Configurado" if len(self.x_datos) >= 3 else "⚠ Insuficientes (min. 3)"
            )
            if len(self.x_datos) >= 3:
                tabla.add_row(
                    "Segmentos cuadráticos",
                    f"{len(self.x_datos) - 2} segmentos posibles",
                    "✓ Configurado"
                )
                tabla.add_row(
                    "Rango de datos",
                    f"[{self.x_datos.min():.3f}, {self.x_datos.max():.3f}]",
                    "✓ Configurado"
                )
        else:
            tabla.add_row("Datos base", "No configurados", "⚠ Pendiente")
        
        # Puntos de interpolación
        if self.puntos_interpolacion is not None:
            tabla.add_row(
                "Puntos a interpolar",
                f"{len(self.puntos_interpolacion)} puntos",
                "✓ Configurado"
            )
        else:
            tabla.add_row("Puntos a interpolar", "No configurados", "⚠ Pendiente")
        
        # Método de ingreso
        if self.metodo_ingreso:
            tabla.add_row(
                "Método de ingreso",
                self.metodo_ingreso.capitalize(),
                "✓ Configurado"
            )
        
        console.print(tabla)
        console.print()
    
    def ingresar_datos(self):
        """Menú para ingresar o cargar datos base"""
        while True:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]CONFIGURACIÓN DE DATOS BASE[/bold cyan]",
                style="cyan"
            ))
            
            console.print("[yellow]Nota: Se requieren al menos 3 puntos para interpolación cuadrática[/yellow]")
            
            opciones = [
                "Ingresar datos manualmente",
                "Cargar desde archivo CSV",
                "Generar datos de prueba",
                "Volver al menú principal"
            ]
            
            mostrar_menu_opciones(opciones)
            opcion = validar_opcion_menu([1, 2, 3, 4])
            
            if opcion == 1:
                self._ingresar_manual()
                break
            elif opcion == 2:
                self._cargar_archivo()
                break
            elif opcion == 3:
                self._generar_datos_prueba()
                break
            elif opcion == 4:
                break
    
    def _ingresar_manual(self):
        """Ingreso manual de datos"""
        try:
            console.print("\n[cyan]Ingreso manual de datos[/cyan]")
            
            n = validar_numero("Número de puntos (mínimo 3): ", tipo=int, minimo=3, maximo=50)
            
            x_datos = []
            y_datos = []
            
            console.print(f"\n[yellow]Ingrese {n} pares de datos (x, y):[/yellow]")
            console.print("[red]Nota: Los valores de x deben estar ordenados de menor a mayor[/red]")
            
            for i in range(n):
                while True:
                    console.print(f"\n[cyan]Punto {i+1}:[/cyan]")
                    x = validar_numero(f"  x{i+1}: ", tipo=float)
                    
                    # Verificar orden creciente
                    if i > 0 and x <= x_datos[-1]:
                        mostrar_mensaje_error(f"x debe ser mayor que {x_datos[-1]:.3f}")
                        continue
                    
                    y = validar_numero(f"  y{i+1}: ", tipo=float)
                    x_datos.append(x)
                    y_datos.append(y)
                    break
            
            self.x_datos = np.array(x_datos)
            self.y_datos = np.array(y_datos)
            self.n_datos = n
            self.metodo_ingreso = "manual"
            
            mostrar_mensaje_exito("Datos ingresados correctamente")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al ingresar datos: {str(e)}")
        
        esperar_enter()
    
    def _cargar_archivo(self):
        """Cargar datos desde archivo CSV"""
        try:
            archivo = input("\nRuta del archivo CSV: ").strip()
            
            if not os.path.exists(archivo):
                mostrar_mensaje_error("El archivo no existe")
                esperar_enter()
                return
            
            # Intentar cargar el archivo
            try:
                datos = pd.read_csv(archivo)
                if len(datos.columns) < 2:
                    raise ValueError("El archivo debe tener al menos 2 columnas")
                
                x_datos = datos.iloc[:, 0].values
                y_datos = datos.iloc[:, 1].values
                
                # Ordenar por x
                indices_orden = np.argsort(x_datos)
                self.x_datos = x_datos[indices_orden]
                self.y_datos = y_datos[indices_orden]
                self.n_datos = len(self.x_datos)
                
                if self.n_datos < 3:
                    raise ValueError("Se necesitan al menos 3 datos para interpolación cuadrática")
                
                self.metodo_ingreso = "archivo"
                mostrar_mensaje_exito(f"Datos cargados: {self.n_datos} puntos")
                
            except Exception as e:
                mostrar_mensaje_error(f"Error al leer archivo: {str(e)}")
        
        except Exception as e:
            mostrar_mensaje_error(f"Error inesperado: {str(e)}")
        
        esperar_enter()
    
    def _generar_datos_prueba(self):
        """Generar datos de prueba"""
        try:
            console.print("\n[cyan]Generación de datos de prueba[/cyan]")
            
            n = validar_numero("Número de puntos (5-20): ", tipo=int, minimo=5, maximo=20)
            x_min = validar_numero("Valor mínimo de x: ", tipo=float)
            x_max = validar_numero("Valor máximo de x: ", tipo=float)
            
            if x_max <= x_min:
                mostrar_mensaje_error("x_max debe ser mayor que x_min")
                esperar_enter()
                return
            
            # Generar función base
            console.print("\n[yellow]Seleccione tipo de función:[/yellow]")
            funciones = [
                "Cuadrática: ax² + bx + c",
                "Cúbica: ax³ + bx² + cx + d",
                "Senoidal: a*sin(bx) + c",
                "Exponencial: ae^(bx)",
                "Función con cambios de curvatura"
            ]
            
            for i, func in enumerate(funciones, 1):
                console.print(f"{i}. {func}")
            
            tipo_func = validar_opcion_menu([1, 2, 3, 4, 5])
            
            # Generar datos dispersos (no equidistantes)
            x_datos = np.sort(np.random.uniform(x_min, x_max, n))
            
            if tipo_func == 1:  # Cuadrática
                y_datos = 0.5 * x_datos**2 - 2 * x_datos + 3
            elif tipo_func == 2:  # Cúbica
                y_datos = 0.1 * x_datos**3 - 0.5 * x_datos**2 + x_datos + 2
            elif tipo_func == 3:  # Senoidal
                y_datos = 3 * np.sin(x_datos) + 0.5 * x_datos
            elif tipo_func == 4:  # Exponencial
                y_datos = 2 * np.exp(0.3 * x_datos)
            else:  # Función compleja
                y_datos = x_datos**2 * np.sin(x_datos) + np.cos(2 * x_datos)
            
            # Agregar un poco de ruido
            ruido = validar_numero("Nivel de ruido (0.0-0.2): ", tipo=float, minimo=0, maximo=0.2)
            if ruido > 0:
                noise = np.random.normal(0, ruido * np.std(y_datos), n)
                y_datos += noise
            
            self.x_datos = x_datos
            self.y_datos = y_datos
            self.n_datos = n
            self.metodo_ingreso = "generado"
            
            mostrar_mensaje_exito(f"Datos generados: {n} puntos")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar datos: {str(e)}")
        
        esperar_enter()
    
    def configurar_interpolacion(self):
        """Configurar puntos donde interpolar"""
        if not self.configuracion_completa():
            mostrar_mensaje_error("Configure al menos 3 datos base primero")
            esperar_enter()
            return
        
        while True:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]CONFIGURACIÓN DE INTERPOLACIÓN[/bold cyan]",
                style="cyan"
            ))
            
            console.print(f"[yellow]Rango de datos: [{self.x_datos.min():.3f}, {self.x_datos.max():.3f}][/yellow]")
            console.print(f"[yellow]Segmentos cuadráticos disponibles: {len(self.x_datos) - 2}[/yellow]")
            
            opciones = [
                "Puntos específicos",
                "Rango uniforme",
                "Puntos aleatorios en el rango",
                "Malla densa para visualización",
                "Volver al menú principal"
            ]
            
            mostrar_menu_opciones(opciones)
            opcion = validar_opcion_menu([1, 2, 3, 4, 5])
            
            if opcion == 1:
                self._configurar_puntos_especificos()
                break
            elif opcion == 2:
                self._configurar_rango_uniforme()
                break
            elif opcion == 3:
                self._configurar_puntos_aleatorios()
                break
            elif opcion == 4:
                self._configurar_malla_densa()
                break
            elif opcion == 5:
                break
    
    def _configurar_puntos_especificos(self):
        """Configurar puntos específicos para interpolación"""
        try:
            console.print("\n[cyan]Configuración de puntos específicos[/cyan]")
            
            n_puntos = validar_numero("Número de puntos a interpolar: ", tipo=int, minimo=1, maximo=100)
            
            puntos = []
            for i in range(n_puntos):
                while True:
                    punto = validar_numero(f"Punto {i+1}: ", tipo=float)
                    
                    if punto < self.x_datos.min() or punto > self.x_datos.max():
                        console.print(f"[red]Advertencia: Punto fuera del rango de datos [{self.x_datos.min():.3f}, {self.x_datos.max():.3f}][/red]")
                        console.print(f"[red]La extrapolación cuadrática puede ser muy imprecisa[/red]")
                        if confirmar_accion("¿Continuar con este punto?"):
                            puntos.append(punto)
                            break
                    else:
                        puntos.append(punto)
                        break
            
            self.puntos_interpolacion = np.array(puntos)
            mostrar_mensaje_exito(f"Configurados {len(puntos)} puntos para interpolación")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al configurar puntos: {str(e)}")
        
        esperar_enter()
    
    def _configurar_rango_uniforme(self):
        """Configurar rango uniforme para interpolación"""
        try:
            console.print("\n[cyan]Configuración de rango uniforme[/cyan]")
            
            x_min_interp = validar_numero(f"Valor mínimo (datos: {self.x_datos.min():.3f}): ", tipo=float)
            x_max_interp = validar_numero(f"Valor máximo (datos: {self.x_datos.max():.3f}): ", tipo=float)
            
            if x_max_interp <= x_min_interp:
                mostrar_mensaje_error("El valor máximo debe ser mayor que el mínimo")
                esperar_enter()
                return
            
            n_puntos = validar_numero("Número de puntos uniformes: ", tipo=int, minimo=2, maximo=1000)
            
            self.puntos_interpolacion = np.linspace(x_min_interp, x_max_interp, n_puntos)
            
            # Verificar extrapolación
            if (x_min_interp < self.x_datos.min() or x_max_interp > self.x_datos.max()):
                console.print("[red]Advertencia: Algunos puntos requieren extrapolación cuadrática[/red]")
                console.print("[red]La extrapolación cuadrática puede diverger rápidamente[/red]")
            
            mostrar_mensaje_exito(f"Configurados {n_puntos} puntos uniformes")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al configurar rango: {str(e)}")
        
        esperar_enter()
    
    def _configurar_puntos_aleatorios(self):
        """Configurar puntos aleatorios en el rango"""
        try:
            console.print("\n[cyan]Configuración de puntos aleatorios[/cyan]")
            
            n_puntos = validar_numero("Número de puntos aleatorios: ", tipo=int, minimo=1, maximo=1000)
            
            # Usar el rango de los datos por defecto
            x_min = self.x_datos.min()
            x_max = self.x_datos.max()
            
            if confirmar_accion("¿Desea especificar un rango diferente?"):
                x_min = validar_numero(f"Valor mínimo (actual: {x_min:.3f}): ", tipo=float)
                x_max = validar_numero(f"Valor máximo (actual: {x_max:.3f}): ", tipo=float)
            
            if x_max <= x_min:
                mostrar_mensaje_error("El valor máximo debe ser mayor que el mínimo")
                esperar_enter()
                return
            
            # Generar puntos aleatorios
            np.random.seed(42)  # Para reproducibilidad
            puntos_aleatorios = np.random.uniform(x_min, x_max, n_puntos)
            self.puntos_interpolacion = np.sort(puntos_aleatorios)
            
            mostrar_mensaje_exito(f"Generados {n_puntos} puntos aleatorios")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar puntos aleatorios: {str(e)}")
        
        esperar_enter()
    
    def _configurar_malla_densa(self):
        """Configurar malla densa para visualización suave"""
        try:
            console.print("\n[cyan]Configuración de malla densa[/cyan]")
            console.print("[yellow]Ideal para generar gráficas suaves de la interpolación[/yellow]")
            
            # Usar el rango de datos con pequeña extensión
            x_min = self.x_datos.min()
            x_max = self.x_datos.max()
            rango = x_max - x_min
            
            # Sugerir extensión del 5%
            extension = 0.05 * rango
            x_min_sugerido = x_min - extension
            x_max_sugerido = x_max + extension
            
            console.print(f"[yellow]Rango sugerido: [{x_min_sugerido:.3f}, {x_max_sugerido:.3f}][/yellow]")
            
            if confirmar_accion("¿Usar rango sugerido?"):
                x_min_interp = x_min_sugerido
                x_max_interp = x_max_sugerido
            else:
                x_min_interp = validar_numero("Valor mínimo: ", tipo=float)
                x_max_interp = validar_numero("Valor máximo: ", tipo=float)
            
            if x_max_interp <= x_min_interp:
                mostrar_mensaje_error("El valor máximo debe ser mayor que el mínimo")
                esperar_enter()
                return
            
            # Sugerir número de puntos basado en el rango
            n_sugerido = max(100, int(rango * 50))  # Aproximadamente 50 puntos por unidad
            n_puntos = validar_numero(f"Número de puntos (sugerido: {n_sugerido}): ", 
                                    tipo=int, minimo=50, maximo=5000)
            
            self.puntos_interpolacion = np.linspace(x_min_interp, x_max_interp, n_puntos)
            
            mostrar_mensaje_exito(f"Configurada malla densa: {n_puntos} puntos")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al configurar malla: {str(e)}")
        
        esperar_enter()
    
    def calcular_interpolacion(self):
        """Calcular la interpolación cuadrática"""
        if not self.configuracion_completa() or self.puntos_interpolacion is None:
            mostrar_mensaje_error("Configure datos y puntos de interpolación primero")
            esperar_enter()
            return
        
        try:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]CÁLCULO DE INTERPOLACIÓN CUADRÁTICA[/bold cyan]",
                style="cyan"
            ))
            
            with mostrar_progreso_ejecucion("Calculando interpolación cuadrática..."):
                time.sleep(0.5)
                
                # Calcular los polinomios cuadráticos para cada segmento
                self._calcular_polinomios_segmentos()
                
                # Calcular interpolación para cada punto
                valores_interpolados = []
                
                for x_interp in self.puntos_interpolacion:
                    y_interp = self._interpolar_punto(x_interp)
                    valores_interpolados.append(y_interp)
                
                self.valores_interpolados = np.array(valores_interpolados)
                
                # Analizar resultados
                interpolacion_count = np.sum((self.puntos_interpolacion >= self.x_datos.min()) & 
                                           (self.puntos_interpolacion <= self.x_datos.max()))
                extrapolacion_count = len(self.puntos_interpolacion) - interpolacion_count
                
                # Guardar resultados
                self.resultados = {
                    'valores_interpolados': self.valores_interpolados,
                    'interpolacion_count': interpolacion_count,
                    'extrapolacion_count': extrapolacion_count,
                    'rango_original': [self.x_datos.min(), self.x_datos.max()],
                    'rango_interpolacion': [self.puntos_interpolacion.min(), self.puntos_interpolacion.max()],
                    'num_segmentos': len(self.polinomios_segmentos)
                }
            
            mostrar_mensaje_exito("Interpolación cuadrática calculada exitosamente")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error en el cálculo: {str(e)}")
        
        esperar_enter()
    
    def _calcular_polinomios_segmentos(self):
        """Calcular los polinomios cuadráticos para cada segmento"""
        self.polinomios_segmentos = []
        
        # Para cada trio de puntos consecutivos, calcular polinomio cuadrático
        for i in range(len(self.x_datos) - 2):
            # Tomar tres puntos consecutivos
            x_trio = self.x_datos[i:i+3]
            y_trio = self.y_datos[i:i+3]
            
            # Calcular polinomio cuadrático usando numpy.polyfit
            coeficientes = np.polyfit(x_trio, y_trio, 2)
            
            # Guardar información del segmento
            segmento = {
                'coeficientes': coeficientes,  # [a, b, c] para ax² + bx + c
                'x_inicio': x_trio[0],
                'x_fin': x_trio[2],
                'x_medio': x_trio[1],
                'puntos_x': x_trio,
                'puntos_y': y_trio
            }
            
            self.polinomios_segmentos.append(segmento)
    
    def _interpolar_punto(self, x):
        """Interpolar un punto específico usando interpolación cuadrática"""
        # Si x está fuera del rango, usar extrapolación con el segmento más cercano
        if x <= self.x_datos[0]:
            # Extrapolación hacia la izquierda usando el primer segmento
            segmento = self.polinomios_segmentos[0]
        elif x >= self.x_datos[-1]:
            # Extrapolación hacia la derecha usando el último segmento
            segmento = self.polinomios_segmentos[-1]
        else:
            # Interpolación: encontrar el segmento apropiado
            # Buscar el segmento que contiene x
            segmento = None
            for seg in self.polinomios_segmentos:
                if seg['x_inicio'] <= x <= seg['x_fin']:
                    segmento = seg
                    break
            
            # Si no se encuentra (caso raro), usar el segmento más cercano
            if segmento is None:
                distancias = [abs(x - seg['x_medio']) for seg in self.polinomios_segmentos]
                idx_cercano = np.argmin(distancias)
                segmento = self.polinomios_segmentos[idx_cercano]
        
        # Evaluar el polinomio cuadrático
        coefs = segmento['coeficientes']
        return np.polyval(coefs, x)  # a*x² + b*x + c
    
    def mostrar_resultados(self):
        """Mostrar resultados detallados"""
        if self.resultados is None:
            mostrar_mensaje_error("No hay resultados para mostrar. Ejecute el cálculo primero.")
            esperar_enter()
            return
        
        while True:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]RESULTADOS DE INTERPOLACIÓN CUADRÁTICA[/bold cyan]",
                style="cyan"
            ))
            
            opciones = [
                "Ver tabla de resultados",
                "Ver polinomios de segmentos",
                "Análisis de interpolación/extrapolación",
                "Generar gráfica",
                "Evaluar punto específico",
                "Comparar con interpolación lineal",
                "Exportar resultados",
                "Volver al menú principal"
            ]
            
            mostrar_menu_opciones(opciones)
            opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6, 7, 8])
            
            if opcion == 1:
                self._mostrar_tabla_resultados()
            elif opcion == 2:
                self._mostrar_polinomios_segmentos()
            elif opcion == 3:
                self._mostrar_analisis()
            elif opcion == 4:
                self._generar_grafica()
            elif opcion == 5:
                self._evaluar_punto_especifico()
            elif opcion == 6:
                self._comparar_con_lineal()
            elif opcion == 7:
                self._exportar_resultados()
            elif opcion == 8:
                break
    
    def _mostrar_tabla_resultados(self):
        """Mostrar tabla de resultados de interpolación"""
        tabla = Table(title="Resultados de Interpolación Cuadrática")
        tabla.add_column("Punto", style="cyan")
        tabla.add_column("Valor Interpolado", style="yellow")
        tabla.add_column("Segmento Usado", style="green")
        tabla.add_column("Tipo", style="magenta")
        
        for i, (x, y) in enumerate(zip(self.puntos_interpolacion, self.valores_interpolados)):
            # Determinar segmento usado
            segmento_usado = self._obtener_segmento_para_punto(x)
            
            # Determinar si es interpolación o extrapolación
            if self.x_datos.min() <= x <= self.x_datos.max():
                tipo = "Interpolación"
            else:
                tipo = "Extrapolación"
            
            tabla.add_row(
                formatear_numero(x),
                formatear_numero(y),
                f"Seg. {segmento_usado + 1}",
                tipo
            )
            
            # Mostrar solo los primeros 20 resultados en pantalla
            if i >= 19 and len(self.puntos_interpolacion) > 20:
                tabla.add_row("...", "...", "...", "...")
                tabla.add_row(
                    f"(y {len(self.puntos_interpolacion) - 20} más)",
                    "",
                    "",
                    ""
                )
                break
        
        console.print(tabla)
        esperar_enter()
    
    def _obtener_segmento_para_punto(self, x):
        """Obtener el índice del segmento usado para interpolar un punto"""
        if x <= self.x_datos[0]:
            return 0
        elif x >= self.x_datos[-1]:
            return len(self.polinomios_segmentos) - 1
        else:
            for i, seg in enumerate(self.polinomios_segmentos):
                if seg['x_inicio'] <= x <= seg['x_fin']:
                    return i
            # Si no se encuentra, devolver el más cercano
            distancias = [abs(x - seg['x_medio']) for seg in self.polinomios_segmentos]
            return np.argmin(distancias)
    
    def _mostrar_polinomios_segmentos(self):
        """Mostrar los polinomios de cada segmento"""
        tabla = Table(title="Polinomios Cuadráticos por Segmento")
        tabla.add_column("Segmento", style="cyan")
        tabla.add_column("Rango", style="yellow")
        tabla.add_column("Polinomio", style="green")
        tabla.add_column("R² Local", style="magenta")
        
        for i, seg in enumerate(self.polinomios_segmentos):
            # Construir ecuación del polinomio
            a, b, c = seg['coeficientes']
            ecuacion = f"{a:.4f}x² + {b:.4f}x + {c:.4f}"
            
            # Calcular R² local (siempre será 1.0 para 3 puntos exactos)
            x_seg = seg['puntos_x']
            y_seg = seg['puntos_y']
            y_pred = np.polyval(seg['coeficientes'], x_seg)
            
            ss_res = np.sum((y_seg - y_pred) ** 2)
            ss_tot = np.sum((y_seg - np.mean(y_seg)) ** 2)
            r2_local = 1 - (ss_res / ss_tot) if ss_tot != 0 else 1.0
            
            tabla.add_row(
                f"{i + 1}",
                f"[{seg['x_inicio']:.3f}, {seg['x_fin']:.3f}]",
                ecuacion,
                f"{r2_local:.6f}"
            )
        
        console.print(tabla)
        
        # Mostrar puntos usados para cada segmento
        if confirmar_accion("¿Ver puntos usados para cada segmento?"):
            for i, seg in enumerate(self.polinomios_segmentos):
                console.print(f"\n[cyan]Segmento {i + 1}:[/cyan]")
                for j, (x, y) in enumerate(zip(seg['puntos_x'], seg['puntos_y'])):
                    console.print(f"  Punto {j + 1}: ({formatear_numero(x)}, {formatear_numero(y)})")
        
        esperar_enter()
    
    def _mostrar_analisis(self):
        """Mostrar análisis de interpolación/extrapolación"""
        tabla = Table(title="Análisis de Interpolación Cuadrática")
        tabla.add_column("Aspecto", style="cyan")
        tabla.add_column("Valor", style="yellow")
        tabla.add_column("Descripción", style="green")
        
        # Estadísticas básicas
        tabla.add_row("Datos base", str(self.n_datos), "Puntos de datos originales")
        tabla.add_row("Segmentos cuadráticos", str(self.resultados['num_segmentos']), "Polinomios cuadráticos calculados")
        tabla.add_row("Puntos calculados", str(len(self.puntos_interpolacion)), "Puntos interpolados/extrapolados")
        tabla.add_row("Interpolaciones", str(self.resultados['interpolacion_count']), "Dentro del rango de datos")
        tabla.add_row("Extrapolaciones", str(self.resultados['extrapolacion_count']), "Fuera del rango de datos")
        
        # Rangos
        rango_orig = self.resultados['rango_original']
        rango_interp = self.resultados['rango_interpolacion']
        
        tabla.add_row("Rango original", f"[{rango_orig[0]:.3f}, {rango_orig[1]:.3f}]", "Rango de datos base")
        tabla.add_row("Rango interpolación", f"[{rango_interp[0]:.3f}, {rango_interp[1]:.3f}]", "Rango de puntos calculados")
        
        # Estadísticas de valores interpolados
        tabla.add_row("Valor mínimo", formatear_numero(self.valores_interpolados.min()), "Mínimo valor interpolado")
        tabla.add_row("Valor máximo", formatear_numero(self.valores_interpolados.max()), "Máximo valor interpolado")
        tabla.add_row("Valor promedio", formatear_numero(self.valores_interpolados.mean()), "Promedio de valores interpolados")
        
        console.print(tabla)
        
        # Advertencias
        if self.resultados['extrapolacion_count'] > 0:
            console.print(Panel(
                "[red]Advertencia: La extrapolación cuadrática puede diverger rápidamente.\n"
                "Los polinomios cuadráticos pueden producir valores muy grandes fuera del rango de datos.\n"
                "Use con extrema precaución para extrapolación.[/red]",
                title="Advertencia de Extrapolación",
                style="red"
            ))
        
        esperar_enter()
    
    def _generar_grafica(self):
        """Generar gráfica de la interpolación"""
        try:
            self._graficar_interpolacion_cuadratica()
            mostrar_mensaje_exito("Gráfica generada exitosamente")
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {str(e)}")
        
        esperar_enter()
    
    def _graficar_interpolacion_cuadratica(self):
        """Crear gráfica de la interpolación cuadrática"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfica principal
        # Datos originales
        ax1.scatter(self.x_datos, self.y_datos, color='blue', s=100, zorder=5,
                   label='Datos originales', marker='o', edgecolors='black', linewidth=2)
        
        # Puntos interpolados
        mask_interp = ((self.puntos_interpolacion >= self.x_datos.min()) & 
                      (self.puntos_interpolacion <= self.x_datos.max()))
        mask_extrap = ~mask_interp
        
        if np.any(mask_interp):
            ax1.scatter(self.puntos_interpolacion[mask_interp], 
                       self.valores_interpolados[mask_interp],
                       color='green', s=20, alpha=0.7, label='Interpolación', marker='.')
        
        if np.any(mask_extrap):
            ax1.scatter(self.puntos_interpolacion[mask_extrap], 
                       self.valores_interpolados[mask_extrap],
                       color='red', s=20, alpha=0.7, label='Extrapolación', marker='.')
        
        # Curva de interpolación cuadrática
        x_smooth = np.linspace(self.puntos_interpolacion.min(), self.puntos_interpolacion.max(), 1000)
        y_smooth = [self._interpolar_punto(x) for x in x_smooth]
        ax1.plot(x_smooth, y_smooth, 'purple', linewidth=2, alpha=0.8, label='Interpolación cuadrática')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Interpolación Cuadrática')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfica de segmentos individuales
        colores = plt.cm.Set3(np.linspace(0, 1, len(self.polinomios_segmentos)))
        
        for i, (seg, color) in enumerate(zip(self.polinomios_segmentos, colores)):
            # Dibujar el segmento
            x_seg = np.linspace(seg['x_inicio'], seg['x_fin'], 100)
            y_seg = np.polyval(seg['coeficientes'], x_seg)
            ax2.plot(x_seg, y_seg, color=color, linewidth=2, 
                    label=f'Segmento {i+1}', alpha=0.8)
            
            # Marcar los puntos usados para este segmento
            ax2.scatter(seg['puntos_x'], seg['puntos_y'], color=color, s=60, 
                       edgecolors='black', linewidth=1, zorder=5)
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Segmentos Cuadráticos Individuales')
        if len(self.polinomios_segmentos) <= 8:
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Gráfica de derivadas (pendientes)
        x_derivadas = []
        pendientes = []
        
        for seg in self.polinomios_segmentos:
            x_medio = seg['x_medio']
            # Derivada de ax² + bx + c es 2ax + b
            a, b, c = seg['coeficientes']
            pendiente = 2 * a * x_medio + b
            x_derivadas.append(x_medio)
            pendientes.append(pendiente)
        
        ax3.plot(x_derivadas, pendientes, 'ro-', markersize=8, linewidth=2)
        ax3.set_xlabel('x')
        ax3.set_ylabel('Pendiente')
        ax3.set_title('Pendientes en Puntos Medios de Segmentos')
        ax3.grid(True, alpha=0.3)
        
        # Gráfica de curvaturas (segunda derivada)
        x_curvaturas = []
        curvaturas = []
        
        for seg in self.polinomios_segmentos:
            x_medio = seg['x_medio']
            # Segunda derivada de ax² + bx + c es 2a
            a, b, c = seg['coeficientes']
            curvatura = 2 * a
            x_curvaturas.append(x_medio)
            curvaturas.append(curvatura)
        
        ax4.plot(x_curvaturas, curvaturas, 'bo-', markersize=8, linewidth=2)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
        ax4.set_xlabel('x')
        ax4.set_ylabel('Curvatura (2a)')
        ax4.set_title('Curvatura de Segmentos Cuadráticos')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _evaluar_punto_especifico(self):
        """Evaluar interpolación en un punto específico"""
        try:
            console.print("\n[cyan]Evaluación de punto específico[/cyan]")
            
            x_eval = validar_numero("Punto a evaluar: ", tipo=float)
            y_eval = self._interpolar_punto(x_eval)
            
            # Determinar segmento usado
            segmento_idx = self._obtener_segmento_para_punto(x_eval)
            segmento = self.polinomios_segmentos[segmento_idx]
            
            # Determinar tipo (interpolación/extrapolación)
            if self.x_datos.min() <= x_eval <= self.x_datos.max():
                tipo = "Interpolación"
                confianza = "Alta"
            else:
                tipo = "Extrapolación"
                confianza = "Muy baja (usar con precaución)"
            
            # Crear tabla con resultado
            tabla = Table(title="Evaluación de Punto")
            tabla.add_column("Aspecto", style="cyan")
            tabla.add_column("Valor", style="yellow")
            
            tabla.add_row("Punto evaluado", formatear_numero(x_eval))
            tabla.add_row("Valor interpolado", formatear_numero(y_eval))
            tabla.add_row("Tipo de cálculo", tipo)
            tabla.add_row("Confianza", confianza)
            tabla.add_row("Segmento usado", f"Segmento {segmento_idx + 1}")
            
            console.print(tabla)
            
            # Mostrar detalles del segmento
            console.print(f"\n[green]Detalles del segmento {segmento_idx + 1}:[/green]")
            a, b, c = segmento['coeficientes']
            console.print(f"  Polinomio: {a:.6f}x² + {b:.6f}x + {c:.6f}")
            console.print(f"  Rango: [{segmento['x_inicio']:.3f}, {segmento['x_fin']:.3f}]")
            console.print(f"  Puntos base:")
            for i, (x, y) in enumerate(zip(segmento['puntos_x'], segmento['puntos_y'])):
                console.print(f"    P{i+1}: ({formatear_numero(x)}, {formatear_numero(y)})")
            
            # Calcular derivadas en el punto
            pendiente = 2 * a * x_eval + b
            curvatura = 2 * a
            
            console.print(f"\n[cyan]Análisis local:[/cyan]")
            console.print(f"  Pendiente en x={x_eval:.3f}: {pendiente:.6f}")
            console.print(f"  Curvatura (2a): {curvatura:.6f}")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al evaluar punto: {str(e)}")
        
        esperar_enter()
    
    def _comparar_con_lineal(self):
        """Comparar con interpolación lineal"""
        try:
            console.print("\n[cyan]Comparación con interpolación lineal[/cyan]")
            
            # Calcular interpolación lineal para los mismos puntos
            y_lineal = []
            for x in self.puntos_interpolacion:
                y_lin = self._interpolar_lineal(x)
                y_lineal.append(y_lin)
            
            y_lineal = np.array(y_lineal)
            
            # Calcular diferencias
            diferencias = np.abs(self.valores_interpolados - y_lineal)
            
            # Estadísticas de comparación
            tabla = Table(title="Comparación: Cuadrática vs Lineal")
            tabla.add_column("Métrica", style="cyan")
            tabla.add_column("Valor", style="yellow")
            tabla.add_column("Descripción", style="green")
            
            tabla.add_row("Diferencia promedio", formatear_numero(np.mean(diferencias)), "Promedio de |cuadrática - lineal|")
            tabla.add_row("Diferencia máxima", formatear_numero(np.max(diferencias)), "Máxima diferencia absoluta")
            tabla.add_row("Diferencia mínima", formatear_numero(np.min(diferencias)), "Mínima diferencia absoluta")
            tabla.add_row("Desv. estándar dif.", formatear_numero(np.std(diferencias)), "Variabilidad de las diferencias")
            
            console.print(tabla)
            
            # Mostrar algunos ejemplos
            if confirmar_accion("¿Ver ejemplos de diferencias punto a punto?"):
                tabla_ejemplos = Table(title="Ejemplos de Diferencias")
                tabla_ejemplos.add_column("Punto x", style="cyan")
                tabla_ejemplos.add_column("Cuadrática", style="yellow")
                tabla_ejemplos.add_column("Lineal", style="green")
                tabla_ejemplos.add_column("Diferencia", style="red")
                
                # Mostrar 10 ejemplos espaciados
                indices = np.linspace(0, len(self.puntos_interpolacion) - 1, 10, dtype=int)
                for i in indices:
                    x = self.puntos_interpolacion[i]
                    y_cuad = self.valores_interpolados[i]
                    y_lin = y_lineal[i]
                    dif = diferencias[i]
                    
                    tabla_ejemplos.add_row(
                        formatear_numero(x),
                        formatear_numero(y_cuad),
                        formatear_numero(y_lin),
                        formatear_numero(dif)
                    )
                
                console.print(tabla_ejemplos)
            
            # Generar gráfica comparativa
            if confirmar_accion("¿Generar gráfica comparativa?"):
                self._graficar_comparacion_lineal(y_lineal, diferencias)
            
        except Exception as e:
            mostrar_mensaje_error(f"Error en comparación: {str(e)}")
        
        esperar_enter()
    
    def _interpolar_lineal(self, x):
        """Interpolación lineal simple para comparación"""
        if x <= self.x_datos[0]:
            # Extrapolación izquierda
            x1, x2 = self.x_datos[0], self.x_datos[1]
            y1, y2 = self.y_datos[0], self.y_datos[1]
        elif x >= self.x_datos[-1]:
            # Extrapolación derecha
            x1, x2 = self.x_datos[-2], self.x_datos[-1]
            y1, y2 = self.y_datos[-2], self.y_datos[-1]
        else:
            # Interpolación
            i = np.searchsorted(self.x_datos, x) - 1
            x1, x2 = self.x_datos[i], self.x_datos[i + 1]
            y1, y2 = self.y_datos[i], self.y_datos[i + 1]
        
        if x2 - x1 == 0:
            return y1
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    def _graficar_comparacion_lineal(self, y_lineal, diferencias):
        """Generar gráfica comparativa con interpolación lineal"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Comparación de métodos
        ax1.scatter(self.x_datos, self.y_datos, color='blue', s=100, zorder=5,
                   label='Datos originales', marker='o', edgecolors='black', linewidth=2)
        
        # Solo mostrar una muestra de puntos si hay muchos
        if len(self.puntos_interpolacion) > 200:
            indices = np.linspace(0, len(self.puntos_interpolacion) - 1, 200, dtype=int)
            x_muestra = self.puntos_interpolacion[indices]
            y_cuad_muestra = self.valores_interpolados[indices]
            y_lin_muestra = y_lineal[indices]
        else:
            x_muestra = self.puntos_interpolacion
            y_cuad_muestra = self.valores_interpolados
            y_lin_muestra = y_lineal
        
        ax1.plot(x_muestra, y_cuad_muestra, 'purple', linewidth=2, 
                label='Interpolación cuadrática', alpha=0.8)
        ax1.plot(x_muestra, y_lin_muestra, 'orange', linewidth=2, 
                label='Interpolación lineal', alpha=0.8, linestyle='--')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Comparación: Cuadrática vs Lineal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfica de diferencias
        ax2.plot(self.puntos_interpolacion, diferencias, 'red', linewidth=1, alpha=0.7)
        ax2.fill_between(self.puntos_interpolacion, 0, diferencias, alpha=0.3, color='red')
        ax2.set_xlabel('x')
        ax2.set_ylabel('|Cuadrática - Lineal|')
        ax2.set_title('Diferencia Absoluta entre Métodos')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _exportar_resultados(self):
        """Exportar resultados a archivo"""
        try:
            nombre_archivo = input("\nNombre del archivo (sin extensión): ").strip()
            if not nombre_archivo:
                nombre_archivo = "interpolacion_cuadratica"
            
            # Crear DataFrame con resultados
            tipos = []
            segmentos_usados = []
            
            for x in self.puntos_interpolacion:
                if self.x_datos.min() <= x <= self.x_datos.max():
                    tipos.append("Interpolación")
                else:
                    tipos.append("Extrapolación")
                
                segmento_idx = self._obtener_segmento_para_punto(x)
                segmentos_usados.append(segmento_idx + 1)
            
            df_resultados = pd.DataFrame({
                'x': self.puntos_interpolacion,
                'y_cuadratica': self.valores_interpolados,
                'segmento_usado': segmentos_usados,
                'tipo': tipos
            })
            
            # Crear DataFrame con datos originales
            df_originales = pd.DataFrame({
                'x_original': self.x_datos,
                'y_original': self.y_datos
            })
            
            # Crear DataFrame con polinomios de segmentos
            segmentos_data = []
            for i, seg in enumerate(self.polinomios_segmentos):
                a, b, c = seg['coeficientes']
                segmentos_data.append({
                    'segmento': i + 1,
                    'x_inicio': seg['x_inicio'],
                    'x_fin': seg['x_fin'],
                    'coef_a': a,
                    'coef_b': b,
                    'coef_c': c,
                    'ecuacion': f"{a:.6f}x² + {b:.6f}x + {c:.6f}"
                })
            
            df_segmentos = pd.DataFrame(segmentos_data)
            
            # Guardar archivos
            archivo_resultados = f"{nombre_archivo}_resultados.csv"
            archivo_originales = f"{nombre_archivo}_datos_originales.csv"
            archivo_segmentos = f"{nombre_archivo}_polinomios.csv"
            
            df_resultados.to_csv(archivo_resultados, index=False)
            df_originales.to_csv(archivo_originales, index=False)
            df_segmentos.to_csv(archivo_segmentos, index=False)
            
            # Crear archivo de resumen
            with open(f"{nombre_archivo}_resumen.txt", 'w') as f:
                f.write("INTERPOLACIÓN CUADRÁTICA - RESUMEN\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Método: Interpolación cuadrática por tramos\n")
                f.write(f"Datos base: {self.n_datos} puntos\n")
                f.write(f"Segmentos cuadráticos: {len(self.polinomios_segmentos)}\n")
                f.write(f"Puntos calculados: {len(self.puntos_interpolacion)}\n")
                f.write(f"Interpolaciones: {self.resultados['interpolacion_count']}\n")
                f.write(f"Extrapolaciones: {self.resultados['extrapolacion_count']}\n\n")
                
                f.write(f"Rango original: [{self.x_datos.min():.6f}, {self.x_datos.max():.6f}]\n")
                f.write(f"Rango calculado: [{self.puntos_interpolacion.min():.6f}, {self.puntos_interpolacion.max():.6f}]\n\n")
                
                f.write("Polinomios por segmento:\n")
                for i, seg in enumerate(self.polinomios_segmentos):
                    a, b, c = seg['coeficientes']
                    f.write(f"  Segmento {i+1}: {a:.6f}x² + {b:.6f}x + {c:.6f}\n")
                    f.write(f"    Rango: [{seg['x_inicio']:.6f}, {seg['x_fin']:.6f}]\n")
                
                f.write(f"\nValores interpolados:\n")
                f.write(f"  Mínimo: {self.valores_interpolados.min():.6f}\n")
                f.write(f"  Máximo: {self.valores_interpolados.max():.6f}\n")
                f.write(f"  Promedio: {self.valores_interpolados.mean():.6f}\n")
            
            mostrar_mensaje_exito(f"Resultados exportados:")
            console.print(f"  • {archivo_resultados}")
            console.print(f"  • {archivo_originales}")
            console.print(f"  • {archivo_segmentos}")
            console.print(f"  • {nombre_archivo}_resumen.txt")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {str(e)}")
        
        esperar_enter()
    
    def mostrar_ayuda(self):
        """Mostrar ayuda del método"""
        mostrar_ayuda_metodo(
            "INTERPOLACIÓN CUADRÁTICA",
            "Interpolación cuadrática por tramos usando polinomios de segundo grado",
            {
                "Objetivo": "Estimar valores usando polinomios cuadráticos que pasan por grupos de tres puntos consecutivos",
                "Método": "Construye polinomios de grado 2 para cada trio de puntos y evalúa según la ubicación del punto deseado",
                "Ventajas": [
                    "Mayor suavidad que interpolación lineal",
                    "Captura mejor las curvaturas locales",
                    "Proporciona derivadas continuas",
                    "Cada segmento es exacto en los puntos de datos"
                ],
                "Limitaciones": [
                    "Requiere al menos 3 puntos de datos",
                    "Puede oscilar entre segmentos",
                    "Extrapolación cuadrática puede diverger rápidamente",
                    "No garantiza continuidad de la segunda derivada"
                ]
            }
        )

def main():
    """Función principal del programa"""
    interpolacion = InterpolacionCuadratica()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "INTERPOLACIÓN CUADRÁTICA",
            "Interpolación cuadrática por tramos"
        )
        
        mostrar_banner_metodo(
            "Interpolación Cuadrática",
            "Estima valores usando polinomios cuadráticos por tramos"
        )
        
        interpolacion.mostrar_configuracion()
        
        opciones = [
            "Ingresar/cargar datos base",
            "Configurar puntos de interpolación",
            "Calcular interpolación cuadrática",
            "Ver resultados y análisis",
            "Ver ayuda del método",
            "Salir"
        ]
        
        mostrar_menu_opciones(opciones)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6])
        
        if opcion == 1:
            interpolacion.ingresar_datos()
        elif opcion == 2:
            interpolacion.configurar_interpolacion()
        elif opcion == 3:
            interpolacion.calcular_interpolacion()
        elif opcion == 4:
            interpolacion.mostrar_resultados()
        elif opcion == 5:
            interpolacion.mostrar_ayuda()
        elif opcion == 6:
            console.print("\n[green]¡Gracias por usar interpolación cuadrática![/green]")
            break

if __name__ == "__main__":
    main()
