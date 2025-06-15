#!/usr/bin/env python3
"""
Regresión Lineal - Implementación con menús interactivos
Ajusta una recta y = mx + b a un conjunto de datos usando mínimos cuadrados
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
from rich.layout import Layout

# Agregar directorio padre al path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_numero, validar_opcion_menu, confirmar_accion, limpiar_pantalla,
    mostrar_titulo_principal, mostrar_menu_opciones, mostrar_banner_metodo,
    mostrar_estado_configuracion, mostrar_progreso_ejecucion,
    mostrar_resultado_final, mostrar_mensaje_error, mostrar_mensaje_exito,
    mostrar_ayuda_metodo, esperar_enter, formatear_numero,
    mostrar_tabla_datos, mostrar_estadisticas_datos, graficar_regresion
)

console = Console()

class RegresionLineal:
    def __init__(self):
        self.x_datos = None
        self.y_datos = None
        self.n_datos = 0
        self.metodo_ingreso = None  # 'manual', 'archivo', 'generar'
        self.pendiente = None
        self.ordenada = None
        self.r_cuadrado = None
        self.resultados = None
        
    def configuracion_completa(self) -> bool:
        """Verifica si la configuración está completa"""
        return (self.x_datos is not None and 
                self.y_datos is not None and 
                len(self.x_datos) >= 2)
    
    def mostrar_configuracion(self):
        """Muestra el estado actual de la configuración"""
        config = {
            "Datos": f"{self.n_datos} puntos" if self.n_datos > 0 else None,
            "Método de ingreso": self.metodo_ingreso,
            "Rango X": f"[{np.min(self.x_datos):.3f}, {np.max(self.x_datos):.3f}]" if self.x_datos is not None else None,
            "Rango Y": f"[{np.min(self.y_datos):.3f}, {np.max(self.y_datos):.3f}]" if self.y_datos is not None else None
        }
        mostrar_estado_configuracion(config)
    
    def ingresar_datos(self):
        """Menú principal para ingreso de datos"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Ingreso de Datos")
        
        console.print(Panel(
            "[bold cyan]Seleccione el método para ingresar los datos[/bold cyan]\n\n"
            "[yellow]Opciones disponibles:[/yellow]\n"
            "• Ingreso manual punto por punto\n"
            "• Cargar desde archivo CSV/TXT\n"
            "• Generar datos sintéticos para pruebas\n"
            "• Usar datos de ejemplo predefinidos",
            title="📊 Métodos de Ingreso de Datos",
            border_style="blue"
        ))
        
        opciones = [
            "Ingreso manual de puntos",
            "Cargar desde archivo",
            "Generar datos sintéticos",
            "Usar datos de ejemplo",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Seleccione método de ingreso", False)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5])
        
        if opcion == 1:
            self._ingresar_datos_manual()
        elif opcion == 2:
            self._cargar_desde_archivo()
        elif opcion == 3:
            self._generar_datos_sinteticos()
        elif opcion == 4:
            self._usar_datos_ejemplo()
        elif opcion == 5:
            return
    
    def _ingresar_datos_manual(self):
        """Ingreso manual de datos punto por punto"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Ingreso Manual de Datos")
        
        console.print(Panel(
            "[bold cyan]Ingreso manual de puntos (x, y)[/bold cyan]\n\n"
            "[yellow]Instrucciones:[/yellow]\n"
            "• Ingrese cada punto por separado\n"
            "• Se necesitan mínimo 2 puntos\n"
            "• Más puntos dan mejor ajuste\n"
            "• Presione Enter vacío para terminar",
            title="✏️ Ingreso Manual",
            border_style="yellow"
        ))
        
        puntos_x = []
        puntos_y = []
        i = 1
        
        while True:
            console.print(f"\n[cyan]Punto {i}:[/cyan]")
            
            try:
                x_input = input("  x: ").strip()
                if not x_input:
                    if len(puntos_x) >= 2:
                        break
                    else:
                        console.print("[red]❌ Se necesitan al menos 2 puntos[/red]")
                        continue
                
                x = float(x_input)
                y = validar_numero("  y", "float")
                
                puntos_x.append(x)
                puntos_y.append(y)
                
                console.print(f"[green]✓ Punto ({formatear_numero(x)}, {formatear_numero(y)}) agregado[/green]")
                i += 1
                
            except ValueError:
                console.print("[red]❌ Valor de x inválido[/red]")
        
        if len(puntos_x) >= 2:
            self.x_datos = np.array(puntos_x)
            self.y_datos = np.array(puntos_y)
            self.n_datos = len(puntos_x)
            self.metodo_ingreso = "manual"
            
            mostrar_mensaje_exito(f"✓ {self.n_datos} puntos ingresados correctamente")
            
            # Mostrar resumen de datos
            self._mostrar_resumen_datos()
            
        esperar_enter()
    
    def _cargar_desde_archivo(self):
        """Carga datos desde archivo CSV o TXT"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Cargar desde Archivo")
        
        console.print(Panel(
            "[bold cyan]Cargar datos desde archivo[/bold cyan]\n\n"
            "[yellow]Formatos soportados:[/yellow]\n"
            "• CSV: valores separados por comas\n"
            "• TXT: valores separados por espacios o tabs\n"
            "• Primera columna: valores X\n"
            "• Segunda columna: valores Y\n"
            "• Sin encabezados (solo números)",
            title="📁 Carga de Archivos",
            border_style="blue"
        ))
        
        while True:
            nombre_archivo = input("\nIngrese el nombre del archivo: ").strip()
            
            if not nombre_archivo:
                console.print("[red]❌ Debe ingresar un nombre de archivo[/red]")
                continue
            
            try:
                # Intentar cargar como CSV primero
                if nombre_archivo.endswith('.csv'):
                    datos = pd.read_csv(nombre_archivo, header=None)
                else:
                    # Cargar como archivo delimitado por espacios/tabs
                    datos = pd.read_csv(nombre_archivo, sep=r'\s+', header=None)
                
                if datos.shape[1] < 2:
                    mostrar_mensaje_error("El archivo debe tener al menos 2 columnas (X, Y)")
                    continue
                
                if datos.shape[0] < 2:
                    mostrar_mensaje_error("El archivo debe tener al menos 2 filas de datos")
                    continue
                
                # Tomar las primeras dos columnas
                self.x_datos = datos.iloc[:, 0].values
                self.y_datos = datos.iloc[:, 1].values
                self.n_datos = len(self.x_datos)
                self.metodo_ingreso = "archivo"
                
                mostrar_mensaje_exito(f"✓ {self.n_datos} puntos cargados desde {nombre_archivo}")
                
                # Mostrar resumen de datos
                self._mostrar_resumen_datos()
                break
                
            except FileNotFoundError:
                mostrar_mensaje_error(f"Archivo '{nombre_archivo}' no encontrado")
                if not confirmar_accion("¿Desea intentar con otro archivo?"):
                    break
            except Exception as e:
                mostrar_mensaje_error(f"Error al cargar archivo: {e}")
                if not confirmar_accion("¿Desea intentar con otro archivo?"):
                    break
        
        esperar_enter()
    
    def _generar_datos_sinteticos(self):
        """Genera datos sintéticos para pruebas"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Datos Sintéticos")
        
        console.print(Panel(
            "[bold cyan]Generación de datos sintéticos[/bold cyan]\n\n"
            "[yellow]Parámetros de la recta real:[/yellow]\n"
            "y = mx + b + ruido\n\n"
            "Donde el ruido sigue una distribución normal",
            title="🎲 Generador de Datos",
            border_style="green"
        ))
        
        # Configurar parámetros
        m_real = validar_numero("Pendiente real (m)", "float")
        b_real = validar_numero("Ordenada real (b)", "float")
        n_puntos = validar_numero("Número de puntos", "int", min_val=5, max_val=1000)
        x_min = validar_numero("X mínimo", "float")
        x_max = validar_numero("X máximo", "float")
        
        if x_min >= x_max:
            mostrar_mensaje_error("X máximo debe ser mayor que X mínimo")
            esperar_enter()
            return
        
        ruido_std = validar_numero("Desviación estándar del ruido", "float", min_val=0)
        
        # Generar datos
        np.random.seed(42)  # Para reproducibilidad
        self.x_datos = np.linspace(x_min, x_max, n_puntos)
        y_teorico = m_real * self.x_datos + b_real
        ruido = np.random.normal(0, ruido_std, n_puntos)
        self.y_datos = y_teorico + ruido
        
        self.n_datos = n_puntos
        self.metodo_ingreso = "sintético"
        
        mostrar_mensaje_exito(f"✓ {n_puntos} puntos generados con y = {m_real}x + {b_real} + ruido")
        
        # Mostrar parámetros reales para comparación
        console.print(Panel(
            f"[bold yellow]Parámetros reales (para comparación):[/bold yellow]\n"
            f"Pendiente: {m_real}\n"
            f"Ordenada: {b_real}\n"
            f"Ruido: σ = {ruido_std}",
            title="📊 Valores Teóricos",
            border_style="yellow"
        ))
        
        # Mostrar resumen de datos
        self._mostrar_resumen_datos()
        esperar_enter()
    
    def _usar_datos_ejemplo(self):
        """Usa conjuntos de datos de ejemplo predefinidos"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Datos de Ejemplo")
        
        ejemplos = {
            1: {
                "nombre": "Temperatura vs Ventas de helado",
                "x": [20, 25, 30, 35, 40, 45, 50],
                "y": [10, 15, 25, 35, 50, 65, 80],
                "descripcion": "Relación positiva fuerte"
            },
            2: {
                "nombre": "Años de experiencia vs Salario",
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [30, 35, 38, 42, 48, 52, 58, 64, 70, 75],
                "descripcion": "Crecimiento casi lineal"
            },
            3: {
                "nombre": "Precio vs Demanda",
                "x": [10, 15, 20, 25, 30, 35, 40],
                "y": [100, 85, 70, 55, 40, 25, 10],
                "descripcion": "Relación negativa"
            },
            4: {
                "nombre": "Datos con ruido moderado",
                "x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "y": [2.1, 3.9, 6.2, 7.8, 10.1, 11.8, 14.2, 15.9, 18.1, 19.8],
                "descripcion": "Relación lineal con ruido"
            }
        }
        
        # Mostrar ejemplos disponibles
        table = Table(title="Conjuntos de Datos de Ejemplo", border_style="green")
        table.add_column("Opción", style="cyan", width=8)
        table.add_column("Nombre", style="yellow")
        table.add_column("Puntos", style="white", width=8)
        table.add_column("Descripción", style="dim")
        
        for key, ejemplo in ejemplos.items():
            table.add_row(
                str(key),
                ejemplo["nombre"],
                str(len(ejemplo["x"])),
                ejemplo["descripcion"]
            )
        
        console.print(table)
        
        opciones_validas = list(ejemplos.keys()) + [len(ejemplos) + 1]
        console.print(f"\n[dim]{len(ejemplos) + 1}. Volver al menú anterior[/dim]")
        
        opcion = validar_opcion_menu(opciones_validas, "Seleccione un ejemplo")
        
        if opcion in ejemplos:
            ejemplo = ejemplos[opcion]
            self.x_datos = np.array(ejemplo["x"])
            self.y_datos = np.array(ejemplo["y"])
            self.n_datos = len(ejemplo["x"])
            self.metodo_ingreso = "ejemplo"
            
            mostrar_mensaje_exito(f"✓ Cargado: {ejemplo['nombre']}")
            
            # Mostrar resumen de datos
            self._mostrar_resumen_datos()
            esperar_enter()
    
    def _mostrar_resumen_datos(self):
        """Muestra un resumen de los datos cargados"""
        if self.x_datos is None or self.y_datos is None:
            return
        
        console.print("\n" + "="*60)
        mostrar_estadisticas_datos(self.x_datos, self.y_datos)
        
        if confirmar_accion("¿Desea ver los primeros datos?"):
            mostrar_tabla_datos(self.x_datos, self.y_datos, "Datos Cargados", max_filas=10)
        
        if confirmar_accion("¿Desea ver una gráfica de dispersión?"):
            try:
                plt.figure(figsize=(10, 6))
                plt.scatter(self.x_datos, self.y_datos, alpha=0.7, s=50)
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.title('Gráfica de Dispersión de los Datos')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
            except Exception as e:
                mostrar_mensaje_error(f"Error al mostrar gráfica: {e}")
    
    def calcular_regresion(self):
        """Calcula la regresión lineal usando mínimos cuadrados"""
        if not self.configuracion_completa():
            mostrar_mensaje_error("No hay datos suficientes para calcular la regresión")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Cálculo de Regresión")
        
        self.mostrar_configuracion()
        
        if not confirmar_accion("¿Desea calcular la regresión lineal?"):
            return
        
        mostrar_progreso_ejecucion("Calculando regresión lineal por mínimos cuadrados...")
        
        tiempo_inicio = time.time()
        
        # Cálculos de regresión lineal
        n = len(self.x_datos)
        
        # Sumas necesarias
        sum_x = np.sum(self.x_datos)
        sum_y = np.sum(self.y_datos)
        sum_xy = np.sum(self.x_datos * self.y_datos)
        sum_x2 = np.sum(self.x_datos**2)
        sum_y2 = np.sum(self.y_datos**2)
        
        # Medias
        mean_x = sum_x / n
        mean_y = sum_y / n
        
        # Cálculo de pendiente (m) y ordenada (b)
        denominador = n * sum_x2 - sum_x**2
        
        if abs(denominador) < 1e-15:
            mostrar_mensaje_error("Error: denominador muy pequeño. Los datos X son constantes.")
            esperar_enter()
            return
        
        self.pendiente = (n * sum_xy - sum_x * sum_y) / denominador
        self.ordenada = (sum_y - self.pendiente * sum_x) / n
        
        # Cálculo del coeficiente de determinación R²
        y_pred = self.pendiente * self.x_datos + self.ordenada
        ss_res = np.sum((self.y_datos - y_pred)**2)  # Suma de cuadrados residuales
        ss_tot = np.sum((self.y_datos - mean_y)**2)   # Suma total de cuadrados
        
        if abs(ss_tot) < 1e-15:
            self.r_cuadrado = 1.0  # Todos los Y son iguales
        else:
            self.r_cuadrado = 1 - (ss_res / ss_tot)
        
        # Estadísticas adicionales
        residuos = self.y_datos - y_pred
        error_estandar = np.sqrt(ss_res / (n - 2)) if n > 2 else 0
        
        # Correlación de Pearson
        correlacion = np.corrcoef(self.x_datos, self.y_datos)[0, 1]
        
        # Intervalos de confianza para los coeficientes (aproximados)
        s_xx = sum_x2 - (sum_x**2 / n)
        error_m = error_estandar / np.sqrt(s_xx) if s_xx > 0 else 0
        error_b = error_estandar * np.sqrt(1/n + mean_x**2/s_xx) if s_xx > 0 else 0
        
        tiempo_ejecucion = time.time() - tiempo_inicio
        
        # Preparar resultados
        self.resultados = {
            "pendiente": self.pendiente,
            "ordenada": self.ordenada,
            "r_cuadrado": self.r_cuadrado,
            "correlacion": correlacion,
            "error_estandar": error_estandar,
            "residuos": residuos,
            "y_predichos": y_pred,
            "suma_residuos_cuadrados": ss_res,
            "suma_total_cuadrados": ss_tot,
            "error_pendiente": error_m,
            "error_ordenada": error_b,
            "tiempo_ejecucion": tiempo_ejecucion,
            "estadisticas": {
                "n_puntos": n,
                "sum_x": sum_x,
                "sum_y": sum_y,
                "sum_xy": sum_xy,
                "sum_x2": sum_x2,
                "sum_y2": sum_y2,
                "mean_x": mean_x,
                "mean_y": mean_y
            }
        }
        
        mostrar_mensaje_exito("¡Regresión calculada exitosamente!")
        esperar_enter()
    
    def mostrar_resultados(self):
        """Muestra los resultados de la regresión"""
        if self.resultados is None:
            mostrar_mensaje_error("No hay resultados disponibles. Calcule la regresión primero.")
            esperar_enter()
            return
        
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Resultados")
        
        # Ecuación de la recta
        m, b = self.pendiente, self.ordenada
        signo = "+" if b >= 0 else "-"
        ecuacion = f"y = {formatear_numero(m)}x {signo} {formatear_numero(abs(b))}"
        
        console.print(Panel(
            f"[bold green]{ecuacion}[/bold green]",
            title="📈 Ecuación de la Recta Ajustada",
            border_style="green"
        ))
        
        # Mostrar resultado principal
        resultado_display = {
            "pendiente_m": self.pendiente,
            "ordenada_b": self.ordenada,
            "r_cuadrado": self.r_cuadrado,
            "correlacion": self.resultados["correlacion"],
            "error_estandar": self.resultados["error_estandar"],
            "puntos_datos": self.n_datos
        }
        
        mostrar_resultado_final("Regresión Lineal", resultado_display, self.resultados["tiempo_ejecucion"])
        
        # Interpretación del R²
        r2 = self.r_cuadrado
        if r2 > 0.9:
            interpretacion = "Excelente ajuste"
            color = "green"
        elif r2 > 0.7:
            interpretacion = "Buen ajuste"
            color = "yellow"
        elif r2 > 0.5:
            interpretacion = "Ajuste moderado"
            color = "orange"
        else:
            interpretacion = "Ajuste pobre"
            color = "red"
        
        console.print(Panel(
            f"[bold {color}]R² = {formatear_numero(r2)} ({interpretacion})[/bold {color}]\n\n"
            f"Esto significa que el {formatear_numero(r2*100)}% de la variabilidad\n"
            f"en Y es explicada por la relación lineal con X.",
            title="📊 Interpretación del Ajuste",
            border_style=color
        ))
        
        # Menú de opciones para resultados
        opciones = [
            "Ver gráfica de regresión",
            "Ver análisis de residuos",
            "Ver estadísticas detalladas",
            "Hacer predicciones",
            "Exportar resultados",
            "Volver al menú principal"
        ]
        
        mostrar_menu_opciones(opciones, "Opciones de resultados", False)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6])
        
        if opcion == 1:
            self.mostrar_grafica_regresion()
        elif opcion == 2:
            self.analizar_residuos()
        elif opcion == 3:
            self.mostrar_estadisticas_detalladas()
        elif opcion == 4:
            self.hacer_predicciones()
        elif opcion == 5:
            self.exportar_resultados()
        elif opcion == 6:
            return
    
    def mostrar_grafica_regresion(self):
        """Muestra la gráfica de regresión con datos y recta ajustada"""
        console.print("[yellow]Generando gráfica de regresión...[/yellow]")
        
        try:
            # Crear puntos para la recta ajustada
            x_min, x_max = np.min(self.x_datos), np.max(self.x_datos)
            margen = (x_max - x_min) * 0.1
            x_ajuste = np.linspace(x_min - margen, x_max + margen, 100)
            y_ajuste = self.pendiente * x_ajuste + self.ordenada
            
            # Ecuación para el título
            m, b = self.pendiente, self.ordenada
            signo = "+" if b >= 0 else "-"
            ecuacion = f"y = {formatear_numero(m)}x {signo} {formatear_numero(abs(b))}"
            titulo = f"Regresión Lineal: {ecuacion} (R² = {formatear_numero(self.r_cuadrado)})"
            
            graficar_regresion(
                self.x_datos,
                self.y_datos,
                x_ajuste,
                y_ajuste,
                titulo,
                "Datos observados",
                f"Ajuste lineal (R² = {formatear_numero(self.r_cuadrado)})"
            )
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {e}")
        
        esperar_enter()
    
    def analizar_residuos(self):
        """Analiza los residuos para validar el modelo"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Análisis de Residuos")
        
        residuos = self.resultados["residuos"]
        y_pred = self.resultados["y_predichos"]
        
        # Estadísticas de residuos
        media_residuos = np.mean(residuos)
        std_residuos = np.std(residuos)
        residuo_max = np.max(np.abs(residuos))
        
        console.print(Panel(
            f"[bold cyan]Estadísticas de Residuos[/bold cyan]\n\n"
            f"Media: {formatear_numero(media_residuos)} (ideal: ≈ 0)\n"
            f"Desviación estándar: {formatear_numero(std_residuos)}\n"
            f"Residuo máximo: {formatear_numero(residuo_max)}\n"
            f"Error estándar: {formatear_numero(self.resultados['error_estandar'])}",
            title="📊 Análisis de Residuos",
            border_style="blue"
        ))
        
        # Gráficas de residuos
        if confirmar_accion("¿Desea ver las gráficas de residuos?"):
            try:
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                
                # 1. Residuos vs Valores predichos
                ax1.scatter(y_pred, residuos, alpha=0.7)
                ax1.axhline(y=0, color='red', linestyle='--')
                ax1.set_xlabel('Valores Predichos')
                ax1.set_ylabel('Residuos')
                ax1.set_title('Residuos vs Valores Predichos')
                ax1.grid(True, alpha=0.3)
                
                # 2. Residuos vs Valores X
                ax2.scatter(self.x_datos, residuos, alpha=0.7)
                ax2.axhline(y=0, color='red', linestyle='--')
                ax2.set_xlabel('Valores X')
                ax2.set_ylabel('Residuos')
                ax2.set_title('Residuos vs Valores X')
                ax2.grid(True, alpha=0.3)
                
                # 3. Histograma de residuos
                ax3.hist(residuos, bins=min(10, len(residuos)//2), alpha=0.7, edgecolor='black')
                ax3.set_xlabel('Residuos')
                ax3.set_ylabel('Frecuencia')
                ax3.set_title('Distribución de Residuos')
                ax3.grid(True, alpha=0.3)
                
                # 4. Q-Q plot (aproximado)
                residuos_ordenados = np.sort(residuos)
                n = len(residuos_ordenados)
                teoricos = np.random.normal(0, std_residuos, n)
                teoricos = np.sort(teoricos)
                
                ax4.scatter(teoricos, residuos_ordenados, alpha=0.7)
                # Línea de referencia
                min_val = min(np.min(teoricos), np.min(residuos_ordenados))
                max_val = max(np.max(teoricos), np.max(residuos_ordenados))
                ax4.plot([min_val, max_val], [min_val, max_val], 'r--')
                ax4.set_xlabel('Cuantiles Teóricos')
                ax4.set_ylabel('Cuantiles Observados')
                ax4.set_title('Q-Q Plot (Normalidad)')
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.show()
                
            except Exception as e:
                mostrar_mensaje_error(f"Error al generar gráficas: {e}")
        
        esperar_enter()
    
    def mostrar_estadisticas_detalladas(self):
        """Muestra estadísticas detalladas del ajuste"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Estadísticas Detalladas")
        
        # Crear layout con dos paneles
        layout = Layout()
        layout.split_row(
            Layout(name="coeficientes"),
            Layout(name="ajuste")
        )
        
        # Panel de coeficientes
        coef_table = Table(title="Coeficientes de Regresión", border_style="blue")
        coef_table.add_column("Coeficiente", style="cyan")
        coef_table.add_column("Valor", style="white")
        coef_table.add_column("Error Estándar", style="yellow")
        
        coef_table.add_row("Pendiente (m)", formatear_numero(self.pendiente), formatear_numero(self.resultados["error_pendiente"]))
        coef_table.add_row("Ordenada (b)", formatear_numero(self.ordenada), formatear_numero(self.resultados["error_ordenada"]))
        
        # Panel de bondad de ajuste
        ajuste_table = Table(title="Bondad de Ajuste", border_style="green")
        ajuste_table.add_column("Estadística", style="cyan")
        ajuste_table.add_column("Valor", style="white")
        
        ajuste_table.add_row("R²", formatear_numero(self.r_cuadrado))
        ajuste_table.add_row("R (correlación)", formatear_numero(self.resultados["correlacion"]))
        ajuste_table.add_row("Error estándar", formatear_numero(self.resultados["error_estandar"]))
        ajuste_table.add_row("SSR (residuos)", formatear_numero(self.resultados["suma_residuos_cuadrados"]))
        ajuste_table.add_row("SST (total)", formatear_numero(self.resultados["suma_total_cuadrados"]))
        
        layout["coeficientes"].update(coef_table)
        layout["ajuste"].update(ajuste_table)
        
        console.print(layout)
        console.print()
        
        # Tabla de sumas para cálculos
        sumas_table = Table(title="Sumas para Cálculos", border_style="yellow")
        sumas_table.add_column("Suma", style="cyan")
        sumas_table.add_column("Valor", style="white")
        
        stats = self.resultados["estadisticas"]
        sumas_table.add_row("n (puntos)", str(stats["n_puntos"]))
        sumas_table.add_row("Σx", formatear_numero(stats["sum_x"]))
        sumas_table.add_row("Σy", formatear_numero(stats["sum_y"]))
        sumas_table.add_row("Σxy", formatear_numero(stats["sum_xy"]))
        sumas_table.add_row("Σx²", formatear_numero(stats["sum_x2"]))
        sumas_table.add_row("x̄ (media)", formatear_numero(stats["mean_x"]))
        sumas_table.add_row("ȳ (media)", formatear_numero(stats["mean_y"]))
        
        console.print(sumas_table)
        
        esperar_enter()
    
    def hacer_predicciones(self):
        """Permite hacer predicciones con el modelo"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Predicciones")
        
        m, b = self.pendiente, self.ordenada
        signo = "+" if b >= 0 else "-"
        ecuacion = f"y = {formatear_numero(m)}x {signo} {formatear_numero(abs(b))}"
        
        console.print(Panel(
            f"[bold green]Modelo: {ecuacion}[/bold green]\n"
            f"R² = {formatear_numero(self.r_cuadrado)}",
            title="📈 Modelo de Predicción",
            border_style="green"
        ))
        
        while True:
            opciones = [
                "Predecir Y para un valor X",
                "Predecir X para un valor Y",
                "Predicciones múltiples",
                "Volver al menú de resultados"
            ]
            
            mostrar_menu_opciones(opciones, "Tipo de predicción", False)
            opcion = validar_opcion_menu([1, 2, 3, 4])
            
            if opcion == 1:
                x_pred = validar_numero("Ingrese el valor de X para predecir Y", "float")
                y_pred = self.pendiente * x_pred + self.ordenada
                
                console.print(Panel(
                    f"[bold cyan]Para X = {formatear_numero(x_pred)}:[/bold cyan]\n"
                    f"Y predicho = {formatear_numero(y_pred)}",
                    title="🎯 Predicción Y",
                    border_style="cyan"
                ))
                
            elif opcion == 2:
                if abs(self.pendiente) < 1e-15:
                    mostrar_mensaje_error("No se puede predecir X cuando la pendiente es cero")
                else:
                    y_pred = validar_numero("Ingrese el valor de Y para predecir X", "float")
                    x_pred = (y_pred - self.ordenada) / self.pendiente
                    
                    console.print(Panel(
                        f"[bold cyan]Para Y = {formatear_numero(y_pred)}:[/bold cyan]\n"
                        f"X predicho = {formatear_numero(x_pred)}",
                        title="🎯 Predicción X",
                        border_style="cyan"
                    ))
                    
            elif opcion == 3:
                self._predicciones_multiples()
                
            elif opcion == 4:
                break
            
            if opcion != 4:
                esperar_enter()
    
    def _predicciones_multiples(self):
        """Realiza múltiples predicciones"""
        console.print("\n[yellow]Predicciones múltiples - Ingrese valores separados por comas[/yellow]")
        
        valores_input = input("Valores de X: ").strip()
        
        try:
            valores_x = [float(x.strip()) for x in valores_input.split(',')]
            
            table = Table(title="Predicciones Múltiples", border_style="green")
            table.add_column("X", style="cyan")
            table.add_column("Y Predicho", style="white")
            
            for x_val in valores_x:
                y_pred = self.pendiente * x_val + self.ordenada
                table.add_row(formatear_numero(x_val), formatear_numero(y_pred))
            
            console.print(table)
            
        except ValueError:
            mostrar_mensaje_error("Error en el formato de los valores. Use números separados por comas.")
    
    def exportar_resultados(self):
        """Exporta los resultados a archivos"""
        limpiar_pantalla()
        mostrar_titulo_principal("Regresión Lineal", "Exportar Resultados")
        
        opciones = [
            "Exportar resumen a TXT",
            "Exportar datos y predicciones a CSV",
            "Exportar todo (TXT + CSV)",
            "Volver al menú de resultados"
        ]
        
        mostrar_menu_opciones(opciones, "Formato de exportación", False)
        opcion = validar_opcion_menu([1, 2, 3, 4])
        
        if opcion == 4:
            return
        
        timestamp = int(time.time())
        
        try:
            if opcion in [1, 3]:
                # Exportar resumen TXT
                nombre_txt = f"regresion_lineal_resumen_{timestamp}.txt"
                self._exportar_resumen_txt(nombre_txt)
                
            if opcion in [2, 3]:
                # Exportar datos CSV
                nombre_csv = f"regresion_lineal_datos_{timestamp}.csv"
                self._exportar_datos_csv(nombre_csv)
                
            mostrar_mensaje_exito("Archivos exportados exitosamente")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {e}")
        
        esperar_enter()
    
    def _exportar_resumen_txt(self, nombre_archivo):
        """Exporta resumen completo a TXT"""
        with open(nombre_archivo, 'w', encoding='utf-8') as f:
            f.write("RESULTADOS DE REGRESIÓN LINEAL\n")
            f.write("=" * 50 + "\n\n")
            
            # Ecuación
            m, b = self.pendiente, self.ordenada
            signo = "+" if b >= 0 else "-"
            f.write(f"ECUACIÓN DE LA RECTA: y = {formatear_numero(m)}x {signo} {formatear_numero(abs(b))}\n\n")
            
            # Coeficientes
            f.write("COEFICIENTES:\n")
            f.write(f"Pendiente (m): {formatear_numero(self.pendiente)}\n")
            f.write(f"Ordenada (b): {formatear_numero(self.ordenada)}\n")
            f.write(f"Error estándar de m: {formatear_numero(self.resultados['error_pendiente'])}\n")
            f.write(f"Error estándar de b: {formatear_numero(self.resultados['error_ordenada'])}\n\n")
            
            # Bondad de ajuste
            f.write("BONDAD DE AJUSTE:\n")
            f.write(f"R² (coeficiente de determinación): {formatear_numero(self.r_cuadrado)}\n")
            f.write(f"R (correlación): {formatear_numero(self.resultados['correlacion'])}\n")
            f.write(f"Error estándar: {formatear_numero(self.resultados['error_estandar'])}\n\n")
            
            # Estadísticas
            stats = self.resultados["estadisticas"]
            f.write("ESTADÍSTICAS:\n")
            f.write(f"Número de puntos: {stats['n_puntos']}\n")
            f.write(f"Media de X: {formatear_numero(stats['mean_x'])}\n")
            f.write(f"Media de Y: {formatear_numero(stats['mean_y'])}\n")
            f.write(f"Suma de cuadrados residuales: {formatear_numero(self.resultados['suma_residuos_cuadrados'])}\n")
            f.write(f"Suma total de cuadrados: {formatear_numero(self.resultados['suma_total_cuadrados'])}\n")
            
        console.print(f"[green]Resumen exportado a: {nombre_archivo}[/green]")
    
    def _exportar_datos_csv(self, nombre_archivo):
        """Exporta datos y predicciones a CSV"""
        df = pd.DataFrame({
            'X': self.x_datos,
            'Y_Observado': self.y_datos,
            'Y_Predicho': self.resultados['y_predichos'],
            'Residuo': self.resultados['residuos']
        })
        
        df.to_csv(nombre_archivo, index=False)
        console.print(f"[green]Datos exportados a: {nombre_archivo}[/green]")
    
    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre regresión lineal"""
        mostrar_ayuda_metodo(
            "Regresión Lineal",
            "La regresión lineal encuentra la mejor recta que se ajusta a un conjunto de datos "
            "usando el método de mínimos cuadrados. Minimiza la suma de cuadrados de las "
            "diferencias entre valores observados y predichos.",
            [
                "Modelar relaciones lineales entre variables",
                "Predicción de valores futuros",
                "Análisis de tendencias en datos",
                "Base para modelos más complejos",
                "Control de calidad en procesos",
                "Análisis financiero y económico"
            ],
            [
                "Fácil de interpretar y explicar",
                "Cálculos simples y rápidos",
                "Buena base estadística",
                "Permite predicciones",
                "Funciona bien con relaciones lineales",
                "Proporciona medidas de bondad de ajuste"
            ],
            [
                "Solo modela relaciones lineales",
                "Sensible a valores atípicos",
                "Asume homocedasticidad",
                "Requiere independencia de residuos",
                "No funciona bien con datos no lineales"
            ]
        )

def main():
    """Función principal del programa"""
    regresion = RegresionLineal()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "REGRESIÓN LINEAL",
            "Ajuste de recta por mínimos cuadrados"
        )
        
        mostrar_banner_metodo(
            "Regresión Lineal",
            "Encuentra la mejor recta y = mx + b que se ajusta a los datos"
        )
        
        regresion.mostrar_configuracion()
        
        opciones = [
            "Ingresar/cargar datos",
            "Calcular regresión lineal",
            "Ver resultados y análisis",
            "Ver ayuda del método",
            "Salir"
        ]
        
        mostrar_menu_opciones(opciones)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5])
        
        if opcion == 1:
            regresion.ingresar_datos()
        elif opcion == 2:
            regresion.calcular_regresion()
        elif opcion == 3:
            regresion.mostrar_resultados()
        elif opcion == 4:
            regresion.mostrar_ayuda()
        elif opcion == 5:
            console.print("\n[green]¡Gracias por usar regresión lineal![/green]")
            break

if __name__ == "__main__":
    main()
