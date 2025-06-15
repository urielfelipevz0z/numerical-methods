#!/usr/bin/env python3
"""
Regresión No Lineal - Implementación con menús interactivos
Ajusta funciones no lineales a datos usando optimización de Levenberg-Marquardt
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit, least_squares
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

class RegresionNoLineal:
    def __init__(self):
        self.x_datos = None
        self.y_datos = None
        self.n_datos = 0
        self.funcion_tipo = None
        self.funcion_modelo = None
        self.nombre_funcion = None
        self.parametros_iniciales = None
        self.parametros_optimizados = None
        self.r_cuadrado = None
        self.resultados = None
        self.metodo_ingreso = None
        
        # Funciones predefinidas
        self.funciones_disponibles = {
            1: {
                'nombre': 'Exponencial: a*exp(b*x)',
                'funcion': lambda x, a, b: a * np.exp(b * x),
                'parametros': ['a', 'b'],
                'iniciales': [1.0, 0.1]
            },
            2: {
                'nombre': 'Potencial: a*x^b',
                'funcion': lambda x, a, b: a * np.power(np.abs(x), b),
                'parametros': ['a', 'b'],
                'iniciales': [1.0, 1.0]
            },
            3: {
                'nombre': 'Logarítmica: a*ln(x) + b',
                'funcion': lambda x, a, b: a * np.log(np.abs(x)) + b,
                'parametros': ['a', 'b'],
                'iniciales': [1.0, 0.0]
            },
            4: {
                'nombre': 'Gaussiana: a*exp(-((x-b)/c)²)',
                'funcion': lambda x, a, b, c: a * np.exp(-((x - b) / c)**2),
                'parametros': ['a', 'b', 'c'],
                'iniciales': [1.0, 0.0, 1.0]
            },
            5: {
                'nombre': 'Sigmoidal: a/(1+exp(-b*(x-c)))',
                'funcion': lambda x, a, b, c: a / (1 + np.exp(-b * (x - c))),
                'parametros': ['a', 'b', 'c'],
                'iniciales': [1.0, 1.0, 0.0]
            },
            6: {
                'nombre': 'Michaelis-Menten: (a*x)/(b+x)',
                'funcion': lambda x, a, b: (a * x) / (b + x),
                'parametros': ['a', 'b'],
                'iniciales': [1.0, 1.0]
            }
        }
    
    def configuracion_completa(self) -> bool:
        """Verifica si la configuración está completa"""
        return (self.x_datos is not None and 
                self.y_datos is not None and 
                self.funcion_modelo is not None and
                len(self.x_datos) >= len(self.parametros_iniciales))
    
    def mostrar_configuracion(self):
        """Muestra el estado actual de la configuración"""
        tabla = Table(title="Estado de la Configuración")
        tabla.add_column("Parámetro", style="cyan")
        tabla.add_column("Valor", style="yellow")
        tabla.add_column("Estado", style="green")
        
        # Datos
        if self.x_datos is not None:
            tabla.add_row(
                "Datos",
                f"{len(self.x_datos)} puntos cargados",
                "✓ Configurado"
            )
        else:
            tabla.add_row("Datos", "No configurados", "⚠ Pendiente")
        
        # Función modelo
        if self.funcion_modelo is not None:
            tabla.add_row(
                "Función modelo",
                self.nombre_funcion,
                "✓ Configurado"
            )
        else:
            tabla.add_row("Función modelo", "No seleccionada", "⚠ Pendiente")
        
        # Parámetros iniciales
        if self.parametros_iniciales is not None:
            params_str = ", ".join([f"{p}={v:.3f}" for p, v in 
                                  zip(self.funciones_disponibles[self.funcion_tipo]['parametros'], 
                                      self.parametros_iniciales)])
            tabla.add_row(
                "Parámetros iniciales",
                params_str,
                "✓ Configurado"
            )
        
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
        """Menú para ingresar o cargar datos"""
        while True:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]CONFIGURACIÓN DE DATOS[/bold cyan]",
                style="cyan"
            ))
            
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
            
            n = validar_numero("Número de puntos (mínimo 3): ", tipo=int, minimo=3, maximo=100)
            
            x_datos = []
            y_datos = []
            
            console.print(f"\n[yellow]Ingrese {n} pares de datos (x, y):[/yellow]")
            
            for i in range(n):
                console.print(f"\n[cyan]Punto {i+1}:[/cyan]")
                x = validar_numero(f"  x{i+1}: ", tipo=float)
                y = validar_numero(f"  y{i+1}: ", tipo=float)
                x_datos.append(x)
                y_datos.append(y)
            
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
                
                self.x_datos = datos.iloc[:, 0].values
                self.y_datos = datos.iloc[:, 1].values
                self.n_datos = len(self.x_datos)
                
                if self.n_datos < 3:
                    raise ValueError("Se necesitan al menos 3 datos")
                
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
            
            n = validar_numero("Número de puntos (10-100): ", tipo=int, minimo=10, maximo=100)
            x_min = validar_numero("Valor mínimo de x: ", tipo=float)
            x_max = validar_numero("Valor máximo de x: ", tipo=float)
            
            if x_max <= x_min:
                mostrar_mensaje_error("x_max debe ser mayor que x_min")
                esperar_enter()
                return
            
            # Mostrar funciones disponibles
            console.print("\n[yellow]Seleccione tipo de función para generar:[/yellow]")
            for key, func_info in self.funciones_disponibles.items():
                console.print(f"{key}. {func_info['nombre']}")
            
            tipo_func = validar_opcion_menu(list(self.funciones_disponibles.keys()))
            
            # Generar datos
            x_datos = np.linspace(x_min, x_max, n)
            
            # Usar función seleccionada con parámetros predeterminados
            func_info = self.funciones_disponibles[tipo_func]
            y_datos = func_info['funcion'](x_datos, *func_info['iniciales'])
            
            # Agregar ruido
            ruido = validar_numero("Nivel de ruido (0.0-0.5): ", tipo=float, minimo=0, maximo=0.5)
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
    
    def seleccionar_funcion(self):
        """Seleccionar función modelo para el ajuste"""
        while True:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]SELECCIÓN DE FUNCIÓN MODELO[/bold cyan]",
                style="cyan"
            ))
            
            console.print("\n[yellow]Funciones disponibles:[/yellow]")
            for key, func_info in self.funciones_disponibles.items():
                console.print(f"{key}. {func_info['nombre']}")
            
            console.print(f"{len(self.funciones_disponibles) + 1}. Volver al menú principal")
            
            opciones_validas = list(self.funciones_disponibles.keys()) + [len(self.funciones_disponibles) + 1]
            opcion = validar_opcion_menu(opciones_validas)
            
            if opcion == len(self.funciones_disponibles) + 1:
                break
            
            # Configurar función seleccionada
            self.funcion_tipo = opcion
            func_info = self.funciones_disponibles[opcion]
            self.funcion_modelo = func_info['funcion']
            self.nombre_funcion = func_info['nombre']
            self.parametros_iniciales = func_info['iniciales'].copy()
            
            mostrar_mensaje_exito(f"Función seleccionada: {self.nombre_funcion}")
            
            # Opción para modificar parámetros iniciales
            if confirmar_accion("¿Desea modificar los parámetros iniciales?"):
                self._configurar_parametros_iniciales()
            
            break
    
    def _configurar_parametros_iniciales(self):
        """Configurar parámetros iniciales para el ajuste"""
        try:
            console.print("\n[cyan]Configuración de parámetros iniciales[/cyan]")
            func_info = self.funciones_disponibles[self.funcion_tipo]
            
            nuevos_parametros = []
            for i, (nombre, valor_actual) in enumerate(zip(func_info['parametros'], self.parametros_iniciales)):
                nuevo_valor = validar_numero(
                    f"Valor inicial para {nombre} (actual: {valor_actual}): ",
                    tipo=float
                )
                nuevos_parametros.append(nuevo_valor)
            
            self.parametros_iniciales = nuevos_parametros
            mostrar_mensaje_exito("Parámetros iniciales actualizados")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al configurar parámetros: {str(e)}")
        
        esperar_enter()
    
    def calcular_regresion(self):
        """Calcular la regresión no lineal"""
        if not self.configuracion_completa():
            mostrar_mensaje_error("Configure los datos y función antes de calcular")
            esperar_enter()
            return
        
        try:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]CÁLCULO DE REGRESIÓN NO LINEAL[/bold cyan]",
                style="cyan"
            ))
            
            with mostrar_progreso_ejecucion("Optimizando parámetros..."):
                time.sleep(0.5)
                
                # Usar curve_fit de scipy para optimización
                try:
                    # Intentar ajuste con curve_fit
                    self.parametros_optimizados, covarianza = curve_fit(
                        self.funcion_modelo,
                        self.x_datos,
                        self.y_datos,
                        p0=self.parametros_iniciales,
                        maxfev=5000
                    )
                    
                    # Calcular predicciones
                    y_pred = self.funcion_modelo(self.x_datos, *self.parametros_optimizados)
                    
                    # Calcular R²
                    ss_res = np.sum((self.y_datos - y_pred) ** 2)
                    ss_tot = np.sum((self.y_datos - np.mean(self.y_datos)) ** 2)
                    self.r_cuadrado = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Calcular errores estándar de los parámetros
                    errores_std = np.sqrt(np.diag(covarianza))
                    
                    # Guardar resultados
                    self.resultados = {
                        'parametros': self.parametros_optimizados,
                        'errores_std': errores_std,
                        'covarianza': covarianza,
                        'y_predichos': y_pred,
                        'r_cuadrado': self.r_cuadrado,
                        'error_cuadratico_medio': np.mean((self.y_datos - y_pred) ** 2),
                        'error_absoluto_medio': np.mean(np.abs(self.y_datos - y_pred)),
                        'exitoso': True
                    }
                    
                except Exception as e:
                    mostrar_mensaje_error(f"Error en optimización: {str(e)}")
                    esperar_enter()
                    return
            
            mostrar_mensaje_exito("Regresión calculada exitosamente")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error en el cálculo: {str(e)}")
        
        esperar_enter()
    
    def mostrar_resultados(self):
        """Mostrar resultados detallados"""
        if self.resultados is None:
            mostrar_mensaje_error("No hay resultados para mostrar. Ejecute el cálculo primero.")
            esperar_enter()
            return
        
        while True:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]RESULTADOS DE REGRESIÓN NO LINEAL[/bold cyan]",
                style="cyan"
            ))
            
            opciones = [
                "Ver parámetros optimizados",
                "Ver estadísticas del ajuste",
                "Ver tabla de datos vs predicciones",
                "Generar gráfica",
                "Análisis de sensibilidad",
                "Exportar resultados",
                "Volver al menú principal"
            ]
            
            mostrar_menu_opciones(opciones)
            opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6, 7])
            
            if opcion == 1:
                self._mostrar_parametros()
            elif opcion == 2:
                self._mostrar_estadisticas()
            elif opcion == 3:
                self._mostrar_tabla_comparacion()
            elif opcion == 4:
                self._generar_grafica()
            elif opcion == 5:
                self._analisis_sensibilidad()
            elif opcion == 6:
                self._exportar_resultados()
            elif opcion == 7:
                break
    
    def _mostrar_parametros(self):
        """Mostrar parámetros optimizados"""
        tabla = Table(title="Parámetros Optimizados")
        tabla.add_column("Parámetro", style="cyan")
        tabla.add_column("Valor Inicial", style="yellow")
        tabla.add_column("Valor Optimizado", style="green")
        tabla.add_column("Error Estándar", style="red")
        tabla.add_column("Intervalo 95% Conf.", style="magenta")
        
        func_info = self.funciones_disponibles[self.funcion_tipo]
        
        for i, nombre in enumerate(func_info['parametros']):
            valor_inicial = self.parametros_iniciales[i]
            valor_optimo = self.parametros_optimizados[i]
            error_std = self.resultados['errores_std'][i]
            
            # Intervalo de confianza 95% (aproximado)
            intervalo_inf = valor_optimo - 1.96 * error_std
            intervalo_sup = valor_optimo + 1.96 * error_std
            
            tabla.add_row(
                nombre,
                f"{valor_inicial:.6f}",
                f"{valor_optimo:.6f}",
                f"±{error_std:.6f}",
                f"[{intervalo_inf:.4f}, {intervalo_sup:.4f}]"
            )
        
        console.print(tabla)
        
        # Mostrar función ajustada
        console.print(f"\n[green]Función ajustada:[/green]")
        funcion_str = self._construir_funcion_ajustada()
        panel = Panel(
            funcion_str,
            title="Función con Parámetros Optimizados",
            style="green"
        )
        console.print(panel)
        
        esperar_enter()
    
    def _construir_funcion_ajustada(self):
        """Construir representación string de la función ajustada"""
        func_info = self.funciones_disponibles[self.funcion_tipo]
        nombre_base = func_info['nombre'].split(':')[1].strip()
        
        # Reemplazar parámetros por valores optimizados
        funcion_str = nombre_base
        for i, param in enumerate(func_info['parametros']):
            valor = self.parametros_optimizados[i]
            funcion_str = funcion_str.replace(param, f"{valor:.4f}")
        
        return funcion_str
    
    def _mostrar_estadisticas(self):
        """Mostrar estadísticas del ajuste"""
        tabla = Table(title="Estadísticas del Ajuste")
        tabla.add_column("Métrica", style="cyan")
        tabla.add_column("Valor", style="yellow")
        tabla.add_column("Interpretación", style="green")
        
        r2 = self.resultados['r_cuadrado']
        mse = self.resultados['error_cuadratico_medio']
        mae = self.resultados['error_absoluto_medio']
        
        # R²
        if r2 > 0.9:
            interpretacion_r2 = "Excelente ajuste"
        elif r2 > 0.7:
            interpretacion_r2 = "Buen ajuste"
        elif r2 > 0.5:
            interpretacion_r2 = "Ajuste moderado"
        else:
            interpretacion_r2 = "Ajuste pobre"
        
        tabla.add_row("R² (Coef. determinación)", f"{r2:.6f}", interpretacion_r2)
        tabla.add_row("Error cuadrático medio", f"{mse:.6f}", "Menor es mejor")
        tabla.add_row("Error absoluto medio", f"{mae:.6f}", "Menor es mejor")
        tabla.add_row("Número de datos", str(self.n_datos), "")
        tabla.add_row("Función modelo", self.nombre_funcion, "")
        
        console.print(tabla)
        esperar_enter()
    
    def _mostrar_tabla_comparacion(self):
        """Mostrar tabla de datos originales vs predicciones"""
        mostrar_tabla_datos(
            self.x_datos,
            self.y_datos,
            y_predichos=self.resultados['y_predichos'],
            titulo="Datos vs Predicciones"
        )
        esperar_enter()
    
    def _generar_grafica(self):
        """Generar gráfica de la regresión"""
        try:
            self._graficar_regresion_no_lineal()
            mostrar_mensaje_exito("Gráfica generada exitosamente")
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {str(e)}")
        
        esperar_enter()
    
    def _graficar_regresion_no_lineal(self):
        """Crear gráfica de la regresión no lineal"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Gráfica principal
        ax1.scatter(self.x_datos, self.y_datos, alpha=0.7, color='blue', 
                   label='Datos originales', s=50)
        
        # Generar curva suave para la función
        x_smooth = np.linspace(self.x_datos.min(), self.x_datos.max(), 300)
        try:
            y_smooth = self.funcion_modelo(x_smooth, *self.parametros_optimizados)
            ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                    label='Función ajustada')
        except:
            # Si hay problemas con la función en el rango completo, usar solo los puntos de datos
            y_pred = self.resultados['y_predichos']
            indices_orden = np.argsort(self.x_datos)
            ax1.plot(self.x_datos[indices_orden], y_pred[indices_orden], 'r-', linewidth=2, 
                    label='Función ajustada')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Regresión No Lineal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Agregar información
        info_text = f"R² = {self.r_cuadrado:.4f}\n{self.nombre_funcion}"
        ax1.text(0.05, 0.95, info_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Gráfica de residuos vs predicciones
        residuos = self.y_datos - self.resultados['y_predichos']
        ax2.scatter(self.resultados['y_predichos'], residuos, alpha=0.7, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Valores predichos')
        ax2.set_ylabel('Residuos')
        ax2.set_title('Residuos vs Predicciones')
        ax2.grid(True, alpha=0.3)
        
        # Histograma de residuos
        ax3.hist(residuos, bins=min(15, len(residuos)//2), alpha=0.7, color='orange', edgecolor='black')
        ax3.set_xlabel('Residuos')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribución de Residuos')
        ax3.grid(True, alpha=0.3)
        
        # Q-Q plot (aproximado)
        from scipy import stats
        stats.probplot(residuos, dist="norm", plot=ax4)
        ax4.set_title('Q-Q Plot (Normalidad de Residuos)')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _analisis_sensibilidad(self):
        """Análisis de sensibilidad de parámetros"""
        try:
            console.print("\n[cyan]Análisis de Sensibilidad[/cyan]")
            
            # Crear gráficas de sensibilidad
            n_params = len(self.parametros_optimizados)
            fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
            
            if n_params == 1:
                axes = [axes]
            
            func_info = self.funciones_disponibles[self.funcion_tipo]
            
            for i, (ax, param_name) in enumerate(zip(axes, func_info['parametros'])):
                # Variar el parámetro i mientras mantener otros constantes
                param_base = self.parametros_optimizados[i]
                variacion = 0.2 * abs(param_base) if param_base != 0 else 0.1
                
                valores_param = np.linspace(param_base - variacion, 
                                          param_base + variacion, 50)
                
                x_test = np.linspace(self.x_datos.min(), self.x_datos.max(), 100)
                
                for j, val in enumerate(valores_param[::10]):  # Solo algunas curvas
                    params_temp = self.parametros_optimizados.copy()
                    params_temp[i] = val
                    try:
                        y_test = self.funcion_modelo(x_test, *params_temp)
                        alpha = 0.3 if j != len(valores_param[::10])//2 else 1.0
                        linewidth = 1 if j != len(valores_param[::10])//2 else 2
                        color = 'gray' if j != len(valores_param[::10])//2 else 'red'
                        ax.plot(x_test, y_test, color=color, alpha=alpha, linewidth=linewidth)
                    except:
                        continue
                
                # Agregar datos originales
                ax.scatter(self.x_datos, self.y_datos, alpha=0.7, color='blue', s=30)
                ax.set_title(f'Sensibilidad de {param_name}')
                ax.set_xlabel('x')
                ax.set_ylabel('y')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            mostrar_mensaje_exito("Análisis de sensibilidad completado")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error en análisis de sensibilidad: {str(e)}")
        
        esperar_enter()
    
    def _exportar_resultados(self):
        """Exportar resultados a archivo"""
        try:
            nombre_archivo = input("\nNombre del archivo (sin extensión): ").strip()
            if not nombre_archivo:
                nombre_archivo = f"regresion_no_lineal_{self.funcion_tipo}"
            
            # Crear DataFrame con resultados
            df_resultados = pd.DataFrame({
                'x': self.x_datos,
                'y_original': self.y_datos,
                'y_predicho': self.resultados['y_predichos'],
                'residuo': self.y_datos - self.resultados['y_predichos']
            })
            
            # Guardar como CSV
            archivo_csv = f"{nombre_archivo}.csv"
            df_resultados.to_csv(archivo_csv, index=False)
            
            # Crear archivo de resumen
            with open(f"{nombre_archivo}_resumen.txt", 'w') as f:
                f.write("REGRESIÓN NO LINEAL - RESUMEN\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Función modelo: {self.nombre_funcion}\n")
                f.write(f"Número de datos: {self.n_datos}\n")
                f.write(f"R²: {self.r_cuadrado:.6f}\n")
                f.write(f"Error cuadrático medio: {self.resultados['error_cuadratico_medio']:.6f}\n")
                f.write(f"Error absoluto medio: {self.resultados['error_absoluto_medio']:.6f}\n\n")
                
                f.write("Parámetros optimizados:\n")
                func_info = self.funciones_disponibles[self.funcion_tipo]
                for i, nombre in enumerate(func_info['parametros']):
                    f.write(f"{nombre}: {self.parametros_optimizados[i]:.6f} ± {self.resultados['errores_std'][i]:.6f}\n")
                
                f.write(f"\nFunción ajustada:\n{self._construir_funcion_ajustada()}\n")
            
            mostrar_mensaje_exito(f"Resultados exportados: {archivo_csv}, {nombre_archivo}_resumen.txt")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {str(e)}")
        
        esperar_enter()
    
    def mostrar_ayuda(self):
        """Mostrar ayuda del método"""
        mostrar_ayuda_metodo(
            "REGRESIÓN NO LINEAL",
            "Ajusta funciones no lineales a datos usando algoritmos de optimización",
            {
                "Objetivo": "Encontrar los parámetros óptimos de una función no lineal que minimice el error con los datos",
                "Método": "Levenberg-Marquardt (combinación de Gauss-Newton y descenso de gradiente)",
                "Ventajas": [
                    "Puede ajustar funciones complejas y realistas",
                    "Proporciona estimaciones de incertidumbre de parámetros",
                    "Incluye análisis de residuos y sensibilidad"
                ],
                "Limitaciones": [
                    "Requiere buenas estimaciones iniciales de parámetros",
                    "Puede converger a mínimos locales",
                    "Sensible a datos atípicos"
                ]
            }
        )

def main():
    """Función principal del programa"""
    regresion = RegresionNoLineal()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "REGRESIÓN NO LINEAL",
            "Ajuste de funciones no lineales por optimización"
        )
        
        mostrar_banner_metodo(
            "Regresión No Lineal",
            "Encuentra los mejores parámetros para funciones no lineales"
        )
        
        regresion.mostrar_configuracion()
        
        opciones = [
            "Ingresar/cargar datos",
            "Seleccionar función modelo",
            "Calcular regresión no lineal",
            "Ver resultados y análisis",
            "Ver ayuda del método",
            "Salir"
        ]
        
        mostrar_menu_opciones(opciones)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6])
        
        if opcion == 1:
            regresion.ingresar_datos()
        elif opcion == 2:
            regresion.seleccionar_funcion()
        elif opcion == 3:
            regresion.calcular_regresion()
        elif opcion == 4:
            regresion.mostrar_resultados()
        elif opcion == 5:
            regresion.mostrar_ayuda()
        elif opcion == 6:
            console.print("\n[green]¡Gracias por usar regresión no lineal![/green]")
            break

if __name__ == "__main__":
    main()
