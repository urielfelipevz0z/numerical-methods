#!/usr/bin/env python3
"""
Regresión Polinomial - Implementación con menús interactivos
Ajusta un polinomio de grado n a un conjunto de datos usando mínimos cuadrados
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

class RegresionPolinomial:
    def __init__(self):
        self.x_datos = None
        self.y_datos = None
        self.n_datos = 0
        self.grado = 2
        self.metodo_ingreso = None
        self.coeficientes = None
        self.r_cuadrado = None
        self.resultados = None
        
    def configuracion_completa(self) -> bool:
        """Verifica si la configuración está completa"""
        return (self.x_datos is not None and 
                self.y_datos is not None and 
                len(self.x_datos) >= self.grado + 1)
    
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
        
        # Grado del polinomio
        tabla.add_row(
            "Grado del polinomio",
            str(self.grado),
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
            
            # Validar que el número de datos sea suficiente
            min_datos = self.grado + 1
            n = validar_numero(
                f"Número de puntos (mínimo {min_datos}): ",
                tipo=int,
                minimo=min_datos,
                maximo=100
            )
            
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
                
                if self.n_datos < self.grado + 1:
                    raise ValueError(f"Se necesitan al menos {self.grado + 1} datos para grado {self.grado}")
                
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
            
            # Generar función base
            console.print("\n[yellow]Seleccione tipo de función:[/yellow]")
            funciones = [
                "Cuadrática: ax² + bx + c",
                "Cúbica: ax³ + bx² + cx + d",
                "Exponencial: ae^(bx)",
                "Senoidal: a*sin(bx) + c"
            ]
            
            for i, func in enumerate(funciones, 1):
                console.print(f"{i}. {func}")
            
            tipo_func = validar_opcion_menu([1, 2, 3, 4])
            
            # Generar datos
            x_datos = np.linspace(x_min, x_max, n)
            
            if tipo_func == 1:  # Cuadrática
                y_datos = 2 * x_datos**2 + 3 * x_datos + 1
            elif tipo_func == 2:  # Cúbica
                y_datos = x_datos**3 - 2 * x_datos**2 + x_datos + 5
            elif tipo_func == 3:  # Exponencial
                y_datos = 2 * np.exp(0.5 * x_datos)
            else:  # Senoidal
                y_datos = 3 * np.sin(x_datos) + 2
            
            # Agregar ruido
            ruido = validar_numero("Nivel de ruido (0.0-1.0): ", tipo=float, minimo=0, maximo=1)
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
    
    def configurar_grado(self):
        """Configurar el grado del polinomio"""
        try:
            console.print("\n[cyan]Configuración del grado del polinomio[/cyan]")
            
            if self.x_datos is not None:
                max_grado = min(len(self.x_datos) - 1, 10)
                console.print(f"[yellow]Máximo grado recomendado: {max_grado}[/yellow]")
            else:
                max_grado = 10
            
            grado = validar_numero(
                f"Grado del polinomio (1-{max_grado}): ",
                tipo=int,
                minimo=1,
                maximo=max_grado
            )
            
            self.grado = grado
            
            if self.x_datos is not None and len(self.x_datos) < grado + 1:
                mostrar_mensaje_error(f"Se necesitan al menos {grado + 1} datos para grado {grado}")
                self.x_datos = None
                self.y_datos = None
            else:
                mostrar_mensaje_exito(f"Grado configurado: {grado}")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al configurar grado: {str(e)}")
        
        esperar_enter()
    
    def calcular_regresion(self):
        """Calcular la regresión polinomial"""
        if not self.configuracion_completa():
            mostrar_mensaje_error("Configure los datos antes de calcular")
            esperar_enter()
            return
        
        try:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]CÁLCULO DE REGRESIÓN POLINOMIAL[/bold cyan]",
                style="cyan"
            ))
            
            with mostrar_progreso_ejecucion("Calculando regresión polinomial..."):
                time.sleep(0.5)  # Simular procesamiento
                
                # Usar numpy.polyfit para obtener coeficientes
                self.coeficientes = np.polyfit(self.x_datos, self.y_datos, self.grado)
                
                # Calcular predicciones
                y_pred = np.polyval(self.coeficientes, self.x_datos)
                
                # Calcular R²
                ss_res = np.sum((self.y_datos - y_pred) ** 2)
                ss_tot = np.sum((self.y_datos - np.mean(self.y_datos)) ** 2)
                self.r_cuadrado = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                
                # Guardar resultados
                self.resultados = {
                    'coeficientes': self.coeficientes,
                    'y_predichos': y_pred,
                    'r_cuadrado': self.r_cuadrado,
                    'error_cuadratico_medio': np.mean((self.y_datos - y_pred) ** 2),
                    'error_absoluto_medio': np.mean(np.abs(self.y_datos - y_pred))
                }
            
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
                "[bold cyan]RESULTADOS DE REGRESIÓN POLINOMIAL[/bold cyan]",
                style="cyan"
            ))
            
            opciones = [
                "Ver ecuación del polinomio",
                "Ver estadísticas del ajuste",
                "Ver tabla de datos vs predicciones",
                "Generar gráfica",
                "Exportar resultados",
                "Volver al menú principal"
            ]
            
            mostrar_menu_opciones(opciones)
            opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6])
            
            if opcion == 1:
                self._mostrar_ecuacion()
            elif opcion == 2:
                self._mostrar_estadisticas()
            elif opcion == 3:
                self._mostrar_tabla_comparacion()
            elif opcion == 4:
                self._generar_grafica()
            elif opcion == 5:
                self._exportar_resultados()
            elif opcion == 6:
                break
    
    def _mostrar_ecuacion(self):
        """Mostrar la ecuación del polinomio"""
        console.print("\n[cyan]Ecuación del polinomio:[/cyan]")
        
        # Construir ecuación
        ecuacion = "y = "
        coefs = self.coeficientes
        grado = len(coefs) - 1
        
        for i, coef in enumerate(coefs):
            potencia = grado - i
            
            if i > 0:
                if coef >= 0:
                    ecuacion += " + "
                else:
                    ecuacion += " - "
                    coef = abs(coef)
            
            if potencia == 0:
                ecuacion += f"{formatear_numero(coef)}"
            elif potencia == 1:
                ecuacion += f"{formatear_numero(coef)}x"
            else:
                ecuacion += f"{formatear_numero(coef)}x^{potencia}"
        
        panel = Panel(
            ecuacion,
            title="Ecuación del Polinomio",
            style="green"
        )
        console.print(panel)
        
        # Mostrar coeficientes
        tabla = Table(title="Coeficientes")
        tabla.add_column("Término", style="cyan")
        tabla.add_column("Coeficiente", style="yellow")
        
        for i, coef in enumerate(coefs):
            potencia = grado - i
            if potencia == 0:
                termino = "Constante"
            elif potencia == 1:
                termino = "x"
            else:
                termino = f"x^{potencia}"
            
            tabla.add_row(termino, formatear_numero(coef))
        
        console.print(tabla)
        esperar_enter()
    
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
        tabla.add_row("Grado del polinomio", str(self.grado), "")
        
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
            self._graficar_regresion_polinomial()
            mostrar_mensaje_exito("Gráfica generada exitosamente")
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {str(e)}")
        
        esperar_enter()
    
    def _graficar_regresion_polinomial(self):
        """Crear gráfica de la regresión polinomial"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfica principal
        ax1.scatter(self.x_datos, self.y_datos, alpha=0.7, color='blue', 
                   label='Datos originales', s=50)
        
        # Generar curva suave para el polinomio
        x_smooth = np.linspace(self.x_datos.min(), self.x_datos.max(), 300)
        y_smooth = np.polyval(self.coeficientes, x_smooth)
        
        ax1.plot(x_smooth, y_smooth, 'r-', linewidth=2, 
                label=f'Polinomio grado {self.grado}')
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title(f'Regresión Polinomial (Grado {self.grado})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Agregar ecuación
        ecuacion = self._construir_ecuacion_corta()
        ax1.text(0.05, 0.95, ecuacion, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Gráfica de residuos
        residuos = self.y_datos - self.resultados['y_predichos']
        ax2.scatter(self.resultados['y_predichos'], residuos, alpha=0.7, color='green')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Valores predichos')
        ax2.set_ylabel('Residuos')
        ax2.set_title('Análisis de Residuos')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _construir_ecuacion_corta(self):
        """Construir versión corta de la ecuación para mostrar en gráfica"""
        coefs = self.coeficientes
        grado = len(coefs) - 1
        
        ecuacion = f"R² = {self.r_cuadrado:.3f}\n"
        
        if grado <= 3:  # Mostrar ecuación completa solo para grados bajos
            ecuacion += "y = "
            for i, coef in enumerate(coefs):
                potencia = grado - i
                
                if i > 0:
                    if coef >= 0:
                        ecuacion += " + "
                    else:
                        ecuacion += " - "
                        coef = abs(coef)
                
                if potencia == 0:
                    ecuacion += f"{coef:.2f}"
                elif potencia == 1:
                    ecuacion += f"{coef:.2f}x"
                else:
                    ecuacion += f"{coef:.2f}x^{potencia}"
        else:
            ecuacion += f"Polinomio grado {grado}"
        
        return ecuacion
    
    def _exportar_resultados(self):
        """Exportar resultados a archivo"""
        try:
            nombre_archivo = input("\nNombre del archivo (sin extensión): ").strip()
            if not nombre_archivo:
                nombre_archivo = f"regresion_polinomial_grado_{self.grado}"
            
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
                f.write("REGRESIÓN POLINOMIAL - RESUMEN\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Grado del polinomio: {self.grado}\n")
                f.write(f"Número de datos: {self.n_datos}\n")
                f.write(f"R²: {self.r_cuadrado:.6f}\n")
                f.write(f"Error cuadrático medio: {self.resultados['error_cuadratico_medio']:.6f}\n")
                f.write(f"Error absoluto medio: {self.resultados['error_absoluto_medio']:.6f}\n\n")
                
                f.write("Coeficientes del polinomio:\n")
                for i, coef in enumerate(self.coeficientes):
                    potencia = self.grado - i
                    f.write(f"x^{potencia}: {coef:.6f}\n")
            
            mostrar_mensaje_exito(f"Resultados exportados: {archivo_csv}, {nombre_archivo}_resumen.txt")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {str(e)}")
        
        esperar_enter()
    
    def mostrar_ayuda(self):
        """Mostrar ayuda del método"""
        mostrar_ayuda_metodo(
            "REGRESIÓN POLINOMIAL",
            "Ajusta un polinomio de grado n a un conjunto de datos usando mínimos cuadrados",
            {
                "Objetivo": "Encontrar el polinomio P(x) = a₀ + a₁x + a₂x² + ... + aₙxⁿ que minimice la suma de errores cuadráticos",
                "Método": "Mínimos cuadrados ordinarios (resuelve sistema de ecuaciones normales)",
                "Ventajas": [
                    "Flexibilidad para ajustar diferentes tipos de curvas",
                    "Solución analítica (no iterativa)",
                    "Proporciona medidas de bondad de ajuste"
                ],
                "Limitaciones": [
                    "Susceptible a sobreajuste con grados altos",
                    "Puede ser inestable numéricamente para grados muy altos",
                    "Oscilaciones indeseadas en los extremos (fenómeno de Runge)"
                ]
            }
        )

def main():
    """Función principal del programa"""
    regresion = RegresionPolinomial()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "REGRESIÓN POLINOMIAL",
            "Ajuste de polinomios por mínimos cuadrados"
        )
        
        mostrar_banner_metodo(
            "Regresión Polinomial",
            f"Encuentra el mejor polinomio de grado {regresion.grado} que se ajusta a los datos"
        )
        
        regresion.mostrar_configuracion()
        
        opciones = [
            "Ingresar/cargar datos",
            "Configurar grado del polinomio",
            "Calcular regresión polinomial",
            "Ver resultados y análisis",
            "Ver ayuda del método",
            "Salir"
        ]
        
        mostrar_menu_opciones(opciones)
        opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6])
        
        if opcion == 1:
            regresion.ingresar_datos()
        elif opcion == 2:
            regresion.configurar_grado()
        elif opcion == 3:
            regresion.calcular_regresion()
        elif opcion == 4:
            regresion.mostrar_resultados()
        elif opcion == 5:
            regresion.mostrar_ayuda()
        elif opcion == 6:
            console.print("\n[green]¡Gracias por usar regresión polinomial![/green]")
            break

if __name__ == "__main__":
    main()
