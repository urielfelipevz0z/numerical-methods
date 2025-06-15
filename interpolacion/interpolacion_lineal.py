#!/usr/bin/env python3
"""
Interpolación Lineal - Implementación con menús interactivos
Interpolación lineal por tramos entre puntos de datos
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

class InterpolacionLineal:
    def __init__(self):
        self.x_datos = None
        self.y_datos = None
        self.n_datos = 0
        self.puntos_interpolacion = None
        self.valores_interpolados = None
        self.metodo_ingreso = None
        self.resultados = None
        
    def configuracion_completa(self) -> bool:
        """Verifica si la configuración está completa"""
        return (self.x_datos is not None and 
                self.y_datos is not None and 
                len(self.x_datos) >= 2)
    
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
            
            n = validar_numero("Número de puntos (mínimo 2): ", tipo=int, minimo=2, maximo=50)
            
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
                
                if self.n_datos < 2:
                    raise ValueError("Se necesitan al menos 2 datos")
                
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
                "Lineal: mx + b",
                "Cuadrática: ax² + bx + c",
                "Senoidal: a*sin(bx) + c",
                "Exponencial: ae^(bx)"
            ]
            
            for i, func in enumerate(funciones, 1):
                console.print(f"{i}. {func}")
            
            tipo_func = validar_opcion_menu([1, 2, 3, 4])
            
            # Generar datos dispersos (no equidistantes)
            x_datos = np.sort(np.random.uniform(x_min, x_max, n))
            
            if tipo_func == 1:  # Lineal
                y_datos = 2 * x_datos + 1
            elif tipo_func == 2:  # Cuadrática
                y_datos = 0.5 * x_datos**2 - x_datos + 3
            elif tipo_func == 3:  # Senoidal
                y_datos = 2 * np.sin(x_datos) + 1
            else:  # Exponencial
                y_datos = np.exp(0.5 * x_datos)
            
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
            mostrar_mensaje_error("Configure los datos base primero")
            esperar_enter()
            return
        
        while True:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]CONFIGURACIÓN DE INTERPOLACIÓN[/bold cyan]",
                style="cyan"
            ))
            
            console.print(f"[yellow]Rango de datos: [{self.x_datos.min():.3f}, {self.x_datos.max():.3f}][/yellow]")
            
            opciones = [
                "Puntos específicos",
                "Rango uniforme",
                "Puntos aleatorios en el rango",
                "Volver al menú principal"
            ]
            
            mostrar_menu_opciones(opciones)
            opcion = validar_opcion_menu([1, 2, 3, 4])
            
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
                console.print("[red]Advertencia: Algunos puntos requieren extrapolación[/red]")
            
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
    
    def calcular_interpolacion(self):
        """Calcular la interpolación lineal"""
        if not self.configuracion_completa() or self.puntos_interpolacion is None:
            mostrar_mensaje_error("Configure datos y puntos de interpolación primero")
            esperar_enter()
            return
        
        try:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]CÁLCULO DE INTERPOLACIÓN LINEAL[/bold cyan]",
                style="cyan"
            ))
            
            with mostrar_progreso_ejecucion("Calculando interpolación lineal..."):
                time.sleep(0.3)
                
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
                    'rango_interpolacion': [self.puntos_interpolacion.min(), self.puntos_interpolacion.max()]
                }
            
            mostrar_mensaje_exito("Interpolación calculada exitosamente")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error en el cálculo: {str(e)}")
        
        esperar_enter()
    
    def _interpolar_punto(self, x):
        """Interpolar un punto específico usando interpolación lineal"""
        # Si x está fuera del rango, usar extrapolación lineal
        if x <= self.x_datos[0]:
            # Extrapolación hacia la izquierda usando los dos primeros puntos
            x1, x2 = self.x_datos[0], self.x_datos[1]
            y1, y2 = self.y_datos[0], self.y_datos[1]
        elif x >= self.x_datos[-1]:
            # Extrapolación hacia la derecha usando los dos últimos puntos
            x1, x2 = self.x_datos[-2], self.x_datos[-1]
            y1, y2 = self.y_datos[-2], self.y_datos[-1]
        else:
            # Interpolación: encontrar los dos puntos que rodean x
            i = np.searchsorted(self.x_datos, x) - 1
            x1, x2 = self.x_datos[i], self.x_datos[i + 1]
            y1, y2 = self.y_datos[i], self.y_datos[i + 1]
        
        # Fórmula de interpolación lineal
        if x2 - x1 == 0:  # Evitar división por cero
            return y1
        
        return y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    
    def mostrar_resultados(self):
        """Mostrar resultados detallados"""
        if self.resultados is None:
            mostrar_mensaje_error("No hay resultados para mostrar. Ejecute el cálculo primero.")
            esperar_enter()
            return
        
        while True:
            limpiar_pantalla()
            console.print(Panel.fit(
                "[bold cyan]RESULTADOS DE INTERPOLACIÓN LINEAL[/bold cyan]",
                style="cyan"
            ))
            
            opciones = [
                "Ver tabla de resultados",
                "Análisis de interpolación/extrapolación",
                "Generar gráfica",
                "Evaluar punto específico",
                "Exportar resultados",
                "Volver al menú principal"
            ]
            
            mostrar_menu_opciones(opciones)
            opcion = validar_opcion_menu([1, 2, 3, 4, 5, 6])
            
            if opcion == 1:
                self._mostrar_tabla_resultados()
            elif opcion == 2:
                self._mostrar_analisis()
            elif opcion == 3:
                self._generar_grafica()
            elif opcion == 4:
                self._evaluar_punto_especifico()
            elif opcion == 5:
                self._exportar_resultados()
            elif opcion == 6:
                break
    
    def _mostrar_tabla_resultados(self):
        """Mostrar tabla de resultados de interpolación"""
        tabla = Table(title="Resultados de Interpolación")
        tabla.add_column("Punto", style="cyan")
        tabla.add_column("Valor Interpolado", style="yellow")
        tabla.add_column("Tipo", style="green")
        
        for i, (x, y) in enumerate(zip(self.puntos_interpolacion, self.valores_interpolados)):
            # Determinar si es interpolación o extrapolación
            if self.x_datos.min() <= x <= self.x_datos.max():
                tipo = "Interpolación"
            else:
                tipo = "Extrapolación"
            
            tabla.add_row(
                formatear_numero(x),
                formatear_numero(y),
                tipo
            )
            
            # Mostrar solo los primeros 20 resultados en pantalla
            if i >= 19 and len(self.puntos_interpolacion) > 20:
                tabla.add_row("...", "...", "...")
                tabla.add_row(
                    f"(y {len(self.puntos_interpolacion) - 20} más)",
                    "",
                    ""
                )
                break
        
        console.print(tabla)
        esperar_enter()
    
    def _mostrar_analisis(self):
        """Mostrar análisis de interpolación/extrapolación"""
        tabla = Table(title="Análisis de Interpolación")
        tabla.add_column("Aspecto", style="cyan")
        tabla.add_column("Valor", style="yellow")
        tabla.add_column("Descripción", style="green")
        
        # Estadísticas básicas
        tabla.add_row("Datos base", str(self.n_datos), "Puntos de datos originales")
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
        
        # Advertencias sobre extrapolación
        if self.resultados['extrapolacion_count'] > 0:
            console.print(Panel(
                "[red]Advertencia: La extrapolación lineal puede no ser precisa fuera del rango de datos.\n"
                "Los valores extrapolados asumen que la tendencia lineal local continúa indefinidamente.[/red]",
                title="Advertencia de Extrapolación",
                style="red"
            ))
        
        esperar_enter()
    
    def _generar_grafica(self):
        """Generar gráfica de la interpolación"""
        try:
            self._graficar_interpolacion_lineal()
            mostrar_mensaje_exito("Gráfica generada exitosamente")
        except Exception as e:
            mostrar_mensaje_error(f"Error al generar gráfica: {str(e)}")
        
        esperar_enter()
    
    def _graficar_interpolacion_lineal(self):
        """Crear gráfica de la interpolación lineal"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Gráfica principal
        # Datos originales
        ax1.scatter(self.x_datos, self.y_datos, color='blue', s=80, zorder=5,
                   label='Datos originales', marker='o', edgecolors='black', linewidth=1)
        
        # Puntos interpolados
        mask_interp = ((self.puntos_interpolacion >= self.x_datos.min()) & 
                      (self.puntos_interpolacion <= self.x_datos.max()))
        mask_extrap = ~mask_interp
        
        if np.any(mask_interp):
            ax1.scatter(self.puntos_interpolacion[mask_interp], 
                       self.valores_interpolados[mask_interp],
                       color='green', s=30, alpha=0.7, label='Interpolación', marker='x')
        
        if np.any(mask_extrap):
            ax1.scatter(self.puntos_interpolacion[mask_extrap], 
                       self.valores_interpolados[mask_extrap],
                       color='red', s=30, alpha=0.7, label='Extrapolación', marker='+')
        
        # Líneas de interpolación
        # Conectar datos originales con líneas
        ax1.plot(self.x_datos, self.y_datos, 'b-', alpha=0.3, linewidth=1, zorder=1)
        
        # Si hay extrapolación, mostrar las líneas extendidas
        if np.any(mask_extrap):
            # Extensión a la izquierda
            if self.puntos_interpolacion.min() < self.x_datos.min():
                x_ext_izq = [self.puntos_interpolacion.min(), self.x_datos[0]]
                y_ext_izq = [self._interpolar_punto(self.puntos_interpolacion.min()), self.y_datos[0]]
                ax1.plot(x_ext_izq, y_ext_izq, 'r--', alpha=0.7, linewidth=1)
            
            # Extensión a la derecha
            if self.puntos_interpolacion.max() > self.x_datos.max():
                x_ext_der = [self.x_datos[-1], self.puntos_interpolacion.max()]
                y_ext_der = [self.y_datos[-1], self._interpolar_punto(self.puntos_interpolacion.max())]
                ax1.plot(x_ext_der, y_ext_der, 'r--', alpha=0.7, linewidth=1)
        
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        ax1.set_title('Interpolación Lineal')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Gráfica de segmentos lineales
        # Mostrar cada segmento lineal individualmente
        colores = plt.cm.Set3(np.linspace(0, 1, len(self.x_datos) - 1))
        
        for i in range(len(self.x_datos) - 1):
            x_seg = [self.x_datos[i], self.x_datos[i + 1]]
            y_seg = [self.y_datos[i], self.y_datos[i + 1]]
            ax2.plot(x_seg, y_seg, color=colores[i], linewidth=3, 
                    label=f'Segmento {i+1}', alpha=0.8)
        
        # Datos originales
        ax2.scatter(self.x_datos, self.y_datos, color='black', s=80, zorder=5,
                   marker='o', edgecolors='white', linewidth=2)
        
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Segmentos de Interpolación')
        if len(self.x_datos) <= 10:  # Solo mostrar leyenda si no hay muchos segmentos
            ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def _evaluar_punto_especifico(self):
        """Evaluar interpolación en un punto específico"""
        try:
            console.print("\n[cyan]Evaluación de punto específico[/cyan]")
            
            x_eval = validar_numero("Punto a evaluar: ", tipo=float)
            y_eval = self._interpolar_punto(x_eval)
            
            # Determinar tipo (interpolación/extrapolación)
            if self.x_datos.min() <= x_eval <= self.x_datos.max():
                tipo = "Interpolación"
                confianza = "Alta"
            else:
                tipo = "Extrapolación"
                confianza = "Baja (usar con precaución)"
            
            # Crear tabla con resultado
            tabla = Table(title="Evaluación de Punto")
            tabla.add_column("Aspecto", style="cyan")
            tabla.add_column("Valor", style="yellow")
            
            tabla.add_row("Punto evaluado", formatear_numero(x_eval))
            tabla.add_row("Valor interpolado", formatear_numero(y_eval))
            tabla.add_row("Tipo de cálculo", tipo)
            tabla.add_row("Confianza", confianza)
            
            console.print(tabla)
            
            # Mostrar segmento utilizado
            if self.x_datos.min() <= x_eval <= self.x_datos.max():
                i = np.searchsorted(self.x_datos, x_eval) - 1
                if i < 0:
                    i = 0
                elif i >= len(self.x_datos) - 1:
                    i = len(self.x_datos) - 2
                
                console.print(f"\n[green]Segmento utilizado:[/green]")
                console.print(f"  Punto 1: ({formatear_numero(self.x_datos[i])}, {formatear_numero(self.y_datos[i])})")
                console.print(f"  Punto 2: ({formatear_numero(self.x_datos[i+1])}, {formatear_numero(self.y_datos[i+1])})")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al evaluar punto: {str(e)}")
        
        esperar_enter()
    
    def _exportar_resultados(self):
        """Exportar resultados a archivo"""
        try:
            nombre_archivo = input("\nNombre del archivo (sin extensión): ").strip()
            if not nombre_archivo:
                nombre_archivo = "interpolacion_lineal"
            
            # Crear DataFrame con resultados
            tipos = []
            for x in self.puntos_interpolacion:
                if self.x_datos.min() <= x <= self.x_datos.max():
                    tipos.append("Interpolación")
                else:
                    tipos.append("Extrapolación")
            
            df_resultados = pd.DataFrame({
                'x': self.puntos_interpolacion,
                'y_interpolado': self.valores_interpolados,
                'tipo': tipos
            })
            
            # Crear DataFrame con datos originales
            df_originales = pd.DataFrame({
                'x_original': self.x_datos,
                'y_original': self.y_datos
            })
            
            # Guardar archivos
            archivo_resultados = f"{nombre_archivo}_resultados.csv"
            archivo_originales = f"{nombre_archivo}_datos_originales.csv"
            
            df_resultados.to_csv(archivo_resultados, index=False)
            df_originales.to_csv(archivo_originales, index=False)
            
            # Crear archivo de resumen
            with open(f"{nombre_archivo}_resumen.txt", 'w') as f:
                f.write("INTERPOLACIÓN LINEAL - RESUMEN\n")
                f.write("=" * 40 + "\n\n")
                f.write(f"Método: Interpolación lineal por tramos\n")
                f.write(f"Datos base: {self.n_datos} puntos\n")
                f.write(f"Puntos calculados: {len(self.puntos_interpolacion)}\n")
                f.write(f"Interpolaciones: {self.resultados['interpolacion_count']}\n")
                f.write(f"Extrapolaciones: {self.resultados['extrapolacion_count']}\n\n")
                
                f.write(f"Rango original: [{self.x_datos.min():.6f}, {self.x_datos.max():.6f}]\n")
                f.write(f"Rango calculado: [{self.puntos_interpolacion.min():.6f}, {self.puntos_interpolacion.max():.6f}]\n\n")
                
                f.write("Valores interpolados:\n")
                f.write(f"  Mínimo: {self.valores_interpolados.min():.6f}\n")
                f.write(f"  Máximo: {self.valores_interpolados.max():.6f}\n")
                f.write(f"  Promedio: {self.valores_interpolados.mean():.6f}\n")
            
            mostrar_mensaje_exito(f"Resultados exportados: {archivo_resultados}, {archivo_originales}, {nombre_archivo}_resumen.txt")
            
        except Exception as e:
            mostrar_mensaje_error(f"Error al exportar: {str(e)}")
        
        esperar_enter()
    
    def mostrar_ayuda(self):
        """Mostrar ayuda del método"""
        mostrar_ayuda_metodo(
            "INTERPOLACIÓN LINEAL",
            "Interpolación lineal por tramos entre puntos de datos",
            {
                "Objetivo": "Estimar valores entre puntos conocidos usando segmentos de línea recta",
                "Método": "Conecta puntos consecutivos con líneas rectas y calcula valores intermedios",
                "Ventajas": [
                    "Método simple y robusto",
                    "Siempre produce resultados continuos",
                    "Computacionalmente eficiente",
                    "No presenta oscilaciones indeseadas"
                ],
                "Limitaciones": [
                    "No es diferenciable en los puntos de datos",
                    "Puede no capturar curvaturas suaves",
                    "La extrapolación es puramente lineal",
                    "Precisión limitada para funciones muy curvadas"
                ]
            }
        )

def main():
    """Función principal del programa"""
    interpolacion = InterpolacionLineal()
    
    while True:
        limpiar_pantalla()
        mostrar_titulo_principal(
            "INTERPOLACIÓN LINEAL",
            "Interpolación lineal por tramos"
        )
        
        mostrar_banner_metodo(
            "Interpolación Lineal",
            "Estima valores entre puntos usando segmentos de línea recta"
        )
        
        interpolacion.mostrar_configuracion()
        
        opciones = [
            "Ingresar/cargar datos base",
            "Configurar puntos de interpolación",
            "Calcular interpolación",
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
            console.print("\n[green]¡Gracias por usar interpolación lineal![/green]")
            break

if __name__ == "__main__":
    main()
