#!/usr/bin/env python3
"""
Módulo de formatos para métodos numéricos
Funciones para formatear resultados, tablas y reportes
"""

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from typing import List, Dict, Any, Optional, Union
import sympy as sp

console = Console()

def formatear_numero(numero: float, decimales: int = 6) -> str:
    """
    Formatea un número con el número especificado de decimales
    
    Args:
        numero: Número a formatear
        decimales: Número de decimales
    
    Returns:
        String formateado
    """
    if abs(numero) < 1e-15:
        return "0.000000"
    elif abs(numero) > 1e6 or abs(numero) < 1e-4:
        return f"{numero:.{decimales}e}"
    else:
        return f"{numero:.{decimales}f}"

def crear_tabla_iteraciones(
    iteraciones: List[int],
    valores: List[Dict[str, Any]],
    titulo: str = "Tabla de Iteraciones",
    mostrar_indices: bool = True
) -> Table:
    """
    Crea una tabla formateada para mostrar iteraciones
    
    Args:
        iteraciones: Lista de números de iteración
        valores: Lista de diccionarios con los valores de cada iteración
        titulo: Título de la tabla
        mostrar_indices: Si True, muestra columna de índices
    
    Returns:
        Tabla formateada de Rich
    """
    table = Table(
        title=titulo,
        title_style="bold bright_green",
        border_style="green",
        show_header=True,
        header_style="bold bright_cyan"
    )
    
    # Agregar columna de iteración si se solicita
    if mostrar_indices:
        table.add_column("Iter", style="bright_yellow", width=6, justify="center")
    
    # Determinar columnas basándose en el primer elemento
    if valores:
        columnas = list(valores[0].keys())
        for col in columnas:
            table.add_column(col, style="white", justify="center")
    
    # Agregar filas
    for i, valor_dict in enumerate(valores):
        fila = []
        
        if mostrar_indices:
            fila.append(str(iteraciones[i]) if i < len(iteraciones) else str(i))
        
        for col in columnas:
            val = valor_dict.get(col, "")
            if isinstance(val, (int, float, complex)):
                fila.append(formatear_numero(float(val.real if isinstance(val, complex) else val)))
            else:
                fila.append(str(val))
        
        table.add_row(*fila)
    
    return table

def mostrar_resultado_final(
    metodo: str,
    resultado: Dict[str, Any],
    tiempo_ejecucion: Optional[float] = None
):
    """
    Muestra el resultado final de un método de forma elegante
    
    Args:
        metodo: Nombre del método
        resultado: Diccionario con los resultados
        tiempo_ejecucion: Tiempo de ejecución en segundos
    """
    # Panel principal con el resultado
    titulo_panel = Panel(
        Text(f"Resultado Final - {metodo}", style="bold bright_green", justify="center"),
        border_style="bright_green",
        padding=(1, 2)
    )
    console.print(titulo_panel)
    console.print()
    
    # Crear tabla de resultados
    table = Table(
        border_style="green",
        show_header=True,
        header_style="bold bright_cyan",
        title="📊 Resumen de Resultados",
        title_style="bold bright_green"
    )
    
    table.add_column("Parámetro", style="bright_cyan", width=25)
    table.add_column("Valor", style="white", width=30)
    table.add_column("Descripción", style="dim", width=35)
    
    # Mapeo de descripciones comunes
    descripciones = {
        "raiz": "Valor de la raíz encontrada",
        "solucion": "Solución del sistema",
        "valor_optimo": "Valor óptimo encontrado",
        "iteraciones": "Número de iteraciones realizadas",
        "error_final": "Error absoluto final",
        "convergencia": "Estado de convergencia",
        "tolerancia": "Tolerancia utilizada",
        "tiempo": "Tiempo de ejecución",
        "punto_fijo": "Punto fijo encontrado",
        "coeficientes": "Coeficientes calculados",
        "r_cuadrado": "Coeficiente de determinación",
        "residuos": "Suma de residuos cuadrados"
    }
    
    for key, value in resultado.items():
        descripcion = descripciones.get(key, "")
        
        if isinstance(value, (int, float, complex)):
            valor_formateado = formatear_numero(float(value.real if isinstance(value, complex) else value))
        elif isinstance(value, (list, np.ndarray)):
            if len(value) <= 3:
                valor_formateado = ", ".join([formatear_numero(float(v)) for v in value])
            else:
                valor_formateado = f"Array de {len(value)} elementos"
        elif isinstance(value, bool):
            valor_formateado = "✅ Sí" if value else "❌ No"
        else:
            valor_formateado = str(value)
        
        table.add_row(key.replace("_", " ").title(), valor_formateado, descripcion)
    
    # Agregar tiempo de ejecución si está disponible
    if tiempo_ejecucion is not None:
        table.add_row(
            "Tiempo de Ejecución", 
            f"{tiempo_ejecucion:.4f} segundos", 
            "Tiempo total de cálculo"
        )
    
    console.print(table)
    console.print()

def mostrar_tabla_datos(
    x_datos: np.ndarray,
    y_datos: np.ndarray,
    titulo: str = "Datos de Entrada",
    max_filas: int = 20
):
    """
    Muestra una tabla con los datos de entrada
    
    Args:
        x_datos: Array de valores X
        y_datos: Array de valores Y
        titulo: Título de la tabla
        max_filas: Máximo número de filas a mostrar
    """
    table = Table(
        title=titulo,
        title_style="bold bright_blue",
        border_style="blue",
        show_header=True,
        header_style="bold bright_cyan"
    )
    
    table.add_column("Índice", style="bright_yellow", width=8, justify="center")
    table.add_column("X", style="white", width=15, justify="center")
    table.add_column("Y", style="white", width=15, justify="center")
    
    n_datos = min(len(x_datos), max_filas)
    
    for i in range(n_datos):
        table.add_row(
            str(i + 1),
            formatear_numero(x_datos[i]),
            formatear_numero(y_datos[i])
        )
    
    if len(x_datos) > max_filas:
        table.add_row("...", "...", "...", style="dim")
        table.add_row(
            str(len(x_datos)),
            formatear_numero(x_datos[-1]),
            formatear_numero(y_datos[-1])
        )
    
    console.print(table)
    console.print()

def mostrar_estadisticas_datos(
    x_datos: np.ndarray,
    y_datos: np.ndarray
):
    """
    Muestra estadísticas básicas de los datos
    
    Args:
        x_datos: Array de valores X
        y_datos: Array de valores Y
    """
    stats_x = {
        "Mínimo": np.min(x_datos),
        "Máximo": np.max(x_datos),
        "Media": np.mean(x_datos),
        "Desv. Estándar": np.std(x_datos)
    }
    
    stats_y = {
        "Mínimo": np.min(y_datos),
        "Máximo": np.max(y_datos),
        "Media": np.mean(y_datos),
        "Desv. Estándar": np.std(y_datos)
    }
    
    # Crear layout para mostrar lado a lado
    layout = Layout()
    layout.split_row(
        Layout(name="stats_x"),
        Layout(name="stats_y")
    )
    
    # Tabla para X
    table_x = Table(
        title="Estadísticas de X",
        border_style="blue",
        show_header=True,
        header_style="bold bright_cyan"
    )
    table_x.add_column("Estadística", style="bright_cyan")
    table_x.add_column("Valor", style="white")
    
    for stat, valor in stats_x.items():
        table_x.add_row(stat, formatear_numero(valor))
    
    # Tabla para Y
    table_y = Table(
        title="Estadísticas de Y",
        border_style="green",
        show_header=True,
        header_style="bold bright_cyan"
    )
    table_y.add_column("Estadística", style="bright_cyan")
    table_y.add_column("Valor", style="white")
    
    for stat, valor in stats_y.items():
        table_y.add_row(stat, formatear_numero(valor))
    
    layout["stats_x"].update(table_x)
    layout["stats_y"].update(table_y)
    
    console.print(layout)
    console.print()

def crear_reporte_metodo(
    metodo: str,
    configuracion: Dict[str, Any],
    resultado: Dict[str, Any],
    iteraciones_tabla: Optional[Table] = None
) -> str:
    """
    Crea un reporte completo del método ejecutado
    
    Args:
        metodo: Nombre del método
        configuracion: Configuración utilizada
        resultado: Resultados obtenidos
        iteraciones_tabla: Tabla de iteraciones opcional
    
    Returns:
        String con el reporte formateado
    """
    reporte = []
    reporte.append("=" * 80)
    reporte.append(f"REPORTE DE EJECUCIÓN - {metodo.upper()}")
    reporte.append("=" * 80)
    reporte.append("")
    
    # Configuración
    reporte.append("CONFIGURACIÓN UTILIZADA:")
    reporte.append("-" * 30)
    for key, value in configuracion.items():
        reporte.append(f"{key.replace('_', ' ').title()}: {value}")
    reporte.append("")
    
    # Resultados
    reporte.append("RESULTADOS OBTENIDOS:")
    reporte.append("-" * 30)
    for key, value in resultado.items():
        if isinstance(value, (int, float, complex)):
            valor_str = formatear_numero(float(value.real if isinstance(value, complex) else value))
        else:
            valor_str = str(value)
        reporte.append(f"{key.replace('_', ' ').title()}: {valor_str}")
    
    return "\n".join(reporte)

def mostrar_matriz(
    matriz: np.ndarray,
    titulo: str = "Matriz",
    formato: str = "6.3f"
):
    """
    Muestra una matriz de forma elegante
    
    Args:
        matriz: Matriz numpy a mostrar
        titulo: Título de la matriz
        formato: Formato para los números
    """
    filas, columnas = matriz.shape
    
    table = Table(
        title=titulo,
        title_style="bold bright_magenta",
        border_style="magenta",
        show_header=True,
        header_style="bold bright_cyan"
    )
    
    # Agregar columnas
    table.add_column("Fila", style="bright_yellow", width=6, justify="center")
    for j in range(columnas):
        table.add_column(f"Col {j+1}", style="white", width=12, justify="center")
    
    # Agregar filas
    for i in range(filas):
        fila = [str(i + 1)]
        for j in range(columnas):
            fila.append(f"{matriz[i, j]:{formato}}")
        table.add_row(*fila)
    
    console.print(table)
    console.print()
