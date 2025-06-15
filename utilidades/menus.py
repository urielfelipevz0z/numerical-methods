#!/usr/bin/env python3
"""
Módulo de menús para métodos numéricos
Funciones para crear menús consistentes y atractivos con Rich
"""

import os
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.layout import Layout
from rich.align import Align
from typing import List, Dict, Optional, Any

console = Console()

def limpiar_pantalla():
    """Limpia la pantalla de la consola"""
    os.system('clear' if os.name == 'posix' else 'cls')

def mostrar_titulo_principal(titulo: str, subtitulo: str = ""):
    """
    Muestra un título principal elegante
    
    Args:
        titulo: Título principal
        subtitulo: Subtítulo opcional
    """
    limpiar_pantalla()
    
    # Crear el texto del título
    titulo_text = Text(titulo, style="bold blue", justify="center")
    if subtitulo:
        titulo_text.append(f"\n{subtitulo}", style="italic cyan")
    
    # Crear panel con el título
    panel = Panel(
        titulo_text,
        border_style="bright_blue",
        padding=(1, 2),
        title="🔢 Métodos Numéricos",
        title_align="center"
    )
    
    console.print(panel)
    console.print()

def mostrar_menu_opciones(
    opciones: List[str], 
    titulo: str = "Seleccione una opción",
    mostrar_salir: bool = True,
    color_titulo: str = "bright_green"
) -> None:
    """
    Muestra un menú de opciones numeradas
    
    Args:
        opciones: Lista de opciones del menú
        titulo: Título del menú
        mostrar_salir: Si True, agrega opción de salir
        color_titulo: Color del título
    """
    # Crear tabla para el menú
    table = Table(
        show_header=False,
        show_edge=False,
        padding=(0, 2),
        border_style="dim"
    )
    
    table.add_column("Opción", style="bold bright_cyan", width=8)
    table.add_column("Descripción", style="white")
    
    # Agregar opciones
    for i, opcion in enumerate(opciones, 1):
        table.add_row(f"[{i}]", opcion)
    
    # Agregar opción de salir si se solicita
    if mostrar_salir:
        numero_salir = len(opciones) + 1
        table.add_row(f"[{numero_salir}]", "Salir", style="dim")
    
    # Crear panel con el menú
    panel = Panel(
        table,
        title=titulo,
        title_align="left",
        border_style=color_titulo,
        padding=(1, 2)
    )
    
    console.print(panel)

def mostrar_submenu(
    titulo: str,
    opciones: List[str],
    descripcion: str = "",
    color: str = "bright_yellow"
) -> None:
    """
    Muestra un submenú con formato consistente
    
    Args:
        titulo: Título del submenú
        opciones: Lista de opciones
        descripcion: Descripción opcional
        color: Color del borde
    """
    content = []
    
    if descripcion:
        content.append(Text(descripcion, style="italic"))
        content.append("")
    
    # Crear tabla para opciones
    table = Table(show_header=False, show_edge=False, padding=(0, 1))
    table.add_column("", style="bold bright_cyan", width=6)
    table.add_column("", style="white")
    
    for i, opcion in enumerate(opciones, 1):
        table.add_row(f"[{i}]", opcion)
    
    content.append(table)
    
    panel = Panel(
        "\n".join(str(item) for item in content) if isinstance(content[0], str) else table,
        title=titulo,
        border_style=color,
        padding=(1, 2)
    )
    
    console.print(panel)

def mostrar_banner_metodo(nombre_metodo: str, descripcion: str):
    """
    Muestra un banner para el método seleccionado
    
    Args:
        nombre_metodo: Nombre del método
        descripcion: Descripción breve del método
    """
    texto_banner = Text()
    texto_banner.append(nombre_metodo, style="bold bright_magenta")
    texto_banner.append(f"\n\n{descripcion}", style="italic bright_white")
    
    panel = Panel(
        texto_banner,
        border_style="bright_magenta",
        padding=(2, 4),
        title="📊 Método Seleccionado",
        title_align="center"
    )
    
    console.print(panel)
    console.print()

def mostrar_estado_configuracion(configuracion: Dict[str, Any]):
    """
    Muestra el estado actual de la configuración
    
    Args:
        configuracion: Diccionario con la configuración actual
    """
    table = Table(
        title="🔧 Configuración Actual",
        title_style="bold bright_green",
        border_style="green",
        show_header=True
    )
    
    table.add_column("Parámetro", style="bright_cyan", width=20)
    table.add_column("Valor", style="white", width=30)
    table.add_column("Estado", style="bold", width=15)
    
    for key, value in configuracion.items():
        if value is not None:
            valor_str = str(value)
            estado = "[green]✓ Configurado[/green]"
        else:
            valor_str = "[dim]No configurado[/dim]"
            estado = "[red]✗ Pendiente[/red]"
        
        table.add_row(key, valor_str, estado)
    
    console.print(table)
    console.print()

def mostrar_progreso_ejecucion(mensaje: str):
    """
    Muestra un mensaje de progreso durante la ejecución
    
    Args:
        mensaje: Mensaje a mostrar
    """
    panel = Panel(
        Text(mensaje, style="bright_yellow", justify="center"),
        border_style="yellow",
        padding=(1, 2),
        title="⚙️ Ejecutando",
        title_align="center"
    )
    
    console.print(panel)

def mostrar_mensaje_exito(mensaje: str):
    """
    Muestra un mensaje de éxito
    
    Args:
        mensaje: Mensaje de éxito
    """
    panel = Panel(
        Text(mensaje, style="bold bright_green", justify="center"),
        border_style="bright_green",
        padding=(1, 2),
        title="✅ Éxito",
        title_align="center"
    )
    
    console.print(panel)

def mostrar_mensaje_error(mensaje: str):
    """
    Muestra un mensaje de error
    
    Args:
        mensaje: Mensaje de error
    """
    panel = Panel(
        Text(mensaje, style="bold bright_red", justify="center"),
        border_style="bright_red",
        padding=(1, 2),
        title="❌ Error",
        title_align="center"
    )
    
    console.print(panel)

def mostrar_mensaje_advertencia(mensaje: str):
    """
    Muestra un mensaje de advertencia
    
    Args:
        mensaje: Mensaje de advertencia
    """
    panel = Panel(
        Text(mensaje, style="bold bright_yellow", justify="center"),
        border_style="bright_yellow",
        padding=(1, 2),
        title="⚠️ Advertencia",
        title_align="center"
    )
    
    console.print(panel)

def mostrar_separador(caracter: str = "─", longitud: int = 80):
    """
    Muestra un separador visual
    
    Args:
        caracter: Carácter para el separador
        longitud: Longitud del separador
    """
    console.print(caracter * longitud, style="dim")

def mostrar_ayuda_metodo(
    nombre: str,
    descripcion: str,
    aplicaciones: List[str],
    ventajas: List[str],
    limitaciones: List[str]
):
    """
    Muestra información de ayuda para un método
    
    Args:
        nombre: Nombre del método
        descripcion: Descripción del método
        aplicaciones: Lista de aplicaciones
        ventajas: Lista de ventajas
        limitaciones: Lista de limitaciones
    """
    limpiar_pantalla()
    
    # Título
    titulo = Panel(
        Text(f"Ayuda: {nombre}", style="bold bright_blue", justify="center"),
        border_style="bright_blue",
        padding=(1, 2)
    )
    console.print(titulo)
    console.print()
    
    # Descripción
    desc_panel = Panel(
        descripcion,
        title="📖 Descripción",
        border_style="blue",
        padding=(1, 2)
    )
    console.print(desc_panel)
    console.print()
    
    # Layout para ventajas y limitaciones
    layout = Layout()
    layout.split_row(
        Layout(name="ventajas"),
        Layout(name="limitaciones")
    )
    
    # Ventajas
    ventajas_text = "\n".join(f"• {ventaja}" for ventaja in ventajas)
    ventajas_panel = Panel(
        ventajas_text,
        title="✅ Ventajas",
        border_style="green",
        padding=(1, 2)
    )
    
    # Limitaciones
    limitaciones_text = "\n".join(f"• {limitacion}" for limitacion in limitaciones)
    limitaciones_panel = Panel(
        limitaciones_text,
        title="⚠️ Limitaciones",
        border_style="yellow",
        padding=(1, 2)
    )
    
    layout["ventajas"].update(ventajas_panel)
    layout["limitaciones"].update(limitaciones_panel)
    console.print(layout)
    console.print()
    
    # Aplicaciones
    aplicaciones_text = "\n".join(f"• {aplicacion}" for aplicacion in aplicaciones)
    aplicaciones_panel = Panel(
        aplicaciones_text,
        title="🎯 Aplicaciones",
        border_style="magenta",
        padding=(1, 2)
    )
    console.print(aplicaciones_panel)
    
    console.print("\n" + "═" * 80)
    input("Presione Enter para volver al menú principal...")

def esperar_enter(mensaje: str = "Presione Enter para continuar..."):
    """
    Pausa la ejecución esperando que el usuario presione Enter
    
    Args:
        mensaje: Mensaje a mostrar
    """
    console.print(f"\n[dim]{mensaje}[/dim]")
    input()
