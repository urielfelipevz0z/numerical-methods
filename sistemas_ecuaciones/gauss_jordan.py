#!/usr/bin/env python3
"""
Gauss-Jordan - Implementación con menús interactivos
Método de eliminación para resolver sistemas de ecuaciones lineales
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import sympy as sp
import os
from typing import List, Tuple, Optional, Union
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_flotante, validar_entero, validar_polinomio,
    crear_menu, limpiar_pantalla, mostrar_progreso,
    formatear_numero, formatear_tabla_resultados
)

console = Console()

class GaussJordan:
    def __init__(self):
        self.matriz_coeficientes = None
        self.vector_terminos = None
        self.matriz_ampliada = None
        self.solucion = None
        self.pasos = []
        self.intercambios = []
        self.pivoteo = True
        self.mostrar_fracciones = False
        self.num_variables = 0

    def mostrar_menu_principal(self):
        """Muestra el menú principal del método"""
        opciones = [
            "Ingresar sistema de ecuaciones",
            "Configurar opciones de solución",
            "Ejecutar eliminación Gauss-Jordan",
            "Ver pasos detallados",
            "Analizar matriz y solución",
            "Mostrar gráficos (sistemas 2x2)",
            "Ver ayuda",
            "Salir"
        ]
        return crear_menu("ELIMINACIÓN GAUSS-JORDAN", opciones)

    def ingresar_sistema(self):
        """Menú para ingreso del sistema de ecuaciones"""
        limpiar_pantalla()
        console.print(Panel.fit("INGRESO DEL SISTEMA", style="bold green"))
        
        console.print("\n[bold]Configuración del sistema:[/bold]")
        
        # Determinar tamaño del sistema
        self.num_variables = validar_entero("Número de variables/ecuaciones (2-6): ", 2, 6)
        
        console.print(f"\n[yellow]Sistema de {self.num_variables}x{self.num_variables}[/yellow]")
        console.print("Sistema: Ax = b")
        
        # Opción de ingreso
        console.print("\n[bold]Opciones de ingreso:[/bold]")
        console.print("1. Ingresar matriz y vector por separado")
        console.print("2. Ingresar matriz ampliada")
        console.print("3. Generar sistema aleatorio")
        
        opcion = validar_entero("Seleccione opción (1-3): ", 1, 3)
        
        if opcion == 1:
            self._ingresar_matriz_vector()
        elif opcion == 2:
            self._ingresar_matriz_ampliada()
        else:
            self._generar_sistema_aleatorio()

    def _ingresar_matriz_vector(self):
        """Ingresa matriz de coeficientes y vector de términos independientes"""
        console.print(f"\n[bold]Matriz de coeficientes A ({self.num_variables}x{self.num_variables}):[/bold]")
        
        self.matriz_coeficientes = np.zeros((self.num_variables, self.num_variables))
        
        for i in range(self.num_variables):
            console.print(f"\nFila {i+1}:")
            for j in range(self.num_variables):
                valor = validar_flotante(f"  A[{i+1},{j+1}] = ", puede_ser_cero=True)
                self.matriz_coeficientes[i, j] = valor
        
        console.print(f"\n[bold]Vector de términos independientes b:[/bold]")
        self.vector_terminos = np.zeros(self.num_variables)
        
        for i in range(self.num_variables):
            valor = validar_flotante(f"b[{i+1}] = ", puede_ser_cero=True)
            self.vector_terminos[i] = valor
        
        # Crear matriz ampliada
        self.matriz_ampliada = np.column_stack([self.matriz_coeficientes, self.vector_terminos])
        
        self._mostrar_sistema()

    def _ingresar_matriz_ampliada(self):
        """Ingresa la matriz ampliada directamente"""
        console.print(f"\n[bold]Matriz ampliada [A|b] ({self.num_variables}x{self.num_variables+1}):[/bold]")
        
        self.matriz_ampliada = np.zeros((self.num_variables, self.num_variables + 1))
        
        for i in range(self.num_variables):
            console.print(f"\nFila {i+1}:")
            for j in range(self.num_variables + 1):
                if j < self.num_variables:
                    etiqueta = f"A[{i+1},{j+1}]"
                else:
                    etiqueta = f"b[{i+1}]"
                
                valor = validar_flotante(f"  {etiqueta} = ", puede_ser_cero=True)
                self.matriz_ampliada[i, j] = valor
        
        # Separar matriz y vector
        self.matriz_coeficientes = self.matriz_ampliada[:, :-1]
        self.vector_terminos = self.matriz_ampliada[:, -1]
        
        self._mostrar_sistema()

    def _generar_sistema_aleatorio(self):
        """Genera un sistema aleatorio para pruebas"""
        console.print(f"\n[cyan]Generando sistema aleatorio {self.num_variables}x{self.num_variables}...[/cyan]")
        
        # Generar matriz con solución conocida
        np.random.seed(42)  # Para reproducibilidad
        
        # Crear solución conocida
        solucion_verdadera = np.random.randint(-5, 6, self.num_variables)
        
        # Crear matriz de coeficientes
        self.matriz_coeficientes = np.random.randint(-3, 4, (self.num_variables, self.num_variables))
        
        # Asegurar que no sea singular
        while abs(np.linalg.det(self.matriz_coeficientes)) < 1e-10:
            self.matriz_coeficientes = np.random.randint(-3, 4, (self.num_variables, self.num_variables))
        
        # Calcular vector de términos independientes
        self.vector_terminos = self.matriz_coeficientes @ solucion_verdadera
        
        # Crear matriz ampliada
        self.matriz_ampliada = np.column_stack([self.matriz_coeficientes, self.vector_terminos])
        
        console.print(f"[green]✓ Sistema generado con solución conocida: {solucion_verdadera}[/green]")
        self._mostrar_sistema()

    def _mostrar_sistema(self):
        """Muestra el sistema actual"""
        console.print(f"\n[bold cyan]Sistema de ecuaciones:[/bold cyan]")
        
        # Crear tabla para mostrar el sistema
        tabla = Table(title=f"Matriz Ampliada {self.num_variables}x{self.num_variables+1}")
        
        # Agregar columnas
        for j in range(self.num_variables):
            tabla.add_column(f"x₍{j+1}₎", style="blue")
        tabla.add_column("=", style="white")
        tabla.add_column("b", style="green")
        
        # Agregar filas
        for i in range(self.num_variables):
            fila = []
            for j in range(self.num_variables):
                fila.append(formatear_numero(self.matriz_ampliada[i, j]))
            fila.append("=")
            fila.append(formatear_numero(self.matriz_ampliada[i, -1]))
            tabla.add_row(*fila)
        
        console.print(tabla)
        
        # Información adicional
        det_A = np.linalg.det(self.matriz_coeficientes)
        console.print(f"\n[bold]Determinante de A:[/bold] {formatear_numero(det_A)}")
        
        if abs(det_A) < 1e-10:
            console.print("[red]⚠ Matriz singular - el sistema puede no tener solución única[/red]")
        else:
            console.print("[green]✓ Matriz no singular - solución única esperada[/green]")
        
        input("\nPresione Enter para continuar...")

    def configurar_opciones(self):
        """Configura opciones del método"""
        limpiar_pantalla()
        console.print(Panel.fit("OPCIONES DE SOLUCIÓN", style="bold blue"))
        
        console.print(f"\n[bold]Configuración actual:[/bold]")
        console.print(f"Pivoteo parcial: {'Activado' if self.pivoteo else 'Desactivado'}")
        console.print(f"Mostrar fracciones: {'Sí' if self.mostrar_fracciones else 'No'}")
        
        if input("\n¿Cambiar configuración? (s/n): ").lower() == 's':
            console.print("\n[bold]Pivoteo parcial:[/bold]")
            console.print("Mejora la estabilidad numérica intercambiando filas")
            self.pivoteo = input("¿Activar pivoteo parcial? (s/n): ").lower() == 's'
            
            console.print("\n[bold]Mostrar fracciones:[/bold]")
            console.print("Muestra resultados como fracciones exactas cuando es posible")
            self.mostrar_fracciones = input("¿Mostrar fracciones? (s/n): ").lower() == 's'
            
            console.print(f"\n[green]Configuración actualizada[/green]")
            input("Presione Enter para continuar...")

    def ejecutar_gauss_jordan(self):
        """Ejecuta la eliminación Gauss-Jordan"""
        if self.matriz_ampliada is None:
            console.print("[red]Primero debe ingresar un sistema de ecuaciones[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("EJECUTANDO GAUSS-JORDAN", style="bold yellow"))
        
        # Mostrar configuración
        console.print(f"\n[bold]Configuración:[/bold]")
        console.print(f"Pivoteo: {'Activado' if self.pivoteo else 'Desactivado'}")
        console.print(f"Tamaño: {self.num_variables}x{self.num_variables}")
        
        try:
            # Trabajar con copia de la matriz
            matriz_trabajo = self.matriz_ampliada.copy()
            self.pasos = []
            self.intercambios = []
            
            console.print(f"\n[cyan]Iniciando eliminación...[/cyan]")
            
            # Fase de eliminación hacia adelante
            for k in track(range(self.num_variables), description="Eliminando..."):
                
                # Pivoteo parcial
                if self.pivoteo:
                    self._pivoteo_parcial(matriz_trabajo, k)
                
                # Verificar elemento diagonal
                if abs(matriz_trabajo[k, k]) < 1e-15:
                    console.print(f"[red]⚠ Elemento diagonal muy pequeño en posición ({k+1},{k+1})[/red]")
                    console.print("El sistema puede ser singular o mal condicionado")
                    break
                
                # Normalizar fila k (hacer 1 el elemento diagonal)
                if abs(matriz_trabajo[k, k] - 1.0) > 1e-15:
                    factor = matriz_trabajo[k, k]
                    matriz_trabajo[k, :] = matriz_trabajo[k, :] / factor
                    self.pasos.append(f"R₍{k+1}₎ → R₍{k+1}₎/{formatear_numero(factor)}")
                
                # Eliminar elementos en la columna k (arriba y abajo del pivote)
                for i in range(self.num_variables):
                    if i != k and abs(matriz_trabajo[i, k]) > 1e-15:
                        factor = matriz_trabajo[i, k]
                        matriz_trabajo[i, :] = matriz_trabajo[i, :] - factor * matriz_trabajo[k, :]
                        self.pasos.append(f"R₍{i+1}₎ → R₍{i+1}₎ - ({formatear_numero(factor)})R₍{k+1}₎")
            
            # Extraer solución
            self.solucion = matriz_trabajo[:, -1].copy()
            
            # Verificar solución
            self._verificar_solucion()
            
            console.print(f"\n[bold green]✓ Eliminación Gauss-Jordan completada[/bold green]")
            console.print(f"[bold]Pasos realizados:[/bold] {len(self.pasos)}")
            
        except Exception as e:
            console.print(f"[red]Error durante la eliminación: {e}[/red]")
        
        input("Presione Enter para continuar...")

    def _pivoteo_parcial(self, matriz: np.ndarray, k: int):
        """Realiza pivoteo parcial para mejorar estabilidad"""
        # Encontrar la fila con el mayor elemento en valor absoluto en la columna k
        max_fila = k
        for i in range(k + 1, self.num_variables):
            if abs(matriz[i, k]) > abs(matriz[max_fila, k]):
                max_fila = i
        
        # Intercambiar filas si es necesario
        if max_fila != k:
            matriz[[k, max_fila], :] = matriz[[max_fila, k], :]
            self.intercambios.append((k, max_fila))
            self.pasos.append(f"R₍{k+1}₎ ↔ R₍{max_fila+1}₎")

    def _verificar_solucion(self):
        """Verifica la calidad de la solución"""
        if self.solucion is None:
            return
        
        # Calcular residuo Ax - b
        residuo = self.matriz_coeficientes @ self.solucion - self.vector_terminos
        norma_residuo = np.linalg.norm(residuo)
        
        console.print(f"\n[bold]Verificación:[/bold]")
        console.print(f"||Ax - b|| = {formatear_numero(norma_residuo)}")
        
        if norma_residuo < 1e-10:
            console.print("[green]✓ Solución verificada correctamente[/green]")
        elif norma_residuo < 1e-6:
            console.print("[yellow]⚠ Solución aceptable (pequeños errores numéricos)[/yellow]")
        else:
            console.print("[red]⚠ Posibles errores en la solución[/red]")

    def mostrar_pasos(self):
        """Muestra los pasos detallados de la eliminación"""
        if not self.pasos:
            console.print("[red]No hay pasos para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("PASOS DETALLADOS", style="bold cyan"))
        
        console.print(f"\n[bold]Total de operaciones realizadas:[/bold] {len(self.pasos)}")
        
        if self.intercambios:
            console.print(f"\n[bold]Intercambios de filas (pivoteo):[/bold]")
            for i, (fila1, fila2) in enumerate(self.intercambios):
                console.print(f"{i+1}. R₍{fila1+1}₎ ↔ R₍{fila2+1}₎")
        
        console.print(f"\n[bold]Operaciones elementales:[/bold]")
        
        # Mostrar pasos en grupos
        pasos_por_pagina = 15
        total_paginas = (len(self.pasos) + pasos_por_pagina - 1) // pasos_por_pagina
        
        for pagina in range(total_paginas):
            inicio = pagina * pasos_por_pagina
            fin = min(inicio + pasos_por_pagina, len(self.pasos))
            
            console.print(f"\n[bold cyan]Pasos {inicio+1}-{fin}:[/bold cyan]")
            
            for i in range(inicio, fin):
                console.print(f"{i+1:3d}. {self.pasos[i]}")
            
            if pagina < total_paginas - 1:
                if input(f"\nPágina {pagina+1}/{total_paginas}. ¿Continuar? (s/n): ").lower() != 's':
                    break
        
        input("\nPresione Enter para continuar...")

    def analizar_matriz(self):
        """Analiza propiedades de la matriz y solución"""
        if self.matriz_coeficientes is None or self.solucion is None:
            console.print("[red]Primero debe resolver un sistema[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("ANÁLISIS DE MATRIZ Y SOLUCIÓN", style="bold purple"))
        
        try:
            # Propiedades de la matriz
            det_A = np.linalg.det(self.matriz_coeficientes)
            cond_A = np.linalg.cond(self.matriz_coeficientes)
            rango_A = np.linalg.matrix_rank(self.matriz_coeficientes)
            
            # Tabla de propiedades
            tabla = Table(title="Propiedades de la Matriz A")
            tabla.add_column("Propiedad", style="cyan")
            tabla.add_column("Valor", style="yellow")
            tabla.add_column("Interpretación", style="green")
            
            tabla.add_row(
                "Determinante",
                formatear_numero(det_A),
                "Singular" if abs(det_A) < 1e-10 else "No singular"
            )
            
            tabla.add_row(
                "Número de condición",
                formatear_numero(cond_A),
                "Bien condicionada" if cond_A < 100 else 
                "Moderadamente condicionada" if cond_A < 1e6 else 
                "Mal condicionada"
            )
            
            tabla.add_row(
                "Rango",
                str(rango_A),
                f"Completo ({self.num_variables})" if rango_A == self.num_variables else "Deficiente"
            )
            
            console.print(tabla)
            
            # Análisis de la solución
            console.print(f"\n[bold]Solución encontrada:[/bold]")
            
            tabla_sol = Table(title="Vector Solución")
            tabla_sol.add_column("Variable", style="cyan")
            tabla_sol.add_column("Valor", style="yellow")
            
            variables = [f"x₍{i+1}₎" for i in range(self.num_variables)]
            
            for var, val in zip(variables, self.solucion):
                if self.mostrar_fracciones:
                    # Intentar mostrar como fracción
                    try:
                        fraccion = sp.Rational(val).limit_denominator(1000)
                        if abs(float(fraccion) - val) < 1e-10 and fraccion.denominator <= 100:
                            valor_str = str(fraccion)
                        else:
                            valor_str = formatear_numero(val)
                    except:
                        valor_str = formatear_numero(val)
                else:
                    valor_str = formatear_numero(val)
                
                tabla_sol.add_row(var, valor_str)
            
            console.print(tabla_sol)
            
            # Verificación adicional
            residuo = self.matriz_coeficientes @ self.solucion - self.vector_terminos
            norma_residuo = np.linalg.norm(residuo)
            
            console.print(f"\n[bold]Verificación numérica:[/bold]")
            console.print(f"||Ax - b|| = {formatear_numero(norma_residuo)}")
            console.print(f"Error relativo ≈ {formatear_numero(norma_residuo / np.linalg.norm(self.vector_terminos))}")
            
        except Exception as e:
            console.print(f"[red]Error en análisis: {e}[/red]")
        
        input("\nPresione Enter para continuar...")

    def mostrar_graficos(self):
        """Muestra gráficos para sistemas 2x2"""
        if self.num_variables != 2 or self.solucion is None:
            console.print("[red]Gráficos solo disponibles para sistemas 2x2 resueltos[/red]")
            input("Presione Enter para continuar...")
            return
        
        console.print("[cyan]Generando gráfico del sistema 2x2...[/cyan]")
        
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Gráfico 1: Rectas del sistema
            x_vals = np.linspace(-10, 10, 400)
            
            # Ecuación 1: a11*x + a12*y = b1 => y = (b1 - a11*x) / a12
            # Ecuación 2: a21*x + a22*y = b2 => y = (b2 - a21*x) / a22
            
            A = self.matriz_coeficientes
            b = self.vector_terminos
            
            if abs(A[0, 1]) > 1e-10:  # a12 ≠ 0
                y1_vals = (b[0] - A[0, 0] * x_vals) / A[0, 1]
                ax1.plot(x_vals, y1_vals, 'b-', linewidth=2, 
                        label=f'{formatear_numero(A[0,0])}x + {formatear_numero(A[0,1])}y = {formatear_numero(b[0])}')
            
            if abs(A[1, 1]) > 1e-10:  # a22 ≠ 0
                y2_vals = (b[1] - A[1, 0] * x_vals) / A[1, 1]
                ax1.plot(x_vals, y2_vals, 'r-', linewidth=2,
                        label=f'{formatear_numero(A[1,0])}x + {formatear_numero(A[1,1])}y = {formatear_numero(b[1])}')
            
            # Marcar solución
            ax1.plot(self.solucion[0], self.solucion[1], 'go', markersize=10, 
                    label=f'Solución: ({formatear_numero(self.solucion[0])}, {formatear_numero(self.solucion[1])})')
            
            ax1.set_xlabel('x')
            ax1.set_ylabel('y')
            ax1.set_title('Sistema de Ecuaciones Lineales 2x2')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
            ax1.axis('equal')
            
            # Establecer límites razonables
            sol_x, sol_y = self.solucion[0], self.solucion[1]
            margen = max(5, abs(sol_x), abs(sol_y)) + 2
            ax1.set_xlim(sol_x - margen, sol_x + margen)
            ax1.set_ylim(sol_y - margen, sol_y + margen)
            
            # Gráfico 2: Interpretación geométrica de la eliminación
            ax2.text(0.1, 0.9, 'Interpretación del Método:', transform=ax2.transAxes, 
                    fontsize=14, fontweight='bold')
            
            info_text = f"""
Sistema original:
{formatear_numero(A[0,0])}x + {formatear_numero(A[0,1])}y = {formatear_numero(b[0])}
{formatear_numero(A[1,0])}x + {formatear_numero(A[1,1])}y = {formatear_numero(b[1])}

Solución:
x = {formatear_numero(self.solucion[0])}
y = {formatear_numero(self.solucion[1])}

Operaciones realizadas: {len(self.pasos)}
Intercambios de filas: {len(self.intercambios)}

Determinante: {formatear_numero(np.linalg.det(A))}
Número de condición: {formatear_numero(np.linalg.cond(A))}
            """
            
            ax2.text(0.1, 0.8, info_text, transform=ax2.transAxes, 
                    fontsize=10, verticalalignment='top', fontfamily='monospace')
            
            ax2.set_xlim(0, 1)
            ax2.set_ylim(0, 1)
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            console.print(f"[red]Error generando gráfico: {e}[/red]")

    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        limpiar_pantalla()
        ayuda_texto = """
[bold blue]ELIMINACIÓN GAUSS-JORDAN[/bold blue]

[bold]¿Qué es?[/bold]
Método directo para resolver sistemas de ecuaciones lineales Ax = b
mediante operaciones elementales que transforman la matriz ampliada
a la forma escalonada reducida por filas.

[bold]Algoritmo:[/bold]
1. Formar matriz ampliada [A|b]
2. Para cada columna k:
   • Pivoteo parcial (opcional)
   • Normalizar fila k (hacer 1 el elemento diagonal)
   • Eliminar todos los elementos de la columna k excepto el pivote
3. La última columna contiene la solución

[bold]Operaciones elementales:[/bold]
• Intercambio de filas: Ri ↔ Rj
• Multiplicación por escalar: Ri → c·Ri (c ≠ 0)
• Combinación lineal: Ri → Ri + c·Rj

[bold]Ventajas:[/bold]
• Método directo (no iterativo)
• Encuentra solución exacta (sin errores de redondeo)
• Funciona para cualquier sistema no singular
• Proporciona información sobre rango y determinante

[bold]Desventajas:[/bold]
• Costo computacional O(n³)
• Sensible a errores de redondeo para matrices grandes
• No eficiente para matrices dispersas
• Requiere almacenar toda la matriz

[bold]Pivoteo parcial:[/bold]
• Intercambia filas para poner el elemento de mayor valor absoluto
  como pivote
• Mejora la estabilidad numérica
• Reduce errores de redondeo

[bold]Aplicaciones:[/bold]
• Sistemas de ecuaciones lineales
• Cálculo de matriz inversa
• Determinante de matrices
• Análisis de rango de matrices

[bold]Casos especiales:[/bold]
• Determinante = 0: Sistema singular (sin solución única)
• Matriz mal condicionada: Solución sensible a perturbaciones
• Sistema homogéneo: Al menos la solución trivial x = 0
        """
        
        console.print(Panel(ayuda_texto, title="AYUDA", border_style="blue"))
        input("\nPresione Enter para continuar...")

    def main(self):
        """Función principal con el bucle del menú"""
        while True:
            try:
                limpiar_pantalla()
                opcion = self.mostrar_menu_principal()
                
                if opcion == 1:
                    self.ingresar_sistema()
                elif opcion == 2:
                    self.configurar_opciones()
                elif opcion == 3:
                    self.ejecutar_gauss_jordan()
                elif opcion == 4:
                    self.mostrar_pasos()
                elif opcion == 5:
                    self.analizar_matriz()
                elif opcion == 6:
                    self.mostrar_graficos()
                elif opcion == 7:
                    self.mostrar_ayuda()
                elif opcion == 8:
                    console.print("[bold green]¡Hasta luego![/bold green]")
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[bold red]Programa interrumpido[/bold red]")
                break
            except Exception as e:
                console.print(f"[bold red]Error inesperado: {e}[/bold red]")
                input("Presione Enter para continuar...")

if __name__ == "__main__":
    gauss_jordan = GaussJordan()
    gauss_jordan.main()
