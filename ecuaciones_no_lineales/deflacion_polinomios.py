#!/usr/bin/env python3
"""
Deflación de Polinomios - Implementación con menús interactivos
Método para encontrar múltiples raíces de polinomios mediante deflación sintética
"""

import numpy as np
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track
import sympy as sp
import os
from typing import List, Tuple, Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utilidades import (
    validar_flotante, validar_entero, validar_polinomio,
    crear_menu, limpiar_pantalla, mostrar_progreso,
    formatear_numero, formatear_tabla_resultados,
    graficar_polinomio_deflacion
)

console = Console()

class DeflacionPolinomios:
    def __init__(self):
        self.coeficientes = []
        self.raices_conocidas = []
        self.polinomio_original = None
        self.polinomios_deflactados = []
        self.tolerancia = 1e-10
        self.max_iteraciones = 100
        self.resultados = []

    def mostrar_menu_principal(self):
        """Muestra el menú principal del método"""
        opciones = [
            "Ingresar polinomio",
            "Configurar parámetros",
            "Ejecutar deflación",
            "Ver resultados",
            "Mostrar gráficos",
            "Ver ayuda",
            "Salir"
        ]
        return crear_menu("DEFLACIÓN DE POLINOMIOS", opciones)

    def ingresar_polinomio(self):
        """Menú para ingreso del polinomio"""
        limpiar_pantalla()
        console.print(Panel.fit("INGRESO DE POLINOMIO", style="bold green"))
        
        while True:
            try:
                console.print("\n[bold]Opciones de ingreso:[/bold]")
                console.print("1. Ingresar coeficientes directamente")
                console.print("2. Ingresar polinomio simbólico")
                
                opcion = validar_entero("Seleccione opción (1-2): ", 1, 2)
                
                if opcion == 1:
                    self._ingresar_coeficientes()
                else:
                    self._ingresar_simbolico()
                
                # Mostrar el polinomio ingresado
                self._mostrar_polinomio()
                
                if input("\n¿Confirmar polinomio? (s/n): ").lower() == 's':
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Operación cancelada[/yellow]")
                return

    def _ingresar_coeficientes(self):
        """Ingresa coeficientes del polinomio directamente"""
        console.print("\n[bold]Ingreso de coeficientes (de mayor a menor grado):[/bold]")
        
        grado = validar_entero("Grado del polinomio: ", 1, 20)
        self.coeficientes = []
        
        for i in range(grado + 1):
            exponente = grado - i
            coef = validar_flotante(f"Coeficiente de x^{exponente}: ")
            self.coeficientes.append(coef)
        
        # Crear polinomio simbólico para verificación
        x = sp.Symbol('x')
        self.polinomio_original = sum(
            coef * x**i for i, coef in enumerate(reversed(self.coeficientes))
        )

    def _ingresar_simbolico(self):
        """Ingresa polinomio en forma simbólica"""
        console.print("\n[bold]Ejemplos:[/bold] x^3 - 2*x^2 + x - 1, 2*x^4 + 3*x^2 - 5")
        
        while True:
            expr_str = input("Ingrese el polinomio: ").strip()
            try:
                x = sp.Symbol('x')
                self.polinomio_original = sp.expand(sp.sympify(expr_str))
                
                # Extraer coeficientes
                poly = sp.Poly(self.polinomio_original, x)
                self.coeficientes = [float(coef) for coef in poly.all_coeffs()]
                break
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

    def _mostrar_polinomio(self):
        """Muestra el polinomio actual"""
        if self.polinomio_original:
            console.print(f"\n[bold cyan]Polinomio:[/bold cyan] {self.polinomio_original}")
            console.print(f"[bold]Grado:[/bold] {len(self.coeficientes) - 1}")

    def configurar_parametros(self):
        """Configura parámetros del método"""
        limpiar_pantalla()
        console.print(Panel.fit("CONFIGURACIÓN DE PARÁMETROS", style="bold blue"))
        
        console.print(f"\n[bold]Parámetros actuales:[/bold]")
        console.print(f"Tolerancia: {self.tolerancia}")
        console.print(f"Máximo iteraciones: {self.max_iteraciones}")
        
        if input("\n¿Cambiar parámetros? (s/n): ").lower() == 's':
            self.tolerancia = validar_flotante(
                "Nueva tolerancia (1e-10): ", 1e-15, 1e-3, self.tolerancia
            )
            self.max_iteraciones = validar_entero(
                "Nuevas máximo iteraciones (100): ", 10, 1000, self.max_iteraciones
            )

    def ingresar_raices_conocidas(self):
        """Permite ingresar raíces conocidas para deflación"""
        limpiar_pantalla()
        console.print(Panel.fit("RAÍCES CONOCIDAS", style="bold magenta"))
        
        if not self.coeficientes:
            console.print("[red]Primero debe ingresar un polinomio[/red]")
            input("Presione Enter para continuar...")
            return
        
        console.print("\nIngrese las raíces conocidas (una por línea, Enter vacío para terminar):")
        self.raices_conocidas = []
        
        i = 1
        while True:
            try:
                raiz_str = input(f"Raíz {i} (o Enter para terminar): ").strip()
                if not raiz_str:
                    break
                
                # Intentar evaluar como número complejo
                if 'j' in raiz_str or 'i' in raiz_str:
                    raiz_str = raiz_str.replace('i', 'j')
                    raiz = complex(raiz_str)
                else:
                    raiz = float(raiz_str)
                
                # Verificar que sea raíz
                if self._verificar_raiz(raiz):
                    self.raices_conocidas.append(raiz)
                    console.print(f"[green]✓ Raíz {raiz} agregada[/green]")
                    i += 1
                else:
                    console.print(f"[red]⚠ {raiz} no es raíz del polinomio[/red]")
                    
            except ValueError:
                console.print("[red]Formato inválido. Ejemplos: 1.5, -2, 1+2j[/red]")
            except KeyboardInterrupt:
                break

    def _verificar_raiz(self, raiz: complex, tolerancia: float = 1e-8) -> bool:
        """Verifica si un valor es raíz del polinomio"""
        if not self.polinomio_original:
            return False
        
        x = sp.Symbol('x')
        valor = complex(self.polinomio_original.subs(x, raiz))
        return abs(valor) < tolerancia

    def ejecutar_deflacion(self):
        """Ejecuta el proceso de deflación"""
        if not self.coeficientes:
            console.print("[red]Primero debe ingresar un polinomio[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("EJECUTANDO DEFLACIÓN", style="bold yellow"))
        
        # Si no hay raíces conocidas, intentar encontrar algunas
        if not self.raices_conocidas:
            console.print("\n[yellow]No hay raíces conocidas. Buscando raíces automáticamente...[/yellow]")
            self._buscar_raices_automaticamente()
        
        if not self.raices_conocidas:
            console.print("[red]No se encontraron raíces para realizar deflación[/red]")
            input("Presione Enter para continuar...")
            return
        
        self.resultados = []
        self.polinomios_deflactados = []
        coefs_actuales = self.coeficientes.copy()
        
        console.print(f"\n[bold]Deflactando con {len(self.raices_conocidas)} raíces...[/bold]")
        
        for i, raiz in enumerate(track(self.raices_conocidas, description="Deflactando...")):
            try:
                # Realizar deflación sintética
                coefs_deflactados = self._deflacion_sintetica(coefs_actuales, raiz)
                
                # Guardar resultado
                resultado = {
                    'raiz': raiz,
                    'polinomio_antes': coefs_actuales.copy(),
                    'polinomio_despues': coefs_deflactados,
                    'grado_reducido': len(coefs_deflactados) - 1
                }
                
                self.resultados.append(resultado)
                self.polinomios_deflactados.append(coefs_deflactados)
                
                # Actualizar coeficientes para siguiente deflación
                coefs_actuales = coefs_deflactados
                
                console.print(f"[green]✓ Deflación {i+1} completada[/green]")
                
            except Exception as e:
                console.print(f"[red]Error en deflación {i+1}: {e}[/red]")
                break
        
        console.print(f"\n[bold green]Deflación completada. {len(self.resultados)} etapas realizadas.[/bold green]")
        input("Presione Enter para continuar...")

    def _buscar_raices_automaticamente(self):
        """Busca raíces automáticamente usando numpy"""
        try:
            # Usar numpy para encontrar raíces
            raices_numpy = np.roots(self.coeficientes)
            
            # Filtrar raíces válidas
            for raiz in raices_numpy:
                if abs(raiz.imag) < self.tolerancia:
                    raiz = raiz.real
                
                if self._verificar_raiz(raiz):
                    self.raices_conocidas.append(raiz)
            
            if self.raices_conocidas:
                console.print(f"[green]Se encontraron {len(self.raices_conocidas)} raíces automáticamente[/green]")
            
        except Exception as e:
            console.print(f"[yellow]No se pudieron encontrar raíces automáticamente: {e}[/yellow]")

    def _deflacion_sintetica(self, coeficientes: List[float], raiz: complex) -> List[float]:
        """Realiza deflación sintética con una raíz"""
        n = len(coeficientes)
        if n < 2:
            raise ValueError("Polinomio muy simple para deflactar")
        
        # Algoritmo de deflación sintética (división sintética)
        resultado = [coeficientes[0]]  # Primer coeficiente se mantiene
        
        for i in range(1, n-1):  # Excluir el último término (residuo)
            siguiente_coef = coeficientes[i] + raiz * resultado[i-1]
            resultado.append(siguiente_coef)
        
        return resultado

    def mostrar_resultados(self):
        """Muestra los resultados de la deflación"""
        if not self.resultados:
            console.print("[red]No hay resultados para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("RESULTADOS DE DEFLACIÓN", style="bold green"))
        
        # Tabla resumen
        tabla = Table(title="Resumen de Deflación")
        tabla.add_column("Etapa", style="cyan")
        tabla.add_column("Raíz", style="magenta")
        tabla.add_column("Grado Original", style="blue")
        tabla.add_column("Grado Deflactado", style="green")
        tabla.add_column("Tipo", style="yellow")
        
        for i, resultado in enumerate(self.resultados):
            raiz = resultado['raiz']
            tipo_raiz = "Real" if isinstance(raiz, (int, float)) or abs(raiz.imag) < 1e-10 else "Compleja"
            
            tabla.add_row(
                str(i + 1),
                formatear_numero(raiz),
                str(len(resultado['polinomio_antes']) - 1),
                str(resultado['grado_reducido']),
                tipo_raiz
            )
        
        console.print(tabla)
        
        # Mostrar polinomios deflactados
        console.print("\n[bold]Polinomios deflactados:[/bold]")
        x = sp.Symbol('x')
        
        for i, coefs in enumerate(self.polinomios_deflactados):
            if len(coefs) > 1:
                poly = sum(coef * x**j for j, coef in enumerate(reversed(coefs)))
                console.print(f"Etapa {i+1}: {poly}")
        
        input("\nPresione Enter para continuar...")

    def mostrar_graficos(self):
        """Muestra gráficos del proceso de deflación"""
        if not self.resultados:
            console.print("[red]No hay resultados para graficar[/red]")
            input("Presione Enter para continuar...")
            return
        
        console.print("[cyan]Generando gráficos...[/cyan]")
        graficar_polinomio_deflacion(
            self.coeficientes,
            self.raices_conocidas,
            self.polinomios_deflactados
        )

    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        limpiar_pantalla()
        ayuda_texto = """
[bold blue]DEFLACIÓN DE POLINOMIOS[/bold blue]

[bold]¿Qué es?[/bold]
La deflación de polinomios es un método para reducir el grado de un polinomio
eliminando raíces conocidas, facilitando la búsqueda de las raíces restantes.

[bold]¿Cómo funciona?[/bold]
1. Se conoce una raíz r del polinomio P(x)
2. Se divide P(x) entre (x - r) usando división sintética
3. El resultado es un polinomio de grado menor
4. Se repite el proceso con otras raíces conocidas

[bold]Algoritmo de División Sintética:[/bold]
- Para P(x) = aₙxⁿ + aₙ₋₁xⁿ⁻¹ + ... + a₁x + a₀
- Y raíz conocida r
- El polinomio deflactado es Q(x) = bₙ₋₁xⁿ⁻¹ + ... + b₁x + b₀
- Donde: bₙ₋₁ = aₙ, bᵢ = aᵢ₊₁ + r·bᵢ₊₁

[bold]Ventajas:[/bold]
- Reduce la complejidad del problema
- Permite encontrar todas las raíces sistemáticamente
- Numéricamente estable para raíces bien separadas

[bold]Desventajas:[/bold]
- Requiere conocer al menos una raíz
- Los errores numéricos se pueden acumular
- Sensible a la precisión de las raíces conocidas

[bold]Casos de uso:[/bold]
- Factorización de polinomios
- Solución completa de ecuaciones polinómicas
- Análisis de sistemas dinámicos
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
                    self.ingresar_polinomio()
                elif opcion == 2:
                    self.configurar_parametros()
                elif opcion == 3:
                    if not self.coeficientes:
                        self.ingresar_polinomio()
                    self.ingresar_raices_conocidas()
                    self.ejecutar_deflacion()
                elif opcion == 4:
                    self.mostrar_resultados()
                elif opcion == 5:
                    self.mostrar_graficos()
                elif opcion == 6:
                    self.mostrar_ayuda()
                elif opcion == 7:
                    console.print("[bold green]¡Hasta luego![/bold green]")
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[bold red]Programa interrumpido[/bold red]")
                break
            except Exception as e:
                console.print(f"[bold red]Error inesperado: {e}[/bold red]")
                input("Presione Enter para continuar...")

if __name__ == "__main__":
    deflacion = DeflacionPolinomios()
    deflacion.main()
