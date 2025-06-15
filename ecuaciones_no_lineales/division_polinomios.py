#!/usr/bin/env python3
"""
División de Polinomios - Implementación con menús interactivos
Método para dividir polinomios usando algoritmo sintético y tradicional
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
    graficar_division_polinomios
)

console = Console()

class DivisionPolinomios:
    def __init__(self):
        self.dividendo = []
        self.divisor = []
        self.cociente = []
        self.residuo = []
        self.polinomio_dividendo = None
        self.polinomio_divisor = None
        self.metodo = "sintetico"  # "sintetico" o "tradicional"
        self.pasos = []

    def mostrar_menu_principal(self):
        """Muestra el menú principal del método"""
        opciones = [
            "Ingresar dividendo",
            "Ingresar divisor", 
            "Seleccionar método de división",
            "Ejecutar división",
            "Ver resultados",
            "Mostrar gráficos",
            "Ver ayuda",
            "Salir"
        ]
        return crear_menu("DIVISIÓN DE POLINOMIOS", opciones)

    def ingresar_polinomio(self, tipo: str):
        """Menú para ingreso de polinomios (dividendo o divisor)"""
        limpiar_pantalla()
        console.print(Panel.fit(f"INGRESO DE {tipo.upper()}", style="bold green"))
        
        while True:
            try:
                console.print("\n[bold]Opciones de ingreso:[/bold]")
                console.print("1. Ingresar coeficientes directamente")
                console.print("2. Ingresar polinomio simbólico")
                
                opcion = validar_entero("Seleccione opción (1-2): ", 1, 2)
                
                if opcion == 1:
                    coeficientes = self._ingresar_coeficientes(tipo)
                else:
                    coeficientes, polinomio = self._ingresar_simbolico(tipo)
                
                # Mostrar el polinomio ingresado
                self._mostrar_polinomio(coeficientes, tipo)
                
                if input(f"\n¿Confirmar {tipo}? (s/n): ").lower() == 's':
                    if tipo == "dividendo":
                        self.dividendo = coeficientes
                        if opcion == 2:
                            self.polinomio_dividendo = polinomio
                    else:
                        self.divisor = coeficientes
                        if opcion == 2:
                            self.polinomio_divisor = polinomio
                    break
                    
            except KeyboardInterrupt:
                console.print("\n[yellow]Operación cancelada[/yellow]")
                return

    def _ingresar_coeficientes(self, tipo: str) -> List[float]:
        """Ingresa coeficientes del polinomio directamente"""
        console.print(f"\n[bold]Ingreso de coeficientes del {tipo} (de mayor a menor grado):[/bold]")
        
        grado = validar_entero(f"Grado del {tipo}: ", 0, 20)
        coeficientes = []
        
        for i in range(grado + 1):
            exponente = grado - i
            coef = validar_flotante(f"Coeficiente de x^{exponente}: ")
            coeficientes.append(coef)
        
        return coeficientes

    def _ingresar_simbolico(self, tipo: str) -> Tuple[List[float], sp.Expr]:
        """Ingresa polinomio en forma simbólica"""
        console.print(f"\n[bold]Ejemplos de {tipo}:[/bold] x^3 - 2*x^2 + x - 1, 2*x^4 + 3*x^2 - 5")
        
        while True:
            expr_str = input(f"Ingrese el {tipo}: ").strip()
            try:
                x = sp.Symbol('x')
                polinomio = sp.expand(sp.sympify(expr_str))
                
                # Extraer coeficientes
                poly = sp.Poly(polinomio, x)
                coeficientes = [float(coef) for coef in poly.all_coeffs()]
                
                return coeficientes, polinomio
                
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                continue

    def _mostrar_polinomio(self, coeficientes: List[float], tipo: str):
        """Muestra el polinomio actual"""
        if coeficientes:
            x = sp.Symbol('x')
            poly = sum(coef * x**i for i, coef in enumerate(reversed(coeficientes)))
            console.print(f"\n[bold cyan]{tipo.capitalize()}:[/bold cyan] {poly}")
            console.print(f"[bold]Grado:[/bold] {len(coeficientes) - 1}")

    def seleccionar_metodo(self):
        """Selecciona el método de división"""
        limpiar_pantalla()
        console.print(Panel.fit("SELECCIÓN DE MÉTODO", style="bold blue"))
        
        console.print("\n[bold]Métodos disponibles:[/bold]")
        console.print("1. División sintética (para divisores lineales)")
        console.print("2. División tradicional (algoritmo general)")
        
        console.print(f"\n[bold]Método actual:[/bold] {self.metodo}")
        
        if input("\n¿Cambiar método? (s/n): ").lower() == 's':
            opcion = validar_entero("Seleccione método (1-2): ", 1, 2)
            self.metodo = "sintetico" if opcion == 1 else "tradicional"
            
            console.print(f"[green]Método cambiado a: {self.metodo}[/green]")
            input("Presione Enter para continuar...")

    def ejecutar_division(self):
        """Ejecuta la división de polinomios"""
        if not self.dividendo or not self.divisor:
            console.print("[red]Debe ingresar tanto dividendo como divisor[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("EJECUTANDO DIVISIÓN", style="bold yellow"))
        
        # Verificar si es posible la división
        if len(self.divisor) > len(self.dividendo):
            console.print("[red]El grado del divisor no puede ser mayor al del dividendo[/red]")
            input("Presione Enter para continuar...")
            return
        
        try:
            if self.metodo == "sintetico" and len(self.divisor) == 2:
                self._division_sintetica()
            else:
                self._division_tradicional()
                
            console.print(f"\n[bold green]División completada exitosamente[/bold green]")
            
        except Exception as e:
            console.print(f"[red]Error durante la división: {e}[/red]")
        
        input("Presione Enter para continuar...")

    def _division_sintetica(self):
        """Realiza división sintética (solo para divisores lineales)"""
        if len(self.divisor) != 2:
            raise ValueError("División sintética solo funciona con divisores lineales")
        
        # El divisor debe ser de la forma (x - a), extraer 'a'
        if abs(self.divisor[0] - 1.0) > 1e-10:
            raise ValueError("División sintética requiere divisor mónico (coeficiente principal = 1)")
        
        raiz = -self.divisor[1] / self.divisor[0]
        
        console.print(f"[cyan]Realizando división sintética con raíz: {formatear_numero(raiz)}[/cyan]")
        
        # Algoritmo de división sintética
        self.pasos = []
        n = len(self.dividendo)
        resultado = [0] * (n - 1)
        
        # Primer coeficiente se copia directamente
        resultado[0] = self.dividendo[0]
        self.pasos.append(f"Paso 1: b₀ = a₀ = {formatear_numero(resultado[0])}")
        
        # Calcular resto de coeficientes
        for i in range(1, n - 1):
            resultado[i] = self.dividendo[i] + raiz * resultado[i - 1]
            self.pasos.append(
                f"Paso {i+1}: b₍{i}₎ = a₍{i}₎ + ({formatear_numero(raiz)}) × b₍{i-1}₎ = "
                f"{formatear_numero(self.dividendo[i])} + {formatear_numero(raiz * resultado[i-1])} = "
                f"{formatear_numero(resultado[i])}"
            )
        
        # El residuo es el último cálculo
        residuo_valor = self.dividendo[-1] + raiz * resultado[-1]
        self.pasos.append(
            f"Residuo: {formatear_numero(self.dividendo[-1])} + "
            f"{formatear_numero(raiz)} × {formatear_numero(resultado[-1])} = "
            f"{formatear_numero(residuo_valor)}"
        )
        
        self.cociente = resultado
        self.residuo = [residuo_valor] if abs(residuo_valor) > 1e-12 else []

    def _division_tradicional(self):
        """Realiza división tradicional de polinomios"""
        console.print("[cyan]Realizando división tradicional...[/cyan]")
        
        self.pasos = []
        dividendo_trabajo = self.dividendo.copy()
        divisor = self.divisor.copy()
        
        # Grados
        grado_dividendo = len(dividendo_trabajo) - 1
        grado_divisor = len(divisor) - 1
        grado_cociente = grado_dividendo - grado_divisor
        
        if grado_cociente < 0:
            self.cociente = []
            self.residuo = dividendo_trabajo
            return
        
        # Inicializar cociente
        self.cociente = [0] * (grado_cociente + 1)
        
        for i in track(range(grado_cociente + 1), description="Dividiendo..."):
            if len(dividendo_trabajo) < len(divisor):
                break
            
            # Calcular término del cociente
            coef_cociente = dividendo_trabajo[0] / divisor[0]
            self.cociente[i] = coef_cociente
            
            self.pasos.append(f"Paso {i+1}: Término del cociente = {formatear_numero(coef_cociente)}")
            
            # Multiplicar divisor por el término del cociente
            multiplicacion = [coef_cociente * coef for coef in divisor]
            
            # Restar del dividendo
            for j in range(len(multiplicacion)):
                if j < len(dividendo_trabajo):
                    dividendo_trabajo[j] -= multiplicacion[j]
            
            # Eliminar término principal (que ahora es 0)
            dividendo_trabajo = dividendo_trabajo[1:]
        
        # Lo que queda es el residuo
        self.residuo = dividendo_trabajo if dividendo_trabajo else []

    def mostrar_resultados(self):
        """Muestra los resultados de la división"""
        if not self.cociente and not self.residuo:
            console.print("[red]No hay resultados para mostrar[/red]")
            input("Presione Enter para continuar...")
            return
        
        limpiar_pantalla()
        console.print(Panel.fit("RESULTADOS DE LA DIVISIÓN", style="bold green"))
        
        # Mostrar polinomios originales
        x = sp.Symbol('x')
        
        dividendo_poly = sum(coef * x**i for i, coef in enumerate(reversed(self.dividendo)))
        divisor_poly = sum(coef * x**i for i, coef in enumerate(reversed(self.divisor)))
        
        console.print(f"[bold]Dividendo:[/bold] {dividendo_poly}")
        console.print(f"[bold]Divisor:[/bold] {divisor_poly}")
        
        # Mostrar cociente
        if self.cociente:
            cociente_poly = sum(coef * x**i for i, coef in enumerate(reversed(self.cociente)))
            console.print(f"[bold cyan]Cociente:[/bold cyan] {cociente_poly}")
        else:
            console.print(f"[bold cyan]Cociente:[/bold cyan] 0")
        
        # Mostrar residuo
        if self.residuo:
            residuo_poly = sum(coef * x**i for i, coef in enumerate(reversed(self.residuo)))
            console.print(f"[bold magenta]Residuo:[/bold magenta] {residuo_poly}")
        else:
            console.print(f"[bold magenta]Residuo:[/bold magenta] 0")
        
        # Verificación
        console.print("\n[bold]Verificación:[/bold]")
        console.print("Dividendo = Divisor × Cociente + Residuo")
        
        # Tabla de información
        tabla = Table(title="Información de la División")
        tabla.add_column("Componente", style="cyan")
        tabla.add_column("Grado", style="magenta")
        tabla.add_column("Método", style="blue")
        
        tabla.add_row("Dividendo", str(len(self.dividendo) - 1), self.metodo.capitalize())
        tabla.add_row("Divisor", str(len(self.divisor) - 1), "")
        tabla.add_row("Cociente", str(len(self.cociente) - 1) if self.cociente else "0", "")
        tabla.add_row("Residuo", str(len(self.residuo) - 1) if self.residuo else "0", "")
        
        console.print(tabla)
        
        # Mostrar pasos si están disponibles
        if self.pasos:
            console.print("\n[bold]Pasos del algoritmo:[/bold]")
            for paso in self.pasos:
                console.print(f"  • {paso}")
        
        input("\nPresione Enter para continuar...")

    def mostrar_graficos(self):
        """Muestra gráficos de la división"""
        if not self.cociente and not self.residuo:
            console.print("[red]No hay resultados para graficar[/red]")
            input("Presione Enter para continuar...")
            return
        
        console.print("[cyan]Generando gráficos...[/cyan]")
        graficar_division_polinomios(
            self.dividendo,
            self.divisor,
            self.cociente,
            self.residuo
        )

    def mostrar_ayuda(self):
        """Muestra información de ayuda sobre el método"""
        limpiar_pantalla()
        ayuda_texto = """
[bold blue]DIVISIÓN DE POLINOMIOS[/bold blue]

[bold]¿Qué es?[/bold]
La división de polinomios es una operación fundamental que permite expresar
un polinomio como el producto de dos polinomios más simples, más un residuo.

[bold]Fórmula General:[/bold]
P(x) = Q(x) × D(x) + R(x)

Donde:
- P(x): Polinomio dividendo
- D(x): Polinomio divisor  
- Q(x): Polinomio cociente
- R(x): Polinomio residuo (grado menor que D(x))

[bold]Métodos Implementados:[/bold]

[bold cyan]1. División Sintética:[/bold cyan]
- Solo para divisores lineales de la forma (x - a)
- Método más rápido y eficiente
- Menos propenso a errores de redondeo
- Algoritmo: bᵢ = aᵢ₊₁ + a × bᵢ₊₁

[bold cyan]2. División Tradicional:[/bold cyan]
- Funciona con cualquier divisor
- Algoritmo general más versátil
- Similar a la división larga de números

[bold]Aplicaciones:[/bold]
- Factorización de polinomios
- Simplificación de fracciones racionales
- Encontrar raíces de polinomios
- Análisis de funciones racionales

[bold]Casos Especiales:[/bold]
- Si el residuo es 0, el divisor es factor del dividendo
- El grado del cociente = grado dividendo - grado divisor
- El grado del residuo < grado del divisor

[bold]Ventajas de cada método:[/bold]
- Sintético: Rápido, preciso para divisores lineales
- Tradicional: General, funciona con cualquier polinomio
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
                    self.ingresar_polinomio("dividendo")
                elif opcion == 2:
                    self.ingresar_polinomio("divisor")
                elif opcion == 3:
                    self.seleccionar_metodo()
                elif opcion == 4:
                    self.ejecutar_division()
                elif opcion == 5:
                    self.mostrar_resultados()
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
    division = DivisionPolinomios()
    division.main()
