#!/usr/bin/env python3
"""
Módulo de validación para métodos numéricos
Funciones para validar entrada de usuario, rangos, funciones matemáticas, etc.
"""

import re
import numpy as np
import sympy as sp
from typing import Union, Tuple, Optional, Any
from rich.console import Console

console = Console()

def validar_numero(
    prompt: str, 
    tipo: str = "float", 
    min_val: Optional[float] = None, 
    max_val: Optional[float] = None,
    excluir_cero: bool = False
) -> Union[int, float]:
    """
    Valida entrada numérica del usuario
    
    Args:
        prompt: Mensaje para solicitar entrada
        tipo: 'int' o 'float'
        min_val: Valor mínimo permitido
        max_val: Valor máximo permitido
        excluir_cero: Si True, no permite cero
    
    Returns:
        Número validado
    """
    while True:
        try:
            entrada = input(f"{prompt}: ").strip()
            
            if tipo == "int":
                numero = int(entrada)
            else:
                numero = float(entrada)
            
            # Validar rango
            if min_val is not None and numero < min_val:
                console.print(f"[red]❌ El valor debe ser mayor o igual a {min_val}[/red]")
                continue
                
            if max_val is not None and numero > max_val:
                console.print(f"[red]❌ El valor debe ser menor o igual a {max_val}[/red]")
                continue
                
            # Validar cero
            if excluir_cero and numero == 0:
                console.print("[red]❌ El valor no puede ser cero[/red]")
                continue
                
            return numero
            
        except ValueError:
            console.print(f"[red]❌ Por favor ingrese un número {tipo} válido[/red]")

def validar_opcion_menu(opciones: list[int], prompt: str = "Seleccione una opción") -> int:
    """
    Valida selección de menú
    
    Args:
        opciones: Lista de opciones válidas
        prompt: Mensaje personalizado
    
    Returns:
        Opción válida seleccionada
    """
    while True:
        try:
            opcion = int(input(f"\n{prompt}: ").strip())
            if opcion in opciones:
                return opcion
            else:
                console.print(f"[red]❌ Opción inválida. Opciones válidas: {opciones}[/red]")
        except ValueError:
            console.print("[red]❌ Por favor ingrese un número entero[/red]")

def validar_funcion(funcion_str: str) -> Tuple[bool, Optional[sp.Expr], str]:
    """
    Valida y parsea una función matemática ingresada como string
    
    Args:
        funcion_str: String de la función (ej: "x**2 + 2*x - 1")
    
    Returns:
        Tuple(es_valida, expresion_sympy, mensaje_error)
    """
    try:
        # Limpiar la función
        funcion_str = funcion_str.replace('^', '**')  # Convertir ^ a **
        funcion_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', funcion_str)  # 2x -> 2*x
        
        # Variables permitidas
        x = sp.Symbol('x')
        variables_permitidas = {
            'x': x,
            'sin': sp.sin,
            'cos': sp.cos,
            'tan': sp.tan,
            'exp': sp.exp,
            'log': sp.log,
            'ln': sp.log,
            'sqrt': sp.sqrt,
            'pi': sp.pi,
            'e': sp.E,
            'abs': sp.Abs
        }
        
        # Parsear la expresión
        expr = sp.sympify(funcion_str, locals=variables_permitidas)
        
        # Verificar que solo contenga la variable x
        variables = expr.free_symbols
        if len(variables) > 1 or (len(variables) == 1 and x not in variables):
            return False, None, "La función debe contener solo la variable 'x'"
        
        # Probar evaluación en algunos puntos
        for test_val in [0, 1, -1, 0.5]:
            try:
                float(expr.subs(x, test_val))
            except:
                return False, None, f"Error al evaluar la función en x={test_val}"
        
        return True, expr, ""
        
    except Exception as e:
        return False, None, f"Error de sintaxis: {str(e)}"

def validar_intervalo(a: float, b: float) -> Tuple[bool, str]:
    """
    Valida que el intervalo [a,b] sea válido
    
    Args:
        a: Extremo izquierdo
        b: Extremo derecho
    
    Returns:
        Tuple(es_valido, mensaje_error)
    """
    if a >= b:
        return False, "El extremo izquierdo debe ser menor que el derecho"
    
    if abs(b - a) < 1e-15:
        return False, "El intervalo es demasiado pequeño"
    
    return True, ""

def validar_tolerancia(tolerancia: float) -> Tuple[bool, str]:
    """
    Valida tolerancia para métodos iterativos
    
    Args:
        tolerancia: Valor de tolerancia
    
    Returns:
        Tuple(es_valida, mensaje_error)
    """
    if tolerancia <= 0:
        return False, "La tolerancia debe ser positiva"
    
    if tolerancia >= 1:
        return False, "La tolerancia debe ser menor que 1"
    
    if tolerancia < 1e-15:
        return False, "La tolerancia es demasiado pequeña (mínimo: 1e-15)"
    
    return True, ""

def validar_max_iteraciones(max_iter: int) -> Tuple[bool, str]:
    """
    Valida número máximo de iteraciones
    
    Args:
        max_iter: Número máximo de iteraciones
    
    Returns:
        Tuple(es_valido, mensaje_error)
    """
    if max_iter <= 0:
        return False, "El número de iteraciones debe ser positivo"
    
    if max_iter > 10000:
        return False, "Número de iteraciones demasiado alto (máximo: 10000)"
    
    return True, ""

def es_numero_complejo_valido(numero: complex) -> bool:
    """
    Verifica si un número complejo es válido (no NaN o infinito)
    
    Args:
        numero: Número complejo a verificar
    
    Returns:
        True si es válido, False en caso contrario
    """
    return (not np.isnan(numero.real) and 
            not np.isnan(numero.imag) and 
            not np.isinf(numero.real) and 
            not np.isinf(numero.imag))

def limpiar_pantalla():
    """Limpia la pantalla de la consola"""
    import os
    os.system('clear' if os.name == 'posix' else 'cls')

def confirmar_accion(mensaje: str = "¿Desea continuar?") -> bool:
    """
    Solicita confirmación del usuario
    
    Args:
        mensaje: Mensaje de confirmación
    
    Returns:
        True si confirma, False en caso contrario
    """
    while True:
        respuesta = input(f"{mensaje} (s/n): ").strip().lower()
        if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
            return True
        elif respuesta in ['n', 'no']:
            return False
        else:
            console.print("[red]❌ Por favor responda 's' o 'n'[/red]")
