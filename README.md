# 🔢 Proyecto de Métodos Numéricos
## 📁 Estructura del Proyecto

```
métodos_numéricos/
├── ecuaciones_no_lineales/     # Métodos para ecuaciones no lineales
│   ├── biseccion.py           # Método de bisección
│   ├── falsa_posicion.py      # Método de falsa posición
│   ├── punto_fijo.py          # Método de punto fijo
│   ├── newton_raphson.py      # Método de Newton-Raphson
│   ├── deflacion_polinomios.py # Deflación de polinomios
│   ├── division_polinomios.py  # División sintética
│   ├── polinomio_orden2.py    # Fórmula cuadrática
│   └── bairstow.py            # Método de Bairstow
├── sistemas_ecuaciones/        # Sistemas de ecuaciones
│   ├── punto_fijo_sistemas.py # Punto fijo para sistemas
│   ├── newton_raphson_sistemas.py # Newton-Raphson para sistemas
│   └── gauss_jordan.py        # Eliminación Gauss-Jordan
├── optimizacion/              # Métodos de optimización
│   ├── newton_optimizacion.py # Newton para optimización
│   ├── seccion_dorada.py      # Sección dorada
│   └── simplex.py             # Método Simplex
├── ajuste_curvas/             # Ajuste e interpolación
│   ├── regresion_lineal.py    # Regresión lineal
│   ├── regresion_no_lineal.py # Regresión no lineal
│   ├── regresion_polinomial.py # Regresión polinomial
│   ├── interpolacion_lineal.py # Interpolación lineal
│   ├── interpolacion_cuadratica.py # Interpolación cuadrática
│   └── interpolacion_lagrange.py # Interpolación de Lagrange
├── utilidades/                # Módulos de apoyo
│   ├── validacion.py          # Validación de entrada
│   ├── graficas.py            # Funciones de graficación
│   ├── menus.py               # Menús interactivos
│   └── formatos.py            # Formateo de resultados
├── documentacion/             # Documentación en Quarto
└── requirements.txt           # Dependencias del proyecto
```

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.8 o superior
- pip (administrador de paquetes de Python)

### Instalación

1. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Características Principales

### ✨ Interfaz de Usuario
- **Menús numerados interactivos** con Rich
- **Validación robusta** de entrada del usuario
- **Mensajes de error descriptivos** y ayuda contextual
- **Limpieza automática** de pantalla entre menús

### 📊 Visualización
- **Gráficos en tiempo real** con Matplotlib
- **Animaciones de convergencia** para métodos iterativos
- **Visualización de funciones** y resultados
- **Comparación de métodos** lado a lado

### 🔧 Funcionalidades Técnicas
- **Evaluación simbólica** de funciones con SymPy
- **Manejo de errores** y casos extremos
- **Progreso visual** con barras de tqdm
- **Formato elegante** de tablas y resultados
- **Exportación de resultados** en múltiples formatos

### 📈 Análisis de Convergencia
- **Tablas de iteraciones** detalladas
- **Gráficos de error** y convergencia
- **Análisis de estabilidad** numérica
- **Comparación de velocidad** entre métodos

## 📚 Métodos Implementados

### 🔍 Ecuaciones No Lineales
- **Bisección**: Método garantizado para funciones continuas
- **Falsa Posición**: Mejor aproximación que bisección
- **Punto Fijo**: Para ecuaciones de la forma x = g(x)
- **Newton-Raphson**: Convergencia cuadrática (requiere derivada)
- **Deflación de Polinomios**: Para múltiples raíces
- **División Sintética**: Evaluación eficiente de polinomios
- **Fórmula Cuadrática**: Solución directa para grado 2
- **Bairstow**: Para raíces complejas de polinomios

### 🔢 Sistemas de Ecuaciones
- **Punto Fijo para Sistemas**: Iteración para sistemas no lineales
- **Newton-Raphson para Sistemas**: Método del Jacobiano
- **Gauss-Jordan**: Eliminación con pivoteo

### 📊 Optimización
- **Newton No Restringida**: Segunda derivada para optimización
- **Sección Dorada**: Búsqueda unidimensional
- **Simplex**: Programación lineal

### 📈 Ajuste de Curvas
- **Regresión Lineal**: Mínimos cuadrados ordinarios
- **Regresión No Lineal**: Levenberg-Marquardt
- **Regresión Polinomial**: Ajuste de grado n
- **Interpolación Lineal**: Entre puntos consecutivos
- **Interpolación Cuadrática**: Parábolas locales
- **Interpolación de Lagrange**: Polinomio único

- **NumPy**: Cálculos numéricos y arrays
- **SymPy**: Matemática simbólica y evaluación de funciones
- **Matplotlib**: Gráficos y visualizaciones
- **Rich**: Interfaz de terminal elegante y colorida
- **tqdm**: Barras de progreso para iteraciones
- **Seaborn**: Estilos de gráficos mejorados