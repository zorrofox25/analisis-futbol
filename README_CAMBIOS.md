# SportIQ AI — estabilización inicial

## Cambios incluidos

1. `fase2_poisson.py`
   - Valida que el optimizador haya convergido antes de aceptar el modelo.
   - Calcula la masa omitida usando la matriz truncada sin normalizar, para que la normalización no esconda la probabilidad fuera de la matriz visible.
   - Valida valores no finitos en la matriz Dixon–Coles.
2. `main.py`
   - Corrige la llamada a `calcular_partido_poisson`: pasa el DataFrame histórico y los nombres de equipos según la firma real.
   - Corrige el nombre de la clave `masa_omitida_matriz`.
3. `app.py`
   - Añade confirmación explícita para recordar que las cuotas iniciales son ejemplos y deben reemplazarse por cuotas reales actualizadas antes de calcular/guardar el análisis.

## Validaciones ejecutadas

- `py_compile` sobre todos los módulos Python.
- Prueba sintética del modelo: probabilidades 1X2 suman 1, masa omitida válida y análisis EV+ ejecutable.

## Importante

Estos cambios no demuestran que las probabilidades estén calibradas ni que exista rentabilidad. La aplicación sigue usando los partidos que entregue la API actual; para validar precisión y rendimiento se necesitarán datos históricos suficientes y pruebas cronológicas fuera de muestra.

## Despliegue

Reemplaza los archivos actuales por los incluidos en este paquete y despliega. No sobrescribas `.streamlit/secrets.toml` ni compartas tokens de API o `DATABASE_URL`. Después prueba una liga y dos equipos; confirma que las cuotas sean las actuales antes de calcular el EV+.
