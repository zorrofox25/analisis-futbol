"""Orquestador CLI del MVP de análisis y predicción 1X2.

Ejemplo:
    python main.py --csv partidos.csv --local "Real Madrid" \
        --visitante "Barcelona" --cuota-1 2.10 --cuota-x 3.40 --cuota-2 3.60
"""

from __future__ import annotations

import argparse
import math
import os
from typing import Any

from fase1_estadisticas import calcular_estadisticas, cargar_partidos_api
from fase2_poisson import calcular_partido_poisson
from fase3_valor_esperado import analizar_valor_esperado


ETIQUETAS_MERCADOS = {
    "1": "Victoria local",
    "X": "Empate",
    "2": "Victoria visitante",
}


def crear_parser() -> argparse.ArgumentParser:
    """Define y devuelve el parser de argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Analiza un partido de fútbol con Poisson y detecta valor +EV."
    )
    parser.add_argument("--liga", required=True, help="Código de liga de football-data.org (p. ej. PD o PL)")
    parser.add_argument(
        "--token",
        help=(
            "API token (alternativa; preferible usar la variable "
            "FOOTBALL_DATA_API_TOKEN para no dejarlo en el historial del shell)"
        ),
    )
    parser.add_argument("--local", required=True, help="Nombre exacto del equipo local")
    parser.add_argument("--visitante", required=True, help="Nombre exacto del equipo visitante")
    parser.add_argument("--cuota-1", required=True, type=float, help="Cuota decimal para victoria local")
    parser.add_argument("--cuota-x", required=True, type=float, help="Cuota decimal para empate")
    parser.add_argument("--cuota-2", required=True, type=float, help="Cuota decimal para victoria visitante")
    parser.add_argument(
        "--max-goles",
        type=int,
        default=10,
        help="Máximo de goles por equipo en la matriz Poisson (por defecto: 10)",
    )
    return parser


def _porcentaje(valor: float) -> str:
    """Formatea un valor en escala 0-1 como porcentaje."""
    return f"{valor * 100:.2f}%"


def imprimir_reporte(
    prediccion: dict[str, Any],
    analisis_ev: dict[str, Any],
) -> None:
    """Imprime en terminal un resumen del modelo y del análisis de cuotas."""
    fuerzas = prediccion["fuerzas"]
    print("\n" + "=" * 72)
    print(f"ANÁLISIS 1X2: {prediccion['equipo_local']} vs {prediccion['equipo_visitante']}")
    print("=" * 72)

    print("\nFuerzas (1.00 = promedio de liga):")
    print(f"  Ataque local:       {fuerzas['ataque_local']:.3f}")
    print(f"  Defensa local:      {fuerzas['defensa_local']:.3f}")
    print(f"  Ataque visitante:   {fuerzas['ataque_visitante']:.3f}")
    print(f"  Defensa visitante:  {fuerzas['defensa_visitante']:.3f}")
    print("\nGoles esperados (lambda):")
    print(f"  Local: {prediccion['lambda_local']:.3f}")
    print(f"  Visitante: {prediccion['lambda_visitante']:.3f}")

    print("\nProbabilidades y cuotas:")
    print(f"  {'Mercado':<21} {'Modelo':>10} {'Cuota justa':>13} {'Cuota casa':>12} {'Implícita':>11} {'EV':>10}")
    print("  " + "-" * 81)
    probabilidades = prediccion["probabilidades_1x2"]
    cuotas_justas = prediccion["cuotas_justas"]
    for mercado in ("1", "X", "2"):
        datos = analisis_ev["mercados"][mercado]
        print(
            f"  {mercado} - {ETIQUETAS_MERCADOS[mercado]:<17} "
            f"{_porcentaje(probabilidades[mercado]):>10} "
            f"{cuotas_justas[mercado]:>13.2f} "
            f"{datos['cuota_casa']:>12.2f} "
            f"{_porcentaje(datos['probabilidad_implicita']):>11} "
            f"{datos['ev_porcentaje']:>+9.2f}%"
        )

    print(f"\nOverround de la casa: {analisis_ev['overround_porcentaje']:+.2f}%")
    print(f"Masa de probabilidad fuera de la matriz: {_porcentaje(prediccion['masa_omitida'])}")

    mercados_valor = analisis_ev["mercados_con_valor_positivo"]
    if not mercados_valor:
        print("\nRecomendación: no se detecta apuesta con valor esperado positivo (+EV).")
        return

    mejor = max(
        mercados_valor,
        key=lambda mercado: analisis_ev["mercados"][mercado]["ev"],
    )
    mejor_dato = analisis_ev["mercados"][mejor]
    print(
        "\nRecomendación: existe valor positivo en "
        f"{ETIQUETAS_MERCADOS[mejor]} ({mejor})."
    )
    print(f"  Mayor EV: +{mejor_dato['ev_porcentaje']:.2f}% a cuota {mejor_dato['cuota_casa']:.2f}.")
    if len(mercados_valor) > 1:
        otros = [
            f"{mercado} (+{analisis_ev['mercados'][mercado]['ev_porcentaje']:.2f}%)"
            for mercado in mercados_valor
            if mercado != mejor
        ]
        print("  Otros mercados +EV: " + ", ".join(otros))


def main() -> None:
    """Ejecuta carga, estadísticas, Poisson, análisis +EV y reporte."""
    parser = crear_parser()
    args = parser.parse_args()

    # La variable de entorno es preferible: evita incluir el secreto en el comando.
    api_token = args.token or os.environ.get("FOOTBALL_DATA_API_TOKEN")
    if not api_token:
        parser.error(
            "Falta el token: define FOOTBALL_DATA_API_TOKEN o pásalo con --token."
        )

    cuotas_casa = {"1": args.cuota_1, "X": args.cuota_x, "2": args.cuota_2}
    if args.max_goles < 1:
        parser.error("--max-goles debe ser un entero mayor o igual a 1.")
    if any(not math.isfinite(cuota) or cuota <= 1 for cuota in cuotas_casa.values()):
        parser.error("Todas las cuotas decimales deben ser finitas y mayores que 1.")

    try:
        partidos = cargar_partidos_api(args.liga, api_token)
        equipos, liga = calcular_estadisticas(partidos)
        prediccion = calcular_partido_poisson(
            equipos,
            liga,
            args.local,
            args.visitante,
            max_goles=args.max_goles,
        )
        analisis_ev = analizar_valor_esperado(
            prediccion["probabilidades_1x2"],
            cuotas_casa,
        )
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Error: {exc}\n")

    imprimir_reporte(prediccion, analisis_ev)


if __name__ == "__main__":
    main()
