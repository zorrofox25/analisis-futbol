"""Fase 3: detector de valor esperado (+EV) para mercados 1X2."""

from __future__ import annotations

import math
from typing import Any, Mapping


MERCADOS_1X2 = ("1", "X", "2")


def analizar_valor_esperado(
    probabilidades_1x2: Mapping[str, float],
    cuotas_casa: Mapping[str, float],
) -> dict[str, Any]:
    """Compara probabilidades del modelo con cuotas decimales de la casa.

    EV se expresa como retorno neto por unidad apostada:
    ``EV = (probabilidad_modelo * cuota_decimal) - 1``.
    El campo ``ev_porcentaje`` convierte ese valor a porcentaje; por ejemplo,
    EV 0.05 equivale a +5% de valor esperado.

    El overround es la suma de probabilidades implícitas de la casa menos 1.
    Puede expresarse como porcentaje y no representa el margen real exacto
    sobre beneficios, ya que depende de cómo la casa distribuya su margen.
    """
    faltan_probabilidades = [
        mercado for mercado in MERCADOS_1X2 if mercado not in probabilidades_1x2
    ]
    faltan_cuotas = [mercado for mercado in MERCADOS_1X2 if mercado not in cuotas_casa]
    if faltan_probabilidades:
        raise ValueError(
            "Faltan probabilidades para los mercados: "
            + ", ".join(faltan_probabilidades)
        )
    if faltan_cuotas:
        raise ValueError("Faltan cuotas para los mercados: " + ", ".join(faltan_cuotas))

    resultados: dict[str, dict[str, float | bool]] = {}
    for mercado in MERCADOS_1X2:
        try:
            probabilidad = float(probabilidades_1x2[mercado])
            cuota = float(cuotas_casa[mercado])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Probabilidad o cuota no numérica para el mercado {mercado}.") from exc

        if not math.isfinite(probabilidad) or not 0.0 <= probabilidad <= 1.0:
            raise ValueError(
                f"La probabilidad del mercado {mercado} debe ser finita y estar entre 0 y 1."
            )
        if not math.isfinite(cuota) or cuota <= 1.0:
            raise ValueError(
                f"La cuota decimal del mercado {mercado} debe ser finita y mayor que 1."
            )

        implicita = 1.0 / cuota
        ev = (probabilidad * cuota) - 1.0
        resultados[mercado] = {
            "probabilidad_modelo": probabilidad,
            "cuota_casa": cuota,
            "probabilidad_implicita": implicita,
            "ev": ev,
            "ev_porcentaje": ev * 100.0,
            "tiene_valor_positivo": ev > 0.0,
        }

    suma_implicitas = sum(
        float(resultados[mercado]["probabilidad_implicita"])
        for mercado in MERCADOS_1X2
    )
    overround = suma_implicitas - 1.0
    mercados_valor = [
        mercado
        for mercado in MERCADOS_1X2
        if resultados[mercado]["tiene_valor_positivo"]
    ]

    return {
        "mercados": resultados,
        "mercados_con_valor_positivo": mercados_valor,
        "hay_valor_positivo": bool(mercados_valor),
        "probabilidad_implicita_total": suma_implicitas,
        "overround": overround,
        "overround_porcentaje": overround * 100.0,
    }
