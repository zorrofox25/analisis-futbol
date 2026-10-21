"""Interfaz web local para el MVP de análisis de fútbol.

Arranque desde la carpeta del proyecto:
    streamlit run app.py
"""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from fase1_estadisticas import calcular_estadisticas, cargar_partidos_api
from fase2_poisson import calcular_partido_poisson
from fase3_valor_esperado import analizar_valor_esperado


LIGAS = {
    "La Liga (España)": "PD",
    "Premier League (Inglaterra)": "PL",
    "Bundesliga (Alemania)": "BL1",
    "Serie A (Italia)": "SA",
    "Ligue 1 (Francia)": "FL1",
    "Champions League": "CL",
}


@st.cache_data(ttl=1800, show_spinner=False)
def cargar_datos_liga(codigo_liga: str, api_token: str) -> tuple[pd.DataFrame, dict[str, float]]:
    """Descarga los partidos y calcula métricas; cachea el resultado 30 minutos."""
    partidos = cargar_partidos_api(codigo_liga, api_token)
    return calcular_estadisticas(partidos)


def obtener_token_configurado() -> str:
    """Lee el token de Streamlit Secrets o del entorno, si está configurado."""
    try:
        token = st.secrets.get("FOOTBALL_DATA_API_TOKEN", "")
    except FileNotFoundError:
        token = ""
    return str(token or os.environ.get("FOOTBALL_DATA_API_TOKEN", "")).strip()


def main() -> None:
    st.set_page_config(page_title="Análisis de fútbol +EV", page_icon="⚽", layout="wide")
    st.title("⚽ Análisis de fútbol y valor esperado")
    st.caption("Estadísticas históricas, distribución de Poisson y detector de valor 1X2.")

    with st.sidebar:
        st.header("Datos de la liga")
        api_token = obtener_token_configurado()
        if api_token:
            st.caption("API Token cargado desde Secrets o configuración del entorno.")
        else:
            api_token = st.text_input(
                "API Token de football-data.org",
                type="password",
                help="Se usa solo para consultar la API y no se muestra en pantalla.",
            ).strip()
        nombre_liga = st.selectbox("Liga", list(LIGAS.keys()))
        codigo_liga = LIGAS[nombre_liga]
        st.caption(f"Código de competición: `{codigo_liga}`")

    if not api_token:
        st.info("Introduce tu API Token en el menú lateral para cargar los equipos.")
        st.stop()

    try:
        with st.spinner(f"Cargando partidos finalizados de {nombre_liga}…"):
            equipos, liga = cargar_datos_liga(codigo_liga, api_token)
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(f"No se pudieron cargar los datos: {exc}")
        st.stop()

    # El modelo necesita promedios de cada equipo tanto como local como visitante.
    columnas_necesarias = ["GF_local", "GC_local", "GF_visitante", "GC_visitante"]
    equipos_disponibles = sorted(
        equipos.dropna(subset=columnas_necesarias).index.astype(str).tolist()
    )
    if len(equipos_disponibles) < 2:
        st.error("La liga no tiene al menos dos equipos con promedios completos de local y visitante.")
        st.stop()

    st.subheader("Partido y cuotas")
    with st.form("formulario_analisis"):
        col_local, col_visitante = st.columns(2)
        with col_local:
            equipo_local = st.selectbox("Equipo local", equipos_disponibles, key="equipo_local")
        with col_visitante:
            opciones_visitante = [equipo for equipo in equipos_disponibles if equipo != equipo_local]
            equipo_visitante = st.selectbox(
                "Equipo visitante", opciones_visitante, key="equipo_visitante"
            )

        st.markdown("**Cuotas decimales de la casa**")
        col_1, col_x, col_2 = st.columns(3)
        with col_1:
            cuota_1 = st.number_input("Cuota 1 · Local", min_value=1.01, value=2.10, step=0.01)
        with col_x:
            cuota_x = st.number_input("Cuota X · Empate", min_value=1.01, value=3.40, step=0.01)
        with col_2:
            cuota_2 = st.number_input("Cuota 2 · Visitante", min_value=1.01, value=3.60, step=0.01)

        calcular = st.form_submit_button("Calcular análisis", type="primary", use_container_width=True)

    if not calcular:
        st.caption(f"Datos cargados: {len(equipos)} equipos · medias de liga calculadas desde partidos finalizados.")
        return

    try:
        prediccion = calcular_partido_poisson(
            equipos,
            liga,
            equipo_local,
            equipo_visitante,
        )
        resumen_ev = analizar_valor_esperado(
            prediccion["probabilidades_1x2"],
            {"1": cuota_1, "X": cuota_x, "2": cuota_2},
        )
    except (RuntimeError, ValueError, KeyError) as exc:
        st.error(f"No se pudo calcular el análisis: {exc}")
        return

    st.divider()
    st.subheader(f"{equipo_local} vs {equipo_visitante}")

    st.markdown("#### Fuerzas y goles esperados")
    fuerzas = prediccion["fuerzas"]
    metrica_cols = st.columns(6)
    metricas = [
        ("Ataque local", fuerzas["ataque_local"]),
        ("Defensa local", fuerzas["defensa_local"]),
        ("Ataque visitante", fuerzas["ataque_visitante"]),
        ("Defensa visitante", fuerzas["defensa_visitante"]),
        ("λ local", prediccion["lambda_local"]),
        ("λ visitante", prediccion["lambda_visitante"]),
    ]
    for columna, (etiqueta, valor) in zip(metrica_cols, metricas):
        columna.metric(etiqueta, f"{valor:.3f}")
    st.caption("Las fuerzas de 1.00 equivalen al promedio de liga; λ representa goles esperados.")

    st.markdown("#### Probabilidades, cuotas y valor esperado")
    nombres = {"1": "Victoria local", "X": "Empate", "2": "Victoria visitante"}
    filas = []
    for mercado in ("1", "X", "2"):
        datos = resumen_ev["mercados"][mercado]
        filas.append(
            {
                "Mercado": f"{mercado} · {nombres[mercado]}",
                "Prob. modelo": f"{prediccion['probabilidades_1x2'][mercado] * 100:.2f}%",
                "Cuota justa": f"{prediccion['cuotas_justas'][mercado]:.2f}",
                "Cuota casa": f"{datos['cuota_casa']:.2f}",
                "Prob. implícita": f"{datos['probabilidad_implicita'] * 100:.2f}%",
                "EV": f"{datos['ev_porcentaje']:+.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(filas), hide_index=True, use_container_width=True)
    st.metric("Overround de la casa", f"{resumen_ev['overround_porcentaje']:.2f}%")
    if prediccion["masa_omitida"] > 0.001:
        st.caption(
            f"Masa de probabilidad fuera de la matriz de marcadores: "
            f"{prediccion['masa_omitida'] * 100:.3f}%."
        )

    mercados_con_valor = resumen_ev["mercados_con_valor_positivo"]
    if mercados_con_valor:
        mejor = max(
            mercados_con_valor,
            key=lambda mercado: resumen_ev["mercados"][mercado]["ev"],
        )
        mejor_ev = resumen_ev["mercados"][mejor]["ev_porcentaje"]
        otros = [mercado for mercado in mercados_con_valor if mercado != mejor]
        texto = (
            f"**Mayor valor detectado: {mejor} · {nombres[mejor]}** "
            f"con EV de **+{mejor_ev:.2f}%**."
        )
        if otros:
            texto += " También hay EV positivo en: " + ", ".join(
                f"{mercado} (+{resumen_ev['mercados'][mercado]['ev_porcentaje']:.2f}%)"
                for mercado in otros
            ) + "."
        st.success(texto)
    else:
        st.info("No se detectó ningún mercado con valor esperado positivo (+EV) a estas cuotas.")


if __name__ == "__main__":
    main()
