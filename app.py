"""Interfaz Streamlit para explorar datos y predicciones de fútbol.

Arranque: streamlit run app.py
"""

from __future__ import annotations

import os
import sqlite3
from typing import Any

import pandas as pd
import streamlit as st

from fase1_estadisticas import calcular_estadisticas, cargar_partidos_api
from fase2_poisson import calcular_partido_poisson
from fase3_valor_esperado import analizar_valor_esperado
from historial_predicciones import PrediccionGuardada, guardar_prediccion, leer_historial


LIGAS = {
    "La Liga (España)": "PD",
    "Premier League (Inglaterra)": "PL",
    "Bundesliga (Alemania)": "BL1",
    "Serie A (Italia)": "SA",
    "Ligue 1 (Francia)": "FL1",
    "Champions League": "CL",
}
NOMBRES_1X2 = {"1": "Victoria local", "X": "Empate", "2": "Victoria visitante"}


@st.cache_data(ttl=1800, show_spinner=False)
def cargar_datos_liga(
    codigo_liga: str,
    api_token: str,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Descarga y cachea partidos, estadísticas y metadatos durante 30 minutos."""
    partidos = cargar_partidos_api(codigo_liga, api_token)
    equipos, liga = calcular_estadisticas(partidos)
    return equipos, liga, partidos


def obtener_token_configurado() -> str:
    """Lee el token de Streamlit Secrets o del entorno, si está configurado."""
    try:
        token = st.secrets.get("FOOTBALL_DATA_API_TOKEN", "")
    except FileNotFoundError:
        token = ""
    return str(token or os.environ.get("FOOTBALL_DATA_API_TOKEN", "")).strip()


def fecha_bogota(valor: Any) -> str:
    """Formatea una fecha de API (normalmente UTC) en hora de Bogotá."""
    if valor is None or valor == "":
        return "No disponible"
    fecha = pd.to_datetime(valor, utc=True, errors="coerce")
    if pd.isna(fecha):
        return str(valor)
    return fecha.tz_convert("America/Bogota").strftime("%Y-%m-%d %H:%M %Z")


def construir_ultimos_resultados(
    partidos: pd.DataFrame,
    equipo: str,
    cantidad: int = 5,
) -> pd.DataFrame:
    """Prepara los últimos partidos del equipo con resultado desde su perspectiva."""
    filas: list[dict[str, Any]] = []
    del_equipo = partidos[
        (partidos["HomeTeam"] == equipo) | (partidos["AwayTeam"] == equipo)
    ].copy()
    if "MatchDate" in del_equipo:
        del_equipo = del_equipo.sort_values("MatchDate", ascending=False, na_position="last")

    for _, partido in del_equipo.head(cantidad).iterrows():
        es_local = partido["HomeTeam"] == equipo
        goles_equipo = int(partido["FTHG"] if es_local else partido["FTAG"])
        goles_rival = int(partido["FTAG"] if es_local else partido["FTHG"])
        resultado = "G" if goles_equipo > goles_rival else "E" if goles_equipo == goles_rival else "P"
        filas.append(
            {
                "Fecha (Bogotá)": fecha_bogota(partido.get("MatchDate")),
                "Condición": "Local" if es_local else "Visitante",
                "Rival": partido["AwayTeam"] if es_local else partido["HomeTeam"],
                "Marcador": f"{goles_equipo}-{goles_rival}",
                "Resultado": resultado,
            }
        )
    return pd.DataFrame(filas, columns=["Fecha (Bogotá)", "Condición", "Rival", "Marcador", "Resultado"])


def construir_tabla_rendimiento(equipos: pd.DataFrame, equipo: str) -> pd.DataFrame:
    """Devuelve el balance del equipo separado por partidos en casa y fuera."""
    metricas = equipos.loc[equipo]
    filas = []
    for condicion, sufijo in (("Local", "local"), ("Visitante", "visitante")):
        filas.append(
            {
                "Condición": condicion,
                "PJ": int(metricas[f"PJ_{sufijo}"]),
                "G-E-P": (
                    f"{int(metricas[f'PG_{sufijo}'])}-"
                    f"{int(metricas[f'PE_{sufijo}'])}-"
                    f"{int(metricas[f'PP_{sufijo}'])}"
                ),
                "Puntos": int(metricas[f"Puntos_{sufijo}"]),
                "GF": int(metricas[f"GF_total_{sufijo}"]),
                "GC": int(metricas[f"GC_total_{sufijo}"]),
                "GF/partido": float(metricas[f"GF_{sufijo}"]),
                "GC/partido": float(metricas[f"GC_{sufijo}"]),
            }
        )
    return pd.DataFrame(filas)


def mostrar_rendimiento_y_forma(
    equipos: pd.DataFrame,
    partidos: pd.DataFrame,
    equipo_local: str,
    equipo_visitante: str,
) -> None:
    """Muestra rendimiento por condición y los últimos cinco resultados."""
    st.markdown("#### Rendimiento detallado por equipo")
    col_local, col_visitante = st.columns(2)
    for columna, equipo in ((col_local, equipo_local), (col_visitante, equipo_visitante)):
        with columna:
            st.markdown(f"**{equipo}**")
            st.dataframe(
                construir_tabla_rendimiento(equipos, equipo),
                hide_index=True,
                use_container_width=True,
            )

    st.markdown("#### Últimos resultados")
    col_local, col_visitante = st.columns(2)
    for columna, equipo in ((col_local, equipo_local), (col_visitante, equipo_visitante)):
        with columna:
            st.markdown(f"**{equipo} · últimos 5**")
            resultados = construir_ultimos_resultados(partidos, equipo)
            if resultados.empty:
                st.caption("No hay resultados recientes disponibles.")
            else:
                st.dataframe(resultados, hide_index=True, use_container_width=True)


def mostrar_pronostico(prediccion: dict[str, Any], resumen_ev: dict[str, Any]) -> None:
    """Renderiza lambdas, 1X2, totales, ambos marcan y el veredicto +EV."""
    equipo_local = prediccion["equipo_local"]
    equipo_visitante = prediccion["equipo_visitante"]
    st.markdown("#### Fuerzas y goles esperados")
    fuerzas = prediccion["fuerzas"]
    metricas = [
        ("Ataque local", fuerzas["ataque_local"]),
        ("Defensa local", fuerzas["defensa_local"]),
        ("Ataque visitante", fuerzas["ataque_visitante"]),
        ("Defensa visitante", fuerzas["defensa_visitante"]),
        (f"λ {equipo_local}", prediccion["lambda_local"]),
        (f"λ {equipo_visitante}", prediccion["lambda_visitante"]),
    ]
    for fila in (metricas[:3], metricas[3:]):
        columnas = st.columns(len(fila))
        for columna, (etiqueta, valor) in zip(columnas, fila):
            columna.metric(etiqueta, f"{valor:.3f}")
    st.caption("Fuerza 1.00 = promedio de liga. Los goles esperados se calculan así:")
    desglose = prediccion["desglose_lambda"]
    st.code(
        f"λ local = {desglose['local']['media_liga_local']:.3f} × "
        f"{desglose['local']['fuerza_ataque_local']:.3f} × "
        f"{desglose['local']['fuerza_defensa_visitante']:.3f} = "
        f"{prediccion['lambda_local']:.3f}\n"
        f"λ visitante = {desglose['visitante']['media_liga_visitante']:.3f} × "
        f"{desglose['visitante']['fuerza_ataque_visitante']:.3f} × "
        f"{desglose['visitante']['fuerza_defensa_local']:.3f} = "
        f"{prediccion['lambda_visitante']:.3f}",
        language="text",
    )

    st.markdown("#### Probabilidades 1X2 y cuotas")
    filas_1x2 = []
    for mercado in ("1", "X", "2"):
        datos = resumen_ev["mercados"][mercado]
        filas_1x2.append(
            {
                "Mercado": f"{mercado} · {NOMBRES_1X2[mercado]}",
                "Prob. modelo": f"{prediccion['probabilidades_1x2'][mercado] * 100:.2f}%",
                "Cuota justa": f"{prediccion['cuotas_justas'][mercado]:.2f}",
                "Cuota casa": f"{datos['cuota_casa']:.2f}",
                "Prob. implícita": f"{datos['probabilidad_implicita'] * 100:.2f}%",
                "EV": f"{datos['ev_porcentaje']:+.2f}%",
            }
        )
    st.dataframe(pd.DataFrame(filas_1x2), hide_index=True, use_container_width=True)
    st.caption(
        f"La suma de goles esperados es λ total = {prediccion['lambda_total']:.3f}. "
        f"El 1X2 se calculó ampliando el soporte hasta 0–{prediccion['soporte_1x2_max_goles']} goles."
    )

    st.markdown("#### Mercados de goles")
    col_totales, col_ambos = st.columns([2, 1])
    with col_totales:
        filas_totales = [
            {
                "Línea": f"{linea} goles",
                "Más de": f"{valores['over'] * 100:.2f}%",
                "Menos de": f"{valores['under'] * 100:.2f}%",
            }
            for linea, valores in prediccion["totales_goles"].items()
        ]
        st.dataframe(pd.DataFrame(filas_totales), hide_index=True, use_container_width=True)
    with col_ambos:
        st.markdown("**Ambos equipos marcan**")
        st.metric("Sí", f"{prediccion['ambos_marcan']['si'] * 100:.2f}%")
        st.metric("No", f"{prediccion['ambos_marcan']['no'] * 100:.2f}%")

    st.markdown("#### Marcadores exactos más probables")
    scores = pd.DataFrame(prediccion["marcadores_mas_probables"])
    if not scores.empty:
        scores["probabilidad"] = scores["probabilidad"].map(lambda valor: f"{valor * 100:.2f}%")
        scores = scores.rename(columns={"marcador": "Marcador", "probabilidad": "Probabilidad"})
        st.dataframe(scores, hide_index=True, use_container_width=True)

    st.metric("Overround de la casa", f"{resumen_ev['overround_porcentaje']:.2f}%")
    mercados_con_valor = resumen_ev["mercados_con_valor_positivo"]
    if mercados_con_valor:
        mejor = max(mercados_con_valor, key=lambda mercado: resumen_ev["mercados"][mercado]["ev"])
        valor = resumen_ev["mercados"][mejor]["ev_porcentaje"]
        texto = f"**Mayor valor detectado: {mejor} · {NOMBRES_1X2[mejor]}** con EV de **+{valor:.2f}%**."
        otros = [mercado for mercado in mercados_con_valor if mercado != mejor]
        if otros:
            texto += " Otros mercados +EV: " + ", ".join(
                f"{m} (+{resumen_ev['mercados'][m]['ev_porcentaje']:.2f}%)" for m in otros
            ) + "."
        st.success(texto)
    else:
        st.info("No se detectó ningún mercado con valor esperado positivo (+EV) a estas cuotas.")


def mostrar_historial() -> None:
    """Muestra los últimos pronósticos guardados en la base de datos local."""
    st.divider()
    st.subheader("Historial de pronósticos")
    try:
        historial = leer_historial(limite=500)
    except (OSError, sqlite3.Error, ValueError) as exc:
        st.error(f"No se pudo leer el historial: {exc}")
        return

    if historial.empty:
        st.caption("Todavía no hay pronósticos guardados. Calcula un análisis para iniciar el historial.")
        return

    fechas = pd.to_datetime(historial["fecha_utc"], utc=True, errors="coerce")
    vista = pd.DataFrame(
        {
            "Fecha (Bogotá)": fechas.dt.tz_convert("America/Bogota").dt.strftime("%Y-%m-%d %H:%M"),
            "Liga": historial["nombre_liga"],
            "Partido": historial["equipo_local"] + " vs " + historial["equipo_visitante"],
            "Cuotas casa (1/X/2)": historial.apply(
                lambda fila: f"{fila['cuota_1']:.2f} / {fila['cuota_x']:.2f} / {fila['cuota_2']:.2f}",
                axis=1,
            ),
            "Prob. modelo (1/X/2)": historial.apply(
                lambda fila: (
                    f"{fila['probabilidad_1'] * 100:.1f}% / "
                    f"{fila['probabilidad_x'] * 100:.1f}% / "
                    f"{fila['probabilidad_2'] * 100:.1f}%"
                ),
                axis=1,
            ),
            "EV (1/X/2)": historial.apply(
                lambda fila: (
                    f"{fila['ev_1_porcentaje']:+.1f}% / "
                    f"{fila['ev_x_porcentaje']:+.1f}% / "
                    f"{fila['ev_2_porcentaje']:+.1f}%"
                ),
                axis=1,
            ),
            "λ local": historial["lambda_local"].map(lambda valor: f"{valor:.2f}"),
            "λ visitante": historial["lambda_visitante"].map(lambda valor: f"{valor:.2f}"),
        }
    )
    st.dataframe(vista, hide_index=True, use_container_width=True)
    st.download_button(
        "Descargar registros cargados (CSV)",
        data=historial.to_csv(index=False).encode("utf-8-sig"),
        file_name="historial_predicciones.csv",
        mime="text/csv",
    )
    st.caption(
        f"Mostrando {len(vista)} registros recientes (máximo 500). El almacenamiento actual es SQLite local."
    )


def main() -> None:
    """Construye la aplicación de selección, consulta y análisis de partidos."""
    st.set_page_config(page_title="Análisis de fútbol +EV", page_icon="⚽", layout="wide")
    st.title("⚽ Análisis de fútbol y valor esperado")
    st.caption("Resultados verificables, rendimiento por condición y probabilidades Poisson.")

    with st.sidebar:
        st.header("Datos de la liga")
        api_token = obtener_token_configurado()
        if api_token:
            st.caption("API Token cargado desde Secrets o configuración del entorno.")
        else:
            api_token = st.text_input(
                "API Token de football-data.org",
                type="password",
                help="Para el despliegue, configura FOOTBALL_DATA_API_TOKEN en Secrets.",
            ).strip()
        nombre_liga = st.selectbox("Liga", list(LIGAS.keys()))
        codigo_liga = LIGAS[nombre_liga]
        st.caption(f"Código: `{codigo_liga}`")

    if not api_token:
        st.info("Configura FOOTBALL_DATA_API_TOKEN en Secrets o escribe el token para esta sesión.")
        st.stop()

    try:
        with st.spinner(f"Cargando partidos finalizados de {nombre_liga}…"):
            equipos, liga, partidos = cargar_datos_liga(codigo_liga, api_token)
    except (RuntimeError, ValueError, OSError) as exc:
        st.error(f"No se pudieron cargar los datos: {exc}")
        st.stop()

    st.markdown("#### Fuente y fecha de los datos")
    fecha_cols = st.columns(4)
    fecha_cols[0].metric("Partidos analizados", f"{liga['partidos_analizados']:,}")
    fecha_cols[1].metric("Consultado (Bogotá)", fecha_bogota(liga.get("consultado_en_utc")))
    fecha_cols[2].metric("Partido más reciente", fecha_bogota(liga.get("ultimo_partido_utc")))
    fecha_cols[3].metric(
        "Última actualización API",
        fecha_bogota(liga.get("ultima_actualizacion_api_utc")),
    )
    if liga.get("fecha_respuesta_api"):
        st.caption(f"Cabecera Date de la API: {liga['fecha_respuesta_api']} · caché de consulta: 30 minutos.")

    columnas_necesarias = ["GF_local", "GC_local", "GF_visitante", "GC_visitante"]
    disponibles = sorted(equipos.dropna(subset=columnas_necesarias).index.astype(str).tolist())
    if len(disponibles) < 2:
        st.error("No hay al menos dos equipos con promedios completos en ambas condiciones.")
        st.stop()

    st.subheader("Partido y cuotas")
    with st.form("formulario_analisis"):
        col_local, col_visitante = st.columns(2)
        with col_local:
            equipo_local = st.selectbox("Equipo local", disponibles, key="equipo_local")
        with col_visitante:
            opciones_visitante = [equipo for equipo in disponibles if equipo != equipo_local]
            equipo_visitante = st.selectbox(
                "Equipo visitante", opciones_visitante, key="equipo_visitante"
            )
        st.markdown("**Cuotas decimales de la casa (1X2)**")
        col_1, col_x, col_2 = st.columns(3)
        with col_1:
            cuota_1 = st.number_input("Cuota 1 · Local", min_value=1.01, value=2.10, step=0.01)
        with col_x:
            cuota_x = st.number_input("Cuota X · Empate", min_value=1.01, value=3.40, step=0.01)
        with col_2:
            cuota_2 = st.number_input("Cuota 2 · Visitante", min_value=1.01, value=3.60, step=0.01)
        calcular = st.form_submit_button(
            "Calcular análisis", type="primary", use_container_width=True
        )

    if calcular:
        try:
            prediccion = calcular_partido_poisson(equipos, liga, equipo_local, equipo_visitante)
            resumen_ev = analizar_valor_esperado(
                prediccion["probabilidades_1x2"],
                {"1": cuota_1, "X": cuota_x, "2": cuota_2},
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            st.error(f"No se pudo calcular el análisis: {exc}")
        else:
            mercados = resumen_ev["mercados"]
            registro = PrediccionGuardada(
                codigo_liga=codigo_liga,
                nombre_liga=nombre_liga,
                equipo_local=equipo_local,
                equipo_visitante=equipo_visitante,
                cuota_1=cuota_1,
                cuota_x=cuota_x,
                cuota_2=cuota_2,
                probabilidad_1=prediccion["probabilidades_1x2"]["1"],
                probabilidad_x=prediccion["probabilidades_1x2"]["X"],
                probabilidad_2=prediccion["probabilidades_1x2"]["2"],
                ev_1=mercados["1"]["ev"],
                ev_x=mercados["X"]["ev"],
                ev_2=mercados["2"]["ev"],
                ev_1_porcentaje=mercados["1"]["ev_porcentaje"],
                ev_x_porcentaje=mercados["X"]["ev_porcentaje"],
                ev_2_porcentaje=mercados["2"]["ev_porcentaje"],
                lambda_local=prediccion["lambda_local"],
                lambda_visitante=prediccion["lambda_visitante"],
            )
            try:
                registro_id = guardar_prediccion(registro)
            except (OSError, sqlite3.Error) as exc:
                st.warning(f"El análisis se calculó, pero no se pudo guardar en el historial: {exc}")
            else:
                st.success(f"Pronóstico guardado en el historial (registro #{registro_id}).")

            st.divider()
            st.subheader(f"{equipo_local} vs {equipo_visitante}")
            mostrar_rendimiento_y_forma(equipos, partidos, equipo_local, equipo_visitante)
            mostrar_pronostico(prediccion, resumen_ev)

    mostrar_historial()


if __name__ == "__main__":
    main()
