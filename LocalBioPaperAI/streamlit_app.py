import hashlib
import os
import tempfile
import pandas as pd
from LocalBioPaperAI.src.extractor import (
    extract_scientific_data,
)
import streamlit as st

from LocalBioPaperAI.src.pdf_reader import read_pdf
from LocalBioPaperAI.src.text_splitter import split_text
from LocalBioPaperAI.src.embeddings import (
    load_embedding_model,
    create_embeddings,
)
from LocalBioPaperAI.src.retriever import find_best_chunks
from LocalBioPaperAI.src.generator import (
    generate_answer,
    summarize_section,
    generate_structured_summary,
    generate_analysis_report,
    generate_focus_analysis,
)
from LocalBioPaperAI.src.paper_parser import (
    split_into_sections,
    get_summary_sections,
)


st.set_page_config(
    page_title="PaperGeneAI",
    page_icon="🧬",
    layout="wide",
)


@st.cache_resource
def get_embedding_model():
    """
    Lädt das Embedding-Modell nur einmal
    und hält es im Arbeitsspeicher.
    """
    return load_embedding_model()


@st.cache_data(show_spinner=False)
def prepare_pdf(file_bytes: bytes):
    """
    Speichert eine hochgeladene PDF kurzzeitig,
    extrahiert den Text und erstellt Textabschnitte.
    """
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_file.write(file_bytes)
            temporary_path = temporary_file.name

        text = read_pdf(temporary_path)

        if not text.strip():
            raise ValueError(
                "Aus dieser PDF konnte kein Text extrahiert werden."
            )

        chunks = split_text(text)

        if not chunks:
            raise ValueError(
                "Der extrahierte Text konnte nicht "
                "in Abschnitte aufgeteilt werden."
            )

        return text, chunks

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


@st.cache_data(show_spinner=False)
def create_cached_embeddings(
    file_hash: str,
    chunks: list[str],
):
    """
    Erstellt die Embeddings nur einmal pro PDF.
    file_hash sorgt dafür, dass verschiedene PDFs
    getrennt gespeichert werden.
    """
    del file_hash

    model = get_embedding_model()

    return create_embeddings(
        chunks,
        model,
    )


def build_source_context(best_chunks: list[dict]) -> str:
    """
    Verbindet die gefundenen Textstellen zu einem
    strukturierten Kontext für das Sprachmodell.
    """
    context_parts = []

    for source_number, item in enumerate(
        best_chunks,
        start=1,
    ):
        context_parts.append(
            f"[Quelle {source_number}]\n"
            f"{item['chunk']}"
        )

    return "\n\n".join(context_parts)


st.title("🧬 PaperGeneAI")

st.write(
    "Lokale Analyse wissenschaftlicher PDFs mit "
    "semantischer Suche und Llama 3.2."
)

st.caption(
    "Die PDF-Verarbeitung, Suche und Antwortgenerierung "
    "erfolgen lokal auf deinem Mac."
)


uploaded_file = st.file_uploader(
    "PDF-Paper hochladen",
    type=["pdf"],
)


if uploaded_file is None:
    st.info(
        "Lade zuerst ein wissenschaftliches Paper "
        "als PDF hoch."
    )
    st.stop()


pdf_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(pdf_bytes).hexdigest()


try:
    with st.spinner("PDF wird verarbeitet ..."):
        full_text, chunks = prepare_pdf(pdf_bytes)

        embedding_model = get_embedding_model()

        chunk_embeddings = create_cached_embeddings(
            file_hash,
            chunks,
        )

except Exception as error:
    st.error("Die PDF konnte nicht verarbeitet werden.")

    with st.expander("Technische Fehlermeldung"):
        st.code(str(error))

    st.stop()


st.success(
    f"PDF erfolgreich verarbeitet: "
    f"{len(chunks)} Textabschnitte."
)


tab_question, tab_summary, tab_extraction, tab_text = st.tabs(
    [
        "💬 Fragen",
        "📄 Zusammenfassung",
        "🧬 Datenextraktion",
        "📝 Extrahierter Text",
    ]
)


# ---------------------------------------------------------
# TAB 1: FRAGEN ZUM PAPER
# ---------------------------------------------------------

with tab_question:
    st.subheader("Fragen zum Paper")

    question = st.text_input(
        "Stelle eine Frage zum Paper",
        placeholder=(
            "Zum Beispiel: "
            "What were the main findings?"
        ),
    )

    top_k = st.slider(
        "Anzahl verwendeter Paper-Abschnitte",
        min_value=1,
        max_value=8,
        value=4,
        help=(
            "Mehr Abschnitte liefern mehr Kontext, "
            "können die Antwort aber langsamer machen."
        ),
    )

    ask_button = st.button(
        "Frage analysieren",
        type="primary",
        key="ask_question",
    )

    if ask_button:
        if not question.strip():
            st.warning("Bitte gib zuerst eine Frage ein.")
            st.stop()

        with st.spinner(
            "Relevante Textstellen werden gesucht ..."
        ):
            best_chunks = find_best_chunks(
                question=question,
                chunks=chunks,
                chunk_embeddings=chunk_embeddings,
                model=embedding_model,
                top_k=top_k,
            )

        if not best_chunks:
            st.warning(
                "Es wurden keine passenden Textstellen gefunden."
            )
            st.stop()

        context = build_source_context(best_chunks)

        try:
            with st.spinner(
                "Die lokale KI formuliert die Antwort ..."
            ):
                answer = generate_answer(
                    question,
                    context,
                )

        except Exception as error:
            st.error(
                "Ollama konnte nicht erreicht werden. "
                "Starte Ollama im Terminal mit:\n\n"
                "`brew services start ollama`"
            )

            with st.expander("Technische Fehlermeldung"):
                st.code(str(error))

            st.stop()

        st.subheader("Antwort der KI")
        st.markdown(answer)

        export_text = (
            f"Datei:\n{uploaded_file.name}\n\n"
            f"Frage:\n{question}\n\n"
            f"Antwort:\n{answer}\n\n"
            f"Verwendete Quellen:\n{context}"
        )

        st.download_button(
            label="Antwort herunterladen",
            data=export_text,
            file_name="papergeneai_antwort.txt",
            mime="text/plain",
        )

        st.subheader("Verwendete Paper-Abschnitte")

        for source_number, item in enumerate(
            best_chunks,
            start=1,
        ):
            score = float(item["score"])

            with st.expander(
                f"Quelle {source_number} – "
                f"Ähnlichkeit: {score:.3f}"
            ):
                st.write(item["chunk"])


# ---------------------------------------------------------
# TAB 2: VERBESSERTE ZUSAMMENFASSUNG
# ---------------------------------------------------------

with tab_summary:
    st.subheader("Wissenschaftliche Paper-Analyse")

    st.write(
        "Das Paper wird kapitelweise verarbeitet. "
        "Referenzen werden von der Analyse ausgeschlossen."
    )

    detected_sections = split_into_sections(full_text)

    summary_sections = get_summary_sections(
        detected_sections
    )

    with st.expander(
        "Erkannte Paper-Kapitel",
        expanded=True,
    ):
        if summary_sections:
            for section_name, section_text in (
                summary_sections.items()
            ):
                st.write(
                    f"**{section_name}:** "
                    f"{len(section_text):,} Zeichen"
                )
        else:
            st.warning(
                "Es konnten keine typischen Kapitel "
                "erkannt werden."
            )

    if "section_summaries" not in st.session_state:
        st.session_state.section_summaries = None

    if "structured_summary" not in st.session_state:
        st.session_state.structured_summary = None

    if "analysis_report" not in st.session_state:
        st.session_state.analysis_report = None

    prepare_button = st.button(
        "Paper vollständig analysieren",
        type="primary",
        key="prepare_paper_analysis",
    )

    if prepare_button:
        if not summary_sections:
            st.error(
                "Es stehen keine geeigneten Kapitel "
                "für die Analyse zur Verfügung."
            )
            st.stop()

        section_summaries = {}

        progress_bar = st.progress(0)
        status_text = st.empty()

        total_sections = len(summary_sections)

        try:
            for index, (
                section_name,
                section_text,
            ) in enumerate(
                summary_sections.items(),
                start=1,
            ):
                status_text.markdown(
                    f"Analysiere Kapitel: "
                    f"**{section_name}**"
                )

                section_summaries[section_name] = (
                    summarize_section(
                        section_name,
                        section_text,
                    )
                )

                progress_bar.progress(
                    index / total_sections
                )

            status_text.markdown(
                "Erstelle strukturierte Zusammenfassung ..."
            )

            structured_summary = (
                generate_structured_summary(
                    section_summaries
                )
            )

            status_text.markdown(
                "Erstelle wissenschaftlichen Analysebericht ..."
            )

            analysis_report = generate_analysis_report(
                section_summaries
            )

            st.session_state.section_summaries = (
                section_summaries
            )

            st.session_state.structured_summary = (
                structured_summary
            )

            st.session_state.analysis_report = (
                analysis_report
            )

        except Exception as error:
            st.error(
                "Die Analyse konnte nicht abgeschlossen werden. "
                "Prüfe, ob Ollama läuft."
            )

            with st.expander("Technische Fehlermeldung"):
                st.code(str(error))

            st.stop()

        finally:
            progress_bar.empty()
            status_text.empty()

    section_summaries = (
        st.session_state.section_summaries
    )

    structured_summary = (
        st.session_state.structured_summary
    )

    analysis_report = (
        st.session_state.analysis_report
    )

    if section_summaries:
        summary_tab, report_tab, focus_tab = st.tabs(
            [
                "📄 Kurzfassung",
                "🔬 Forschungsbericht",
                "🎯 Spezialanalysen",
            ]
        )

        with summary_tab:
            st.subheader(
                "Strukturierte Zusammenfassung"
            )

            st.markdown(structured_summary)

            with st.expander(
                "Einzelne Kapitelzusammenfassungen"
            ):
                for (
                    section_name,
                    section_summary,
                ) in section_summaries.items():
                    st.markdown(
                        f"### {section_name}"
                    )
                    st.markdown(section_summary)

            st.download_button(
                label="Kurzfassung herunterladen",
                data=structured_summary,
                file_name=(
                    "papergeneai_kurzfassung.txt"
                ),
                mime="text/plain",
            )

        with report_tab:
            st.subheader(
                "Wissenschaftlicher Forschungsbericht"
            )

            st.markdown(analysis_report)

            st.download_button(
                label="Forschungsbericht herunterladen",
                data=analysis_report,
                file_name=(
                    "papergeneai_forschungsbericht.txt"
                ),
                mime="text/plain",
            )

        with focus_tab:
            st.write(
                "Wähle einen bestimmten Schwerpunkt "
                "für eine zusätzliche Analyse."
            )

            focus_options = [
                "Studiendesign",
                "Experimentelle Methoden",
                "Wichtigste Ergebnisse",
                "Biologische Mechanismen",
                "Gene und Proteine",
                "Signalwege",
                "Zelltypen und Zellmarker",
                "Stammzelldifferenzierung",
                "Klinische Bedeutung",
                "Limitationen",
                "Zukünftige Forschung",
            ]

            selected_focus = st.selectbox(
                "Analyseschwerpunkt",
                options=focus_options,
            )

            focus_button = st.button(
                "Spezialanalyse erzeugen",
                key="generate_focus_analysis",
            )

            if focus_button:
                try:
                    with st.spinner(
                        f"Analysiere Schwerpunkt: "
                        f"{selected_focus} ..."
                    ):
                        focus_result = (
                            generate_focus_analysis(
                                selected_focus,
                                section_summaries,
                            )
                        )

                except Exception as error:
                    st.error(
                        "Die Spezialanalyse konnte nicht "
                        "erstellt werden."
                    )

                    with st.expander(
                        "Technische Fehlermeldung"
                    ):
                        st.code(str(error))

                    st.stop()

                st.subheader(selected_focus)
                st.markdown(focus_result)

                st.download_button(
                    label="Spezialanalyse herunterladen",
                    data=focus_result,
                    file_name=(
                        "papergeneai_"
                        f"{selected_focus.lower().replace(' ', '_')}"
                        ".txt"
                    ),
                    mime="text/plain",
                )

    else:
        st.info(
            "Klicke auf „Paper vollständig analysieren“, "
            "um die Zusammenfassungen und Forschungsberichte "
            "zu erzeugen."
        )
# ---------------------------------------------------------
# TAB 3: EXTRAHIERTER TEXT
# ---------------------------------------------------------
with tab_extraction:
    st.subheader("Strukturierte wissenschaftliche Daten")

    st.write(
        "Extrahiert Gene, Proteine, Zelltypen, Methoden, "
        "Signalwege, Ergebnisse und Differenzierungsprotokolle."
    )

    if "scientific_extraction" not in st.session_state:
        st.session_state.scientific_extraction = None

    section_summaries = st.session_state.get(
        "section_summaries"
    )

    if not section_summaries:
        st.info(
            "Führe zuerst im Tab „Zusammenfassung“ die "
            "vollständige Paper-Analyse durch."
        )

    else:
        extraction_button = st.button(
            "Wissenschaftliche Daten extrahieren",
            type="primary",
            key="extract_scientific_data_button",
        )

        if extraction_button:
            try:
                with st.spinner(
                    "Wissenschaftliche Daten werden extrahiert ..."
                ):
                    extraction = extract_scientific_data(
                        section_summaries
                    )

                    st.session_state.scientific_extraction = (
                        extraction
                    )

            except Exception as error:
                st.error(
                    "Die wissenschaftlichen Daten konnten "
                    "nicht extrahiert werden."
                )

                with st.expander("Technische Fehlermeldung"):
                    st.code(str(error))

                st.stop()

        extraction = st.session_state.scientific_extraction

        if extraction:
            display_sections = [
                (
                    "Gene und Proteine",
                    "genes_and_proteins",
                ),
                (
                    "Zelltypen",
                    "cell_types",
                ),
                (
                    "Wachstumsfaktoren und Wirkstoffe",
                    "growth_factors_and_compounds",
                ),
                (
                    "Signalwege",
                    "pathways",
                ),
                (
                    "Methoden",
                    "methods",
                ),
                (
                    "Differenzierungsprotokoll",
                    "differentiation_protocol",
                ),
                (
                    "Wichtigste Ergebnisse",
                    "key_results",
                ),
            ]

            for title, extraction_key in display_sections:
                st.markdown(f"### {title}")

                rows = extraction.get(extraction_key, [])

                if rows:
                    dataframe = pd.DataFrame(rows)

                    st.dataframe(
                        dataframe,
                        use_container_width=True,
                        hide_index=True,
                    )

                    csv_data = dataframe.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label=f"{title} als CSV herunterladen",
                        data=csv_data,
                        file_name=(
                            extraction_key + ".csv"
                        ),
                        mime="text/csv",
                        key=(
                            "download_"
                            + extraction_key
                        ),
                    )

                else:
                    st.info(
                        "Keine eindeutig belegten Angaben gefunden."
                    )
with tab_text:
    st.subheader("Extrahierter Paper-Text")

    st.write(
        "Hier siehst du den Text, den PaperGeneAI "
        "aus der PDF extrahiert hat."
    )

    st.text_area(
        "Extrahierter Paper-Text",
        value=full_text,
        height=500,
    )

    st.download_button(
        label="Extrahierten Text herunterladen",
        data=full_text,
        file_name="paper_text.txt",
        mime="text/plain",
    )