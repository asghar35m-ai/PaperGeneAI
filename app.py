import hashlib
import os
import tempfile

import streamlit as st

from LocalBioPaperAI.src.pdf_reader import read_pdf
from LocalBioPaperAI.src.text_splitter import split_text
from LocalBioPaperAI.src.embeddings import (
    load_embedding_model,
    create_embeddings,
)
from LocalBioPaperAI.src.retriever import find_best_chunks
from LocalBioPaperAI.src.generator import generate_answer


# ---------------------------------------------------------
# Streamlit-Seiteneinstellungen
# ---------------------------------------------------------

st.set_page_config(
    page_title="PaperGeneAI",
    page_icon="🧬",
    layout="wide",
)


# ---------------------------------------------------------
# Modell laden
# ---------------------------------------------------------

@st.cache_resource
def get_embedding_model():
    """
    Lädt das Embedding-Modell nur einmal.

    Das Modell bleibt im Speicher, damit es nicht bei jeder
    Streamlit-Aktualisierung erneut geladen werden muss.
    """
    return load_embedding_model()


# ---------------------------------------------------------
# PDF verarbeiten
# ---------------------------------------------------------

@st.cache_data(show_spinner=False)
def prepare_pdf(
    file_bytes: bytes,
    file_hash: str,
):
    """
    Speichert die hochgeladene PDF vorübergehend,
    liest den Text aus und erstellt Textabschnitte.

    file_hash wird für den Streamlit-Cache verwendet.
    """

    del file_hash

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:
            temporary_file.write(file_bytes)
            temporary_path = temporary_file.name

        text = read_pdf(temporary_path)

        if not text or not text.strip():
            raise ValueError(
                "Aus der PDF konnte kein Text extrahiert werden."
            )

        chunks = split_text(text)

        if not chunks:
            raise ValueError(
                "Der extrahierte Text konnte nicht in Abschnitte "
                "aufgeteilt werden."
            )

        return text, chunks

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


def create_context(best_chunks: list[dict]) -> str:
    """
    Verbindet die gefundenen Paper-Abschnitte zu einem Kontext
    für das Sprachmodell.
    """

    context_parts = []

    for index, item in enumerate(best_chunks, start=1):
        chunk = item.get("chunk", "").strip()

        if chunk:
            context_parts.append(
                f"[Quelle {index}]\n{chunk}"
            )

    return "\n\n".join(context_parts)


# ---------------------------------------------------------
# Session-State initialisieren
# ---------------------------------------------------------

if "paper_hash" not in st.session_state:
    st.session_state.paper_hash = None

if "paper_text" not in st.session_state:
    st.session_state.paper_text = None

if "chunks" not in st.session_state:
    st.session_state.chunks = None

if "embeddings" not in st.session_state:
    st.session_state.embeddings = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ---------------------------------------------------------
# Kopfbereich
# ---------------------------------------------------------

st.title("🧬 PaperGeneAI")

st.markdown(
    """
    **Lokale KI-gestützte Analyse wissenschaftlicher Paper**

    Lade ein wissenschaftliches Paper als PDF hoch und stelle
    anschließend Fragen zum Inhalt. Die Antworten werden anhand
    der relevantesten Paper-Abschnitte erzeugt.
    """
)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Einstellungen")

    uploaded_file = st.file_uploader(
        "Paper als PDF hochladen",
        type=["pdf"],
        help="Die PDF wird nur lokal verarbeitet.",
    )

    number_of_sources = st.slider(
        "Anzahl relevanter Textabschnitte",
        min_value=2,
        max_value=10,
        value=5,
        step=1,
    )

    show_sources = st.checkbox(
        "Verwendete Quellen anzeigen",
        value=True,
    )

    if st.button(
        "🗑️ Chatverlauf löschen",
        use_container_width=True,
    ):
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------
# Keine PDF hochgeladen
# ---------------------------------------------------------

if uploaded_file is None:
    st.info(
        "Lade links in der Seitenleiste zuerst ein Paper als PDF hoch."
    )

    st.markdown(
        """
        ### So funktioniert die App

        1. PDF hochladen  
        2. Text wird automatisch extrahiert  
        3. Das Paper wird in Abschnitte zerlegt  
        4. Embeddings werden erstellt  
        5. Zu jeder Frage werden passende Abschnitte gesucht  
        6. Ollama erzeugt eine Antwort anhand dieser Abschnitte
        """
    )

    st.stop()


# ---------------------------------------------------------
# Hochgeladene PDF vorbereiten
# ---------------------------------------------------------

file_bytes = uploaded_file.getvalue()
file_hash = hashlib.sha256(file_bytes).hexdigest()

if st.session_state.paper_hash != file_hash:
    st.session_state.paper_text = None
    st.session_state.chunks = None
    st.session_state.embeddings = None
    st.session_state.chat_history = []

    try:
        with st.spinner("📄 PDF wird gelesen und vorbereitet ..."):
            paper_text, chunks = prepare_pdf(
                file_bytes=file_bytes,
                file_hash=file_hash,
            )

        with st.spinner(
            "🧠 Embeddings werden erstellt. "
            "Beim ersten Start kann das etwas dauern ..."
        ):
            embedding_model = get_embedding_model()
            embeddings = create_embeddings(
                chunks,
                embedding_model,
            )

        st.session_state.paper_hash = file_hash
        st.session_state.paper_text = paper_text
        st.session_state.chunks = chunks
        st.session_state.embeddings = embeddings

        st.success("Das Paper wurde erfolgreich vorbereitet.")

    except Exception as error:
        st.error(
            "Das Paper konnte nicht verarbeitet werden."
        )

        st.exception(error)
        st.stop()


# ---------------------------------------------------------
# Paper-Status
# ---------------------------------------------------------

paper_text = st.session_state.paper_text
chunks = st.session_state.chunks
embeddings = st.session_state.embeddings

status_column_1, status_column_2, status_column_3 = st.columns(3)

with status_column_1:
    st.metric(
        "Datei",
        uploaded_file.name,
    )

with status_column_2:
    st.metric(
        "Textlänge",
        f"{len(paper_text):,} Zeichen",
    )

with status_column_3:
    st.metric(
        "Textabschnitte",
        len(chunks),
    )


# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------

chat_tab, text_tab = st.tabs(
    [
        "💬 Paper fragen",
        "📄 Extrahierter Text",
    ]
)


# ---------------------------------------------------------
# Chat-Tab
# ---------------------------------------------------------

with chat_tab:
    st.subheader("Fragen zum Paper")

    for chat_item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(chat_item["question"])

        with st.chat_message("assistant"):
            st.markdown(chat_item["answer"])

            if (
                show_sources
                and chat_item.get("sources")
            ):
                with st.expander(
                    "Verwendete Paper-Abschnitte"
                ):
                    for source_index, source in enumerate(
                        chat_item["sources"],
                        start=1,
                    ):
                        score = source.get("score")
                        chunk = source.get("chunk", "")

                        st.markdown(
                            f"#### Quelle {source_index}"
                        )

                        if isinstance(score, (int, float)):
                            st.caption(
                                f"Ähnlichkeitsscore: {score:.3f}"
                            )

                        st.write(chunk)

                        if source_index < len(
                            chat_item["sources"]
                        ):
                            st.divider()

    question = st.chat_input(
        "Stelle eine Frage zum Paper ..."
    )

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        try:
            embedding_model = get_embedding_model()

            with st.spinner(
                "🔎 Relevante Paper-Abschnitte werden gesucht ..."
            ):
                best_chunks = find_best_chunks(
                    question,
                    chunks,
                    embeddings,
                    embedding_model,
                )

            if not best_chunks:
                st.warning(
                    "Es wurden keine passenden Paper-Abschnitte gefunden."
                )

                st.stop()

            selected_chunks = best_chunks[:number_of_sources]
            context = create_context(selected_chunks)

            with st.chat_message("assistant"):
                with st.spinner(
                    "🧬 PaperGeneAI formuliert die Antwort ..."
                ):
                    answer = generate_answer(
                        question,
                        context,
                    )

                st.markdown(answer)

                if show_sources:
                    with st.expander(
                        "Verwendete Paper-Abschnitte"
                    ):
                        for source_index, source in enumerate(
                            selected_chunks,
                            start=1,
                        ):
                            score = source.get("score")
                            chunk = source.get("chunk", "")

                            st.markdown(
                                f"#### Quelle {source_index}"
                            )

                            if isinstance(score, (int, float)):
                                st.caption(
                                    "Ähnlichkeitsscore: "
                                    f"{score:.3f}"
                                )

                            st.write(chunk)

                            if source_index < len(
                                selected_chunks
                            ):
                                st.divider()

            st.session_state.chat_history.append(
                {
                    "question": question,
                    "answer": answer,
                    "sources": selected_chunks,
                }
            )

        except Exception as error:
            st.error(
                "Bei der Beantwortung der Frage ist ein Fehler "
                "aufgetreten."
            )

            st.exception(error)


# ---------------------------------------------------------
# Extrahierter Text
# ---------------------------------------------------------

with text_tab:
    st.subheader("Extrahierter Paper-Text")

    st.caption(
        "Dieser Text wurde direkt aus der hochgeladenen PDF gelesen."
    )

    with st.expander(
        "Gesamten extrahierten Text anzeigen",
        expanded=False,
    ):
        st.text_area(
            label="Paper-Text",
            value=paper_text,
            height=600,
            disabled=True,
            label_visibility="collapsed",
        )