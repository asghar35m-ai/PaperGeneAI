import json
import re
from typing import Any

import ollama


MODEL_NAME = "llama3.2"


# -------------------------------------------------------------------
# Grundfunktionen für Ollama
# -------------------------------------------------------------------

def ask_ollama(
    prompt: str,
    system_prompt: str | None = None,
    temperature: float = 0.1,
) -> str:
    """
    Sendet einen Prompt an das lokal laufende Ollama-Modell.

    Parameters
    ----------
    prompt:
        Die konkrete Aufgabe für das Sprachmodell.
    system_prompt:
        Eine übergeordnete Rollenbeschreibung.
    temperature:
        Steuert die Kreativität des Modells.
        Für wissenschaftliche Analysen verwenden wir niedrige Werte.

    Returns
    -------
    str
        Die generierte Antwort des Modells.
    """

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            options={
                "temperature": temperature,
            },
        )

        content = response.get("message", {}).get("content", "")

        if not content:
            return (
                "Das lokale Sprachmodell hat keine Antwort erzeugt. "
                "Bitte überprüfe, ob Ollama korrekt läuft."
            )

        return content.strip()

    except Exception as error:
        return (
            "Beim Aufruf des lokalen Ollama-Modells ist ein Fehler "
            f"aufgetreten:\n\n{error}"
        )


def ask_ollama_json(
    prompt: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Fordert vom Modell eine JSON-Antwort an und wandelt sie
    in ein Python-Dictionary um.

    Falls das Modell zusätzlichen Text um das JSON schreibt,
    versucht die Funktion trotzdem, das JSON zu extrahieren.
    """

    messages = []

    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": system_prompt,
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    try:
        response = ollama.chat(
            model=MODEL_NAME,
            messages=messages,
            format="json",
            options={
                "temperature": 0.0,
            },
        )

        content = response.get("message", {}).get("content", "").strip()

        if not content:
            return {
                "error": (
                    "Das lokale Sprachmodell hat keine JSON-Antwort erzeugt."
                )
            }

        return parse_json_response(content)

    except Exception as error:
        return {
            "error": (
                "Beim Erzeugen der strukturierten Analyse ist ein "
                f"Fehler aufgetreten: {error}"
            )
        }


def parse_json_response(content: str) -> dict[str, Any]:
    """
    Wandelt eine Modellantwort möglichst robust in JSON um.
    """

    try:
        parsed = json.loads(content)

        if isinstance(parsed, dict):
            return parsed

        return {
            "data": parsed,
        }

    except json.JSONDecodeError:
        pass

    json_match = re.search(
        r"\{.*\}",
        content,
        flags=re.DOTALL,
    )

    if not json_match:
        return {
            "error": "In der Modellantwort wurde kein gültiges JSON gefunden.",
            "raw_response": content,
        }

    try:
        parsed = json.loads(json_match.group(0))

        if isinstance(parsed, dict):
            return parsed

        return {
            "data": parsed,
        }

    except json.JSONDecodeError as error:
        return {
            "error": f"Die JSON-Antwort konnte nicht gelesen werden: {error}",
            "raw_response": content,
        }


# -------------------------------------------------------------------
# Frage-Antwort-Funktion
# -------------------------------------------------------------------

def generate_answer(question: str, context: str) -> str:
    """
    Beantwortet eine Frage ausschließlich anhand des
    bereitgestellten Paper-Kontexts.
    """

    system_prompt = """
Du bist ein wissenschaftlicher Assistent für biomedizinische
Fachliteratur.

Du arbeitest streng evidenzbasiert.

Du darfst keine Informationen erfinden, ergänzen oder als Tatsache
darstellen, wenn sie nicht im bereitgestellten Kontext enthalten sind.
"""

    prompt = f"""
Beantworte die Frage ausschließlich anhand des bereitgestellten
Paper-Kontexts.

Regeln:

1. Verwende nur Informationen aus dem Kontext.
2. Erfinde keine Fakten, Zahlen, Methoden oder Schlussfolgerungen.
3. Verwende niemals die Formulierungen „wir haben“ oder „unsere Studie“.
4. Schreibe stattdessen:
   - „Die Autoren berichten ...“
   - „Die Studie zeigt ...“
   - „Im Paper wurde ...“
5. Nenne relevante Zahlen, Zeitpunkte und Messwerte exakt.
6. Trenne direkte Aussagen des Papers von deiner Interpretation.
7. Falls die Antwort nicht enthalten ist, schreibe:
   „Diese Information ist in den gefundenen Paper-Abschnitten
   nicht eindeutig enthalten.“
8. Antworte in derselben Sprache wie die Frage.
9. Antworte verständlich, strukturiert und wissenschaftlich präzise.

PAPER-KONTEXT:

{context}

FRAGE:

{question}

ANTWORT:
"""

    return ask_ollama(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.05,
    )


# -------------------------------------------------------------------
# Zusammenfassung einzelner Kapitel
# -------------------------------------------------------------------

def summarize_section(
    section_name: str,
    section_text: str,
) -> str:
    """
    Erstellt eine präzise Zusammenfassung eines Paper-Kapitels.
    """

    system_prompt = """
Du analysierst biomedizinische Fachliteratur.

Deine Aufgabe ist es, Informationen vollständig, vorsichtig und
wissenschaftlich korrekt zusammenzufassen.
"""

    prompt = f"""
Analysiere den Abschnitt „{section_name}“ eines wissenschaftlichen
Papers.

Erstelle eine präzise Zusammenfassung ausschließlich anhand des
Abschnitts.

Regeln:

- Keine Informationen erfinden.
- Keine Informationen aus allgemeinem Wissen ergänzen.
- Verwende nicht „wir“, sondern „die Autoren“ oder „die Studie“.
- Nenne konkrete Zahlen, Zeitpunkte, Stichprobengrößen und Messwerte.
- Nenne verwendete Methoden und Modelle möglichst exakt.
- Nenne relevante Gene, Proteine, Marker, Zelltypen und Signalwege.
- Unterscheide Ergebnisse von Interpretationen.
- Literaturverzeichnis und reine Zitationslisten ignorieren.
- Falls ein Punkt nicht enthalten ist, lasse ihn weg.
- Maximal 350 Wörter.
- Antworte auf Deutsch.

ABSCHNITT:

{section_text}

ZUSAMMENFASSUNG:
"""

    return ask_ollama(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.05,
    )


# -------------------------------------------------------------------
# Paper-Metadaten
# -------------------------------------------------------------------

def extract_paper_metadata(
    paper_text: str,
) -> dict[str, Any]:
    """
    Extrahiert bibliografische und allgemeine Metadaten aus einem Paper.

    Besonders hilfreich sind dafür die erste und letzte Seite des Papers.
    """

    system_prompt = """
Du extrahierst bibliografische Metadaten aus wissenschaftlichen Papers.

Nutze ausschließlich explizit im Text vorhandene Informationen.
Unbekannte Werte müssen null oder eine leere Liste sein.
"""

    prompt = f"""
Extrahiere die Metadaten aus dem folgenden wissenschaftlichen Paper.

Gib ausschließlich ein gültiges JSON-Objekt zurück.

Verwende exakt diese Struktur:

{{
  "title": null,
  "authors": [],
  "journal": null,
  "publication_year": null,
  "volume": null,
  "article_number": null,
  "doi": null,
  "keywords": [],
  "corresponding_author": null,
  "corresponding_email": null,
  "institutions": [],
  "funding": [],
  "competing_interests": null,
  "ethical_approval": null,
  "received_date": null,
  "accepted_date": null,
  "open_access_license": null
}}

Regeln:

- Keine Informationen erfinden.
- Autorennamen nicht verkürzen, sofern sie vollständig angegeben sind.
- DOI ohne URL-Präfix angeben.
- Jahreszahlen als Zahl angeben.
- Fehlende Einzelwerte als null angeben.
- Fehlende Listen als leere Listen angeben.
- Keine Erklärung außerhalb des JSON ausgeben.

PAPER-TEXT:

{paper_text}
"""

    return ask_ollama_json(
        prompt=prompt,
        system_prompt=system_prompt,
    )


# -------------------------------------------------------------------
# Biologische Entitäten
# -------------------------------------------------------------------

def extract_biological_entities(
    paper_text: str,
) -> dict[str, Any]:
    """
    Extrahiert biologische und biomedizinische Entitäten.

    Die Funktion unterscheidet unter anderem Gene, Proteine,
    Zelltypen, Krankheiten und experimentelle Modelle.
    """

    system_prompt = """
Du bist auf biomedizinische Named-Entity-Recognition spezialisiert.

Extrahiere nur Entitäten, die ausdrücklich im Paper genannt werden.
Ordne Begriffe vorsichtig der richtigen Kategorie zu.
"""

    prompt = f"""
Extrahiere biologische und medizinische Entitäten aus dem folgenden
Paper-Text.

Gib ausschließlich ein gültiges JSON-Objekt zurück.

Verwende exakt diese Struktur:

{{
  "genes": [],
  "proteins": [],
  "biomarkers": [],
  "cell_types": [],
  "stem_cell_types": [],
  "tissues": [],
  "organs": [],
  "diseases": [],
  "drugs_or_compounds": [],
  "antibodies": [],
  "signaling_pathways": [],
  "biological_processes": [],
  "organisms": [],
  "experimental_models": [],
  "laboratory_methods": [],
  "imaging_methods": [],
  "computational_models": [],
  "datasets_or_databases": []
}}

Regeln:

- Keine Begriffe erfinden.
- Abkürzungen zusammen mit dem vollständigen Namen angeben,
  sofern beides im Text vorkommt.
- Doppelte Begriffe entfernen.
- CXCR4 beispielsweise nicht automatisch gleichzeitig als Gen,
  Protein und Biomarker einordnen, außer der Text macht diese
  Rollen ausdrücklich deutlich.
- Software und KI-Modelle unter „computational_models“ einordnen.
- Zellkulturverfahren unter „laboratory_methods“ einordnen.
- Fehlende Kategorien als leere Listen ausgeben.
- Keine Erklärung außerhalb des JSON ausgeben.

PAPER-TEXT:

{paper_text}
"""

    return ask_ollama_json(
        prompt=prompt,
        system_prompt=system_prompt,
    )


# -------------------------------------------------------------------
# Strukturierte Gesamtanalyse
# -------------------------------------------------------------------

def generate_structured_analysis(
    paper_context: str,
) -> dict[str, Any]:
    """
    Erstellt eine umfangreiche strukturierte Analyse des Papers.

    Das Ergebnis kann später in Streamlit als Karten,
    Tabellen und Kennzahlen dargestellt werden.
    """

    system_prompt = """
Du bist ein wissenschaftlicher Reviewer für biomedizinische Forschung.

Du bewertest Studien vorsichtig, evidenzbasiert und transparent.
Du unterscheidest klar zwischen berichteten Fakten und eigener
kritischer Einordnung.
"""

    prompt = f"""
Analysiere das wissenschaftliche Paper ausschließlich anhand des
bereitgestellten Kontexts.

Gib ausschließlich ein gültiges JSON-Objekt zurück.

Verwende exakt diese Struktur:

{{
  "research_question": null,
  "hypothesis": null,
  "study_type": null,
  "scientific_background": null,

  "sample": {{
    "independent_samples": null,
    "sample_description": null,
    "biological_replicates": null,
    "technical_replicates": null,
    "number_of_images": null,
    "number_of_patients_or_donors": null
  }},

  "experimental_design": {{
    "groups": [],
    "control_groups": [],
    "timepoints": [],
    "intervention_or_exposure": null,
    "primary_endpoint": null,
    "secondary_endpoints": []
  }},

  "methods": {{
    "cell_culture": [],
    "laboratory_methods": [],
    "imaging_methods": [],
    "computational_methods": [],
    "statistical_methods": [],
    "validation_methods": []
  }},

  "machine_learning": {{
    "task": null,
    "model": null,
    "pretraining": null,
    "input_data": null,
    "ground_truth": null,
    "training_strategy": null,
    "data_split": null,
    "cross_validation": null,
    "data_augmentation": [],
    "loss_function": null,
    "optimizer": null,
    "explainability_methods": [],
    "baseline_models": []
  }},

  "main_results": [
    {{
      "result": null,
      "value": null,
      "unit": null,
      "timepoint": null,
      "comparison": null,
      "evidence_location": null
    }}
  ],

  "best_reported_performance": {{
    "metric": null,
    "value": null,
    "timepoint": null,
    "model": null
  }},

  "biological_interpretation": null,
  "clinical_relevance": null,
  "translational_relevance": null,

  "strengths": [],
  "limitations": [],
  "potential_biases": [],
  "confounding_factors": [],
  "open_questions": [],
  "future_research": [],

  "author_conclusion": null,
  "reviewer_assessment": null,

  "evidence_ratings": {{
    "novelty": {{
      "score": null,
      "reason": null
    }},
    "methodological_quality": {{
      "score": null,
      "reason": null
    }},
    "sample_size": {{
      "score": null,
      "reason": null
    }},
    "reproducibility": {{
      "score": null,
      "reason": null
    }},
    "biological_plausibility": {{
      "score": null,
      "reason": null
    }},
    "clinical_relevance": {{
      "score": null,
      "reason": null
    }},
    "overall_evidence": {{
      "score": null,
      "reason": null
    }}
  }}
}}

Bewertungsregeln:

- Die Scores reichen von 1 bis 5.
- 1 bedeutet sehr schwach.
- 3 bedeutet mittel.
- 5 bedeutet sehr stark.
- Scores müssen mit einer kurzen Begründung versehen werden.
- Die Bewertung darf nicht mit der Meinung der Autoren verwechselt werden.

Inhaltliche Regeln:

- Keine Informationen erfinden.
- Zahlen und Zeitpunkte exakt übernehmen.
- Verwende nicht „wir“ oder „unsere Studie“.
- Unterscheide unabhängige biologische Proben von der Anzahl
  technisch erzeugter Bilder oder Patches.
- Eine große Bildzahl darf nicht als große unabhängige
  Stichprobe dargestellt werden.
- Unterscheide Patch-Level- und Clone-Level-Ergebnisse.
- Nenne Unsicherheiten und Standardabweichungen, sofern vorhanden.
- Prüfe, ob externe Validierung vorhanden ist.
- Prüfe, ob Datenleckage durch die Datenteilung verhindert wurde.
- Markiere mögliche Confounder wie Helligkeit, Kontrast,
  Zelltrümmer, Waschschritte oder Batch-Effekte.
- Fehlende Einzelwerte als null angeben.
- Fehlende Listen als leere Listen ausgeben.
- Keine Erklärung außerhalb des JSON ausgeben.

PAPER-KONTEXT:

{paper_context}
"""

    return ask_ollama_json(
        prompt=prompt,
        system_prompt=system_prompt,
    )


# -------------------------------------------------------------------
# Strukturierte Gesamtzusammenfassung als Text
# -------------------------------------------------------------------

def generate_structured_summary(
    section_summaries: dict[str, str],
) -> str:
    """
    Erstellt aus den Kapitelzusammenfassungen eine
    strukturierte Gesamtzusammenfassung.
    """

    summaries_text = "\n\n".join(
        f"## {section_name}\n{summary}"
        for section_name, summary in section_summaries.items()
    )

    system_prompt = """
Du bist ein wissenschaftlicher Assistent für biomedizinische Papers.

Du formulierst präzise, neutral und ausschließlich evidenzbasiert.
"""

    prompt = f"""
Erstelle aus den folgenden Kapitelzusammenfassungen eine
wissenschaftliche Gesamtzusammenfassung.

Nutze genau diese Überschriften:

# Forschungsfrage

# Wissenschaftlicher Hintergrund

# Studiendesign und Stichprobe

# Experimentelle Methoden

# Computationale Methoden

# Wichtigste quantitative Ergebnisse

# Biologische Interpretation

# Klinische und translationale Bedeutung

# Stärken

# Limitationen

# Schlussfolgerung

# Relevante Gene, Proteine, Marker und Zelltypen

Regeln:

- Ausschließlich vorhandene Informationen verwenden.
- Keine Informationen ergänzen.
- Niemals „wir haben“ oder „unsere Studie“ schreiben.
- Verwende „die Autoren“, „die Studie“ oder passive Formulierungen.
- Nenne Zahlen, Stichprobengrößen, Messwerte und Zeitpunkte exakt.
- Unterscheide unabhängige Proben von Bildern oder Patches.
- Unterscheide Patch Accuracy und Clone Accuracy.
- Ergebnisse klar von Interpretation trennen.
- Bei fehlenden Informationen schreiben:
  „Nicht eindeutig angegeben.“
- Antworte auf Deutsch.
- Formuliere übersichtlich und wissenschaftlich präzise.

KAPITELZUSAMMENFASSUNGEN:

{summaries_text}

GESAMTZUSAMMENFASSUNG:
"""

    return ask_ollama(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.05,
    )


# -------------------------------------------------------------------
# Ausführlicher Forschungsbericht
# -------------------------------------------------------------------

def generate_analysis_report(
    section_summaries: dict[str, str],
) -> str:
    """
    Erstellt einen professionellen wissenschaftlichen Analysebericht.
    """

    summaries_text = "\n\n".join(
        f"## {section_name}\n{summary}"
        for section_name, summary in section_summaries.items()
    )

    system_prompt = """
Du bist ein kritischer wissenschaftlicher Reviewer für biomedizinische
Forschung.

Du analysierst Papers neutral und unterscheidest streng zwischen
den Aussagen der Autoren und deiner kritischen Bewertung.
"""

    prompt = f"""
Analysiere das wissenschaftliche Paper ausschließlich anhand
der folgenden Kapitelzusammenfassungen.

Erstelle einen professionellen Forschungsbericht mit genau
diesen Überschriften:

# Executive Summary

# Forschungsfrage und Hypothese

# Studiendesign

# Stichprobe und unabhängige Beobachtungseinheiten

# Experimentelle Methoden

# Computationale und statistische Methoden

# Wichtigste quantitative Ergebnisse

# Biologischer oder molekularer Mechanismus

# Relevante Gene, Proteine, Marker und Signalwege

# Verwendete Zelltypen, Modelle und Organismen

# Klinische oder translationale Bedeutung

# Stärken der Studie

# Limitationen der Studie

# Mögliche Bias- und Confounding-Quellen

# Reproduzierbarkeit und Generalisierbarkeit

# Offene Fragen

# Vorschläge für zukünftige Forschung

# Kritische Gesamtbewertung

Regeln:

- Keine Informationen erfinden.
- Nur Inhalte aus den Kapitelzusammenfassungen verwenden.
- Niemals aus Perspektive der Autoren mit „wir“ schreiben.
- Verwende „die Autoren berichten“ oder „die Studie zeigt“.
- Unterscheide Fakten, Autoreninterpretation und Reviewerbewertung.
- Nenne konkrete Zahlen und Zeitpunkte.
- Unterscheide unabhängige biologische Proben von Bildern,
  Patches und technischen Wiederholungen.
- Erwähne fehlende externe Validierung, sofern keine angegeben ist.
- Keine klinischen Empfehlungen geben.
- Bei fehlenden Informationen schreiben:
  „Nicht eindeutig im Paper angegeben.“
- Antworte auf Deutsch.

KAPITELZUSAMMENFASSUNGEN:

{summaries_text}

FORSCHUNGSBERICHT:
"""

    return ask_ollama(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.05,
    )


# -------------------------------------------------------------------
# Spezialanalysen
# -------------------------------------------------------------------

FOCUS_INSTRUCTIONS = {
    "Studiendesign": """
Analysiere insbesondere:
- Studientyp
- unabhängige Stichprobengröße
- biologische und technische Replikate
- Kontrollgruppen
- Datenteilung
- Endpunkte
- mögliche Datenleckage
- Generalisierbarkeit
""",
    "Methoden": """
Analysiere insbesondere:
- Zellkultur und Laborverfahren
- Bildgebung
- Messverfahren
- statistische Methoden
- KI-Modelle
- Trainingsstrategie
- Validierung
- Reproduzierbarkeit
""",
    "Ergebnisse": """
Analysiere insbesondere:
- wichtigste quantitative Resultate
- Effektgrößen
- Genauigkeitsmetriken
- Zeitpunkte
- Unsicherheiten und Standardabweichungen
- positive und negative Ergebnisse
""",
    "Biologische Mechanismen": """
Analysiere insbesondere:
- Gene und Proteine
- Biomarker
- Zelltypen
- Differenzierungsprozesse
- molekulare oder zelluläre Mechanismen
- biologische Plausibilität
""",
    "Kritische Bewertung": """
Analysiere insbesondere:
- Stärken
- Limitationen
- Bias
- Confounder
- Stichprobengröße
- externe Validierung
- Übertragbarkeit
- mögliche Überinterpretationen
""",
    "Klinische Relevanz": """
Analysiere insbesondere:
- klinische Bedeutung
- translationale Bedeutung
- Abstand zur klinischen Anwendung
- Nutzen für Diagnostik, Therapie oder Forschung
- notwendige nächste Validierungsschritte
""",
}


def generate_focus_analysis(
    focus_name: str,
    section_summaries: dict[str, str],
) -> str:
    """
    Analysiert gezielt einen ausgewählten wissenschaftlichen Bereich.
    """

    summaries_text = "\n\n".join(
        f"## {section_name}\n{summary}"
        for section_name, summary in section_summaries.items()
    )

    focus_instruction = FOCUS_INSTRUCTIONS.get(
        focus_name,
        """
Analysiere den gewählten Schwerpunkt vollständig und
wissenschaftlich kritisch.
""",
    )

    system_prompt = """
Du bist ein kritischer wissenschaftlicher Reviewer für biomedizinische
Fachliteratur.

Du verwendest ausschließlich bereitgestellte Informationen und
kennzeichnest Unsicherheiten klar.
"""

    prompt = f"""
Analysiere das Paper mit dem Schwerpunkt:

„{focus_name}“

{focus_instruction}

Nutze ausschließlich die folgenden Kapitelzusammenfassungen.

Regeln:

- Keine Informationen erfinden.
- Nenne konkrete Methoden, Resultate und Messwerte.
- Verwende nicht „wir“ oder „unsere Studie“.
- Trenne direkte Aussagen des Papers von kritischer Interpretation.
- Unterscheide unabhängige Stichproben von Bildern und Patches.
- Nenne relevante Gene, Proteine, Zelltypen oder Signalwege,
  sofern vorhanden.
- Verwende klare Zwischenüberschriften.
- Nenne sowohl Stärken als auch Schwächen.
- Falls Informationen fehlen, schreibe:
  „Nicht eindeutig im Paper angegeben.“
- Antworte auf Deutsch.

KAPITELZUSAMMENFASSUNGEN:

{summaries_text}

SPEZIALANALYSE:
"""

    return ask_ollama(
        prompt=prompt,
        system_prompt=system_prompt,
        temperature=0.05,
    )