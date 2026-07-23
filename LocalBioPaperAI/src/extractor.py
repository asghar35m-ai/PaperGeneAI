import json
from typing import Any

import ollama


MODEL_NAME = "llama3.2"


EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "genes_and_proteins": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string"},
                    "role": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "name",
                    "type",
                    "role",
                    "evidence",
                ],
            },
        },
        "cell_types": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "role": {"type": "string"},
                    "markers": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "name",
                    "role",
                    "markers",
                    "evidence",
                ],
            },
        },
        "growth_factors_and_compounds": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "concentration": {"type": "string"},
                    "timepoint": {"type": "string"},
                    "purpose": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "name",
                    "concentration",
                    "timepoint",
                    "purpose",
                    "evidence",
                ],
            },
        },
        "pathways": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "effect": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "name",
                    "effect",
                    "evidence",
                ],
            },
        },
        "methods": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "details": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "name",
                    "purpose",
                    "details",
                    "evidence",
                ],
            },
        },
        "differentiation_protocol": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "stage": {"type": "string"},
                    "timepoint": {"type": "string"},
                    "treatment": {"type": "string"},
                    "concentration": {"type": "string"},
                    "expected_cell_state": {"type": "string"},
                    "markers": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "stage",
                    "timepoint",
                    "treatment",
                    "concentration",
                    "expected_cell_state",
                    "markers",
                    "evidence",
                ],
            },
        },
        "key_results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "result": {"type": "string"},
                    "value": {"type": "string"},
                    "interpretation": {"type": "string"},
                    "evidence": {"type": "string"},
                },
                "required": [
                    "result",
                    "value",
                    "interpretation",
                    "evidence",
                ],
            },
        },
    },
    "required": [
        "genes_and_proteins",
        "cell_types",
        "growth_factors_and_compounds",
        "pathways",
        "methods",
        "differentiation_protocol",
        "key_results",
    ],
}


def extract_scientific_data(
    section_summaries: dict[str, str],
) -> dict[str, Any]:
    summaries_text = "\n\n".join(
        f"## {section_name}\n{summary}"
        for section_name, summary in section_summaries.items()
    )

    prompt = f"""
Extrahiere strukturierte wissenschaftliche Informationen
ausschließlich aus den folgenden Kapitelzusammenfassungen.

Regeln:
- Keine Informationen erfinden.
- Nur explizit genannte Angaben verwenden.
- Bei fehlenden Konzentrationen, Zeitpunkten oder Markern
  eine leere Zeichenkette verwenden.
- evidence soll eine kurze Textstelle oder knappe Begründung
  aus dem bereitgestellten Inhalt enthalten.
- Gene und Proteine nicht doppelt aufnehmen.
- Literaturreferenzen ignorieren.
- Gib ausschließlich Daten entsprechend dem JSON-Schema zurück.

KAPITELZUSAMMENFASSUNGEN:
{summaries_text}
"""

    response = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        format=EXTRACTION_SCHEMA,
        options={
            "temperature": 0,
        },
    )

    raw_content = response["message"]["content"]

    try:
        return json.loads(raw_content)

    except json.JSONDecodeError as error:
        raise ValueError(
            "Die strukturierte Antwort konnte nicht "
            "als JSON gelesen werden."
        ) from error