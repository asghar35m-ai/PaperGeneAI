import re


SECTION_PATTERNS = {
    "Abstract": [
        r"^\s*abstract\s*$",
    ],
    "Introduction": [
        r"^\s*(?:\d+[\.\s]*)?introduction\s*$",
        r"^\s*background\s*$",
    ],
    "Methods": [
        r"^\s*(?:\d+[\.\s]*)?materials and methods\s*$",
        r"^\s*(?:\d+[\.\s]*)?methods\s*$",
        r"^\s*methodology\s*$",
        r"^\s*experimental procedures\s*$",
    ],
    "Results": [
        r"^\s*(?:\d+[\.\s]*)?results\s*$",
    ],
    "Discussion": [
        r"^\s*(?:\d+[\.\s]*)?discussion\s*$",
        r"^\s*results and discussion\s*$",
    ],
    "Conclusion": [
        r"^\s*(?:\d+[\.\s]*)?conclusions?\s*$",
        r"^\s*summary and conclusions?\s*$",
    ],
    "References": [
        r"^\s*references\s*$",
        r"^\s*bibliography\s*$",
    ],
}


def detect_section_title(line: str) -> str | None:
    """Erkennt, ob eine Textzeile eine typische Kapitelüberschrift ist."""

    cleaned_line = line.strip()

    if not cleaned_line or len(cleaned_line) > 80:
        return None

    for section_name, patterns in SECTION_PATTERNS.items():
        for pattern in patterns:
            if re.match(pattern, cleaned_line, flags=re.IGNORECASE):
                return section_name

    return None


def split_into_sections(text: str) -> dict[str, str]:
    """Zerlegt einen Paper-Text anhand typischer Kapitelüberschriften."""

    sections: dict[str, list[str]] = {
        "Front Matter": [],
    }

    current_section = "Front Matter"

    for line in text.splitlines():
        detected_section = detect_section_title(line)

        if detected_section:
            current_section = detected_section
            sections.setdefault(current_section, [])
            continue

        sections.setdefault(current_section, []).append(line)

    cleaned_sections: dict[str, str] = {}

    for section_name, lines in sections.items():
        section_text = "\n".join(lines).strip()

        if section_text:
            cleaned_sections[section_name] = section_text

    return cleaned_sections


def get_summary_sections(
    sections: dict[str, str],
    max_characters_per_section: int = 12_000,
) -> dict[str, str]:
    """
    Wählt wissenschaftlich wichtige Kapitel aus.
    References werden bewusst ausgeschlossen.
    """

    preferred_order = [
        "Abstract",
        "Introduction",
        "Methods",
        "Results",
        "Discussion",
        "Conclusion",
    ]

    selected_sections: dict[str, str] = {}

    for section_name in preferred_order:
        section_text = sections.get(section_name)

        if section_text:
            selected_sections[section_name] = section_text[
                :max_characters_per_section
            ]

    if not selected_sections:
        fallback_text = sections.get("Front Matter", "")

        if fallback_text:
            selected_sections["Paper Text"] = fallback_text[
                :max_characters_per_section
            ]

    return selected_sections