from dataclasses import dataclass

from core.config import get_session
from core.models import EntityType

PUBTATOR_URL = "https://www.ncbi.nlm.nih.gov/research/pubtator3-api/publications/export/biocjson"

# Von PubTator3 tatsaechlich max. erlaubte PMIDs pro Anfrage (getestet:
# 300 -> Fehler "can not be longer than 100", 100 -> funktioniert).
MAX_PMIDS_PER_REQUEST = 100

# Offene Annahme: kein dokumentiertes Rate-Limit gefunden (NCBI-Domain, aber
# nicht die klassische E-Utilities-API). Gleiche defensive Haltung wie bei
# Europe PMC/NCBI-ohne-Key: 3 Anfragen/s, anpassbar falls 429 auftritt.
MAX_CALLS_PER_SECOND = 3

# PubTator3-Annotationstyp -> unser EntityType. Nur diese vier werden
# gespeichert (Gene, Chemikalien, Zelllinien, Spezies), andere PubTator-Typen
# (Disease, Mutation, ...) werden ignoriert.
_TYPE_MAP = {
    "Gene": EntityType.GENE,
    "Chemical": EntityType.CHEMICAL,
    "CellLine": EntityType.CELL_LINE,
    "Species": EntityType.SPECIES,
}


@dataclass(frozen=True)
class PubTatorAnnotation:
    entity_type: EntityType
    canonical_id: str
    name: str
    surface_form: str
    span_start: int
    span_end: int


def _parse_annotations(doc: dict) -> list[PubTatorAnnotation]:
    annotations = []
    for passage in doc.get("passages", []):
        for ann in passage.get("annotations", []):
            infons = ann.get("infons", {})
            entity_type = _TYPE_MAP.get(infons.get("type"))
            if entity_type is None:
                continue
            if not infons.get("valid") or not infons.get("identifier") or infons["identifier"] == "-":
                continue  # nicht aufloesbar, kein verlaesslicher canonical_id

            for location in ann.get("locations", []):
                annotations.append(
                    PubTatorAnnotation(
                        entity_type=entity_type,
                        canonical_id=str(infons["identifier"]),
                        name=str(infons.get("name") or infons["identifier"]),
                        surface_form=ann.get("text", ""),
                        span_start=location["offset"],
                        span_end=location["offset"] + location["length"],
                    )
                )
    return annotations


class PubTatorClient:
    """Kennt nur die PubTator3-API, gibt einfache Datenklassen zurueck.

    Kein DB-Zugriff, kein eigenes Caching -- beides sitzt zentral in
    core/config.py (get_session), wie in CLAUDE.md gefordert.
    """

    def __init__(self) -> None:
        self._session = get_session(
            "www.ncbi.nlm.nih.gov", max_calls=MAX_CALLS_PER_SECOND, period=1.0
        )

    def annotate(self, pmids: list[str]) -> dict[str, list[PubTatorAnnotation]]:
        """Holt Annotationen fuer bis zu MAX_PMIDS_PER_REQUEST PMIDs in einer
        Anfrage. PMIDs ohne PubTator-Ergebnis fehlen im Rueckgabe-Dict."""
        if len(pmids) > MAX_PMIDS_PER_REQUEST:
            raise ValueError(
                f"PubTator3 erlaubt maximal {MAX_PMIDS_PER_REQUEST} PMIDs pro Anfrage, "
                f"bekommen: {len(pmids)}"
            )

        response = self._session.get(PUBTATOR_URL, params={"pmids": ",".join(pmids)})
        response.raise_for_status()
        data = response.json()

        result: dict[str, list[PubTatorAnnotation]] = {}
        for doc in data.get("PubTator3", []):
            result[doc["id"]] = _parse_annotations(doc)
        return result
