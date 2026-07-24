from dataclasses import dataclass

from sqlalchemy import select

from core.config import get_session
from core.db import get_sessionmaker
from core.models import Entity, EntityType

MYGENE_URL = "https://mygene.info/v3/gene"
FIELDS = "symbol,ensembl.gene,uniprot.Swiss-Prot"

# mygene.info dokumentiert keine harte Grenze fuer Batch-Groesse, 200 ist
# defensiv und bleibt weit unter dem ueblichen Richtwert von ~1000.
BATCH_SIZE = 200
MAX_CALLS_PER_SECOND = 5  # kein NCBI-Dienst, aber gleiche defensive Haltung


def _first(value):
    """mygene.info liefert bei mehreren Treffern eine Liste statt eines
    einzelnen Werts (auch fuer verschachtelte Felder wie "ensembl" selbst,
    nicht nur "ensembl.gene") -- wir nehmen den ersten Treffer."""
    if isinstance(value, list):
        return value[0] if value else None
    return value


@dataclass
class GeneEnrichmentSummary:
    pending: int
    updated: int
    not_found: int


def enrich_gene_entities() -> GeneEnrichmentSummary:
    """Reichert alle GENE-Entities ohne ensembl_id ueber mygene.info an.

    Idempotent ueber die Spalte selbst: einmal befuellte Entities werden bei
    einem erneuten Lauf nicht mehr angefragt (kein job_runs-Eintrag noetig,
    da eine Gen-Entity ueber viele Dokumente hinweg geteilt wird).
    """
    session_factory = get_sessionmaker()
    http = get_session("mygene.info", max_calls=MAX_CALLS_PER_SECOND, period=1.0)

    with session_factory() as session:
        pending = session.scalars(
            select(Entity).where(Entity.type == EntityType.GENE, Entity.ensembl_id.is_(None))
        ).all()
        pending_ids = [(e.id, e.canonical_id) for e in pending]

    updated = 0
    not_found = 0

    for i in range(0, len(pending_ids), BATCH_SIZE):
        batch = pending_ids[i : i + BATCH_SIZE]
        response = http.post(
            MYGENE_URL,
            data={"ids": ",".join(cid for _, cid in batch), "fields": FIELDS},
        )
        response.raise_for_status()
        results = {r["query"]: r for r in response.json()}

        with session_factory() as session:
            for entity_id, canonical_id in batch:
                result = results.get(canonical_id)
                entity = session.get(Entity, entity_id)

                if result is None or result.get("notfound"):
                    not_found += 1
                    # Leerer String statt NULL markiert "angefragt, nichts
                    # gefunden" -- sonst wuerde diese Entity bei jedem
                    # weiteren Lauf erneut angefragt (ensembl_id waere immer
                    # noch NULL), nie wirklich idempotent abgeschlossen.
                    entity.ensembl_id = ""
                    session.commit()
                    continue

                entity.symbol = result.get("symbol", entity.symbol)
                ensembl = _first(result.get("ensembl")) or {}
                entity.ensembl_id = _first(ensembl.get("gene")) or ""
                uniprot = _first(result.get("uniprot")) or {}
                entity.uniprot_id = _first(uniprot.get("Swiss-Prot"))
                session.commit()
                updated += 1

    return GeneEnrichmentSummary(
        pending=len(pending_ids), updated=updated, not_found=not_found
    )
