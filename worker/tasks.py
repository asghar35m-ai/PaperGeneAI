from dataclasses import dataclass
from datetime import datetime, timezone

from sqlalchemy import select

from core.db import get_sessionmaker
from core.models import Document, Entity, EntityType, JobRun, JobStage, JobStatus, Mention, MentionSource
from normalize.gene_ids import enrich_gene_entities
from normalize.pubtator import MAX_PMIDS_PER_REQUEST, PubTatorAnnotation, PubTatorClient

# Noch keine echte Job-Queue (RQ/Redis) -- passend zur Phase-1-Entscheidung,
# api/ui-Container erst anzulegen, wenn tatsaechlich Code dafuer existiert.
# Diese Funktionen sind normal aufrufbare Python-Funktionen, die spaeter
# 1:1 als RQ-Task registriert werden koennen.


def _pending_documents(session) -> list[tuple[int, str]]:
    """documents mit PMID, die noch keinen erfolgreichen normalize-Job haben."""
    done_subq = select(JobRun.document_id).where(
        JobRun.stage == JobStage.NORMALIZE, JobRun.status == JobStatus.DONE
    )
    rows = session.execute(
        select(Document.id, Document.pmid)
        .where(Document.pmid.isnot(None))
        .where(Document.id.notin_(done_subq))
    ).all()
    return [(row.id, row.pmid) for row in rows]


def _get_or_create_entity(
    session, cache: dict[tuple[EntityType, str], int], ann: PubTatorAnnotation
) -> int:
    key = (ann.entity_type, ann.canonical_id)
    if key in cache:
        return cache[key]

    entity = session.scalar(
        select(Entity).where(Entity.type == ann.entity_type, Entity.canonical_id == ann.canonical_id)
    )
    if entity is None:
        entity = Entity(
            type=ann.entity_type,
            canonical_id=ann.canonical_id,
            symbol=ann.name if ann.entity_type == EntityType.GENE else None,
            name=ann.name,
        )
        session.add(entity)
        session.flush()  # entity.id

    cache[key] = entity.id
    return entity.id


@dataclass
class NormalizationSummary:
    pending: int
    processed: int
    failed: int
    no_pubtator_result: int
    mentions_created: int


def run_pubtator_normalization() -> NormalizationSummary:
    """Holt PubTator3-Annotationen fuer alle PMID-Dokumente ohne
    abgeschlossenen normalize-Job und schreibt sie als entity/mention.

    Idempotent: bereits als 'done' markierte Dokumente werden von
    _pending_documents() gar nicht erst zurueckgegeben. Dokumente ohne PMID
    tauchen dort ebenfalls nie auf (sauberes Ueberspringen ohne Sonderstatus).
    """
    session_factory = get_sessionmaker()
    client = PubTatorClient()

    with session_factory() as session:
        pending = _pending_documents(session)

    entity_cache: dict[tuple[EntityType, str], int] = {}
    processed = 0
    failed = 0
    no_result = 0
    mentions_created = 0

    for i in range(0, len(pending), MAX_PMIDS_PER_REQUEST):
        batch = pending[i : i + MAX_PMIDS_PER_REQUEST]
        annotations_by_pmid = client.annotate([pmid for _, pmid in batch])

        with session_factory() as session:
            for document_id, pmid in batch:
                annotations = annotations_by_pmid.get(pmid, [])  # fehlt = PubTator hat nichts, kein Fehler

                try:
                    for ann in annotations:
                        entity_id = _get_or_create_entity(session, entity_cache, ann)
                        session.add(
                            Mention(
                                document_id=document_id,
                                entity_id=entity_id,
                                span_start=ann.span_start,
                                span_end=ann.span_end,
                                surface_form=ann.surface_form,
                                source=MentionSource.PUBTATOR,
                            )
                        )
                        mentions_created += 1

                    now = datetime.now(timezone.utc)
                    session.add(
                        JobRun(
                            document_id=document_id,
                            stage=JobStage.NORMALIZE,
                            status=JobStatus.DONE,
                            started_at=now,
                            finished_at=now,
                        )
                    )
                    session.commit()
                    processed += 1
                    if pmid not in annotations_by_pmid:
                        no_result += 1
                except Exception as exc:
                    session.rollback()
                    now = datetime.now(timezone.utc)
                    session.add(
                        JobRun(
                            document_id=document_id,
                            stage=JobStage.NORMALIZE,
                            status=JobStatus.FAILED,
                            started_at=now,
                            finished_at=now,
                            error=str(exc)[:2000],
                        )
                    )
                    session.commit()
                    failed += 1

    return NormalizationSummary(
        pending=len(pending),
        processed=processed,
        failed=failed,
        no_pubtator_result=no_result,
        mentions_created=mentions_created,
    )


if __name__ == "__main__":
    norm_summary = run_pubtator_normalization()
    print("--- PubTator-Normalisierung ---")
    print(f"Ausstehend gewesen:        {norm_summary.pending}")
    print(f"Verarbeitet:               {norm_summary.processed}")
    print(f"  davon ohne PubTator-Ergebnis: {norm_summary.no_pubtator_result}")
    print(f"Fehlgeschlagen:            {norm_summary.failed}")
    print(f"Neue Mentions:             {norm_summary.mentions_created}")

    gene_summary = enrich_gene_entities()
    print("--- Gen-ID-Anreicherung (mygene.info) ---")
    print(f"Ausstehend gewesen: {gene_summary.pending}")
    print(f"Angereichert:       {gene_summary.updated}")
    print(f"Nicht gefunden:     {gene_summary.not_found}")
