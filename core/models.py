import enum
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Enum, ForeignKey, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.db import Base

EMBEDDING_DIM = 384  # muss zum verwendeten sentence-transformers-Modell passen


def _enum_column(enum_cls: type[enum.Enum]) -> Enum:
    """Speichert enum.value (z.B. "ingest") statt enum.name (z.B. "INGEST") --
    passend zur durchgaengig kleingeschriebenen Terminologie in CLAUDE.md."""
    return Enum(enum_cls, values_callable=lambda cls: [member.value for member in cls])


class EntityType(str, enum.Enum):
    GENE = "gene"
    PROTEIN = "protein"
    CELLTYPE = "celltype"
    PATHWAY = "pathway"
    FACTOR = "factor"  # Wachstumsfaktor / kleines Molekuel in Protokollen
    CHEMICAL = "chemical"  # PubTator3-Chemical-Annotationen (MESH-ID)
    CELL_LINE = "cell_line"  # PubTator3-CellLine-Annotationen (Cellosaurus), keine Cell-Ontology-Zelltypen
    SPECIES = "species"  # PubTator3-Species-Annotationen (NCBI-Taxonomy-ID)


class MentionSource(str, enum.Enum):
    PUBTATOR = "pubtator"
    LLM = "llm"
    MANUAL = "manual"


class ScoreComponent(str, enum.Enum):
    LITERATURE = "literature"
    REGULON = "regulon"
    CENTRALITY = "centrality"
    EXPRESSION = "expression"
    PERTURBATION = "perturbation"


class JobStage(str, enum.Enum):
    INGEST = "ingest"
    NORMALIZE = "normalize"
    EXTRACT = "extract"
    ENRICH = "enrich"
    SCORE = "score"


class JobStatus(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


class Entity(Base):
    """Kanonisches Gen/Protein/Zelltyp/Pathway -- nie ein roher String.

    canonical_id z.B. Entrez 6657, UniProt P48431, Cell Ontology CL:0000034.
    """

    __tablename__ = "entity"

    id: Mapped[int] = mapped_column(primary_key=True)
    type: Mapped[EntityType] = mapped_column(_enum_column(EntityType))
    canonical_id: Mapped[str] = mapped_column(index=True)
    symbol: Mapped[str | None] = mapped_column(index=True)
    name: Mapped[str | None]
    ensembl_id: Mapped[str | None]  # nur fuer type=GENE, via mygene.info
    uniprot_id: Mapped[str | None]  # nur fuer type=GENE, via mygene.info

    __table_args__ = (UniqueConstraint("type", "canonical_id"),)


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    pmid: Mapped[str | None] = mapped_column(unique=True, index=True)
    doi: Mapped[str | None]
    title: Mapped[str] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str]  # z.B. europepmc, pubmed, biorxiv, upload
    pub_year: Mapped[int | None]
    is_open_access: Mapped[bool] = mapped_column(default=False)
    full_text_url: Mapped[str | None]
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())

    sections: Mapped[list["Section"]] = relationship(back_populates="document")


class Section(Base):
    __tablename__ = "sections"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    name: Mapped[str]  # z.B. Abstract, Methods, Results
    order_index: Mapped[int]
    text: Mapped[str] = mapped_column(Text)

    document: Mapped["Document"] = relationship(back_populates="sections")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    section_id: Mapped[int | None] = mapped_column(ForeignKey("sections.id"))
    text: Mapped[str] = mapped_column(Text)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM))


class Mention(Base):
    """Eine konkrete Erwaehnung einer Entity an einer Textstelle (Provenienz)."""

    __tablename__ = "mention"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    section_id: Mapped[int | None] = mapped_column(ForeignKey("sections.id"))
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id"))
    span_start: Mapped[int]
    span_end: Mapped[int]
    surface_form: Mapped[str]  # der tatsaechlich im Text gefundene String
    source: Mapped[MentionSource] = mapped_column(_enum_column(MentionSource))
    confidence: Mapped[float | None]


class Protocol(Base):
    __tablename__ = "protocols"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    source_celltype_id: Mapped[int | None] = mapped_column(ForeignKey("entity.id"))
    target_celltype_id: Mapped[int] = mapped_column(ForeignKey("entity.id"))
    efficiency: Mapped[float | None]
    efficiency_marker: Mapped[str | None]  # z.B. "C-Peptid+"
    notes: Mapped[str | None] = mapped_column(Text)

    steps: Mapped[list["ProtocolStep"]] = relationship(back_populates="protocol")
    markers: Mapped[list["ProtocolMarker"]] = relationship(back_populates="protocol")


class ProtocolStep(Base):
    __tablename__ = "protocol_steps"

    id: Mapped[int] = mapped_column(primary_key=True)
    protocol_id: Mapped[int] = mapped_column(ForeignKey("protocols.id"))
    order_index: Mapped[int]
    day_start: Mapped[int | None]
    day_end: Mapped[int | None]
    description: Mapped[str | None] = mapped_column(Text)

    protocol: Mapped["Protocol"] = relationship(back_populates="steps")
    factors: Mapped[list["ProtocolFactor"]] = relationship(back_populates="step")


class ProtocolFactor(Base):
    __tablename__ = "protocol_factors"

    id: Mapped[int] = mapped_column(primary_key=True)
    protocol_step_id: Mapped[int] = mapped_column(ForeignKey("protocol_steps.id"))
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id"))  # z.B. Activin A
    concentration: Mapped[float | None]
    concentration_unit: Mapped[str | None]  # z.B. "ng/mL"

    step: Mapped["ProtocolStep"] = relationship(back_populates="factors")


class ProtocolMarker(Base):
    """Ein im Protokoll gemessener Marker (Prüfgen), z.B. PDX1 an Tag 13."""

    __tablename__ = "protocol_markers"

    id: Mapped[int] = mapped_column(primary_key=True)
    protocol_id: Mapped[int] = mapped_column(ForeignKey("protocols.id"))
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id"))
    measured_value: Mapped[float | None]
    measurement_type: Mapped[str | None]  # z.B. "flow cytometry", "qPCR"
    timepoint_day: Mapped[int | None]

    protocol: Mapped["Protocol"] = relationship(back_populates="markers")


class GeneScore(Base):
    """Score-Komponenten einzeln, nie den Endwert -- Gewichtung passiert bei der Abfrage."""

    __tablename__ = "gene_score"

    id: Mapped[int] = mapped_column(primary_key=True)
    entity_id: Mapped[int] = mapped_column(ForeignKey("entity.id"), index=True)
    target_celltype_id: Mapped[int] = mapped_column(ForeignKey("entity.id"), index=True)
    component: Mapped[ScoreComponent] = mapped_column(_enum_column(ScoreComponent))
    value: Mapped[float]
    computed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class JobRun(Base):
    """Ein Statuseintrag pro Dokument und Pipeline-Stufe -- macht jede Stufe idempotent."""

    __tablename__ = "job_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    stage: Mapped[JobStage] = mapped_column(_enum_column(JobStage))
    status: Mapped[JobStatus] = mapped_column(_enum_column(JobStatus), default=JobStatus.PENDING)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    error: Mapped[str | None] = mapped_column(Text)

    __table_args__ = (UniqueConstraint("document_id", "stage"),)
