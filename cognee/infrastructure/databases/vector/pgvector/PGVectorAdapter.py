import asyncio
import os
import re
from typing import List, Optional, get_type_hints
from time import perf_counter
from contextlib import contextmanager
from sqlalchemy.inspection import inspect
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.dialects.postgresql import insert, UUID as PG_UUID
from sqlalchemy import JSON, Column, Table, select, delete, MetaData, func, bindparam, Float, text
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from sqlalchemy.exc import ProgrammingError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from asyncpg import DeadlockDetectedError, DuplicateTableError, UniqueViolationError

from cognee.shared.logging_utils import get_logger
from cognee.infrastructure.engine import DataPoint
from cognee.infrastructure.engine.utils import parse_id
from cognee.infrastructure.databases.relational import get_relational_engine

from distributed.utils import override_distributed
from distributed.tasks.queued_add_data_points import queued_add_data_points
from cognee.infrastructure.databases.exceptions import MissingQueryParameterError

from ...relational.ModelBase import Base
from ...relational.sqlalchemy.SqlAlchemyAdapter import SQLAlchemyAdapter
from ..utils import normalize_distances
from ..models.ScoredResult import ScoredResult
from ..exceptions import CollectionNotFoundError
from ..vector_db_interface import VectorDBInterface
from ..embeddings.EmbeddingEngine import EmbeddingEngine
from .serialize_data import serialize_data

logger = get_logger("PGVectorAdapter")


@contextmanager
def _log_timing(operation: str, **fields):
    start = perf_counter()
    try:
        yield
    finally:
        duration_ms = (perf_counter() - start) * 1000
        field_str = " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
        logger.debug(f"[timing] {operation} {field_str} took {duration_ms:.2f} ms")


class IndexSchema(DataPoint):
    """
    Define a schema for indexing data points with a text field.

    This class inherits from the DataPoint class and specifies the structure of a single
    data point that includes a text attribute. It also includes a metadata field that
    indicates which fields should be indexed.
    """

    text: str

    metadata: dict = {"index_fields": ["text"]}


class PGVectorAdapter(SQLAlchemyAdapter, VectorDBInterface):
    def __init__(
        self,
        connection_string: str,
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
    ):
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.db_uri: str = connection_string
        self.VECTOR_DB_LOCK = asyncio.Lock()

        relational_db = get_relational_engine()

        # If postgreSQL is used we must use the same engine and sessionmaker
        if relational_db.engine.dialect.name == "postgresql":
            self.engine = relational_db.engine
            self.sessionmaker = relational_db.sessionmaker
        else:
            # If not create new instances of engine and sessionmaker
            self.engine = create_async_engine(self.db_uri)
            self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

        # Has to be imported at class level
        # Functions reading tables from database need to know what a Vector column type is
        from pgvector.sqlalchemy import Vector

        self.Vector = Vector
        
        logger.info("Original PGVectorAdapter initialized.")

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into vectors using the specified embedding engine.

        Parameters:
        -----------

            - data (list[str]): A list of strings to be embedded into vectors.

        Returns:
        --------

            - list[list[float]]: A list of lists of floats representing embedded vectors.
        """
        with _log_timing("embed_text", items=len(data)):
            return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a specified collection exists in the database.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: Returns True if the collection exists, False otherwise.
        """
        with _log_timing("has_collection", collection=collection_name):
            async with self.engine.begin() as connection:
                metadata = MetaData()
                await connection.run_sync(metadata.reflect)
                return collection_name in metadata.tables

    @retry(
        retry=retry_if_exception_type(
            (DuplicateTableError, UniqueViolationError, ProgrammingError)
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    async def create_collection(self, collection_name: str, payload_schema=None):
        data_point_types = get_type_hints(DataPoint)
        vector_size = self.embedding_engine.get_vector_size()

        if not await self.has_collection(collection_name):
            async with self.VECTOR_DB_LOCK:
                if not await self.has_collection(collection_name):

                    class PGVectorDataPoint(Base):
                        """
                        Represent a point in a vector data space with associated data and vector representation.

                        This class inherits from Base and is associated with a database table defined by
                        __tablename__. It maintains the following public methods and instance variables:

                        - __init__(self, id, payload, vector): Initializes a new PGVectorDataPoint instance.

                        Instance variables:
                        - id: Identifier for the data point, defined by data_point_types.
                        - payload: JSON data associated with the data point.
                        - vector: Vector representation of the data point, with size defined by vector_size.
                        """

                        __tablename__ = collection_name
                        __table_args__ = {"extend_existing": True}
                        # PGVector requires one column to be the primary key
                        id: Mapped[data_point_types["id"]] = mapped_column(primary_key=True)
                        payload = Column(JSON)
                        vector = Column(self.Vector(vector_size))

                        def __init__(self, id, payload, vector):
                            self.id = id
                            self.payload = payload
                            self.vector = vector

                    async with self.engine.begin() as connection:
                        if len(Base.metadata.tables.keys()) > 0:
                            with _log_timing("create_collection", collection=collection_name):
                                await connection.run_sync(
                                    Base.metadata.create_all, tables=[PGVectorDataPoint.__table__]
                                )

    @retry(
        retry=retry_if_exception_type(DeadlockDetectedError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    @override_distributed(queued_add_data_points)
    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        data_point_types = get_type_hints(DataPoint)
        if not await self.has_collection(collection_name):
            await self.create_collection(
                collection_name=collection_name,
                payload_schema=type(data_points[0]),
            )

        with _log_timing("embed_for_create", collection=collection_name, items=len(data_points)):
            data_vectors = await self.embed_data(
                [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
            )

        vector_size = self.embedding_engine.get_vector_size()

        class PGVectorDataPoint(Base):
            """
            Represents a data point in a PGVector database. This class maps to a table defined by
            the SQLAlchemy ORM.

            It contains the following public instance variables:
            - id: An identifier for the data point.
            - payload: A JSON object containing additional data related to the data point.
            - vector: A vector representation of the data point, configured to the specified size.
            """

            __tablename__ = collection_name
            __table_args__ = {"extend_existing": True}
            # PGVector requires one column to be the primary key
            id: Mapped[data_point_types["id"]] = mapped_column(primary_key=True)
            payload = Column(JSON)
            vector = Column(self.Vector(vector_size))

            def __init__(self, id, payload, vector):
                self.id = id
                self.payload = payload
                self.vector = vector

        async with self.get_async_session() as session:
            pgvector_data_points = []

            for data_index, data_point in enumerate(data_points):
                # Check to see if data should be updated or a new data item should be created
                # data_point_db = (
                #     await session.execute(
                #         select(PGVectorDataPoint).filter(PGVectorDataPoint.id == data_point.id)
                #     )
                # ).scalar_one_or_none()

                # If data point exists update it, if not create a new one
                # if data_point_db:
                #     data_point_db.id = data_point.id
                #     data_point_db.vector = data_vectors[data_index]
                #     data_point_db.payload = serialize_data(data_point.model_dump())
                #     pgvector_data_points.append(data_point_db)
                # else:
                pgvector_data_points.append(
                    PGVectorDataPoint(
                        id=data_point.id,
                        vector=data_vectors[data_index],
                        payload=serialize_data(data_point.model_dump()),
                    )
                )

            def to_dict(obj):
                return {
                    column.key: getattr(obj, column.key)
                    for column in inspect(obj).mapper.column_attrs
                }

            # session.add_all(pgvector_data_points)
            insert_statement = insert(PGVectorDataPoint).values(
                [to_dict(data_point) for data_point in pgvector_data_points]
            )
            insert_statement = insert_statement.on_conflict_do_nothing(index_elements=["id"])
            with _log_timing("insert_batch", collection=collection_name, items=len(pgvector_data_points)):
                await session.execute(insert_statement)
                await session.commit()

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=DataPoint.get_embeddable_data(data_point),
                )
                for data_point in data_points
            ],
        )

    async def get_table(self, collection_name: str) -> Table:
        """
        Dynamically loads a table using the given collection name
        with an async engine.
        """
        with _log_timing("get_table", collection=collection_name):
            async with self.engine.begin() as connection:
                metadata = MetaData()
                await connection.run_sync(metadata.reflect)
                if collection_name in metadata.tables:
                    return metadata.tables[collection_name]
                else:
                    raise CollectionNotFoundError(
                        f"Collection '{collection_name}' not found!",
                    )

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        with _log_timing("retrieve", collection=collection_name, items=len(data_point_ids)):
            async with self.get_async_session() as session:
                results = await session.execute(
                    select(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
                )
                results = results.all()

                return [
                    ScoredResult(id=parse_id(result.id), payload=result.payload, score=0)
                    for result in results
                ]

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
    ) -> List[ScoredResult]:
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        if limit is None:
            async with self.get_async_session() as session:
                query = select(func.count()).select_from(PGVectorDataPoint)
                result = await session.execute(query)
                limit = result.scalar_one()

        # If limit is still 0, no need to do the search, just return empty results
        if limit <= 0:
            return []

        # NOTE: This needs to be initialized in case search doesn't return a value
        closest_items = []

        with _log_timing("search_prepare", collection=collection_name, limit=limit):
            pass

        async with self.get_async_session() as session:
            query = select(
                PGVectorDataPoint,
                PGVectorDataPoint.c.vector.cosine_distance(query_vector).label("similarity"),
            ).order_by("similarity")

            if limit > 0:
                query = query.limit(limit)

            with _log_timing("search_query", collection=collection_name, limit=limit):
                closest_items = await session.execute(query)

        vector_list = []

        # Extract distances and find min/max for normalization
        for vector in closest_items.all():
            vector_list.append(
                {
                    "id": parse_id(str(vector.id)),
                    "payload": vector.payload,
                    "_distance": vector.similarity,
                }
            )

        if len(vector_list) == 0:
            return []

        # Normalize vector distance and add this as score information to vector_list
        normalized_values = normalize_distances(vector_list)
        for i in range(0, len(normalized_values)):
            vector_list[i]["score"] = normalized_values[i]

        # Create and return ScoredResult objects
        return [
            ScoredResult(id=row.get("id"), payload=row.get("payload"), score=row.get("score"))
            for row in vector_list
        ]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        with _log_timing("batch_search_embed", items=len(query_texts)):
            query_vectors = await self.embedding_engine.embed_text(query_texts)

        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                )
                for query_vector in query_vectors
            ]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        async with self.get_async_session() as session:
            PGVectorDataPoint = await self.get_table(collection_name)
            with _log_timing("delete_data_points", collection=collection_name, items=len(data_point_ids)):
                results = await session.execute(
                    delete(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
                )
                await session.commit()
                return results

    async def prune(self):
        # Clean up the database if it was set up as temporary
        await self.delete_database()


class CustomizedPGVectorAdapter(SQLAlchemyAdapter, VectorDBInterface):
    def __init__(
        self,
        connection_string: str,
        api_key: Optional[str],
        embedding_engine: EmbeddingEngine,
    ):
        self.api_key = api_key
        self.embedding_engine = embedding_engine
        self.db_uri: str = connection_string
        self.VECTOR_DB_LOCK = asyncio.Lock()

        relational_db = get_relational_engine()

        # If postgreSQL is used we must use the same engine and sessionmaker
        if relational_db.engine.dialect.name == "postgresql":
            self.engine = relational_db.engine
            self.sessionmaker = relational_db.sessionmaker
        else:
            # If not create new instances of engine and sessionmaker
            self.engine = create_async_engine(self.db_uri)
            self.sessionmaker = async_sessionmaker(bind=self.engine, expire_on_commit=False)

        # Has to be imported at class level
        # Functions reading tables from database need to know what a Vector column type is
        from pgvector.sqlalchemy import Vector

        self.Vector = Vector
        self._collection_models: dict[str, type[Base]] = {}
        self._existing_collections: Optional[set[str]] = None
        self._existing_collections_lock = asyncio.Lock()
        
        logger.info("Customized PGVectorAdapter initialized.")

    async def _get_existing_collections(self) -> set[str]:
        if self._existing_collections is not None:
            return self._existing_collections

        async with self._existing_collections_lock:
            if self._existing_collections is not None:
                return self._existing_collections

            if self.engine.dialect.name == "postgresql":
                async with self.engine.begin() as connection:
                    rows = await connection.execute(
                        text("SELECT tablename FROM pg_tables WHERE schemaname = current_schema()")
                    )
                    self._existing_collections = {r[0] for r in rows.fetchall()}
            else:
                async with self.engine.begin() as connection:
                    metadata = MetaData()
                    await connection.run_sync(metadata.reflect)
                    self._existing_collections = set(metadata.tables.keys())

        return self._existing_collections

    async def embed_data(self, data: list[str]) -> list[list[float]]:
        """
        Embed a list of texts into vectors using the specified embedding engine.

        Parameters:
        -----------

            - data (list[str]): A list of strings to be embedded into vectors.

        Returns:
        --------

            - list[list[float]]: A list of lists of floats representing embedded vectors.
        """
        with _log_timing("embed_text", items=len(data)):
            return await self.embedding_engine.embed_text(data)

    async def has_collection(self, collection_name: str) -> bool:
        """
        Check if a specified collection exists in the database.

        Parameters:
        -----------

            - collection_name (str): The name of the collection to check for existence.

        Returns:
        --------

            - bool: Returns True if the collection exists, False otherwise.
        """
        with _log_timing("has_collection", collection=collection_name):
            existing = await self._get_existing_collections()
            return collection_name in existing

    @retry(
        retry=retry_if_exception_type(
            (DuplicateTableError, UniqueViolationError, ProgrammingError)
        ),
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    async def create_collection(self, collection_name: str, payload_schema=None):
        data_point_types = get_type_hints(DataPoint)
        vector_size = self.embedding_engine.get_vector_size()

        if not await self.has_collection(collection_name):
            async with self.VECTOR_DB_LOCK:
                if not await self.has_collection(collection_name):

                    class PGVectorDataPoint(Base):
                        """
                        Represent a point in a vector data space with associated data and vector representation.

                        This class inherits from Base and is associated with a database table defined by
                        __tablename__. It maintains the following public methods and instance variables:

                        - __init__(self, id, payload, vector): Initializes a new PGVectorDataPoint instance.

                        Instance variables:
                        - id: Identifier for the data point, defined by data_point_types.
                        - payload: JSON data associated with the data point.
                        - vector: Vector representation of the data point, with size defined by vector_size.
                        """

                        __tablename__ = collection_name
                        __table_args__ = {"extend_existing": True}
                        # PGVector requires one column to be the primary key
                        id: Mapped[data_point_types["id"]] = mapped_column(primary_key=True)
                        payload = Column(JSON)
                        vector = Column(self.Vector(vector_size))

                        def __init__(self, id, payload, vector):
                            self.id = id
                            self.payload = payload
                            self.vector = vector

                    async with self.engine.begin() as connection:
                        if len(Base.metadata.tables.keys()) > 0:
                            with _log_timing(
                                "create_collection", collection=collection_name, adapter="customized"
                            ):
                                await connection.run_sync(
                                    Base.metadata.create_all, tables=[PGVectorDataPoint.__table__]
                                )
                    # Keep the in-memory cache consistent for the current process.
                    if self._existing_collections is not None:
                        self._existing_collections.add(collection_name)

    @retry(
        retry=retry_if_exception_type(DeadlockDetectedError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=1, max=6),
    )
    @override_distributed(queued_add_data_points)
    async def create_data_points(self, collection_name: str, data_points: List[DataPoint]):
        data_point_types = get_type_hints(DataPoint)
        if not await self.has_collection(collection_name):
            await self.create_collection(
                collection_name=collection_name,
                payload_schema=type(data_points[0]),
            )

        with _log_timing("embed_for_create", collection=collection_name, items=len(data_points)):
            data_vectors = await self.embed_data(
                [DataPoint.get_embeddable_data(data_point) for data_point in data_points]
            )

        vector_size = self.embedding_engine.get_vector_size()

        class PGVectorDataPoint(Base):
            """
            Represents a data point in a PGVector database. This class maps to a table defined by
            the SQLAlchemy ORM.

            It contains the following public instance variables:
            - id: An identifier for the data point.
            - payload: A JSON object containing additional data related to the data point.
            - vector: A vector representation of the data point, configured to the specified size.
            """

            __tablename__ = collection_name
            __table_args__ = {"extend_existing": True}
            # PGVector requires one column to be the primary key
            id: Mapped[data_point_types["id"]] = mapped_column(primary_key=True)
            payload = Column(JSON)
            vector = Column(self.Vector(vector_size))

            def __init__(self, id, payload, vector):
                self.id = id
                self.payload = payload
                self.vector = vector

        async with self.get_async_session() as session:
            pgvector_data_points = []

            for data_index, data_point in enumerate(data_points):
                # Check to see if data should be updated or a new data item should be created
                # data_point_db = (
                #     await session.execute(
                #         select(PGVectorDataPoint).filter(PGVectorDataPoint.id == data_point.id)
                #     )
                # ).scalar_one_or_none()

                # If data point exists update it, if not create a new one
                # if data_point_db:
                #     data_point_db.id = data_point.id
                #     data_point_db.vector = data_vectors[data_index]
                #     data_point_db.payload = serialize_data(data_point.model_dump())
                #     pgvector_data_points.append(data_point_db)
                # else:
                pgvector_data_points.append(
                    PGVectorDataPoint(
                        id=data_point.id,
                        vector=data_vectors[data_index],
                        payload=serialize_data(data_point.model_dump()),
                    )
                )

            def to_dict(obj):
                return {
                    column.key: getattr(obj, column.key)
                    for column in inspect(obj).mapper.column_attrs
                }

            # session.add_all(pgvector_data_points)
            insert_statement = insert(PGVectorDataPoint).values(
                [to_dict(data_point) for data_point in pgvector_data_points]
            )
            insert_statement = insert_statement.on_conflict_do_nothing(index_elements=["id"])
            with _log_timing(
                "insert_batch", collection=collection_name, items=len(pgvector_data_points)
            ):
                await session.execute(insert_statement)
                await session.commit()

    async def create_vector_index(self, index_name: str, index_property_name: str):
        await self.create_collection(f"{index_name}_{index_property_name}")

    async def index_data_points(
        self, index_name: str, index_property_name: str, data_points: list[DataPoint]
    ):
        await self.create_data_points(
            f"{index_name}_{index_property_name}",
            [
                IndexSchema(
                    id=data_point.id,
                    text=DataPoint.get_embeddable_data(data_point),
                )
                for data_point in data_points
            ],
        )

    async def get_table(self, collection_name: str) -> Table:
        """
        Dynamically loads a table using the given collection name
        with an async engine.
        """
        with _log_timing("get_table", collection=collection_name):
            async with self.engine.begin() as connection:
                metadata = MetaData()
                await connection.run_sync(metadata.reflect)
                if collection_name in metadata.tables:
                    return metadata.tables[collection_name]
                else:
                    raise CollectionNotFoundError(
                        f"Collection '{collection_name}' not found!",
                    )

    async def retrieve(self, collection_name: str, data_point_ids: List[str]):
        # Get PGVectorDataPoint Table from database
        PGVectorDataPoint = await self.get_table(collection_name)

        with _log_timing("retrieve", collection=collection_name, items=len(data_point_ids)):
            async with self.get_async_session() as session:
                results = await session.execute(
                    select(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
                )
                results = results.all()

                return [
                    ScoredResult(id=parse_id(result.id), payload=result.payload, score=0)
                    for result in results
                ]

    async def search(
        self,
        collection_name: str,
        query_text: Optional[str] = None,
        query_vector: Optional[List[float]] = None,
        limit: Optional[int] = 15,
        with_vector: bool = False,
    ) -> List[ScoredResult]:
        if query_text is None and query_vector is None:
            raise MissingQueryParameterError()

        if query_text and not query_vector:
            query_vector = (await self.embedding_engine.embed_text([query_text]))[0]

        from uuid import UUID

        # Fast-path: avoid doing work for missing collections.
        if not await self.has_collection(collection_name):
            return []

        # NOTE: This needs to be initialized in case search doesn't return a value
        closest_items = []

        vector_size = self.embedding_engine.get_vector_size()
        if query_vector is not None and len(query_vector) != vector_size:
            raise ValueError(
                f"Query vector dim {len(query_vector)} does not match embedding dim {vector_size}."
            )

        # Build a lightweight ORM model once per collection to avoid repeated class re-definition
        # (which causes SAWarning spam) and to avoid table reflection overhead.
        PGVectorDataPoint = self._collection_models.get(collection_name)
        if PGVectorDataPoint is None:
            safe_name = re.sub(r"[^0-9A-Za-z_]", "_", collection_name)
            class_name = f"PGVectorDataPoint_{safe_name}"
            PGVectorDataPoint = type(
                class_name,
                (Base,),
                {
                    "__tablename__": collection_name,
                    "__table_args__": {"extend_existing": True},
                    "__annotations__": {"id": Mapped[UUID]},
                    "id": mapped_column(PG_UUID(as_uuid=True), primary_key=True),
                    "payload": Column(JSON),
                    "vector": Column(self.Vector(vector_size)),
                },
            )
            self._collection_models[collection_name] = PGVectorDataPoint

        # Use async session to connect to the database
        with _log_timing("search_prepare", collection=collection_name, limit=limit):
            pass

        async with self.get_async_session() as session:
            # Optional per-session Postgres tuning for vector search.
            # Keep this behind env vars so it can be tested safely in production-like runs.
            work_mem = os.getenv("COGNEE_PGVECTOR_WORK_MEM")
            eff_io = os.getenv("COGNEE_PGVECTOR_EFFECTIVE_IO_CONCURRENCY")
            if work_mem:
                await session.execute(text(f"SET LOCAL work_mem = '{work_mem}';"))
            if eff_io:
                try:
                    eff_io_int = int(eff_io)
                except ValueError:
                    eff_io_int = None
                if eff_io_int is not None:
                    await session.execute(
                        text(f"SET LOCAL effective_io_concurrency = {eff_io_int};")
                    )

            # We only support 1536-dim (text-embedding-3-small) in this deployment.
            # Keep native `vector` so cosine HNSW indexes on `vector_cosine_ops` can be used.
            query_expr = bindparam("qvec", query_vector, type_=self.Vector(vector_size))
            vector_expr = PGVectorDataPoint.vector

            # Force result type to float, otherwise SQLAlchemy may apply the pgvector
            # result processor to the distance column (and crash when it receives a float).
            similarity_expr = func.cast(vector_expr.op("<=>")(query_expr), Float).label("similarity")

            query = (
                select(
                    PGVectorDataPoint.id,
                    PGVectorDataPoint.payload,
                    similarity_expr,
                )
                .where(PGVectorDataPoint.vector.is_not(None))
                .order_by(similarity_expr)
            )

            if limit is not None and limit > 0:
                query = query.limit(limit)

            # Find closest vectors to query_vector
            try:
                with _log_timing("search_query", collection=collection_name, limit=limit):
                    closest_items = await session.execute(query)
            except ProgrammingError as e:
                if "does not exist" in str(e):
                    # Table does not exist, return empty results
                    # This mimics the behavior of filtered search_in_collection handling CollectionNotFoundError
                    # but doing it here prevents the crash without needing overhead of checking existence first
                    return []
                raise e

        vector_list = []

        # Extract distances and find min/max for normalization
        for vector in closest_items.all():
            vector_list.append(
                {
                    "id": parse_id(str(vector.id)),
                    "payload": vector.payload,
                    "_distance": vector.similarity,
                }
            )

        if len(vector_list) == 0:
            return []

        # Normalize vector distance and add this as score information to vector_list
        normalized_values = normalize_distances(vector_list)
        for i in range(0, len(normalized_values)):
            vector_list[i]["score"] = normalized_values[i]

        # Create and return ScoredResult objects
        return [
            ScoredResult(id=row.get("id"), payload=row.get("payload"), score=row.get("score"))
            for row in vector_list
        ]

    async def batch_search(
        self,
        collection_name: str,
        query_texts: List[str],
        limit: int = None,
        with_vectors: bool = False,
    ):
        with _log_timing("batch_search_embed", items=len(query_texts)):
            query_vectors = await self.embedding_engine.embed_text(query_texts)

        return await asyncio.gather(
            *[
                self.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    limit=limit,
                    with_vector=with_vectors,
                )
                for query_vector in query_vectors
            ]
        )

    async def delete_data_points(self, collection_name: str, data_point_ids: list[str]):
        async with self.get_async_session() as session:
            # Get PGVectorDataPoint Table from database
            PGVectorDataPoint = await self.get_table(collection_name)
            with _log_timing(
                "delete_data_points", collection=collection_name, items=len(data_point_ids)
            ):
                results = await session.execute(
                    delete(PGVectorDataPoint).where(PGVectorDataPoint.c.id.in_(data_point_ids))
                )
                await session.commit()
                return results

    async def prune(self):
        # Clean up the database if it was set up as temporary
        await self.delete_database()
