# db_async.py
"""
ASYNC SQLAlchemy + asyncpg
- создаёт таблицы
- содержит async CRUD
- готов для интеграции с нейронкой
"""

import os
from datetime import datetime
from typing import Optional, List

from sqlalchemy import (
    Column, Integer, String, Float, TIMESTAMP,
    ForeignKey
)
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession
)
from sqlalchemy.orm import (
    declarative_base, sessionmaker, relationship
)

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/postgres"
)

engine = create_async_engine(DATABASE_URL, echo=False, future=True)

AsyncSessionLocal = sessionmaker(
    engine, class_=AsyncSession, expire_on_commit=False, autoflush=False
)

Base = declarative_base()

# -------------------------------------------------------
# MODELS
# -------------------------------------------------------

class Train(Base):
    __tablename__ = "train"
    id = Column(Integer, primary_key=True)
    number = Column(String(50), nullable=False)
    arrival_time = Column(TIMESTAMP, nullable=True)
    departure_time = Column(TIMESTAMP, nullable=True)

    seconds = relationship("Second", back_populates="train", cascade="all, delete")


class Second(Base):
    __tablename__ = "seconds"
    id = Column(Integer, primary_key=True)
    people_count = Column(Integer, default=0)
    active_people_count = Column(Integer, default=0)
    activity_index = Column(Float, default=0.0)

    train_id = Column(Integer, ForeignKey("train.id", ondelete="CASCADE"), nullable=False)
    train = relationship("Train", back_populates="seconds")

    seconds_people = relationship("SecondsPeople", back_populates="second", cascade="all, delete")


class People(Base):
    __tablename__ = "people"
    id = Column(Integer, primary_key=True)
    worker_type = Column(String(100), nullable=False)

    seconds_people = relationship("SecondsPeople", back_populates="person", cascade="all, delete")


class SecondsPeople(Base):
    __tablename__ = "seconds_people"
    id = Column(Integer, primary_key=True)

    person_id = Column(Integer, ForeignKey("people.id", ondelete="CASCADE"), nullable=False)
    second_id = Column(Integer, ForeignKey("seconds.id", ondelete="CASCADE"), nullable=False)

    status = Column(String(50), nullable=True)

    person = relationship("People", back_populates="seconds_people")
    second = relationship("Second", back_populates="seconds_people")

# -------------------------------------------------------
# CREATE TABLES
# -------------------------------------------------------

async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("ASYNC: Tables created.")

# -------------------------------------------------------
# CRUD: TRAIN
# -------------------------------------------------------

async def create_train(db: AsyncSession, number: str,
                       arrival_time: Optional[datetime] = None,
                       departure_time: Optional[datetime] = None):
    t = Train(
        number=number,
        arrival_time=arrival_time,
        departure_time=departure_time
    )
    db.add(t)
    await db.commit()
    await db.refresh(t)
    return t


async def get_train(db: AsyncSession, train_id: int):
    return await db.get(Train, train_id)


async def update_train(db: AsyncSession, train_id: int, **fields):
    t = await get_train(db, train_id)
    if not t:
        return None
    for k, v in fields.items():
        if hasattr(t, k):
            setattr(t, k, v)
    await db.commit()
    await db.refresh(t)
    return t


async def delete_train(db: AsyncSession, train_id: int):
    t = await get_train(db, train_id)
    if not t:
        return False
    await db.delete(t)
    await db.commit()
    return True

# -------------------------------------------------------
# CRUD: SECOND
# -------------------------------------------------------

async def create_second(db: AsyncSession, train_id: int,
                        people_count=0, active_people_count=0,
                        activity_index=0.0):
    s = Second(
        train_id=train_id,
        people_count=people_count,
        active_people_count=active_people_count,
        activity_index=activity_index
    )
    db.add(s)
    await db.commit()
    await db.refresh(s)
    return s


async def get_second(db: AsyncSession, second_id: int):
    return await db.get(Second, second_id)


async def update_second(db: AsyncSession, second_id: int, **fields):
    s = await get_second(db, second_id)
    if not s:
        return None
    for k, v in fields.items():
        if hasattr(s, k):
            setattr(s, k, v)
    await db.commit()
    await db.refresh(s)
    return s


async def delete_second(db: AsyncSession, second_id: int):
    s = await get_second(db, second_id)
    if not s:
        return False
    await db.delete(s)
    await db.commit()
    return True

# -------------------------------------------------------
# CRUD: PEOPLE
# -------------------------------------------------------

async def create_person(db: AsyncSession, worker_type: str):
    p = People(worker_type=worker_type)
    db.add(p)
    await db.commit()
    await db.refresh(p)
    return p


async def get_person(db: AsyncSession, person_id: int):
    return await db.get(People, person_id)


async def update_person(db: AsyncSession, person_id: int, **fields):
    p = await get_person(db, person_id)
    if not p:
        return None
    for k, v in fields.items():
        if hasattr(p, k):
            setattr(p, k, v)
    await db.commit()
    await db.refresh(p)
    return p


async def delete_person(db: AsyncSession, person_id: int):
    p = await get_person(db, person_id)
    if not p:
        return False
    await db.delete(p)
    await db.commit()
    return True

# -------------------------------------------------------
# CRUD: SECONDS_PEOPLE
# -------------------------------------------------------

async def create_seconds_people(db: AsyncSession, person_id: int, second_id: int, status: Optional[str] = None):
    sp = SecondsPeople(person_id=person_id, second_id=second_id, status=status)
    db.add(sp)
    await db.commit()
    await db.refresh(sp)
    return sp


async def get_seconds_people(db: AsyncSession, sp_id: int):
    return await db.get(SecondsPeople, sp_id)


async def update_seconds_people(db: AsyncSession, sp_id: int, **fields):
    sp = await get_seconds_people(db, sp_id)
    if not sp:
        return None
    for k, v in fields.items():
        if hasattr(sp, k):
            setattr(sp, k, v)
    await db.commit()
    await db.refresh(sp)
    return sp


async def delete_seconds_people(db: AsyncSession, sp_id: int):
    sp = await get_seconds_people(db, sp_id)
    if not sp:
        return False
    await db.delete(sp)
    await db.commit()
    return True

# -------------------------------------------------------
# FUNCTION for your AI pipeline
# -------------------------------------------------------

async def record_frame_activity(
    db: AsyncSession,
    train_number: str,
    people_info: List[dict],
    activity_index: float
):
    """
    Главная функция для интеграции с нейронкой:
    - найдёт поезд или создаст
    - создаст Second
    - создаст/найдёт людей
    - создаст SecondsPeople
    """

    # найти поезд
    from sqlalchemy.future import select

    result = await db.execute(
        select(Train).where(Train.number == train_number)
    )
    train = result.scalar_one_or_none()

    if not train:
        train = await create_train(db, number=train_number)

    second = await create_second(
        db,
        train_id=train.id,
        people_count=len(people_info),
        active_people_count=sum(1 for p in people_info if p.get("status") == "active"),
        activity_index=activity_index
    )

    # люди
    for p in people_info:
        worker_type = p.get("worker_type", "unknown")

        result = await db.execute(
            select(People).where(People.worker_type == worker_type)
        )
        person = result.scalar_one_or_none()

        if not person:
            person = await create_person(db, worker_type)

        await create_seconds_people(
            db,
            person_id=person.id,
            second_id=second.id,
            status=p.get("status")
        )

    return second

# -------------------------------------------------------
# TEST
# -------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def main():
        await create_tables()
        async with AsyncSessionLocal() as db:

            sec = await record_frame_activity(
                db,
                train_number="Train-001",
                activity_index=0.88,
                people_info=[
                    {"worker_type": "engineer", "status": "active"},
                    {"worker_type": "operator", "status": "idle"}
                ]
            )

            print("Second created:", sec.id)

    asyncio.run(main())
