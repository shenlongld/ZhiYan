"""
SQLModel 数据模型与 upsert 逻辑。
"""
from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from sqlmodel import JSON, Column, Field, Session, SQLModel, create_engine, select


class TrendStatus(str, Enum):
    EXPAND = "扩招"
    SHRINK = "缩招"
    STABLE = "持平"


class SourcePlatform(str, Enum):
    ZHIHU = "知乎"
    XIAOHONGSHU = "小红书"
    FORUM = "保研论坛"
    OTHER = "其他"


class OfficialPolicy(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    school_college: str = Field(index=True)
    announcement_title: str
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    admission_threshold: Dict[str, Any] = Field(
        default_factory=dict, sa_column=Column(JSON)
    )
    official_url: str = Field(unique=True, index=True)
    data_source: str = Field(default="学校官网")
    update_time: datetime = Field(default_factory=datetime.utcnow, index=True)

    @classmethod
    def upsert(cls, session: Session, data: Dict[str, Any]) -> "OfficialPolicy":
        existing = session.exec(select(cls).where(cls.official_url == data["official_url"])).first()
        now = datetime.utcnow()
        if existing:
            existing.school_college = data.get("school_college", existing.school_college)
            existing.announcement_title = data.get("announcement_title", existing.announcement_title)
            existing.start_date = data.get("start_date")
            existing.end_date = data.get("end_date")
            existing.admission_threshold = data.get("admission_threshold", {})
            existing.data_source = data.get("data_source", existing.data_source)
            existing.update_time = now
            obj = existing
        else:
            obj = cls(**data, update_time=now)
            session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj


class QuotaData(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    school_name: str = Field(index=True)
    dept_name: str = Field(index=True)
    major_full_name: str = Field(index=True)
    current_year_quota: Optional[int] = None
    prev_year_quota: Optional[int] = None
    quota_change: Optional[int] = None
    trend_status: TrendStatus = Field(default=TrendStatus.STABLE, index=True)
    is_full_time: bool = Field(default=True, index=True)
    last_sync_date: date = Field(default_factory=date.today, index=True)

    @staticmethod
    def _derive(current: Optional[int], prev: Optional[int]) -> tuple[Optional[int], TrendStatus]:
        if current is None or prev is None:
            return None, TrendStatus.STABLE
        change = current - prev
        if change > 0:
            return change, TrendStatus.EXPAND
        if change < 0:
            return change, TrendStatus.SHRINK
        return 0, TrendStatus.STABLE

    @classmethod
    def upsert(cls, session: Session, data: Dict[str, Any]) -> "QuotaData":
        stmt = select(cls).where(
            cls.school_name == data["school_name"],
            cls.dept_name == data["dept_name"],
            cls.major_full_name == data["major_full_name"],
            cls.is_full_time == data.get("is_full_time", True),
        )
        existing = session.exec(stmt).first()
        change, trend = cls._derive(
            data.get("current_year_quota"), data.get("prev_year_quota")
        )
        payload = dict(data)
        payload["quota_change"] = change
        payload["trend_status"] = trend
        payload["last_sync_date"] = data.get("last_sync_date") or date.today()

        if existing:
            for k, v in payload.items():
                setattr(existing, k, v)
            obj = existing
        else:
            obj = cls(**payload)
            session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj


class ExperienceArchive(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    school_major: str = Field(index=True)
    source_platform: SourcePlatform = Field(index=True)
    assessment_flow: str
    question_bank: Dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
    mentor_tags: List[str] = Field(default_factory=list, sa_column=Column(JSON))
    experience_year: Optional[int] = Field(default=None, index=True)
    original_post_url: str = Field(unique=True, index=True)
    summary_digest: str = ""

    @classmethod
    def upsert(cls, session: Session, data: Dict[str, Any]) -> "ExperienceArchive":
        existing = session.exec(
            select(cls).where(cls.original_post_url == data["original_post_url"])
        ).first()
        if existing:
            existing.school_major = data.get("school_major", existing.school_major)
            existing.source_platform = data.get("source_platform", existing.source_platform)
            existing.assessment_flow = data.get("assessment_flow", existing.assessment_flow)
            existing.question_bank = data.get("question_bank", {})
            existing.mentor_tags = data.get("mentor_tags", [])
            existing.experience_year = data.get("experience_year")
            existing.summary_digest = data.get("summary_digest", "")
            obj = existing
        else:
            obj = cls(**data)
            session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj


def create_db_and_tables(database_url: str = "sqlite:///./postgrad_agent.db"):
    engine = create_engine(database_url, echo=False)
    SQLModel.metadata.create_all(engine)
    return engine
