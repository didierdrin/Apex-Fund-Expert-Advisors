"""Neon PostgreSQL storage for trading alerts."""

import json
import os
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS trading_alerts (
    id UUID PRIMARY KEY,
    symbol TEXT NOT NULL,
    type TEXT NOT NULL,
    signal TEXT,
    confidence TEXT,
    session TEXT,
    price DOUBLE PRECISION,
    zscore DOUBLE PRECISION,
    sma50 DOUBLE PRECISION,
    sma200 DOUBLE PRECISION,
    resistance DOUBLE PRECISION,
    support DOUBLE PRECISION,
    dist_to_resistance DOUBLE PRECISION,
    dist_to_support DOUBLE PRECISION,
    buffer_price DOUBLE PRECISION,
    htf_close DOUBLE PRECISION,
    htf_sma DOUBLE PRECISION,
    fib_15m DOUBLE PRECISION,
    fib_4h DOUBLE PRECISION,
    trend TEXT,
    timeframe TEXT,
    htf_timeframe TEXT,
    strategy TEXT,
    bar_time TIMESTAMPTZ,
    timestamp_ms BIGINT NOT NULL,
    timestamp_iso TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    payload JSONB,
    UNIQUE (symbol, type, signal, bar_time)
);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_timestamp_ms
    ON trading_alerts (timestamp_ms DESC);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_symbol
    ON trading_alerts (symbol);
CREATE INDEX IF NOT EXISTS idx_trading_alerts_strategy
    ON trading_alerts (strategy);
"""

MIGRATE_SQL = """
ALTER TABLE trading_alerts ADD COLUMN IF NOT EXISTS fib_15m DOUBLE PRECISION;
ALTER TABLE trading_alerts ADD COLUMN IF NOT EXISTS fib_4h DOUBLE PRECISION;
ALTER TABLE trading_alerts ADD COLUMN IF NOT EXISTS trend TEXT;
"""


def get_database_url():
    return os.environ.get("DATABASE_URL") or os.environ.get("NEON_DATABASE_URL")


@contextmanager
def get_conn():
    url = get_database_url()
    if not url:
        raise RuntimeError("DATABASE_URL (or NEON_DATABASE_URL) is not set")
    conn = psycopg2.connect(url)
    try:
        yield conn
    finally:
        conn.close()


def _migrate_unique_constraint(cur):
    """Allow multiple Pine marks per bar (15m + 4h + both) via (symbol, type, signal, bar_time)."""
    cur.execute(
        """
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'trading_alerts'::regclass
          AND contype = 'u'
        """
    )
    existing = {row[0] for row in cur.fetchall()}
    # Drop legacy unique that blocked multiple signal types on the same bar
    for name in list(existing):
        if name != "trading_alerts_symbol_type_signal_bar_time_key":
            # Only drop constraints that look like the old (symbol, type, bar_time) unique
            cur.execute(
                """
                SELECT array_agg(a.attname ORDER BY u.attposition)
                FROM pg_constraint c
                JOIN LATERAL unnest(c.conkey) WITH ORDINALITY AS u(attnum, attposition) ON true
                JOIN pg_attribute a ON a.attrelid = c.conrelid AND a.attnum = u.attnum
                WHERE c.conname = %s
                """,
                (name,),
            )
            cols = cur.fetchone()[0]
            if cols == ["symbol", "type", "bar_time"]:
                cur.execute(f'ALTER TABLE trading_alerts DROP CONSTRAINT IF EXISTS "{name}"')
    cur.execute(
        """
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'trading_alerts'::regclass
          AND contype = 'u'
          AND conname = 'trading_alerts_symbol_type_signal_bar_time_key'
        """
    )
    if not cur.fetchone():
        # Deduplicate before adding new unique (keep earliest row per key)
        cur.execute(
            """
            DELETE FROM trading_alerts a
            USING trading_alerts b
            WHERE a.ctid < b.ctid
              AND a.symbol = b.symbol
              AND a.type = b.type
              AND COALESCE(a.signal, '') = COALESCE(b.signal, '')
              AND a.bar_time IS NOT DISTINCT FROM b.bar_time
            """
        )
        cur.execute(
            """
            ALTER TABLE trading_alerts
            ADD CONSTRAINT trading_alerts_symbol_type_signal_bar_time_key
            UNIQUE (symbol, type, signal, bar_time)
            """
        )


def init_db():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA_SQL)
            cur.execute(MIGRATE_SQL)
        conn.commit()

    with get_conn() as conn:
        with conn.cursor() as cur:
            try:
                _migrate_unique_constraint(cur)
                conn.commit()
            except Exception as exc:
                conn.rollback()
                print(f"⚠️ Unique constraint migrate skipped/failed: {exc}")
    print("✅ Neon PostgreSQL schema ready (trading_alerts / fib_sma)")


def alert_exists(symbol, alert_type, bar_time_iso, signal=None):
    if not bar_time_iso:
        return False
    with get_conn() as conn:
        with conn.cursor() as cur:
            if signal is None:
                cur.execute(
                    """
                    SELECT 1 FROM trading_alerts
                    WHERE symbol = %s AND type = %s AND bar_time = %s
                    LIMIT 1
                    """,
                    (symbol, alert_type, bar_time_iso),
                )
            else:
                cur.execute(
                    """
                    SELECT 1 FROM trading_alerts
                    WHERE symbol = %s AND type = %s AND signal = %s AND bar_time = %s
                    LIMIT 1
                    """,
                    (symbol, alert_type, signal, bar_time_iso),
                )
            return cur.fetchone() is not None


def save_alert(alert_data):
    now_utc = datetime.now(timezone.utc)
    alert_id = str(uuid.uuid4())
    timestamp_ms = int(now_utc.timestamp() * 1000)
    timestamp_iso = now_utc.isoformat().replace("+00:00", "Z")

    row = dict(alert_data)
    row.setdefault("timestamp_ms", timestamp_ms)
    row.setdefault("timestamp_iso", timestamp_iso)
    row.setdefault("strategy", "fib_sma")

    known = {
        "symbol", "type", "signal", "confidence", "session", "price", "zscore",
        "sma50", "sma200", "resistance", "support", "dist_to_resistance",
        "dist_to_support", "buffer_price", "htf_close", "htf_sma",
        "fib_15m", "fib_4h", "trend",
        "timeframe", "htf_timeframe", "strategy", "bar_time", "timestamp_ms",
        "timestamp_iso",
    }
    payload = {k: v for k, v in row.items() if k not in known}

    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO trading_alerts (
                    id, symbol, type, signal, confidence, session, price, zscore,
                    sma50, sma200, resistance, support, dist_to_resistance,
                    dist_to_support, buffer_price, htf_close, htf_sma,
                    fib_15m, fib_4h, trend,
                    timeframe, htf_timeframe, strategy, bar_time, timestamp_ms,
                    timestamp_iso, payload
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s, %s,
                    %s, %s::jsonb
                )
                ON CONFLICT (symbol, type, signal, bar_time) DO NOTHING
                RETURNING id
                """,
                (
                    alert_id,
                    row.get("symbol"),
                    row.get("type"),
                    row.get("signal"),
                    row.get("confidence"),
                    row.get("session"),
                    row.get("price"),
                    row.get("zscore"),
                    row.get("sma50"),
                    row.get("sma200"),
                    row.get("resistance"),
                    row.get("support"),
                    row.get("dist_to_resistance"),
                    row.get("dist_to_support"),
                    row.get("buffer_price"),
                    row.get("htf_close"),
                    row.get("htf_sma"),
                    row.get("fib_15m"),
                    row.get("fib_4h"),
                    row.get("trend"),
                    row.get("timeframe"),
                    row.get("htf_timeframe"),
                    row.get("strategy"),
                    row.get("bar_time"),
                    row.get("timestamp_ms"),
                    row.get("timestamp_iso"),
                    json.dumps(payload, default=str),
                ),
            )
            inserted = cur.fetchone()
        conn.commit()

    if inserted:
        print(f"✅ Alert saved to Neon: {row.get('symbol')} — {row.get('type')} / {row.get('signal')}")
        return True
    print(f"ℹ️  Duplicate skipped: {row.get('symbol')} {row.get('type')} {row.get('signal')} @ {row.get('bar_time')}")
    return False


def fetch_alerts(limit=10):
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id::text, symbol, type, signal, confidence, session, price, zscore,
                       sma50, sma200, resistance, support, dist_to_resistance,
                       dist_to_support, buffer_price, htf_close, htf_sma,
                       fib_15m, fib_4h, trend,
                       timeframe, htf_timeframe, strategy, bar_time, timestamp_ms,
                       timestamp_iso, payload, created_at
                FROM trading_alerts
                ORDER BY timestamp_ms DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()

    alerts = []
    for row in rows:
        item = dict(row)
        if item.get("bar_time"):
            item["bar_time"] = item["bar_time"].isoformat().replace("+00:00", "Z")
        if item.get("created_at"):
            item["created_at"] = item["created_at"].isoformat().replace("+00:00", "Z")
        extra = item.pop("payload", None) or {}
        if isinstance(extra, str):
            extra = json.loads(extra)
        for k, v in extra.items():
            item.setdefault(k, v)
        alerts.append(item)
    return alerts
