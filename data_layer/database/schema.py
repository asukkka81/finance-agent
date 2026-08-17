# data_layer/database/schema.py
"""数据库 Schema 定义 — DDL 建表语句 & 迁移版本."""

SCHEMA_VERSION = 1

ALL_TABLES = [
    "stocks",
    "daily_prices",
    "funds",
    "fund_nav",
    "financial_indicators",
    "macro_indicators",
    "fetch_logs",
]

# ============================================
# DDL 建表语句
# ============================================

CREATE_TABLE_STATEMENTS: dict[str, str] = {
    # ------------------------------------------
    # 1. 股票元数据表
    # ------------------------------------------
    "stocks": """
        CREATE TABLE IF NOT EXISTS stocks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol      TEXT    NOT NULL,
            name        TEXT,
            exchange    TEXT,
            market      TEXT    NOT NULL CHECK(market IN ('US', 'CN')),
            sector      TEXT,
            industry    TEXT,
            currency    TEXT    DEFAULT 'USD',
            is_active   INTEGER DEFAULT 1,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """,

    "idx_stocks_symbol": """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_stocks_symbol ON stocks(symbol);
    """,

    "idx_stocks_market": """
        CREATE INDEX IF NOT EXISTS idx_stocks_market ON stocks(market);
    """,

    # ------------------------------------------
    # 2. 日线行情表 (核心表)
    # ------------------------------------------
    "daily_prices": """
        CREATE TABLE IF NOT EXISTS daily_prices (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id        INTEGER NOT NULL,
            trade_date      DATE    NOT NULL,
            open            REAL,
            high            REAL,
            low             REAL,
            close           REAL,
            volume          REAL,
            adj_close       REAL,
            pre_close       REAL,
            change_pct      REAL,
            turnover_rate   REAL,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        );
    """,

    "idx_daily_prices_unique": """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_daily_prices_unique
            ON daily_prices(stock_id, trade_date);
    """,

    "idx_daily_prices_date": """
        CREATE INDEX IF NOT EXISTS idx_daily_prices_date
            ON daily_prices(trade_date);
    """,

    # ------------------------------------------
    # 3. 基金元数据表
    # ------------------------------------------
    "funds": """
        CREATE TABLE IF NOT EXISTS funds (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            code        TEXT    NOT NULL,
            name        TEXT,
            fund_type   TEXT,
            manager     TEXT,
            company     TEXT,
            is_active   INTEGER DEFAULT 1,
            created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
            updated_at  DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """,

    "idx_funds_code": """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_funds_code ON funds(code);
    """,

    # ------------------------------------------
    # 4. 基金净值表
    # ------------------------------------------
    "fund_nav": """
        CREATE TABLE IF NOT EXISTS fund_nav (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            fund_id         INTEGER NOT NULL,
            nav_date        DATE    NOT NULL,
            unit_nav        REAL,
            accumulated_nav REAL,
            daily_return    REAL,
            subscription    TEXT,
            redemption      TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (fund_id) REFERENCES funds(id)
        );
    """,

    "idx_fund_nav_unique": """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fund_nav_unique
            ON fund_nav(fund_id, nav_date);
    """,

    # ------------------------------------------
    # 5. 财务指标表
    # ------------------------------------------
    "financial_indicators": """
        CREATE TABLE IF NOT EXISTS financial_indicators (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            stock_id        INTEGER NOT NULL,
            report_date     DATE    NOT NULL,
            report_type     TEXT    NOT NULL,
            pe_ttm          REAL,
            pb              REAL,
            ps_ttm          REAL,
            roe             REAL,
            roa             REAL,
            gross_margin    REAL,
            net_margin      REAL,
            revenue_yoy     REAL,
            profit_yoy      REAL,
            debt_to_equity  REAL,
            current_ratio   REAL,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (stock_id) REFERENCES stocks(id)
        );
    """,

    "idx_fin_indicators_unique": """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_fin_indicators_unique
            ON financial_indicators(stock_id, report_date, report_type);
    """,

    # ------------------------------------------
    # 6. 宏观指标表
    # ------------------------------------------
    "macro_indicators": """
        CREATE TABLE IF NOT EXISTS macro_indicators (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            indicator_name  TEXT    NOT NULL,
            indicator_value REAL,
            pub_date        DATE    NOT NULL,
            frequency       TEXT,
            source          TEXT,
            created_at      DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """,

    "idx_macro_unique": """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_macro_unique
            ON macro_indicators(indicator_name, pub_date);
    """,

    # ------------------------------------------
    # 7. 数据同步日志表
    # ------------------------------------------
    "fetch_logs": """
        CREATE TABLE IF NOT EXISTS fetch_logs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            data_type       TEXT    NOT NULL,
            symbol          TEXT,
            start_date      DATE,
            end_date        DATE,
            status          TEXT    DEFAULT 'running',
            records_count   INTEGER DEFAULT 0,
            error_message   TEXT,
            fetch_at        DATETIME DEFAULT CURRENT_TIMESTAMP
        );
    """,

    "idx_fetch_logs_type": """
        CREATE INDEX IF NOT EXISTS idx_fetch_logs_type
            ON fetch_logs(data_type, fetch_at);
    """,
}
