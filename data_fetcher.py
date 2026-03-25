import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


class DataFetcher:
    """
    Fetches data from yfinance and normalizes it to the format expected by ChromaDB.

    Each returned document has this shape:
        {
            "text":     str,   # the field that gets embedded
            "metadata": dict   # stored alongside the vector (not embedded)
        }

    content_type values:
        - "news"             -> recent ticker news articles
        - "recommendation"   -> analyst upgrade/downgrade events
        - "company_summary"  -> business description, stored once per ticker
    """

    def get_documents(
        self,
        ticker: str,
        days_back: int = 7,
        include_recommendations: bool = True,
        include_summary: bool = True,
    ) -> list[dict]:
        t = yf.Ticker(ticker)
        documents = []

        documents.extend(self._fetch_news(t, ticker, days_back))

        if include_recommendations:
            documents.extend(self._fetch_recommendations(t, ticker))

        if include_summary:
            summary_doc = self._fetch_company_summary(t, ticker)
            if summary_doc:
                documents.append(summary_doc)

        return documents

    # ------------------------------------------------------------------ #
    #  NEWS                                                                #
    # ------------------------------------------------------------------ #

    def _fetch_news(self, ticker_obj, ticker: str, days_back: int) -> list[dict]:
        cutoff = datetime.now() - timedelta(days=days_back)
        price_today = self._get_current_price(ticker_obj)
        sector = self._get_sector(ticker_obj)

        documents = []
        for item in ticker_obj.news or []:

            # yfinance >= 0.2.x nests fields under "content"
            content = item.get("content", item)

            title = (
                content.get("title")
                or item.get("title", "")
            ).strip()

            summary = (
                content.get("summary")
                or content.get("description")
                or item.get("summary", "")
            ).strip()

            # date can come as a unix epoch int or an ISO string
            raw_date = (
                content.get("pubDate")
                or content.get("publishedAt")
                or item.get("providerPublishTime")
            )
            pub_dt = self._parse_date(raw_date)

            url = (
                content.get("canonicalUrl", {}).get("url")
                or content.get("clickThroughUrl", {}).get("url")
                or item.get("link", "")
            )

            publisher = (
                content.get("provider", {}).get("displayName")
                or item.get("publisher", "unknown")
            )

            if not title or (pub_dt and pub_dt < cutoff):
                continue

            text = f"{title}. {summary}" if summary else title

            documents.append({
                "text": text,
                "metadata": {
                    "content_type":  "news",
                    "ticker":        ticker.upper(),
                    "source":        publisher,
                    "title":         title,
                    "url":           url,
                    "published_at":  pub_dt.strftime("%Y-%m-%d") if pub_dt else "unknown",
                    "price_at_time": price_today,
                    "sector":        sector,
                },
            })

        return documents

    # ------------------------------------------------------------------ #
    #  RECOMMENDATIONS                                                     #
    # ------------------------------------------------------------------ #

    def _fetch_recommendations(self, ticker_obj, ticker: str) -> list[dict]:
        """
        Uses upgrades_downgrades instead of recommendations.
        In recent yfinance versions, recommendations returns aggregated counts
        (strongBuy/buy/hold/...) rather than per-firm entries.
        upgrades_downgrades has Firm, ToGrade, FromGrade, and Action.
        """
        try:
            df = ticker_obj.upgrades_downgrades
            if df is None or df.empty:
                return []
        except Exception:
            return []

        df = df.head(10).copy()

        documents = []
        for date_idx, row in df.iterrows():
            firm    = str(row.get("Firm", "")).strip() or "Unknown firm"
            to_gr   = str(row.get("ToGrade", "")).strip()
            from_gr = str(row.get("FromGrade", "")).strip()
            action  = str(row.get("Action", "")).strip()

            if from_gr and to_gr:
                text = (
                    f"{firm} changed recommendation for {ticker.upper()} "
                    f"from {from_gr} to {to_gr}."
                )
            elif to_gr:
                text = f"{firm} rated {ticker.upper()} as {to_gr}."
            elif action:
                text = f"{firm} issued a '{action}' on {ticker.upper()}."
            else:
                continue  # empty row, skip it

            pub_date = (
                date_idx.strftime("%Y-%m-%d")
                if hasattr(date_idx, "strftime")
                else str(date_idx)
            )

            documents.append({
                "text": text,
                "metadata": {
                    "content_type": "recommendation",
                    "ticker":       ticker.upper(),
                    "source":       firm,
                    "title":        text,
                    "url":          "",
                    "published_at": pub_date,
                    "from_grade":   from_gr,
                    "to_grade":     to_gr,
                    "action":       action,
                },
            })

        return documents

    # ------------------------------------------------------------------ #
    #  COMPANY SUMMARY                                                     #
    # ------------------------------------------------------------------ #

    def _fetch_company_summary(self, ticker_obj, ticker: str) -> Optional[dict]:
        try:
            info = ticker_obj.info
        except Exception:
            return None

        summary = info.get("longBusinessSummary", "").strip()
        if not summary:
            return None

        return {
            "text": summary,
            "metadata": {
                "content_type": "company_summary",
                "ticker":       ticker.upper(),
                "source":       "yfinance",
                "title":        f"{info.get('shortName', ticker)} — business description",
                "url":          "",
                "published_at": "static",
                "sector":       info.get("sector", ""),
                "industry":     info.get("industry", ""),
                "market_cap":   info.get("marketCap", 0),
                "country":      info.get("country", ""),
            },
        }

    # ------------------------------------------------------------------ #
    #  HELPERS                                                             #
    # ------------------------------------------------------------------ #

    def _parse_date(self, raw) -> Optional[datetime]:
        if raw is None:
            return None
        if isinstance(raw, (int, float)):
            try:
                return datetime.fromtimestamp(raw)
            except Exception:
                return None
        if isinstance(raw, str):
            for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
                try:
                    return datetime.strptime(raw, fmt)
                except ValueError:
                    continue
        return None

    def _get_current_price(self, ticker_obj) -> float:
        try:
            return ticker_obj.info.get("currentPrice", 0.0)
        except Exception:
            return 0.0

    def _get_sector(self, ticker_obj) -> str:
        try:
            return ticker_obj.info.get("sector", "")
        except Exception:
            return ""