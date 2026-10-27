def compute_engagement_metrics(clickstream: pd.DataFrame) -> pd.DataFrame:  # noqa: F821
    """Compute engagement metrics from click-stream events.

    This file is the UDF-body sidecar referenced by ``user_click_stats.yaml``
    (``udf.file``). It is intentionally not importable as a spec module (it
    references ``pd`` without importing it, matching the example projects), so
    the loader treats it purely as the UDF source text.

    Args:
        clickstream: Per-event click-stream rows for the entity window.

    Returns:
        A frame with the per-event engagement columns consumed by the FV.
    """
    df = clickstream.copy()

    conversion_events = ["purchase", "signup", "subscribe"]
    df["IS_CONVERSION"] = df["EVENT_TYPE"].isin(conversion_events)

    weights = {
        "page_view": 1.0,
        "click": 2.0,
        "form_submit": 5.0,
        "purchase": 10.0,
        "signup": 8.0,
    }
    df["ENGAGEMENT_SCORE"] = df["EVENT_TYPE"].map(weights).fillna(1.0)

    return df[["USER_ID", "TIMESTAMP", "IS_CONVERSION", "ENGAGEMENT_SCORE"]]
