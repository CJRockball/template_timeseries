import unittest

try:
    from ml_template.dashboard.forecast_time import lead_metrics
except ModuleNotFoundError:
    from forecast_time import lead_metrics


class TestLeadMetrics(unittest.TestCase):
    def test_many_rolling_origins_still_have_one_lead(self):
        self.assertEqual(
            [r["lead"] for r in lead_metrics(view([row("01", "02", 2), row("02", "03", -4)]))], [1]
        )

    def test_fixed_origin_and_common_support(self):
        a, b = lead_metrics(
            view([row("01", "02", 2), row("01", "03", -4), row("01", "03", 9, False)])
        )
        self.assertEqual((a["lead"], b["lead"], b["n"], b["mae"], b["bias"]), (1, 2, 1, 4, -4))


def row(origin, target, error, common=True):
    return dict(
        origin_date="2016-02-" + origin,
        target_date="2016-02-" + target,
        method="xgb",
        status="ok",
        common_supported=common,
        error=error,
    )


def view(rows):
    return dict(rows=rows, methods=["xgb"], support="common_supported")
