"""Exercise the architecture report and its CSV output."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from synapse.experiments.compare_cli import main


class ReportTests(unittest.TestCase):
    def test_csv_matches_report_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "comparison.csv"
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(main(["--epochs", "2", "--seeds", "7", "3",
                                       "--csv", str(path)]), 0)
            with path.open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["seed"] for row in rows], ["7", "3"])
            for row in rows:
                self.assertEqual(row["epochs"], "2")
                self.assertAlmostEqual(float(row["candidate_minus_baseline"]),
                    float(row["candidate_mse"]) - float(row["baseline_mse"]))
            self.assertIn("Training MSE only", output.getvalue())

    def test_invalid_widths_are_reported(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            main(["--baseline", "3", "--candidate", "3"])
        self.assertEqual(error.exception.code, 2)
