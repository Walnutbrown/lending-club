import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data import generate_features_list as generator


class GenerateFeatureListTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.project_dir = Path(self.temp_dir.name)
        (self.project_dir / "data" / "processed").mkdir(parents=True)
        generator.PROJECT_DIR = self.project_dir

    def tearDown(self):
        self.temp_dir.cleanup()

    def write_feature_table(self, filename):
        path = self.project_dir / "data" / "processed" / filename
        pd.DataFrame(
            {
                "loan_amnt": [1000],
                "default": [0],
                "fico_range_low": [700],
                "purpose": ["debt_consolidation"],
            }
        ).to_csv(path, index=False)

    def test_each_model_writes_its_own_feature_list(self):
        for model, input_name in generator.MODEL_FILES.items():
            self.write_feature_table(input_name[0])
            output_path = generator.generate_feature_list(model)
            self.assertTrue(output_path.exists())
            self.assertEqual(
                pd.read_csv(output_path)["feature"].tolist(),
                ["fico_range_low", "purpose"],
            )

    def test_missing_input_has_actionable_error(self):
        with self.assertRaisesRegex(FileNotFoundError, "입력 파일이 없습니다"):
            generator.generate_feature_list("linear")

    def test_unknown_model_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            generator.generate_feature_list("neural_net")


if __name__ == "__main__":
    unittest.main()
