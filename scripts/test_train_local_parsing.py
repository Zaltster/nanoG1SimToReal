import unittest

import train_local


class TrainLocalParsingTest(unittest.TestCase):
    def test_steps_re_parses_billions_suffix(self):
        match = train_local.STEPS_RE.search("Steps             1.0B      Env")
        self.assertIsNotNone(match)
        assert match is not None
        self.assertEqual(train_local.parse_count(match.group(1), match.group(2)), 1_000_000_000.0)

    def test_parse_count_keeps_existing_suffixes(self):
        self.assertEqual(train_local.parse_count("491.8", "M"), 491_800_000.0)
        self.assertEqual(train_local.parse_count("2.5", "G"), 2_500_000_000.0)


if __name__ == "__main__":
    unittest.main()
