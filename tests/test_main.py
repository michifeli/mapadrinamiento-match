import unittest

import main
from src.semantic_mapper import deterministic_map_response


class TestLocalMapping(unittest.TestCase):
    def test_normalize_token_removes_accents(self):
        self.assertEqual(main.normalize_token("ÓsÜ!  "), "osu!")

    def test_dieta_de_todo_maps_to_omnivora(self):
        mapped, _, conf = deterministic_map_response(
            "De todo", "Dieta", main.OFFICIAL_OPTIONS["Dieta"]
        )
        self.assertEqual(mapped, "Omnívora")
        self.assertGreaterEqual(conf, 0.9)

    def test_trago_soft_drink_maps_to_sin_alcohol(self):
        mapped, _, conf = deterministic_map_response(
            "Coca-cola", "Trago", main.OFFICIAL_OPTIONS["Trago"]
        )
        self.assertEqual(mapped, "Sin alcohol")
        self.assertGreaterEqual(conf, 0.9)

    def test_series_ciencia_ficcion_maps_to_scifi(self):
        mapped, _, conf = deterministic_map_response(
            "Ciencia ficción", "Series", main.OFFICIAL_OPTIONS["Series"]
        )
        self.assertEqual(mapped, "Sci-Fi")
        self.assertGreaterEqual(conf, 0.9)


class TestScoringGuardrails(unittest.TestCase):
    def _row(self, **overrides):
        base = {
            "Pref": "",
            "Hobby": "",
            "Juegos": "",
            "Deportes": "",
            "Comida": "",
            "Musica": "",
            "Series": "",
            "Idiomas": "",
            "Dieta": "",
        }
        base.update(overrides)
        return base

    def test_vital_multiplier_is_low_when_pref_and_hobby_do_not_match(self):
        m = self._row(Deportes="Futbol", Comida="Pizza", Dieta="Omnívora")
        p = self._row(Deportes="Futbol", Comida="Pizza", Dieta="Omnívora")
        comp = main.calculate_match_components(m, p)
        self.assertEqual(comp["vital_multiplier"], 0.8)

    def test_vital_multiplier_is_full_when_pref_and_hobby_match(self):
        m = self._row(Pref="Carretear en casa", Hobby="Música")
        p = self._row(Pref="Carretear en casa", Hobby="Música")
        comp = main.calculate_match_components(m, p)
        self.assertEqual(comp["vital_multiplier"], 1.0)


if __name__ == "__main__":
    unittest.main()
