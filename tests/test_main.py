import unittest
import pandas as pd

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


class TestUnmatchedRows(unittest.TestCase):
    def _build_person(self, name: str) -> dict:
        return {
            "Name": name,
            "Pref": "Carretear en casa",
            "Hobby": "Música",
            "Juegos": "Minecraft",
            "Deportes": "Futbol",
            "Comida": "Pizza",
            "Musica": "Rock",
            "Series": "Drama",
            "Idiomas": "Inglés",
            "Dieta": "Omnívora",
        }

    def test_extra_mapadrino_is_listed_without_mechon(self):
        mechon = pd.DataFrame([self._build_person("M1")])
        mapadrinos = pd.DataFrame([self._build_person("P1"), self._build_person("P2")])

        result = main.match_algorithm(mechon, mapadrinos)
        self.assertEqual(len(result), 2)

        last_row = result.iloc[-1]
        self.assertEqual(last_row["Mechon"], "")
        self.assertTrue(last_row["Padrino"] in {"P1", "P2"})
        self.assertIn("sin_emparejar_por_cupo", last_row["Alertas"])

    def test_extra_mechon_is_listed_without_mapadrino(self):
        mechon = pd.DataFrame([self._build_person("M1"), self._build_person("M2")])
        mapadrinos = pd.DataFrame([self._build_person("P1")])

        result = main.match_algorithm(mechon, mapadrinos)
        self.assertEqual(len(result), 2)

        last_row = result.iloc[-1]
        self.assertTrue(last_row["Mechon"] in {"M1", "M2"})
        self.assertEqual(last_row["Padrino"], "")
        self.assertIn("sin_emparejar_por_cupo", last_row["Alertas"])


if __name__ == "__main__":
    unittest.main()
