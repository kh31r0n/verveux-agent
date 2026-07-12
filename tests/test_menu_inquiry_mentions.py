"""Tests for menu_inquiry's product-mention heuristic (image attachment signal)."""

from src.agents.menu_inquiry import (
    _compute_mentioned_product_ids,
    _photo_attachment_note,
)

CATALOG = [
    {"product_id": "p1", "name": "Bandeja Paisa Tradicional", "price": 35000},
    {"product_id": "p2", "name": "Ajiaco Santafereño", "price": 28000},
    {"product_id": "p3", "name": "Arroz con Pollo", "price": 22000},
]


class TestComputeMentionedProductIds:
    def test_single_dish_named_in_burst(self):
        assert _compute_mentioned_product_ids(
            CATALOG, "¿Cuánto cuesta la bandeja paisa tradicional?", []
        ) == ["p1"]

    def test_partial_name_matches(self):
        # "bandeja paisa" (2 of 3 significant tokens) still resolves
        assert _compute_mentioned_product_ids(
            CATALOG, "quiero la bandeja paisa", []
        ) == ["p1"]

    def test_accent_insensitive(self):
        assert _compute_mentioned_product_ids(
            CATALOG, "tienes ajiaco santafereno?", []
        ) == ["p2"]

    def test_falls_back_to_history_when_burst_names_nothing(self):
        # The transcript case: "¿tienes imágenes del plato?" names no dish,
        # but the bot's previous reply described exactly one.
        history = [
            "¿Qué me recomiendas?",
            "Te recomiendo la Bandeja Paisa Tradicional, ¡es espectacular!",
            "¿Tienes imágenes del plato?",
        ]
        assert _compute_mentioned_product_ids(
            CATALOG, "¿Tienes imágenes del plato?", history
        ) == ["p1"]

    def test_multiple_dishes_in_burst_is_ambiguous(self):
        assert (
            _compute_mentioned_product_ids(
                CATALOG, "diferencia entre el ajiaco santafereño y el arroz con pollo", []
            )
            == []
        )

    def test_multiple_dishes_in_history_is_ambiguous(self):
        history = [
            "Tenemos Bandeja Paisa Tradicional y Ajiaco Santafereño",
            "¿Tienes fotos?",
        ]
        assert _compute_mentioned_product_ids(CATALOG, "¿Tienes fotos?", history) == []

    def test_burst_match_wins_over_history(self):
        history = ["Te recomiendo el Ajiaco Santafereño"]
        assert _compute_mentioned_product_ids(
            CATALOG, "mejor muéstrame el arroz con pollo", history
        ) == ["p3"]

    def test_no_catalog(self):
        assert _compute_mentioned_product_ids([], "bandeja paisa", []) == []

    def test_no_match_anywhere(self):
        assert _compute_mentioned_product_ids(CATALOG, "¿a qué hora abren?", []) == []

    def test_single_generic_token_does_not_match_multiword_name(self):
        # "arroz" alone must not resolve "Arroz con Pollo" (needs 2 tokens)
        assert _compute_mentioned_product_ids(CATALOG, "tienen arroz?", []) == []


class TestPhotoAttachmentNote:
    def test_attaching_note_when_image_and_stock(self):
        catalog = [
            {"product_id": "p1", "name": "Bandeja Paisa ", "has_image": True, "stock": 10}
        ]
        note = _photo_attachment_note(catalog, ["p1"])
        assert "se está adjuntando" in note
        assert "Bandeja Paisa" in note
        assert "NUNCA digas que no puedes enviar imágenes" in note

    def test_no_photo_note_when_image_missing(self):
        catalog = [
            {"product_id": "p1", "name": "Trucha al Ajillo", "has_image": False, "stock": 10}
        ]
        note = _photo_attachment_note(catalog, ["p1"])
        assert "no tiene foto" in note

    def test_no_photo_note_when_out_of_stock(self):
        # Mirrors the backend stock gate: agotado products don't get showcased
        catalog = [
            {"product_id": "p1", "name": "Sancocho", "has_image": True, "stock": 0}
        ]
        note = _photo_attachment_note(catalog, ["p1"])
        assert "no tiene foto" in note

    def test_empty_when_no_single_mention(self):
        catalog = [{"product_id": "p1", "name": "X", "has_image": True, "stock": 1}]
        assert _photo_attachment_note(catalog, []) == ""
        assert _photo_attachment_note(catalog, ["p1", "p2"]) == ""
        assert _photo_attachment_note(catalog, ["unknown"]) == ""
