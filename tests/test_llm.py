import requests

from engine.llm import (
    build_adjustment_prompt,
    build_prompt,
    call_groq,
    generate_begruendung,
    parse_adjustment_response,
    propose_adjustment,
)

MATCH_CONTEXT = {
    "home": "Schweiz",
    "away": "Algerien",
    "stage": "Sechzehntelfinale",
    "probabilities": {"home": 0.49, "draw": 0.31, "away": 0.20},
    "expected_goals": (1.73, 1.09),
    "tip": (2, 1),
    "market_probabilities": None,
}


class TestBuildPrompt:
    def test_contains_key_facts(self):
        prompt = build_prompt(MATCH_CONTEXT)
        assert "Schweiz" in prompt
        assert "Algerien" in prompt
        assert "49%" in prompt
        assert "2:1" in prompt

    def test_includes_market_when_present(self):
        context = {**MATCH_CONTEXT, "market_probabilities": {"home": 0.45, "draw": 0.30, "away": 0.25}}
        prompt = build_prompt(context)
        assert "Buchmacherquoten" in prompt

    def test_omits_market_section_when_absent(self):
        # Der Anweisungssatz nennt "Buchmacherquoten" generisch als möglichen
        # Faktor - hier geht es um die konkrete Datenzeile mit Prozentwerten.
        prompt = build_prompt(MATCH_CONTEXT)
        assert "- Buchmacherquoten" not in prompt

    def test_mentions_elo_when_present(self):
        context = {**MATCH_CONTEXT, "elo": {"home": 1683.0, "away": 1608.0}}
        prompt = build_prompt(context)
        assert "1683" in prompt and "1608" in prompt

    def test_omits_elo_when_absent(self):
        prompt = build_prompt(MATCH_CONTEXT)
        assert "- ELO-Bewertung" not in prompt

    def test_includes_llm_adjustment_when_present(self):
        context = {
            **MATCH_CONTEXT,
            "llm_adjustment": {"tip": [1, 1], "grund": "Stammtorwart fehlt", "news_count": 3},
        }
        prompt = build_prompt(context)
        assert "Stammtorwart fehlt" in prompt
        assert "News-Check" in prompt

    def test_includes_news_checked_without_adjustment(self):
        context = {**MATCH_CONTEXT, "news_checked": 2}
        prompt = build_prompt(context)
        assert "2 aktuelle Schlagzeile" in prompt

    def test_asks_for_longer_source_attributed_text(self):
        prompt = build_prompt(MATCH_CONTEXT)
        assert "5-7" in prompt


class TestCallGroq:
    def test_returns_text_on_success(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "  Klarer Heimsieg erwartet.  "}}]}

        monkeypatch.setattr("engine.llm.requests.post", lambda *a, **kw: FakeResponse())
        result = call_groq("prompt", "fake-key")
        assert result == "Klarer Heimsieg erwartet."

    def test_returns_none_on_network_error(self, monkeypatch):
        def raise_error(*args, **kwargs):
            raise requests.ConnectionError("down")

        monkeypatch.setattr("engine.llm.requests.post", raise_error)
        assert call_groq("prompt", "fake-key") is None

    def test_returns_none_on_malformed_response(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"unexpected": "shape"}

        monkeypatch.setattr("engine.llm.requests.post", lambda *a, **kw: FakeResponse())
        assert call_groq("prompt", "fake-key") is None

    def test_returns_none_on_empty_content(self, monkeypatch):
        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {"choices": [{"message": {"content": "   "}}]}

        monkeypatch.setattr("engine.llm.requests.post", lambda *a, **kw: FakeResponse())
        assert call_groq("prompt", "fake-key") is None


class TestGenerateBegruendung:
    def test_no_api_key_falls_back_to_template(self):
        text, source = generate_begruendung(MATCH_CONTEXT, api_key=None)
        assert text is None
        assert source == "template"

    def test_successful_llm_call(self, monkeypatch):
        monkeypatch.setattr("engine.llm.call_groq", lambda prompt, key, model, **kw: "LLM-Text.")
        text, source = generate_begruendung(MATCH_CONTEXT, api_key="fake-key")
        assert text == "LLM-Text."
        assert source == "llm"

    def test_failed_llm_call_falls_back_to_template(self, monkeypatch):
        monkeypatch.setattr("engine.llm.call_groq", lambda prompt, key, model, **kw: None)
        text, source = generate_begruendung(MATCH_CONTEXT, api_key="fake-key")
        assert text is None
        assert source == "template"


ADJUSTMENT_CONTEXT = {"home": "Deutschland", "away": "Portugal", "tip": (2, 1)}
NEWS = [{"source": "kicker", "title": "Kapitän verletzt", "description": "Fällt aus."}]


class TestBuildAdjustmentPrompt:
    def test_lists_news_items(self):
        prompt = build_adjustment_prompt(ADJUSTMENT_CONTEXT, NEWS)
        assert "Kapitän verletzt" in prompt
        assert "kicker" in prompt
        assert "2:1" in prompt


class TestParseAdjustmentResponse:
    def test_valid_adjustment(self):
        text = '{"adjust": true, "home_delta": -1, "away_delta": 0, "grund": "Stammtorwart fehlt"}'
        result = parse_adjustment_response(text)
        assert result == {"home_delta": -1, "away_delta": 0, "grund": "Stammtorwart fehlt"}

    def test_adjust_false_returns_none(self):
        text = '{"adjust": false, "home_delta": 0, "away_delta": 0, "grund": "nichts Relevantes"}'
        assert parse_adjustment_response(text) is None

    def test_no_op_delta_returns_none(self):
        # adjust=true aber beide Deltas 0 -> nichts zu tun
        text = '{"adjust": true, "home_delta": 0, "away_delta": 0, "grund": "x"}'
        assert parse_adjustment_response(text) is None

    def test_clamps_out_of_range_delta(self):
        text = '{"adjust": true, "home_delta": -3, "away_delta": 2, "grund": "x"}'
        result = parse_adjustment_response(text)
        assert result["home_delta"] == -1
        assert result["away_delta"] == 1

    def test_extracts_json_from_surrounding_prose(self):
        text = 'Hier ist meine Antwort: {"adjust": true, "home_delta": 1, "away_delta": 0, "grund": "x"} Danke.'
        result = parse_adjustment_response(text)
        assert result["home_delta"] == 1

    def test_malformed_json_returns_none(self):
        assert parse_adjustment_response("das ist kein JSON") is None
        assert parse_adjustment_response('{"adjust": true, "home_delta":}') is None

    def test_missing_delta_returns_none(self):
        text = '{"adjust": true, "grund": "x"}'
        assert parse_adjustment_response(text) is None


class TestProposeAdjustment:
    def test_no_news_skips_llm_call_entirely(self, monkeypatch):
        called = []
        monkeypatch.setattr("engine.llm.call_groq", lambda *a, **kw: called.append(1))
        result = propose_adjustment(ADJUSTMENT_CONTEXT, [], api_key="fake-key")
        assert result is None
        assert called == []  # kein API-Call ohne News - nichts zu begründen

    def test_no_api_key_returns_none(self):
        assert propose_adjustment(ADJUSTMENT_CONTEXT, NEWS, api_key=None) is None

    def test_successful_proposal(self, monkeypatch):
        monkeypatch.setattr(
            "engine.llm.call_groq",
            lambda *a, **kw: '{"adjust": true, "home_delta": -1, "away_delta": 0, "grund": "Verletzung"}',
        )
        result = propose_adjustment(ADJUSTMENT_CONTEXT, NEWS, api_key="fake-key")
        assert result == {"home_delta": -1, "away_delta": 0, "grund": "Verletzung"}

    def test_llm_failure_returns_none(self, monkeypatch):
        monkeypatch.setattr("engine.llm.call_groq", lambda *a, **kw: None)
        assert propose_adjustment(ADJUSTMENT_CONTEXT, NEWS, api_key="fake-key") is None
