import requests

from engine.llm import build_prompt, call_groq, generate_begruendung

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
        prompt = build_prompt(MATCH_CONTEXT)
        assert "Buchmacherquoten" not in prompt


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
        monkeypatch.setattr("engine.llm.call_groq", lambda prompt, key, model: "LLM-Text.")
        text, source = generate_begruendung(MATCH_CONTEXT, api_key="fake-key")
        assert text == "LLM-Text."
        assert source == "llm"

    def test_failed_llm_call_falls_back_to_template(self, monkeypatch):
        monkeypatch.setattr("engine.llm.call_groq", lambda prompt, key, model: None)
        text, source = generate_begruendung(MATCH_CONTEXT, api_key="fake-key")
        assert text is None
        assert source == "template"
