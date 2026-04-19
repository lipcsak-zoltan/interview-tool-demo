# Synthetic Interview Analysis Assistant

A public Streamlit demo for retrieval-augmented qualitative interview analysis.

The application uses a fully synthetic dataset about the fictional company **Meridian Industrial Services**. Sites, respondents, roles, answers, and scenarios are invented for portfolio demonstration purposes.

## Live Demo Access

Shared demo password: `demo2026`

The password is intentionally low-friction so hiring managers and reviewers can open the demo while reducing casual automated use.

## What The Demo Contains

- 3 fictional sites: Site A, Site B, Site C
- 30 synthetic interviewees
- 10 questions per interviewee
- 300 retrievable interview chunks
- Filters for site, question number, collar type, and role
- Source panels showing the exact retrieved excerpts used for each answer
- Editable session logs with Markdown and PDF export
- Session checkpoint save/load support

## Example Questions

- What do blue-collar employees say about scheduling?
- Compare managers and employees on communication.
- Which sites describe the strongest team collaboration?
- What concerns do white-collar employees raise about career growth?
- What one improvement appears most often across the interviews?

## Local Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .streamlit/secrets.example.toml .streamlit/secrets.toml
```

Set `OPENAI_API_KEY` in `.streamlit/secrets.toml`.

## Dataset And Database

Generate and validate the synthetic JSONL dataset:

```bash
python scripts/generate_synthetic_dataset.py
python scripts/validate_dataset.py
```

Rebuild the Chroma database:

```bash
export OPENAI_API_KEY="your-api-key"
python scripts/rebuild_chroma.py
```

Run the app:

```bash
streamlit run app.py
```

## Deployment Notes

Deploy only this clean demo repository to Streamlit Cloud. Configure these secrets:

- `OPENAI_API_KEY`
- `app_password`
- `OPENAI_CHAT_MODEL` optional, defaults to `gpt-4o`

The generated Chroma database at `db/chroma_demo` is synthetic and safe to publish after validation.
