from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TRANSCRIPT_COLUMNS = ["speaker_id", "slide_id", "question_id", "problem_id", "text"]
CONTEXT_COLUMNS = ["slide_id", "question_id", "problem_id"]
SPECIAL_EMPTY_VALUES = {"", " ", "nan", "NaN", "None", "NONE", "null", "NULL", "~"}

OUTPUT_COLUMNS = [
    "participant_id",
    "label",
    "segment_id",
    "slide_id",
    "question_id",
    "problem_id",
    "segment_text",
    "segment_text_with_context",
    "n_rows_in_segment",
]


@dataclass
class TranscriptBuildConfig:
    input_dir: Path
    output_dir: Path
    transcript_glob: str = "*.csv"
    keep_special_slide_markers: bool = True
    drop_empty_segments: bool = True


# ============================================================
# Pomocnicze funkcje tekstowe
# ============================================================

def normalize_whitespace(text: str) -> str:
    """Usuwa nadmiarowe białe znaki i skleja tekst do jednej linii."""
    if text is None:
        return ""
    text = str(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def normalize_cell(value) -> str | None:
    """Normalizuje pojedynczą komórkę z CSV do postaci tekstowej lub None."""
    if pd.isna(value):
        return None

    value = str(value).strip()
    if value in SPECIAL_EMPTY_VALUES:
        return None

    return value


def extract_label_from_participant_id(participant_id: str) -> str:
    """
    zwraca etykietę jednej z grup: DE / IT / SSH
    """
    parts = participant_id.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Nie mogę wyciągnąć label z participant_id='{participant_id}'. "
            "Oczekiwany format: RR_LABEL_NN"
        )

    label = parts[1]
    allowed = {"DE", "IT", "SSH"}
    if label not in allowed:
        raise ValueError(
            f"Nieznana etykieta grupy '{label}' w participant_id='{participant_id}'. "
            f"Dozwolone: {sorted(allowed)}"
        )
    return label


def build_contextualized_text(
    segment_text: str,
    slide_id: str | None,
    question_id: str | None,
    problem_id: str | None,
) -> str:
    """
    Buduje tekst segmentu z metadanymi w środku,tekst segmentu z prefiksem kontekstowym
    np.:
    [SLIDE=__S4__] [QUESTION=__Q1__] [PROBLEM=None] Jest to wykres...
    """
    slide_token = slide_id if slide_id is not None else "None"
    question_token = question_id if question_id is not None else "None"
    problem_token = problem_id if problem_id is not None else "None"

    prefix = (
        f"[SLIDE={slide_token}] "
        f"[QUESTION={question_token}] "
        f"[PROBLEM={problem_token}]"
    )

    if segment_text:
        return f"{prefix} {segment_text}".strip()
    return prefix


# ============================================================
# Wczytywanie i czyszczenie transcriptu
# ============================================================

def read_transcript_csv(path: Path) -> pd.DataFrame:
    """
    Wczytuje pojedynczy plik transcriptu.
    Oczekiwane kolumny:
    speaker_id, slide_id, question_id, problem_id, text
    """
    df = pd.read_csv(path)

    missing_cols = [col for col in TRANSCRIPT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Plik {path} nie zawiera wymaganych kolumn: {missing_cols}. "
            f"Znalezione kolumny: {list(df.columns)}"
        )

    df = df[TRANSCRIPT_COLUMNS].copy()

    for col in ["speaker_id", "slide_id", "question_id", "problem_id"]:
        df[col] = df[col].apply(normalize_cell)

    df["text"] = df["text"].apply(normalize_whitespace)

    return df


def forward_fill_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    Uzupełnia kontekst na podstawie ostatniej znanej wartości.
    wykonane NA CAŁYM TRANSCRIPCIE,zanim odfiltrujemy badacza, bo to badacz często ustawia nowy kontekst.
    """
    df = df.copy()
    df[CONTEXT_COLUMNS] = df[CONTEXT_COLUMNS].ffill()
    return df


def get_participant_speaker_ids(df: pd.DataFrame) -> list[str]:
    """
    Zwraca listę speaker_id należących do uczestników.
    Dzięki temu obsługujemy także plik, w którym jest więcej niż jeden participant,
    PK_DE_11-12.csv zawierający PK_DE_11 i PK_DE_12.
    """
    speakers = df["speaker_id"].dropna().astype(str).unique().tolist()

    participant_speakers = [
        s for s in speakers
        if "_DE_" in s or "_IT_" in s or "_SSH_" in s
    ]

    return sorted(participant_speakers)


def keep_only_one_participant_rows(
    df: pd.DataFrame,
    participant_id: str,
) -> pd.DataFrame:
    """
    Zostawiamy tylko wypowiedzi jednego uczestnika.
    """
    out = df.loc[df["speaker_id"] == participant_id].copy()
    return out


def optionally_remove_special_markers(
    df: pd.DataFrame,
    keep_special_slide_markers: bool = True,
) -> pd.DataFrame:
    """
    Opcjonalnie można usunąć techniczne znaczniki slajdów typu
    __S00__, __S15__, __S88__, __S99__.
    Domyślnie je zostawiamy, bo mogą nieść informację o etapie badania.
    """
    if keep_special_slide_markers:
        return df.copy()

    special_slide_ids = {"__S00__", "__S15__", "__S88__", "__S99__"}
    out = df.loc[~df["slide_id"].isin(special_slide_ids)].copy()
    return out


def drop_rows_with_empty_text(df: pd.DataFrame) -> pd.DataFrame:
    """Usuwa wiersze bez sensownej treści tekstowej."""
    out = df.loc[df["text"].fillna("").str.strip() != ""].copy()
    return out


# ============================================================
# Segmentacja
# ============================================================

def build_segment_breaks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Wyznacza granice segmentów.
    Nowy segment zaczyna się, gdy zmienia się kontekst:
    slide_id / question_id / problem_id.

    w pandas NaN != NaN (dwa identyczne braki wyglądają jak zmiana kontekstu),
    więc przed porównaniem zastępujemy braki stałym tokenem technicznym.
    """
    df = df.copy()

    compare_df = df[CONTEXT_COLUMNS].fillna("__NONE__")
    previous_context = compare_df.shift(1) #o jeden wiersz w dół -> dla każdego wiersza „kontekst poprzedniego wiersza"
    context_changed = (compare_df != previous_context).any(axis=1) #porownanie czy kontekst się zmienił, wiersz 0 nie ma poprzednika, więc wszystko wygląda jak zmiana, wiersz 1 ma taki sam kontekst jak 0 -> brak zmiany, itd.
    #True = tutaj zaczyna się nowy segment, False = ten wiersz należy do tego samego segmentu co poprzedni

    if not context_changed.empty: #pierwszy wiersz zawsze ma zacząć nowy segment
        context_changed.iloc[0] = True

    df["segment_id_local"] = context_changed.cumsum()
    return df

def aggregate_segments(
    df: pd.DataFrame,
    participant_id: str,
    label: str,
) -> pd.DataFrame:
    """
    Łączy kolejne wypowiedzi uczestnika o tym samym kontekście w jeden segment.
    """
    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    df = build_segment_breaks(df)

    grouped = (
        df.groupby("segment_id_local", sort=True)
        .agg(
            slide_id=("slide_id", "first"),
            question_id=("question_id", "first"),
            problem_id=("problem_id", "first"),
            segment_text=("text", lambda s: normalize_whitespace(" ".join(s.astype(str)))),
            n_rows_in_segment=("text", "size"),
        )
        .reset_index(drop=True)
    )

    grouped["participant_id"] = participant_id
    grouped["label"] = label
    grouped["segment_id"] = [
        f"{participant_id}_SEG_{i:03d}" for i in range(1, len(grouped) + 1)
    ]

    grouped["segment_text_with_context"] = grouped.apply(
        lambda row: build_contextualized_text(
            segment_text=row["segment_text"],
            slide_id=row["slide_id"],
            question_id=row["question_id"],
            problem_id=row["problem_id"],
        ),
        axis=1,
    )

    grouped = grouped[
        [
            "participant_id",
            "label",
            "segment_id",
            "slide_id",
            "question_id",
            "problem_id",
            "segment_text",
            "segment_text_with_context",
            "n_rows_in_segment",
        ]
    ].copy()

    return grouped


# ============================================================
# Pipeline na pojedynczy plik
# ============================================================

def process_single_transcript(
    transcript_path: Path,
    keep_special_slide_markers: bool = True,
    drop_empty_segments: bool = True,
) -> pd.DataFrame:
    """
    Przetwarza jeden plik transcriptu.
    - najpierw robi ffill na całym transcriptcie,
    - potem wykrywa wszystkich participantów w pliku,
    - dla każdego participant_id buduje segmenty osobno.
    """
    df = read_transcript_csv(transcript_path)
    df = forward_fill_context(df)

    participant_ids = get_participant_speaker_ids(df)

    if not participant_ids:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    all_segments = []

    for participant_id in participant_ids:
        label = extract_label_from_participant_id(participant_id)

        participant_df = keep_only_one_participant_rows(df, participant_id)
        participant_df = optionally_remove_special_markers(
            participant_df,
            keep_special_slide_markers=keep_special_slide_markers,
        )
        participant_df = drop_rows_with_empty_text(participant_df)

        segments = aggregate_segments(
            df=participant_df,
            participant_id=participant_id,
            label=label,
        )

        if drop_empty_segments:
            segments = segments.loc[
                segments["segment_text"].fillna("").str.strip() != ""
            ].copy()

        all_segments.append(segments)

    if not all_segments:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    return pd.concat(all_segments, ignore_index=True)


# ============================================================
# Pipeline na cały folder
# ============================================================

def build_all_segments(config: TranscriptBuildConfig) -> pd.DataFrame:
    transcript_paths = sorted(config.input_dir.glob(config.transcript_glob))

    if not transcript_paths:
        raise FileNotFoundError(
            f"Nie znaleziono plików pasujących do wzorca "
            f"'{config.transcript_glob}' w katalogu: {config.input_dir}"
        )

    all_segments = []

    for path in transcript_paths:
        segments = process_single_transcript(
            transcript_path=path,
            keep_special_slide_markers=config.keep_special_slide_markers,
            drop_empty_segments=config.drop_empty_segments,
        )
        all_segments.append(segments)

    if not all_segments:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    final_segments_df = pd.concat(all_segments, ignore_index=True)

    final_segments_df = final_segments_df.sort_values(
        by=["participant_id", "segment_id"],
        ascending=True,
        kind="stable",
    ).reset_index(drop=True)

    return final_segments_df


def build_participant_level_documents(final_segments_df: pd.DataFrame) -> pd.DataFrame:
    """
    Dodatkowy widok: wszystko dla jednego uczestnika sklejone w jeden dokument.
    Przyda się później do eksperymentów z transformerami na poziomie osoby.
    """
    if final_segments_df.empty:
        return pd.DataFrame(columns=["participant_id", "label", "full_document", "n_segments"])

    participant_docs = (
        final_segments_df.groupby(["participant_id", "label"], sort=True)
        .agg(
            full_document=("segment_text_with_context", lambda s: "\n".join(s.astype(str))),
            n_segments=("segment_id", "size"),
        )
        .reset_index()
    )

    return participant_docs


def save_outputs(
    final_segments_df: pd.DataFrame,
    participant_docs_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    final_segments_path = output_dir / "final_segments.csv"
    participant_docs_path = output_dir / "participant_documents.csv"

    final_segments_df.to_csv(final_segments_path, index=False, encoding="utf-8")
    participant_docs_df.to_csv(participant_docs_path, index=False, encoding="utf-8")

    try:
        final_segments_df.to_parquet(output_dir / "final_segments.parquet", index=False)
        participant_docs_df.to_parquet(output_dir / "participant_documents.parquet", index=False)
    except Exception:
        pass


def print_basic_report(final_segments_df: pd.DataFrame, participant_docs_df: pd.DataFrame) -> None:
    print("\n=== RAPORT ===")
    print(f"Liczba segmentów: {len(final_segments_df)}")
    print(f"Liczba uczestników: {participant_docs_df['participant_id'].nunique()}")

    if not final_segments_df.empty:
        print("\nRozkład segmentów po label:")
        print(final_segments_df["label"].value_counts(dropna=False).sort_index())

        print("\nLiczba uczestników po label:")
        print(
            participant_docs_df[["participant_id", "label"]]
            .drop_duplicates()["label"]
            .value_counts(dropna=False)
            .sort_index()
        )

        print("\nŚrednia liczba segmentów na uczestnika:")
        mean_segments = participant_docs_df["n_segments"].mean()
        print(round(float(mean_segments), 2))

        print("\nPrzykładowe pierwsze 5 segmentów:")
        preview_cols = [
            "participant_id",
            "label",
            "slide_id",
            "question_id",
            "problem_id",
            "segment_text_with_context",
        ]
        print(final_segments_df[preview_cols].head(5).to_string(index=False))


# ============================================================
# CLI
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Buduje początkowe segmenty z transcriptów XAI-FUNGI: "
            "wczytanie CSV, usunięcie wypowiedzi badającego z finalnych segmentów, "
            "ffill kontekstu na całym transcriptcie, segmentacja po "
            "(slide_id, question_id, problem_id)."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Ścieżka do folderu z plikami transcriptów CSV, np. data/transcripts",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Ścieżka do katalogu wyjściowego",
    )
    parser.add_argument(
        "--transcript-glob",
        type=str,
        default="*.csv",
        help="Wzorzec plików do przetworzenia, domyślnie '*.csv'",
    )
    parser.add_argument(
        "--drop-special-slides",
        action="store_true",
        help=(
            "Jeśli podane, usuwa specjalne znaczniki etapów badania "
            "(__S00__, __S15__, __S88__, __S99__). Domyślnie są zachowane."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    config = TranscriptBuildConfig(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        transcript_glob=args.transcript_glob,
        keep_special_slide_markers=not args.drop_special_slides,
        drop_empty_segments=True,
    )

    final_segments_df = build_all_segments(config)
    participant_docs_df = build_participant_level_documents(final_segments_df)

    save_outputs(
        final_segments_df=final_segments_df,
        participant_docs_df=participant_docs_df,
        output_dir=config.output_dir,
    )

    print_basic_report(final_segments_df, participant_docs_df)


if __name__ == "__main__":
    main()