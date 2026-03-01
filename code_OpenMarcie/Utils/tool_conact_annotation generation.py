import argparse
import csv
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Annotation:
    """Single time-stamped annotation row."""
    start_time: float
    end_time: float
    text: str
    hard_label: str = ""           # "tool-contact" or "tool-noncontact"
    confidence: float = 0.0        # 0-1 confidence of the label
    source_format: str = ""        # "whisper" or "softlabel_rich"


# =============================================================================
# CSV Format Detection and Parsing
# =============================================================================

def detect_format(filepath: str) -> str:
    """
    Auto-detect which CSV format the file uses.
    
    Returns:
        "whisper"         WhisperLabel format  (Id, Start_s, End_s, Text, ...)
        "softlabel_rich"  SoftLabels_Rich      (Start_Time, End_Time, Sentence)
    """
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        header_lower = [h.strip().lower() for h in header]

    if "start_s" in header_lower and "end_s" in header_lower:
        return "whisper"
    if "start_time" in header_lower and "end_time" in header_lower:
        return "softlabel_rich"

    raise ValueError(
        f"Unknown CSV format. Headers: {header}\n"
        "Expected either WhisperLabel (Start_s, End_s, Text) "
        "or SoftLabels_Rich (Start_Time, End_Time, Sentence)."
    )


def parse_whisper_csv(filepath: str) -> List[Annotation]:
    """Parse WhisperLabel CSV (Whisper ASR transcription output)."""
    annotations = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("Text", "").strip()
            try:
                start = float(row["Start_s"])
                end = float(row["End_s"])
            except (ValueError, KeyError):
                continue
            if text:
                annotations.append(
                    Annotation(
                        start_time=start,
                        end_time=end,
                        text=text,
                        source_format="whisper",
                    )
                )
    return annotations


def parse_softlabel_rich_csv(filepath: str) -> List[Annotation]:
    """Parse SoftLabels_Rich CSV (action description labels)."""
    annotations = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get("Sentence", "").strip()
            try:
                start = float(row["Start_Time"])
                end = float(row["End_Time"])
            except (ValueError, KeyError):
                continue
            # Keep rows even with empty text (they are unlabelled gaps)
            annotations.append(
                Annotation(
                    start_time=start,
                    end_time=end,
                    text=text,
                    source_format="softlabel_rich",
                )
            )
    return annotations


def load_annotations(filepath: str) -> List[Annotation]:
    """Load annotations from any supported CSV format."""
    fmt = detect_format(filepath)
    print(f"Detected format: {fmt}")
    if fmt == "whisper":
        return parse_whisper_csv(filepath)
    else:
        return parse_softlabel_rich_csv(filepath)


# =============================================================================
# Rule-Based Classifier
# =============================================================================

# Explicit tool keywords (when these appear, it's tool-contact)
TOOL_KEYWORDS = [
    "screwdriver",
    "wrench",
    "plier",
    "pliers",
    "hex key",
    "hex_key",
    "hexkey",
    "pump",
    "pumping",
    "spanner",
    "hammer",
    "drill",
    "ratchet",
    "torque",
    "socket",
    "caliper",
    "clamp",
    "vise",
    "vice",
    "multimeter",
    "soldering",
    "allen key",
    "allen_key",
    "allenkey",
    "crowbar",
    "pry bar",
    "chisel",
    "file",           # only when clearly a tool
    "rasp",
    "saw",
    "knife",
    "cutter",
    "tape measure",
    "level",
    "wire stripper",
    "crimper",
]

# Actions that strongly imply tool use
TOOL_ACTION_PATTERNS = [
    r"using a (?!bare\s*hand)",       # "using a <something>" that is NOT bare hand
    r"screwing\b",                     # screwing implies tool even if tool unstated
    r"unscrewing\b",
    r"tightening\b",
    r"loosening\b",
    r"pumping\b",
    r"drilling\b",
    r"hammering\b",
    r"soldering\b",
    r"cutting\b",
    r"sawing\b",
    r"clamping\b",
]

# Bare-hand / no-tool patterns
NONCONTACT_PATTERNS = [
    r"^$",                            # empty text
    r"^he is walking\.?$",
    r"^he is standing up",
    r"^he is kneeling down\.?$",
    r"^he is sitting",
    r"^he is cycling",
    r"using a bare\s*hand",
    r"^okay\.?$",
    r"^yes\.?$",
    r"^no\.?$",
    r"^wait",
    r"^thank",
    r"^sorry",
]

# Whisper conversational filler – not an activity description
CONVERSATIONAL_KEYWORDS = [
    "camera", "please", "again", "moment", "noise", "heard",
    "stop", "wait", "okay", "yes", "no", "oh", "right",
    "thank you", "sorry", "go ahead", "ready", "done",
    "cannot", "can you", "do you", "don't",
]


def classify_rule_based(annotation: Annotation) -> Tuple[str, float]:
    """
    Classify annotation using keyword/pattern rules.
    
    Returns:
        (label, confidence)
    """
    text = annotation.text.strip().lower()
    
    # Empty text → unlabelled / noncontact
    if not text:
        return "tool-noncontact", 0.5
    
    # ------------------------------------------------------------------
    # Whisper ASR format: these are speech transcriptions, not activity
    # descriptions. We look for mentions of tools in conversation.
    # ------------------------------------------------------------------
    if annotation.source_format == "whisper":
        return _classify_whisper_text(text)
    
    # ------------------------------------------------------------------
    # SoftLabel_Rich format: structured activity descriptions
    # ------------------------------------------------------------------
    return _classify_activity_text(text)


def _classify_whisper_text(text: str) -> Tuple[str, float]:
    """Classify Whisper ASR transcription text."""
    # Check for explicit tool mentions
    for tool in TOOL_KEYWORDS:
        if tool in text:
            return "tool-contact", 0.85
    
    # Check for tool action patterns
    for pattern in TOOL_ACTION_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return "tool-contact", 0.80
    
    # Sounds of tool use (onomatopoeia, metallic sounds)
    tool_sounds = [
        r"\bclank\b", r"\bclang\b", r"\bbuzz\b", r"\bwhir\b",
        r"\bdrill\b", r"\bgrind\b", r"\bclick\b", r"\bsnap\b",
        r"\bratchet\b", r"\btorque\b",
    ]
    for pattern in tool_sounds:
        if re.search(pattern, text, re.IGNORECASE):
            return "tool-contact", 0.70
    
    # Check for activity descriptions embedded in speech
    activity_phrases = [
        r"tighten", r"loosen", r"screw", r"unscrew",
        r"bolt", r"nut", r"adjust", r"torque",
        r"pump", r"inflate", r"deflate",
    ]
    for pattern in activity_phrases:
        if re.search(pattern, text, re.IGNORECASE):
            return "tool-contact", 0.65
    
    # Mostly conversational / not activity → noncontact
    for kw in CONVERSATIONAL_KEYWORDS:
        if kw in text:
            return "tool-noncontact", 0.70
    
    # Default for whisper: ambiguous
    return "tool-noncontact", 0.40


def _classify_activity_text(text: str) -> Tuple[str, float]:
    """Classify activity-description text (SoftLabels_Rich format)."""
    
    # ---- Definite tool-contact ----
    # Explicit tool keyword in text
    for tool in TOOL_KEYWORDS:
        if tool in text:
            return "tool-contact", 0.95
    
    # "using a <something>" that is NOT bare hand
    if re.search(r"using a (?!bare\s*hand)", text, re.IGNORECASE):
        return "tool-contact", 0.90
    
    # "screwing or unscrewing" with a tool mentioned nearby
    if re.search(r"screwing|unscrewing", text, re.IGNORECASE):
        # If also mentions bare hand and no other tool → ambiguous
        if "bare hand" in text:
            # Check if there's ALSO a tool keyword
            for tool in TOOL_KEYWORDS:
                if tool in text:
                    return "tool-contact", 0.90
            # Screwing with bare hand only → could be hand-tightened
            return "tool-contact", 0.60
        return "tool-contact", 0.85
    
    # Pumping
    if "pumping" in text:
        return "tool-contact", 0.90
    
    # ---- Definite tool-noncontact ----
    # Pure locomotion / posture
    for pattern in NONCONTACT_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            # But check there's no hidden tool reference
            has_tool = any(t in text for t in TOOL_KEYWORDS)
            if not has_tool:
                return "tool-noncontact", 0.90
    
    # Bare hand only (no tool keyword)
    if "bare hand" in text and not any(t in text for t in TOOL_KEYWORDS):
        return "tool-noncontact", 0.80
    
    # ---- Default / ambiguous ----
    return "tool-noncontact", 0.50


# =============================================================================
# LLM-Based Classifier
# =============================================================================

SYSTEM_PROMPT = """You are an annotation assistant for a multimodal activity dataset.

Your task is to classify each text annotation into exactly one of two classes:
  - "tool-contact": A physical tool (screwdriver, wrench, plier, hex key, pump, 
    spanner, hammer, drill, etc.) is being actively used or in contact with an object.
  - "tool-noncontact": No tool is being used. This includes bare-hand activities, 
    walking, sitting, standing, kneeling, cycling, and general locomotion.

Rules:
1. "bare hand" alone is NOT a tool → tool-noncontact
2. "screwing or unscrewing" with only bare hands → tool-noncontact (hand-tightened)
3. "screwing or unscrewing" with any named tool → tool-contact
4. Pumping with a pump → tool-contact
5. Speech/conversation without activity description → tool-noncontact
6. If the text is ambiguous or empty → tool-noncontact

Respond ONLY with a JSON object: {"label": "tool-contact" or "tool-noncontact", "confidence": 0.0-1.0}
"""


def classify_with_llm(
    annotations: List[Annotation],
    api_key: str,
    api_base: str = "https://api.openai.com/v1",
    model: str = "gpt-4o-mini",
    batch_size: int = 20,
    max_retries: int = 3,
) -> List[Annotation]:
    """
    Classify annotations using an OpenAI-compatible LLM API.
    Processes in batches for efficiency.
    
    Args:
        annotations: List of annotations to classify
        api_key: API key for authentication
        api_base: Base URL for API (supports OpenAI, local endpoints)
        model: Model name to use
        batch_size: Number of annotations per API call
        max_retries: Retry count on failure
    
    Returns:
        Updated annotations with hard_label and confidence set
    """
    try:
        import requests
    except ImportError:
        print("ERROR: 'requests' package required for LLM mode. Install: pip install requests")
        sys.exit(1)
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    
    url = f"{api_base}/chat/completions"
    total = len(annotations)
    
    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch = annotations[batch_start:batch_end]
        
        # Build batch prompt
        items = []
        for i, ann in enumerate(batch):
            items.append(f"{i+1}. [{ann.start_time:.2f}s - {ann.end_time:.2f}s] \"{ann.text}\"")
        
        user_prompt = (
            f"Classify each annotation below as 'tool-contact' or 'tool-noncontact'.\n\n"
            + "\n".join(items)
            + f"\n\nRespond with a JSON array of {len(batch)} objects, "
            f"each with 'index' (1-based), 'label', and 'confidence' keys."
        )
        
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
            "max_tokens": 2000,
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                
                content = response.json()["choices"][0]["message"]["content"]
                
                # Parse response – extract JSON array
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    results = json.loads(json_match.group())
                else:
                    # Try parsing the whole content
                    results = json.loads(content)
                
                # Apply results
                for result in results:
                    idx = result.get("index", 0) - 1
                    if 0 <= idx < len(batch):
                        label = result.get("label", "tool-noncontact").strip().lower()
                        conf = float(result.get("confidence", 0.5))
                        
                        if label not in ("tool-contact", "tool-noncontact"):
                            label = "tool-noncontact"
                        
                        batch[idx].hard_label = label
                        batch[idx].confidence = conf
                
                break  # Success
                
            except Exception as e:
                print(f"  API attempt {attempt+1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    # Fallback to rule-based for this batch
                    print(f"  Falling back to rule-based for batch {batch_start}-{batch_end}")
                    for ann in batch:
                        label, conf = classify_rule_based(ann)
                        ann.hard_label = label
                        ann.confidence = conf
        
        # Fill any that didn't get classified
        for ann in batch:
            if not ann.hard_label:
                label, conf = classify_rule_based(ann)
                ann.hard_label = label
                ann.confidence = conf
        
        classified = batch_end
        print(f"  Classified {classified}/{total} annotations...")
    
    return annotations


# =============================================================================
# Output
# =============================================================================

def write_output_csv(annotations: List[Annotation], output_path: str):
    """Write classified annotations to output CSV."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Start_Time",
            "End_Time",
            "Sentence",
            "Hard_Label",
            "Confidence",
            "Source_Format",
        ])
        for ann in annotations:
            writer.writerow([
                f"{ann.start_time:.4f}",
                f"{ann.end_time:.4f}",
                ann.text,
                ann.hard_label,
                f"{ann.confidence:.2f}",
                ann.source_format,
            ])


def print_summary(annotations: List[Annotation]):
    """Print classification summary statistics."""
    total = len(annotations)
    tool_contact = sum(1 for a in annotations if a.hard_label == "tool-contact")
    tool_noncontact = sum(1 for a in annotations if a.hard_label == "tool-noncontact")
    avg_confidence = sum(a.confidence for a in annotations) / max(total, 1)
    
    low_conf = sum(1 for a in annotations if a.confidence < 0.6)
    
    total_duration = sum(a.end_time - a.start_time for a in annotations)
    contact_duration = sum(
        a.end_time - a.start_time for a in annotations if a.hard_label == "tool-contact"
    )
    noncontact_duration = total_duration - contact_duration
    
    print("\n" + "=" * 60)
    print("CLASSIFICATION SUMMARY")
    print("=" * 60)
    print(f"  Total annotations:       {total}")
    print(f"  Tool-contact:            {tool_contact} ({100*tool_contact/max(total,1):.1f}%)")
    print(f"  Tool-noncontact:         {tool_noncontact} ({100*tool_noncontact/max(total,1):.1f}%)")
    print(f"  Average confidence:      {avg_confidence:.2f}")
    print(f"  Low confidence (<0.6):   {low_conf}")
    print(f"  Total duration:          {total_duration:.1f}s ({total_duration/60:.1f}min)")
    print(f"  Tool-contact duration:   {contact_duration:.1f}s ({contact_duration/60:.1f}min)")
    print(f"  Tool-noncontact duration:{noncontact_duration:.1f}s ({noncontact_duration/60:.1f}min)")
    print("=" * 60)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convert soft-label CSV annotations to hard tool-contact labels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Rule-based classification (no API needed)
  python softlabel_to_hardlabel.py \\
    --input WhisperLabel_Vol3.csv \\
    --output hardlabel_vol3.csv

  # LLM-based classification
  python softlabel_to_hardlabel.py \\
    --input processed_vol2_SoftLabels_Rich.csv \\
    --output hardlabel_vol2.csv \\
    --use-llm --api-key sk-...

  # Using a local LLM endpoint (e.g., Ollama, vLLM, llama.cpp)
  python softlabel_to_hardlabel.py \\
    --input data.csv --output out.csv \\
    --use-llm --api-base http://localhost:11434/v1 \\
    --model llama3 --api-key dummy
        """,
    )
    
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to input soft-label CSV (WhisperLabel or SoftLabels_Rich format)"
    )
    parser.add_argument(
        "--output", "-o", required=True,
        help="Path to write the hard-label output CSV"
    )
    parser.add_argument(
        "--use-llm", action="store_true",
        help="Use LLM API for classification (default: rule-based)"
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key for LLM service (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--api-base", default="https://api.openai.com/v1",
        help="Base URL for OpenAI-compatible API (default: OpenAI)"
    )
    parser.add_argument(
        "--model", default="gpt-4o-mini",
        help="LLM model name (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=20,
        help="Batch size for LLM API calls (default: 20)"
    )
    parser.add_argument(
        "--skip-empty", action="store_true",
        help="Skip rows with empty text"
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not os.path.isfile(args.input):
        print(f"ERROR: Input file not found: {args.input}")
        sys.exit(1)
    
    # Load annotations
    print(f"Loading: {args.input}")
    annotations = load_annotations(args.input)
    print(f"Loaded {len(annotations)} annotations")
    
    # Optionally skip empty
    if args.skip_empty:
        before = len(annotations)
        annotations = [a for a in annotations if a.text.strip()]
        print(f"Skipped {before - len(annotations)} empty annotations")
    
    # Classify
    if args.use_llm:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("ERROR: --api-key or OPENAI_API_KEY env var required for LLM mode")
            sys.exit(1)
        
        print(f"Classifying with LLM ({args.model}) via {args.api_base}...")
        annotations = classify_with_llm(
            annotations,
            api_key=api_key,
            api_base=args.api_base,
            model=args.model,
            batch_size=args.batch_size,
        )
    else:
        print("Classifying with rule-based engine...")
        for ann in annotations:
            label, conf = classify_rule_based(ann)
            ann.hard_label = label
            ann.confidence = conf
    
    # Write output
    write_output_csv(annotations, args.output)
    print(f"\nOutput written to: {args.output}")
    
    # Summary
    print_summary(annotations)


if __name__ == "__main__":
    main()
