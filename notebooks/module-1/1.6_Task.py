"""
LinkedIn Post Generator + Refiner Pipeline
============================================
Step 1: Generate a raw LinkedIn post from a topic.
Step 2: Refine it using a chosen prompt-engineering technique.

Setup:
    pip install groq
    export GROQ_API_KEY="your_key_here"   # or set it in a .env file

Run:
    python linkedin_post_pipeline.py --topic "أهمية الـ RAG في بناء تطبيقات AI" --technique self_critique
    python linkedin_post_pipeline.py --topic "..." --technique all   # compares every technique
"""
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from dotenv import load_dotenv
import os
import argparse
from groq import Groq

load_dotenv(override=True)



# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL =  "openai/gpt-oss-120b"# fast + strong for creative writing
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
def call_model(messages: list, temperature: float = 0.8) -> str:
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1200,
        reasoning_effort="low",
    )
    content = response.choices[0].message.content
    return content.strip() if content else "[empty response — try raising max_tokens further]"

# ---------------------------------------------------------------------------
# STEP 1 — Generate the raw post
# ---------------------------------------------------------------------------
def generate_post(topic: str, tone: str = "professional yet personal") -> str:
    system = (
        "You are a LinkedIn content writer. Write posts that feel human, "
        "start with a strong hook in the first line, avoid corporate buzzwords, "
        "and end with a light call-to-engagement (a question or an invite to comment)."
    )
    user = (
        f"Write a LinkedIn post about: {topic}\n"
        f"Tone: {tone}\n"
        "Length: 120-180 words. Use short paragraphs (1-3 lines each) and 2-4 relevant hashtags at the end."
    )
    return call_model(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.9,
    )


# ---------------------------------------------------------------------------
# STEP 2 — Refinement techniques (this is the "playground" part of the project)
# ---------------------------------------------------------------------------

def refine_role_prompting(post: str) -> str:
    """Technique: Role / Persona prompting."""
    system = (
        "You are an award-winning LinkedIn ghostwriter who has grown multiple "
        "creator accounts past 100k followers. You know exactly what makes a "
        "post stop the scroll."
    )
    user = f"Rewrite this LinkedIn post to be sharper and more engaging, keep the core message:\n\n{post}"
    return call_model([{"role": "system", "content": system}, {"role": "user", "content": user}])


def refine_chain_of_thought(post: str) -> str:
    """Technique: Chain-of-thought — analyze weaknesses first, then rewrite."""
    system = "You are a LinkedIn editor. Think step by step before rewriting."
    user = (
        "First, list 3 concrete weaknesses in this post (hook strength, clarity, "
        "engagement, structure). Then, based on that analysis, write an improved version.\n"
        "Format your answer as:\nANALYSIS:\n...\nIMPROVED POST:\n...\n\n"
        f"POST:\n{post}"
    )
    return call_model([{"role": "system", "content": system}, {"role": "user", "content": user}])


def refine_few_shot(post: str) -> str:
    """Technique: Few-shot — show examples of high-performing posts as style reference."""
    system = "You are a LinkedIn content writer. Match the style patterns shown in the examples."
    examples = (
        "EXAMPLE 1 (high engagement):\n"
        "I got rejected 47 times before my first client said yes.\n\n"
        "Here's what nobody tells you about starting out...\n\n"
        "EXAMPLE 2 (high engagement):\n"
        "Everyone says 'work smarter, not harder.'\n\n"
        "Nobody tells you what that actually means. So here's my breakdown...\n"
    )
    user = f"{examples}\n\nUsing a similar hook style, rewrite this post:\n\n{post}"
    return call_model([{"role": "system", "content": system}, {"role": "user", "content": user}])


def refine_self_critique(post: str) -> str:
    """Technique: Self-critique loop — model critiques its own output, then fixes it."""
    critique_prompt = (
        "Critique this LinkedIn post harshly but constructively. "
        "Focus on: hook strength, clarity, authenticity, and whether it invites engagement.\n\n"
        f"{post}"
    )
    critique = call_model([{"role": "user", "content": critique_prompt}])

    fix_prompt = (
        f"Original post:\n{post}\n\n"
        f"Critique:\n{critique}\n\n"
        "Now rewrite the post addressing every point in the critique."
    )
    return call_model([{"role": "user", "content": fix_prompt}])


def refine_constraint_based(post: str) -> str:
    """Technique: Explicit constraints — hard rules the model must follow."""
    system = (
        "Rewrite the post following these constraints exactly:\n"
        "- First line must be under 10 words and create curiosity\n"
        "- No sentence longer than 15 words\n"
        "- Exactly one question at the end\n"
        "- Exactly 3 hashtags, no more\n"
        "- No emojis, no buzzwords like 'game-changer' or 'unlock'"
    )
    return call_model([{"role": "system", "content": system}, {"role": "user", "content": post}])


TECHNIQUES = {
    "role_prompting": refine_role_prompting,
    "chain_of_thought": refine_chain_of_thought,
    "few_shot": refine_few_shot,
    "self_critique": refine_self_critique,
    "constraint_based": refine_constraint_based,
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_pdf(topic: str, raw_post: str, refined: dict, output_path: str) -> None:
    """Write the topic, raw post, and each refined version to a PDF file."""
    styles = getSampleStyleSheet()
    heading_style = ParagraphStyle(
        "SectionHeading", parent=styles["Heading2"], textColor=colors.HexColor("#0A66C2"),
        spaceBefore=16, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "Body", parent=styles["Normal"], fontSize=10.5, leading=15, spaceAfter=10,
    )
    title_style = styles["Title"]
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], textColor=colors.grey, fontSize=9)

    def as_paragraphs(text: str):
        blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
        return [Paragraph(b.replace("\n", "<br/>"), body_style) for b in blocks]

    story = [
        Paragraph("LinkedIn Post — Generation &amp; Refinement", title_style),
        Paragraph(f"Topic: {topic}", meta_style),
        Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", meta_style),
        Spacer(1, 12),
        Paragraph("Step 1 — Raw Generated Post", heading_style),
        *as_paragraphs(raw_post),
    ]

    for name, text in refined.items():
        story.append(Paragraph(f"Step 2 — Refined ({name})", heading_style))
        story.extend(as_paragraphs(text))

        doc = SimpleDocTemplate(output_path, pagesize=letter,
                             topMargin=50, bottomMargin=50, leftMargin=50, rightMargin=50)

    parent_dir = os.path.dirname(output_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    doc.build(story)
  
def main():
    parser = argparse.ArgumentParser(description="LinkedIn post generator + refiner")
    parser.add_argument("--topic", required=True, help="Topic to write the post about")
    parser.add_argument("--tone", default="professional yet personal", help="Tone of the post")
    parser.add_argument(
        "--technique",
        default="self_critique",
        choices=list(TECHNIQUES.keys()) + ["all"],
        help="Which refinement technique to apply",
    )
    parser.add_argument(
        "--pdf",
        nargs="?",
        const="linkedin_post.pdf",
        default=None,
        help="Also save the results to a PDF. Optionally pass a file path, "
             "e.g. --pdf output/post.pdf (defaults to linkedin_post.pdf)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("STEP 1 — Raw generated post")
    print("=" * 70)
    raw_post = generate_post(args.topic, args.tone)
    print(raw_post, "\n")

    refined_results = {}

    if args.technique == "all":
        for name, fn in TECHNIQUES.items():
            print("=" * 70)
            print(f"STEP 2 — Refined with technique: {name}")
            print("=" * 70)
            result = fn(raw_post)
            print(result, "\n")
            refined_results[name] = result
    else:
        print("=" * 70)
        print(f"STEP 2 — Refined with technique: {args.technique}")
        print("=" * 70)
        result = TECHNIQUES[args.technique](raw_post)
        print(result)
        refined_results[args.technique] = result

    if args.pdf:
        build_pdf(args.topic, raw_post, refined_results, args.pdf)
        print(f"\nPDF saved to: {os.path.abspath(args.pdf)}")


if __name__ == "__main__":
    main()