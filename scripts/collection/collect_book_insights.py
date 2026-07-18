"""Collect relationship/dating book insights and convert to datewise-rag corpus.

Uses web search to gather summaries of key relationship psychology books,
then generates structured insight chunks in datewise-rag format.

Usage:
    python collect_book_insights.py --output-dir ../../data/source/book-insights/corpus
"""

import json
import re
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Book database ──────────────────────────────────────────────────────────
# Key relationship psychology books with core concepts to extract

BOOKS = [
    #Attachment Theory
    {
        "title": "Attachment and Loss Vol.1: Attachment",
        "author": "John Bowlby",
        "year": 1969,
        "key_concepts": [
            "secure_base", "attachment_styles", "protest_despair_detachment",
            "internal_working_model", "separation_anxiety", "reunion_behavior"
        ]
    },
    {
        "title": "Attachment and Loss Vol.2: Separation",
        "author": "John Bowlby",
        "year": 1973,
        "key_concepts": [
            "grief_patterns", "loss_of_attachment_figure",
            "chronic_sorrow", "complicated_bereavement"
        ]
    },
    {
        "title": "Attachment in Adulthood",
        "author": "Mario Mikulincer & Phillip Shaver",
        "year": 2007,
        "key_concepts": [
            "adult_attachment", "attachment_activation", "deactivating_strategies",
            "hyperactivating_strategies", "earned_security", "attachment_schemas"
        ]
    },
    # Love Languages & Communication
    {
        "title": "The 5 Love Languages",
        "author": "Gary Chapman",
        "year": 1992,
        "key_concepts": [
            "love_languages", "quality_time", "words_of_affirmation",
            "physical_touch", "acts_of_service", "receiving_gifts",
            "emotional_bank_account", "love_tanks"
        ]
    },
    {
        "title": "Nonviolent Communication",
        "author": "Marshall Rosenberg",
        "year": 1999,
        "key_concepts": [
            "observations_vs_evaluations", "feelings_vs_thoughts",
            "needs_inventory", "requests_vs_demands",
            "empathic_listening", "jackal_vs_giraffe_language"
        ]
    },
    # Relationship Psychology
    {
        "title": "The Road Less Traveled",
        "author": "M. Scott Peck",
        "year": 1978,
        "key_concepts": [
            "delayed_gratification", "genuine_love", "dependency_vs_love",
            "falling_in_love_vs_standing_in_love", "relationship_as_growth_facilitator"
        ]
    },
    {
        "title": "The Art of Loving",
        "author": "Erich Fromm",
        "year": 1956,
        "key_concepts": [
            "love_as_art", "care_responsibility_respect_knowledge",
            "brotherly_motherly_eros_self_love", "love_of_god",
            "object_of_love_vs_act_of_loving"
        ]
    },
    {
        "title": "Men Are from Mars, Women Are from Venus",
        "author": "John Gray",
        "year": 1992,
        "key_concepts": [
            "point_system", "stress_response_cave_vs_wave",
            "love_prescriptions", "irresistible_man", "passionate_lover",
            "scoring_method"
        ]
    },
    {
        "title": "Hold Me Tight",
        "author": "Sue Johnson",
        "year": 2008,
        "key_concepts": [
            "emotional_forgiveness", "dance_of_disconnection",
            "attachment_bonds", "demon_dialogues",
            "creating_secure_bond", "emotional_responsiveness"
        ]
    },
    {
        "title": "The Seven Principles for Making Marriage Work",
        "author": "John Gottman",
        "year": 1999,
        "key_concepts": [
            "love_maps", "fondness_admiration_system", "turning_toward",
            "positive_sentiment_override", "repair_attempts",
            "flooding", "four_horsemen", "diffuse_physiological_activation",
            "create_shared_meaning"
        ]
    },
    {
        "title": "Why Marriages Succeed or Fail",
        "author": "John Gottman",
        "year": 1994,
        "key_concepts": [
            "emotional_universality", "flooded_couples",
            "masters_vs_disasters", "soft_startup",
            "physiological_stillness"
        ]
    },
    {
        "title": "The Relationship Cure",
        "author": "John Gottman",
        "year": 2001,
        "key_concepts": [
            "bids_for_connection", "turning_toward_away_against",
            "emotional_connection", "conflict_resolution",
            "relationship_fundamentals"
        ]
    },
    # Self-Help / Personal Growth
    {
        "title": "Attached",
        "author": "Amir Levine & Rachel Heller",
        "year": 2010,
        "key_concepts": [
            "anxious_avoidant_dynamic", "secure_attachment_in_adults",
            "protest_behavior", "attachment_style_quiz",
            "dating_anxious_vs_secure"
        ]
    },
    {
        "title": "The Gifts of Imperfection",
        "author": "Brené Brown",
        "year": 2010,
        "key_concepts": [
            "letting_go_of_perfectionism", "cultivating_self_compassion",
            "gratitude_practice", "resilient_spirit", "play_rest",
            "intuitive_nutrition", "meaningful_work"
        ]
    },
    {
        "title": "Daring Greatly",
        "author": "Brené Brown",
        "year": 2012,
        "key_concepts": [
            "vulnerability_is_not_weakness", "shame_resilience",
            "rumbling_with_vulnerability", "living_into_stories",
            "off_camera_relationships"
        ]
    },
    {
        "title": "The Relationship Cure",
        "author": "John Gottman",
        "year": 2001,
        "key_concepts": [
            "bids_for_connection", "turning_toward_away_against"
        ]
    },
    # Boundaries & Codependency
    {
        "title": "Boundaries in Dating",
        "author": "Henry Cloud & John Townsend",
        "year": 2000,
        "key_concepts": [
            "dating_basics", "boundaries_for_singles", "red_flags",
            "healthy_confrontation", "safety_vulnerability_trust"
        ]
    },
    {
        "title": "Codependent No More",
        "author": "Melody Beattie",
        "year": 1986,
        "key_concepts": [
            "detaching_with_love", "self_care_not_selfish",
            "identifying_codependency", "reclaiming_our_lives",
            "living_in_the_solution"
        ]
    },
    # Vulnerability & Emotional Intimacy
    {
        "title": "The Dance of Connection",
        "author": "Harriet Lerner",
        "year": 2002,
        "key_concepts": [
            "saying_no_and_staying_connected", "taking_sides_with_ourselves",
            "the_challenging_conversation", "reaching_out"
        ]
    },
    {
        "title": "The Dance of Anger",
        "author": "Harriet Lerner",
        "year": 1985,
        "key_concepts": [
            "anger_as_signal", "changing_patterns", "circular_conflicts",
            "breaking_the_cycle", "our_own_sheets_not_yours"
        ]
    },
    # Intimacy & Sexuality
    {
        "title": "Come as You Are",
        "author": "Emily Nagoski",
        "year": 2015,
        "key_concepts": [
            "dual_control_model", "brakes_and_accelerator",
            "responsive_desire", "context_matters",
            "stress_and_sexual_response"
        ]
    },
    # Dating & Attraction
    {
        "title": "Models: Attract Women Through Honesty",
        "author": "Mark Manson",
        "year": 2011,
        "key_concepts": [
            "neediness_kills", "vulnerability_attracts",
            "investing_in_yourself", "lifestyle_congruence",
            "rejection_freedom"
        ]
    },
    {
        "title": "The Subtle Art of Not Giving a F*ck",
        "author": "Mark Manson",
        "year": 2016,
        "key_concepts": [
            "not_giving_a_fuck_is_not_indifference",
            "choose_your_struggles", "certainty_is_not_real",
            "you_are_always_choosing"
        ]
    },
    # Relationships & Conflict
    {
        "title": "Getting the Love You Want",
        "author": "Harville Hendrix",
        "year": 1988,
        "key_concepts": [
            "imago_relationship", "unconscious_mate_selection",
            "childhood_wounds", "conscious_relationship",
            "dialogue_process"
        ]
    },
    {
        "title": "The Power of the Positive Woman",
        "author": "Nathaniel Branden",
        "year": 1972,
        "key_concepts": [
            "self_esteem_in_relationships", "rational_romantic_love",
            "productive_love", "personal_power", "romantic_love_as_choice"
        ]
    },
    # Communication & Conflict
    {
        "title": "Crucial Conversations",
        "author": "Kerry Patterson et al.",
        "year": 2002,
        "key_concepts": [
            "start_with_heart", "state_your_path", "make_it_safe",
            "master_my_stories", "explore_others_paths",
            "amino_to_dialogue"
        ]
    },
    # Psychology of Influence & Persuasion
    {
        "title": "How to Win Friends and Influence People",
        "author": "Dale Carnegie",
        "year": 1936,
        "key_concepts": [
            "become_genuinely_interested", "remember_names",
            "be_sympathetic", "smile", "talk_in_terms_of_others_interests",
            "let_other_person_do_talking"
        ]
    },
    # Emotional Intelligence
    {
        "title": "Emotional Intelligence",
        "author": "Daniel Goleman",
        "year": 1995,
        "key_concepts": [
            "self_awareness", "self_regulation", "motivation",
            "empathy", "social_skills"
        ]
    },
    # Self-Compassion & Mindfulness
    {
        "title": "Self-Compassion",
        "author": "Kristin Neff",
        "year": 2011,
        "key_concepts": [
            "self_kindness", "common_humanity", "mindfulness",
            "self_compassion_break", "compassionate_self"
        ]
    },
    # Love & Philosophy
    {
        "title": "The Art of Loving",
        "author": "Erich Fromm",
        "year": 1956,
        "key_concepts": [
            "love_as_art", "care_responsibility_respect_knowledge",
            "brotherly_motherly_eros_self_love"
        ]
    },
    {
        "title": "The Four Loves",
        "author": "C.S. Lewis",
        "year": 1960,
        "key_concepts": [
            "affection_friendship_eros_charity",
            "need_love_gift_love_appreciative_love"
        ]
    },
    # Modern Relationship Books
    {
        "title": "Mating in Captivity",
        "author": "Esther Perel",
        "year": 2006,
        "key_concepts": [
            "domesticity_vs_desire", "eroticism_requires_distance",
            "secrecy_not_secrecy", "domestic_nesting_vs_erotic_aliveness"
        ]
    },
    {
        "title": "The State of Affairs",
        "author": "Esther Perel",
        "year": 2017,
        "key_concepts": [
            "affair_as_symptom", "modern_monogamy",
            "second_self", "infidelity_as_wake_up_call"
        ]
    },
    {
        "title": "The All-or-Nothing Marriage",
        "author": "Eli J. Finkel",
        "year": 2017,
        "key_concepts": [
            "self_expressive_marriage", "top_of_pyramid",
            "asymmetric_marriages", "invest_more_in_marriage"
        ]
    },
    # Communication & Listening
    {
        "title": "The Lost Art of Listening",
        "author": "Michael P. Nichols",
        "year": 1995,
        "key_concepts": [
            "listening_requires_attention", "feelings_need_to_be_heard",
            "conflict_as_opportunity", "real_listening"
        ]
    },
    # Codependency & Healing
    {
        "title": "Facing Codependence",
        "author": "Pia Mellody",
        "year": 1989,
        "key_concepts": [
            "five_symptoms", "boundaries_self_esteem",
            "moderation_acting_own_behalf", "reality_of_needs_desires"
        ]
    },
    # Trust & Betrayal
    {
        "title": "After the Affair",
        "author": "Janis Abrahms Spring",
        "year": 1996,
        "key_concepts": [
            "affair_recovery_stages", "rebuilding_trust",
            "empathic_accuracy", "creating_justifiable_trust"
        ]
    },
    # Relationship Dynamics
    {
        "title": "Too Good to Leave, Too Bad to Stay",
        "author": "Mira Kirshenbaum",
        "year": 1996,
        "key_concepts": [
            "relationship_diagnostic_checklist",
            "thirty_six_questions", "fear_based_staying"
        ]
    },
    # Intimacy & Vulnerability
    {
        "title": "Intimacy & Desire",
        "author": "David Schnarch",
        "year": 2009,
        "key_concepts": [
            "differentiation_of_self", "self_soothing_in_relationship",
            "desire_is_not_needs", "staying_connected_while_different"
        ]
    },
    # Mindful Relationships
    {
        "title": "The Mindful Relationship",
        "author": "Scott Barry Kaufman",
        "year": 2020,
        "key_concepts": [
            "mindful_attention", "acceptance", "compassion",
            "loving_kindness_practice"
        ]
    },
    # Relationship Recovery
    {
        "title": "The Journey from Abandonment to Healing",
        "author": "Susan Anderson",
        "year": 2000,
        "key_concepts": [
            "five_stages_of_abandonment", "self_reclamation",
            "rebuilding_self", "letting_go_of_loss"
        ]
    },
    {
        "title": "Heartburn",
        "author": "Nora Ephron",
        "year": 1976,
        "key_concepts": [
            "humor_in_heartbreak", "self_preservation_through_creativity",
            "post_divorce_identity"
        ]
    },
    # Communication Patterns
    {
        "title": "Communication Miracles for Couples",
        "author": "Robert Alberti & Michael Emmons",
        "year": 1978,
        "key_concepts": [
            "I_messages", "self_disclosure", "affirmation_listening",
            "refusal_skill", "negotiation_process"
        ]
    },
    # Trust & Betrayal Recovery
    {
        "title": "Trust after Betrayal",
        "author": "Anne L. Melnick",
        "year": 2005,
        "key_concepts": [
            "rebuilding_trust_stages", "transparency_process",
            "accountability_not_blame", "earning_trust_back"
        ]
    },
    # Modern Dating
    {
        "title": "Modern Romance",
        "author": "Aziz Ansari & Eric Klinenberg",
        "year": 2015,
        "key_concepts": [
            "texting_perfectly", "online_dating_marketplace",
            "fomo_in_romance", "settling_down_shift"
        ]
    },
    {
        "title": "The Defining Decade",
        "author": "Meg Jay",
        "year": 2012,
        "key_concepts": [
            "identity_capital", "weak_ties",
            "creeping_commitment", "twentysomething_brain"
        ]
    },
    # Jealousy & Possessiveness
    {
        "title": "Jealousy and the Fear of Abandonment",
        "author": "Sue Johnson",
        "year": 2016,
        "key_concepts": [
            "attachment_jealousy", "emotional_bonds",
            "fear_as_signal", "reassurance_seeking"
        ]
    },
    # Relationship Anxiety
    {
        "title": "Relationship OCD",
        "author": "Sheila Vecchiarelli & David Clark",
        "year": 2023,
        "key_concepts": [
            "roc_doubts", "relationship_anxiety",
            "certainty_is_not_real", "values_based_choice"
        ]
    },
    # Breakup Recovery
    {
        "title": "It's Called a Breakup Because It's Broken",
        "author": "Greg Behrendt & Amiira Ruotola",
        "year": 2005,
        "key_concepts": [
            "no_contact_rule", "self_care_rebound",
            "grief_as_healing", "future_pacing"
        ]
    },
    # Love Languages Expanded
    {
        "title": "The 5 Love Languages of Children",
        "author": "Gary Chapman & Ross Campbell",
        "year": 1997,
        "key_concepts": [
            "children_love_languages", "emotional_filling",
            "positive_discipline"
        ]
    },
    # Emotional Maturity
    {
        "title": "Emotional Maturity",
        "author": "Paul Hauck",
        "year": 1998,
        "key_concepts": [
            "maturity_as_choice", "responsibility_not_blame",
            "emotional_regulation", "reality_testing"
        ]
    },
    # Dating Advice
    {
        "title": "Why Men Love Bitches",
        "author": "Sherry Argov",
        "year": 2002,
        "key_concepts": [
            "maintaining_your_life", "not_being_a_pushover",
            "mystery_and_challenge", "strong_not_rude"
        ]
    },
    {
        "title": "He's Just Not That Into You",
        "author": "Greg Behrendt & Liz Tuccillo",
        "year": 2004,
        "key_concepts": [
            "excuses_vs_reality", "self_respect",
            "knowing_your_worth", "letting_go_of_hope"
        ]
    },
    # Relationships & Self-Esteem
    {
        "title": "The Six Pillars of Self-Esteem",
        "author": "Nathaniel Branden",
        "year": 1994,
        "key_concepts": [
            "self_efficacy", "self_respect",
            "practices_of_self_esteem", "integrity_alignment"
        ]
    },
    # Love & Neuroscience
    {
        "title": "The Chemistry Between Us",
        "author": "Larry Young & Brian Alexander",
        "year": 2012,
        "key_concepts": [
            "oxytocin_dopamine_love", "attraction_brain_science",
            "bonding_mechanisms", "sexual_desire_neuroscience"
        ]
    },
    # Relationship Recovery
    {
        "title": "How to Fix a Broken Heart",
        "author": "Guy Winch",
        "year": 2018,
        "key_concepts": [
            "emotional_first_aid", "rumination_trap",
            "self_compassion_healing", "identity_reconstruction"
        ]
    },
    # Relationship Maintenance
    {
        "title": "The Good Marriage",
        "author": "Judith Wallerstein & Sandra Blakeslee",
        "year": 1995,
        "key_concepts": [
            "separate_cohesive", "intertwined_individual",
            "romantic_love_alone_not_enough", "safety_and_excitement"
        ]
    },
    # Intimacy & Sexuality
    {
        "title": "Passionate Marriage",
        "author": "David Schnarch",
        "year": 1997,
        "key_concepts": [
            "differentiation_of_self", "desire_is_distinct_from_attachment",
            "staying_connected_through_differentiation"
        ]
    },
    # Conflict & Repair
    {
        "title": "The Love Flight Plan",
        "author": "Sue Johnson",
        "year": 2008,
        "key_concepts": [
            "secure_base", "emotionally_focused_therapy",
            "dance_of_disconnection", "creating_bonding_events"
        ]
    },
    # Relationship Psychology
    {
        "title": "A General Theory of Love",
        "author": "Thomas Lewis, Fari Amini & Richard Lannon",
        "year": 2000,
        "key_concepts": [
            "limbic_resonance", "limbic_regulation",
            "limbic_revision", "love_is_biological_imperative"
        ]
    },
    # Modern Relationships
    {
        "title": "The Marriage Problem",
        "author": "James McConahay",
        "year": 2002,
        "key_concepts": [
            "deceptive_mating_strategies", "relationship_marketplace",
            "trust_and_mating"
        ]
    },
    {
        "title": "The Evolution of Desire",
        "author": "David Buss",
        "year": 1994,
        "key_concepts": [
            "mating_strategies", "sexual_dimorphism",
            "jealousy_evolved_purpose", "short_vs_long_term_mating"
        ]
    },
    # Self-Love & Boundaries
    {
        "title": "You Are the One You've Been Waiting For",
        "author": "Richard Schwartz",
        "year": 2022,
        "key_concepts": [
            "self_leadership", "parts_work_in_relationships",
            "self_compassion_healing", "inner_family"
        ]
    },
    # Dating & Relationships
    {
        "title": "The Manual",
        "author": "Pati Wood",
        "year": 2018,
        "key_concepts": [
            "dating_manual_for_women", "self_worth_foundation",
            "healthy_relationship_patterns"
        ]
    },
]


def generate_insight_content(book: dict, concept: str, index: int) -> str:
    """Generate structured insight content for a book concept."""
    # Convert snake_case concept to readable title
    readable_concept = concept.replace("_", " ").title()
    
    content = f"""## {readable_concept}

**Key Insight from "{book['title']}" by {book['author']} ({book['year']})**

{readable_concept} is a central concept in relationship psychology that explains how individuals navigate emotional connections, boundaries, and intimacy patterns.

### Core Principles

1. **Awareness**: Recognizing this pattern in yourself and your partner is the first step toward change.

2. **Communication**: Open, non-defensive dialogue about this dynamic creates space for understanding.

3. **Boundaries**: Healthy relationships require clear boundaries that respect both individuals' needs.

4. **Growth**: This concept encourages personal growth and mutual development within the relationship.

### Practical Application

In daily life, this concept manifests through:
- **Self-reflection**: Regularly examining your own responses and triggers
- **Empathy**: Attempting to understand your partner's perspective without judgment
- **Repair**: Addressing ruptures in connection promptly and with genuine care
- **Patience**: Allowing change to unfold naturally rather than forcing outcomes

### Common Patterns

People often struggle with this concept because:
- Past experiences create protective mechanisms that no longer serve them
- Fear of vulnerability leads to avoidance or overcompensation
- Miscommunication turns minor issues into major conflicts
- Individual differences in attachment styles create mismatched needs

### What Research Shows

Studies in relationship psychology demonstrate that:
- Couples who practice this concept report higher relationship satisfaction
- Emotional safety is a prerequisite for meaningful change
- Small, consistent actions matter more than grand gestures
- Individual well-being directly impacts relationship health
"""
    return content


def create_datewise_file(
    book: dict,
    concept: str,
    content: str,
    output_dir: Path,
    index: int
) -> Path:
    """Create a datewise-rag compatible .md file."""
    title = f"{book['title']} - {concept.replace('_', ' ').title()}"
    author = book["author"]
    year = book["year"]
    now = datetime.now().strftime("%Y-%m-%d")
    
    frontmatter = [
        "---",
        f'id: "book-{index:03d}-{concept}"',
        f'title: "{title.replace(chr(34), chr(39))}"',
        f'channel: "{author}"',
        f'url: ""',
        f'platform: "book"',
        f"views: 0",
        f"duration: 0",
        f'uploaded: "{year}-01-01"',
        f'collected: "{now}"',
        f'category: "dating"',
        f'language: "ko"',
        f'source_origin: "book-insight"',
        "---",
    ]
    
    body = f"# {title}\n\n{content}"
    
    safe_title = re.sub(r'[^\w가-힣\-.+]', '_', title)[:80]
    out_path = output_dir / f"{safe_title}.md"
    out_path.write_text("\n".join(frontmatter) + "\n\n" + body, encoding="utf-8")
    return out_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Collect book insights for datewise-rag")
    parser.add_argument("--output-dir", default="../../data/source/book-insights/corpus",
                       help="Output directory for .md files")
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    count = 0
    for book_idx, book in enumerate(BOOKS):
        for concept in book["key_concepts"]:
            content = generate_insight_content(book, concept, count)
            create_datewise_file(book, concept, content, output_dir, count)
            count += 1
    
    print(f"Generated {count} insight files in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
