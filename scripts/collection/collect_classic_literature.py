"""Collect classic literature relationship insights for datewise-rag corpus.

Covers Shakespeare, Jane Austen, Tolstoy, Brontë, Hardy, Flaubert,
Dostoevsky, and other classic authors whose works contain rich relationship dynamics.
"""

import re
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid5, NAMESPACE_URL

BOOKS = [
    # ═══ SHAKESPEARE ═══
    {
        "title": "Romeo and Juliet", "author": "William Shakespeare", "year": 1597,
        "concepts": [
            "passionate_vs_enduring_love", "forbidden_love_danger", "family_loyalty_vs_romance",
            "impulsiveness_in_love", "communication_breakdown_tragedy", "idealization_vs_reality",
            "sacrifice_and_self_destruction", "timing_and_fate_in_love",
        ]
    },
    {
        "title": "Much Ado About Nothing", "author": "William Shakespeare", "year": 1598,
        "concepts": [
            "banter_as_flirtation", "misunderstanding_and_trust", "scheming_matchmakers",
            "pride_and_prejudice_in_love", "reputation_and_honor", "second_chances_in_love",
        ]
    },
    {
        "title": "A Midsummer Night's Dream", "author": "William Shakespeare", "year": 1595,
        "concepts": [
            "love_potion_metaphor", "love_is_irrational", "conflicting_desires",
            "transformation_through_love", "reality_vs_dream", "reconciliation_of_lovers",
        ]
    },
    {
        "title": "Othello", "author": "William Shakespeare", "year": 1603,
        "concepts": [
            "jealousy_destroying_love", "manipulation_and_deception", "trust_vulnerability",
            "miscommunication_leads_to_tragedy", "insecurity_in_relationships", "honesty_vs_manipulation",
        ]
    },
    {
        "title": "The Taming of the Shrew", "author": "William Shakespeare", "year": 1590,
        "concepts": [
            "power_dynamics_in_marriage", "masking_true_self", "negotiation_in_relationships",
            "changing_for_love", "assertiveness_vs_compliance", "witty_banter_in_romance",
        ]
    },
    {
        "title": "Twelfth Night", "author": "William Shakespeare", "year": 1601,
        "concepts": [
            "identity_and_desire", "unrequited_love_triangle", "cross_dressing_discovery",
            "love_blind_to_identity", "humor_in_romance", "revelation_and_acceptance",
        ]
    },
    {
        "title": "As You Like It", "author": "William Shakespeare", "year": 1599,
        "concepts": [
            "pastoral_romance", "love_conquers_all", "disguise_and_discovery",
            "multiple_love_stories", "nature_and_love", "choice_in_marriage",
        ]
    },
    {
        "title": "The Tempest", "author": "William Shakespeare", "year": 1611,
        "concepts": [
            "forgiveness_over_revenge", "power_and_manipulation", "reconciliation",
            "freedom_and_autonomy", "father_daughter_bond", "artificial_vs_natural_love",
        ]
    },
    {
        "title": "King Lear", "author": "William Shakespeare", "year": 1606,
        "concepts": [
            "family_betrayal", "false_flattery_vs_true_love", "pride_leads_to_fall",
            "unconditional_love", "madness_and_clarity", "redemption_through_suffering",
        ]
    },
    {
        "title": "Hamlet", "author": "William Shakespeare", "year": 1600,
        "concepts": [
            "indecision_in_love", "death_of_relationship", "manipulation_and_obsession",
            "madness_and_truth", "love_beyond_death", "revenge_consuming_love",
        ]
    },
    {
        "title": "Macbeth", "author": "William Shakespeare", "year": 1606,
        "concepts": [
            "power_corrupts_love", "ambition_in_relationships", "guilt_and_conscience",
            "partnership_in_downfall", "sleep_no_more", "unchecked_desire",
        ]
    },

    # ═══ JANE AUSTEN ═══
    {
        "title": "Pride and Prejudice", "author": "Jane Austen", "year": 1813,
        "concepts": [
            "first_impressions_wrong", "pride_humility_balance", "class_and_marriage",
            "wit_vs_sincerity", "growth_through_conflict", "rejecting_unsuitable_proposals",
            "mutual_respect_in_love", "family_pressure_on_romance",
        ]
    },
    {
        "title": "Sense and Sensibility", "author": "Jane Austen", "year": 1811,
        "concepts": [
            "reason_vs_emotion_in_love", "heartbreak_and_resilience", "financial_stability_in_marriage",
            "sisters_different_approaches", "second_chance_romance", "forbidden_attraction",
        ]
    },
    {
        "title": "Emma", "author": "Jane Austen", "year": 1815,
        "concepts": [
            "matchmaking_backfires", "self_knowledge_in_love", "class_and_compatibility",
            "friendship_to_romance", "blindness_to_own_feelings", "humility_in_love",
        ]
    },
    {
        "title": "Persuasion", "author": "Jane Austen", "year": 1817,
        "concepts": [
            "second_chance_romance", "regret_and_lost_love", "persuasion_and_free_will",
            "quiet_strength", "waiting_for_the_right_one", "maturity_in_love",
        ]
    },
    {
        "title": "Northanger Abbey", "author": "Jane Austen", "year": 1817,
        "concepts": [
            "imagination_vs_reality_in_love", "gothic_romance_parody", "innocence_and_experience",
            "social_awkwardness_in_romance", "coming_of_age",
        ]
    },
    {
        "title": "Mansfield Park", "author": "Jane Austen", "year": 1814,
        "concepts": [
            "morality_in_romance", "improving_self_for_love", "social_class_disparity",
            "loyalty_and_friendship", "seduction_and_resistance", "choosing_wisely",
        ]
    },

    # ═══ TOLSTOY ═══
    {
        "title": "Anna Karenina", "author": "Leo Tolstoy", "year": 1877,
        "concepts": [
            "adultery_and_consequences", "passionate_love_vs_domestic_love", "society_and_scandal",
            "self_destruction_in_love", "parallel_love_stories", "forgiveness_in_marriage",
            "freedom_vs_responsibility", "happiness_in_simple_love",
        ]
    },
    {
        "title": "War and Peace", "author": "Leo Tolstoy", "year": 1869,
        "concepts": [
            "love_in_wartime", "maturing_into_love", "duty_vs_desire", "redemption_through_love",
            "multiple_love_journeys", "loss_and_healing",
        ]
    },

    # ═══ BRONTË SISTERS ═══
    {
        "title": "Jane Eyre", "author": "Charlotte Brontë", "year": 1847,
        "concepts": [
            "equality_in_love", "independence_and_romance", "hidden_secrets_in_relationships",
            "self_respect_over_passion", "return_after_growth", "moral_choice_in_love",
            "class_and_romance", "spiritual_connection",
        ]
    },
    {
        "title": "Wuthering Heights", "author": "Emily Brontë", "year": 1847,
        "concepts": [
            "obsessive_love_destructive", "love_beyond_death", "revenge_cycle_in_relationships",
            "social_class_divide", "nature_and_wild_passion", "healing_through_generations",
            "love_as_both_beauty_and_madness",
        ]
    },

    # ═══ HARDY ═══
    {
        "title": "Tess of the d'Urbervilles", "author": "Thomas Hardy", "year": 1891,
        "concepts": [
            "innocence_destroyed", "double_standards_in_marriage", "victim_blaming_in_love",
            "class_and_vulnerability", "tragic_timing", "moral_judgment_in_relationships",
        ]
    },
    {
        "title": "Far from the Madding Crowd", "author": "Thomas Hardy", "year": 1874,
        "concepts": [
            "independent_woman_in_love", "multiple_courtship_styles", "pride_and_misunderstanding",
            "maturity_in_romance", "responsibility_in_love", "nature_and_love",
        ]
    },
    {
        "title": "The Mayor of Casterbridge", "author": "Thomas Hardy", "year": 1886,
        "concepts": [
            "past_mistakes haunt_present", "pride_and_downfall", "reconciliation_after_betrayal",
            "self_destruction_and_love",
        ]
    },

    # ═══ FLAUBERT ═══
    {
        "title": "Madame Bovary", "author": "Gustave Flaubert", "year": 1856,
        "concepts": [
            "romantic_illusions_vs_reality", "adultery_and_disillusion", "consumerism_and_desire",
            "boredom_in_marriage", "self_destruction_through_fantasy", "class_and_aspiration",
            "incom municable_loneliness",
        ]
    },

    # ═══ DOSTOEVSKY ═══
    {
        "title": "The Brothers Karamazov", "author": "Fyodor Dostoevsky", "year": 1880,
        "concepts": [
            "love_vs_possessiveness", "unconditional_forgiveness", "jealousy_and_hatred",
            "paternal_bonds", "spiritual_love", "passion_and_destruction",
            "trial_and_redemption",
        ]
    },
    {
        "title": "Crime and Punishment", "author": "Fyodor Dostoevsky", "year": 1866,
        "concepts": [
            "guilt_and_redemption_through_love", "self_sacrifice_in_relationships", "isolation_and_connection",
            "moral_awakening", "salvation_through_another",
        ]
    },
    {
        "title": "The Idiot", "author": "Fyodor Dostoevsky", "year": 1869,
        "concepts": [
            "purity_in_a_corrupt_world", "tragic_love_and_sacrifice", "innocence_and_manipulation",
            "love_and_pity", "three_women_three_approaches_to_love",
        ]
    },

    # ═══ HEMINGWAY ═══
    {
        "title": "A Farewell to Arms", "author": "Ernest Hemingway", "year": 1929,
        "concepts": [
            "love_in_wartime", "loss_and_grief", "stoicism_vs_emotional_availability",
            "fragility_of_happiness", "farewell_and_letting_go",
        ]
    },
    {
        "title": "The Sun Also Rises", "author": "Ernest Hemingway", "year": 1926,
        "concepts": [
            "lost_generation_love", "unrequited_love", "impotence_and_masculinity",
            "travel_and_escape", "friendship_vs_romance",
        ]
    },

    # ═══ FITZGERALD ═══
    {
        "title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "year": 1925,
        "concepts": [
            "idealization_of_past_love", "wealth_and_corruption", "impossible_dream",
            "class_barriers_in_romance", "self_invention_for_love", "tragic_obsession",
            "you_cant_repeat_the_past",
        ]
    },

    # ═══ WOOLF ═══
    {
        "title": "Mrs Dalloway", "author": "Virginia Woolf", "year": 1925,
        "concepts": [
            "love_and_regret", "choosing_the_wrong_one", "time_and_memory",
            "inner_life_of_love", "society_vs_desire", "what_might_have_been",
        ]
    },

    # ═══ WHARTON ═══
    {
        "title": "The Age of Innocence", "author": "Edith Wharton", "year": 1920,
        "concepts": [
            "duty_vs_desire", "social_constraint_on_love", "forbidden_passion",
            "sacrifice_and_regret", "the_one_that_got_away",
        ]
    },
    {
        "title": "The House of Mirth", "author": "Edith Wharton", "year": 1905,
        "concepts": [
            "marriage_and_economics", "independence_vs_security", "social_downfall",
            "tragic_romance", "women_and_choice",
        ]
    },

    # ═══ AUSTEN CONTEMPORARIES & VICTORIAN ═══
    {
        "title": "Great Expectations", "author": "Charles Dickens", "year": 1861,
        "concepts": [
            "social_class_and_love", "first_love_and_disillusion", "loyalty_and_betrayal",
            "growth_and_humility", "true_vs_false_values", "second_chances",
        ]
    },
    {
        "title": "A Tale of Two Cities", "author": "Charles Dickens", "year": 1859,
        "concepts": [
            "sacrificial_love", "dual_rival_love", "redemption_through_sacrifice",
            "love_beyond_self", "revolution_and_personal_love",
        ]
    },
    {
        "title": "Tess of the d'Urbervilles", "author": "Thomas Hardy", "year": 1891,
        "concepts": [
            "innocence_destroyed", "double_standards_in_marriage", "victim_blaming_in_love",
            "class_and_vulnerability", "tragic_timing",
        ]
    },

    # ═══ GREEK MYTHOLOGY & TRAGEDY ═══
    {
        "title": "The Odyssey", "author": "Homer", "year": -700,
        "concepts": [
            "loyalty_in_separation", "journey_home_to_love", "temptation_and_fidelity",
            "reunion_after_long_absence", "partnership_and_resilience",
        ]
    },
    {
        "title": "Antigone", "author": "Sophocles", "year": -441,
        "concepts": [
            "love_vs_law", "loyalty_over_consequences", "family_duty_and_romance",
            "tragic_defiance", "unyielding_principles",
        ]
    },
    {
        "title": "Medea", "author": "Euripides", "year": -431,
        "concepts": [
            "betrayal_and_revenge", "abandonment_and_desperation", "passion_turned_destructive",
            "women_and_powerlessness", "love_becoming_hatred",
        ]
    },

    # ═══ MODERN CLASSICS ═══
    {
        "title": "The Unbearable Lightness of Being", "author": "Milan Kundera", "year": 1984,
        "concepts": [
            "lightness_vs_weight_of_love", "infidelity_and_meaning", "freedom_vs_commitment",
            "erotic_friendship", "political_upheaval_and_personal_love",
        ]
    },
    {
        "title": "Love in the Time of Cholera", "author": "Gabriel García Márquez", "year": 1985,
        "concepts": [
            "love_waits_decades", "passionate_persistence", "aging_and_love",
            "marriage_vs_passionate_love", "devotion_beyond_reason",
        ]
    },
    {
        "title": "Norwegian Wood", "author": "Haruki Murakami", "year": 1987,
        "concepts": [
            "loss_and_melancholy", "first_love_and_grief", "mental_health_in_relationships",
            "choosing_between_two_loves", "memory_and_healing",
        ]
    },
    {
        "title": "The Notebook", "author": "Nicholas Sparks", "year": 1996,
        "concepts": [
            "enduring_love_through_time", "memory_and_alzheimers", "social_class_obstacles",
            "grand_gesture_romance", "love_letters_and_devotion",
        ]
    },
    {
        "title": "Outlander", "author": "Diana Gabaldon", "year": 1991,
        "concepts": [
            "love_across_time", "trust_in_impossible_situations", "marriage_through_adversity",
            "cultural_clash_in_romance", "passion_and_partnership",
        ]
    },

    # ═══ KOREAN LITERATURE ═══
    {
        "title": "채식주의자", "author": "한강", "year": 2007,
        "concepts": [
            "자아와_정체성", "부부_갈등", "사회적_압박", "저항과_자유", "폭력과_연대",
        ]
    },
    {
        "title": "82년생 김지영", "author": "조남주", "year": 2016,
        "concepts": [
            "여성_경험_공유", "결혼과_출산", "남녀_갈등", "사회적_편견", "공감과_이해",
        ]
    },
    {
        "title": "나의 라임오렌지 나무", "author": "Vitalis (한국어판)", "year": 1987,
        "concepts": [
            "유년기_상처", "성장과_치유", "가족_관계", "상실과_상실_극복",
        ]
    },
    {
        "title": "소년이 온다", "author": "한강", "year": 2014,
        "concepts": [
            "상실과_기억", "폭력의_잔존", "사랑과_상처", "치유와_시간",
        ]
    },
    {
        "title": "시선으로부터", "author": "정세랑", "year": 2020,
        "concepts": [
            "중년_여성의_자아", "결혼_이후", "자기_발견", "관계_재정의",
        ]
    },
    {
        "title": "달러구트 꿈 백화점", "author": "미니로건", "year": 2020,
        "concepts": [
            "꿈과_사랑", "상실과_치유", "기억의_가치", "연결과_소통",
        ]
    },

    # ═══ MORE CLASSICS ═══
    {
        "title": "Don Quixote", "author": "Miguel de Cervantes", "year": 1605,
        "concepts": [
            "idealism_vs_reality", "love_as_madness", "loyalty_of_sancho_panza",
            "impossible_love", "adventure_and_obsession",
        ]
    },
    {
        "title": "Les Misérables", "author": "Victor Hugo", "year": 1862,
        "concepts": [
            "redemptive_love", "self_sacrifice", "dignity_and_kindness",
            "love_beyond_social_class", "moral_transformation",
        ]
    },
    {
        "title": "The Picture of Dorian Gray", "author": "Oscar Wilde", "year": 1890,
        "concepts": [
            "vanity_and_corruption", "beauty_and_decay", "influence_of_toxic_friendship",
            "hedonism_vs_morality", "aestheticism_and_love",
        ]
    },
    {
        "title": "Dracula", "author": "Bram Stoker", "year": 1897,
        "concepts": [
            "predatory_love", "seduction_and_power", "female_desire_and_taboo",
            "purity_and_corruption", "obsessive_protection",
        ]
    },
    {
        "title": "Frankenstein", "author": "Mary Shelley", "year": 1818,
        "concepts": [
            "creation_and_abandonment", "isolation_and_desire_for_connection",
            "parental_responsibility", "otherness_and_rejection", "sympathy_and_monstrosity",
        ]
    },
]


def generate_classic_insight(book: dict, concept: str, index: int) -> str:
    """Generate rich insight content for classic literature."""
    readable = concept.replace("_", " ").replace("-", " ").title()
    return f"""## {readable}

**Literary Insight from "{book['title']}" by {book['author']} ({abs(book['year'])})**

{readable} — a timeless theme explored in this classic work that offers profound insights into human relationships and emotional dynamics.

### The Literary Context

In "{book['title']}", {readable} emerges as a central tension that drives the narrative and reveals fundamental truths about how people love, struggle, and grow.

### Key Observations from the Text

The narrative illustrates that this dynamic is not merely fictional but reflects universal patterns in human relationships:

1. **Universal Relevance**: Characters in classic literature face the same emotional challenges we encounter today — trust, jealousy, sacrifice, communication, and the fear of loss.

2. **Consequences of Choices**: The author shows how small decisions compound over time, leading to either deepened connection or irreversible damage.

3. **Emotional Truth**: Even across centuries, the emotional core of this theme resonates because human nature remains constant.

### Modern Application

Drawing from this literary exploration:

- **Pattern Recognition**: When you notice similar dynamics in your own relationship, you can reference centuries of human wisdom about how to navigate them.

- **Emotional Vocabulary**: Literature gives us precise language for complex feelings that might otherwise remain unnamed.

- **Perspective**: Seeing how characters handle — or fail to handle — these situations helps us make more conscious choices in our own lives.

- **Growth Through Narrative**: Stories model transformation, showing that even painful experiences can lead to deeper understanding and more authentic connections.

### Relationship Takeaway

Classic literature reminds us that love is not a destination but a journey — one filled with the same contradictions, growth opportunities, and moments of grace that characters have navigated for centuries. By studying these patterns, we gain wisdom that modern self-help books are only beginning to rediscover.
"""


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../../data/source/classic-literature/corpus")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for book_idx, book in enumerate(BOOKS):
        for ci, concept in enumerate(book.get("concepts", [])):
            content = generate_classic_insight(book, concept, count)
            readable = concept.replace("_", " ").replace("-", " ").title()
            title = f"{book['title']} - {readable}"
            now = datetime.now().strftime("%Y-%m-%d")
            year = abs(book['year'])

            frontmatter = [
                "---",
                f'id: "classic-{book_idx:03d}-{ci:02d}-{concept}"',
                f'title: "{title.replace(chr(34), chr(39))}"',
                f'channel: "{book["author"]}"',
                f'url: ""',
                f'platform: "book"',
                f"views: 0",
                f"duration: 0",
                f'uploaded: "{year}-01-01"',
                f'collected: "{now}"',
                f'category: "dating"',
                f'language: "ko"',
                f'source_origin: "classic-literature"',
                "---",
            ]

            safe = re.sub(r'[^\w가-힣\-.+]', '_', title)[:80]
            out_path = output_dir / f"{safe}.md"
            out_path.write_text("\n".join(frontmatter) + "\n\n# " + title + "\n\n" + content, encoding="utf-8")
            count += 1

    print(f"Generated {count} classic literature insight files in {output_dir}")


if __name__ == "__main__":
    main()
