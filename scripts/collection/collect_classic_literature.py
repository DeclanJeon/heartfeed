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

    # ═══ EXPANSION 2026-07-18: multi-angle relationship classics ═══
    {
        "title": "The Tale of Genji", "author": "Murasaki Shikibu", "year": 1008,
        "concepts": [
            "courtship_rituals_and_nuance", "fleeting_beauty_and_attachment",
            "jealousy_among_multiple_bonds", "emotional_restraint_vs_desire",
            "memory_of_past_lovers", "status_and_romance",
            "unspoken_longing", "impermanence_of_relationships",
        ]
    },
    {
        "title": "Dream of the Red Chamber", "author": "Cao Xueqin", "year": 1791,
        "concepts": [
            "family_system_and_romance", "arranged_expectations_vs_true_feeling",
            "coming_of_age_love", "grief_after_forced_separation",
            "idealized_lover_vs_reality", "decline_of_household_and_love",
        ]
    },
    {
        "title": "춘향전", "author": "고전 소설 (작자 미상)", "year": 1700,
        "concepts": [
            "신분의_벽을_넘는_사랑", "이별과_재회", "정절과_신뢰",
            "권력에_맞선_연인", "약속과_인내", "사회적_압력과_연애", "그리움의_서사",
        ]
    },
    {
        "title": "구운몽", "author": "김만중", "year": 1689,
        "concepts": [
            "욕망과_각성", "다중_관계의_환상", "집착의_공허함",
            "깨달음_후의_관계관", "꿈과_현실의_사랑",
        ]
    },
    {
        "title": "흥부전", "author": "고전 소설 (작자 미상)", "year": 1700,
        "concepts": [
            "형제_갈등과_화해", "탐욕과_관계_파괴", "베풂과_신뢰", "가족_내_정의",
        ]
    },
    {
        "title": "무정", "author": "이광수", "year": 1917,
        "concepts": [
            "근대적_연애관", "삼각관계와_선택", "계몽과_감정",
            "이별의_이성화", "자아_실현과_사랑",
        ]
    },
    {
        "title": "사랑손님과 어머니", "author": "주요섭", "year": 1925,
        "concepts": [
            "재혼과_아이_시점", "미망인의_감정", "사회적_시선",
            "말하지_못한_호감", "가족_재구성의_불안",
        ]
    },
    {
        "title": "메밀꽃 필 무렵", "author": "이효석", "year": 1936,
        "concepts": [
            "우연한_재회", "과거의_인연", "말하지_못한_진심",
            "여정의_정서", "그리움의_서정",
        ]
    },
    {
        "title": "토지", "author": "박경리", "year": 1969,
        "concepts": [
            "역사_속_사랑과_이별", "가문과_개인_감정", "복수와_연민",
            "운명과_선택", "긴_시간_속의_관계",
        ]
    },
    {
        "title": "난장이가 쏘아올린 작은 공", "author": "조세희", "year": 1978,
        "concepts": [
            "경제적_압박과_관계", "가족_연대", "상처받은_세대의_사랑", "연민과_존엄",
        ]
    },
    {
        "title": "Middlemarch", "author": "George Eliot", "year": 1871,
        "concepts": [
            "idealism_meets_marriage_reality", "mismatched_intellectual_partnership",
            "career_ambition_vs_spouse", "slow_disillusionment",
            "second_chances_after_widowhood", "gossip_and_reputation", "empathy_as_maturity",
        ]
    },
    {
        "title": "The Portrait of a Lady", "author": "Henry James", "year": 1881,
        "concepts": [
            "freedom_then_entrapment", "charming_manipulator",
            "inheritance_and_control", "refusing_safe_proposals",
            "self_betrayal_in_marriage", "seeing_clearly_too_late",
        ]
    },
    {
        "title": "Jude the Obscure", "author": "Thomas Hardy", "year": 1895,
        "concepts": [
            "class_barriers_to_partnership", "failed_marriage_traps",
            "intellectual_companionship_vs_convention", "societal_punishment_of_love",
            "hopelessness_and_attachment",
        ]
    },
    {
        "title": "The Scarlet Letter", "author": "Nathaniel Hawthorne", "year": 1850,
        "concepts": [
            "shame_and_secret_love", "public_judgment_of_relationships",
            "guilt_vs_honesty", "hypocrisy_in_moralizing", "child_as_living_consequence",
        ]
    },
    {
        "title": "Little Women", "author": "Louisa May Alcott", "year": 1868,
        "concepts": [
            "friendship_before_romance", "refusing_proposal_for_self",
            "different_sisters_different_love_paths", "creative_identity_and_love",
            "quiet_steady_partnership", "grief_binding_family",
        ]
    },
    {
        "title": "Ethan Frome", "author": "Edith Wharton", "year": 1911,
        "concepts": [
            "loveless_marriage_stagnation", "affair_as_escape_fantasy",
            "poverty_trapping_relationships", "guilt_after_failed_escape",
            "caregiving_resentment",
        ]
    },
    {
        "title": "Their Eyes Were Watching God", "author": "Zora Neale Hurston", "year": 1937,
        "concepts": [
            "finding_voice_in_love", "three_marriages_three_lessons",
            "passion_with_equality", "leaving_oppressive_partner",
            "selfhood_before_couplehood", "grief_and_storytelling",
        ]
    },
    {
        "title": "Giovanni's Room", "author": "James Baldwin", "year": 1956,
        "concepts": [
            "fear_of_true_desire", "self_denial_hurting_partners",
            "shame_and_intimacy", "choosing_safety_over_authenticity", "jealousy_and_flight",
        ]
    },
    {
        "title": "The Remains of the Day", "author": "Kazuo Ishiguro", "year": 1989,
        "concepts": [
            "repressed_feelings_at_work", "missed_timing_in_love",
            "duty_blocking_vulnerability", "regret_on_looking_back",
            "politeness_as_emotional_armor",
        ]
    },
    {
        "title": "Never Let Me Go", "author": "Kazuo Ishiguro", "year": 2005,
        "concepts": [
            "love_under_limited_future", "memory_as_intimacy",
            "triangular_attachment", "acceptance_and_tenderness", "what_we_owe_each_other",
        ]
    },
    {
        "title": "One Hundred Years of Solitude", "author": "Gabriel García Márquez", "year": 1967,
        "concepts": [
            "family_patterns_repeating_in_love", "solitude_inside_marriage",
            "obsessive_desire", "forgetting_and_remembering_lovers",
        ]
    },
    {
        "title": "Beloved", "author": "Toni Morrison", "year": 1987,
        "concepts": [
            "trauma_shaping_intimacy", "mother_love_and_horror",
            "haunting_past_in_present_bonds", "community_healing",
            "naming_pain_to_love_again",
        ]
    },
    {
        "title": "Normal People", "author": "Sally Rooney", "year": 2018,
        "concepts": [
            "class_gap_in_young_love", "misread_signals",
            "on_again_off_again_cycle", "intimacy_without_words",
            "power_shifts_over_time", "healing_through_being_seen",
            "long_distance_emotional_weather",
        ]
    },
    {
        "title": "Call Me by Your Name", "author": "André Aciman", "year": 2007,
        "concepts": [
            "summer_intensity", "first_same_sex_awakening",
            "silence_and_yearning", "parents_unexpected_wisdom",
            "remembering_without_erasing", "brief_love_lifelong_echo",
        ]
    },
    {
        "title": "The Bell Jar", "author": "Sylvia Plath", "year": 1963,
        "concepts": [
            "identity_crisis_and_dating", "performative_romance",
            "mental_health_and_intimacy", "rejecting_prescribed_roles",
            "aloneness_vs_loneliness",
        ]
    },
    {
        "title": "On Earth We're Briefly Gorgeous", "author": "Ocean Vuong", "year": 2019,
        "concepts": [
            "letter_to_mother_and_love", "queer_first_love",
            "language_barriers_in_family", "tenderness_after_violence",
            "body_memory_in_relationships",
        ]
    },
    {
        "title": "The Song of Achilles", "author": "Madeline Miller", "year": 2011,
        "concepts": [
            "devotion_beyond_glory", "love_and_war_priorities",
            "jealousy_of_fate", "grief_as_epic", "chosen_family_vs_duty",
        ]
    },
    {
        "title": "Circe", "author": "Madeline Miller", "year": 2018,
        "concepts": [
            "becoming_self_after_rejection", "power_imbalance_with_gods",
            "motherhood_and_boundaries", "choosing_mortal_love", "solitude_as_growth",
        ]
    },
    {
        "title": "The Winter's Tale", "author": "William Shakespeare", "year": 1611,
        "concepts": [
            "jealousy_destroying_family", "time_healing_wounds",
            "repentance_and_reunion", "lost_child_and_restoration",
            "forgiveness_after_years",
        ]
    },
    {
        "title": "Antony and Cleopatra", "author": "William Shakespeare", "year": 1607,
        "concepts": [
            "passion_vs_political_duty", "public_image_of_couple",
            "grand_romance_and_self_myth", "loyalty_tested_by_war",
        ]
    },
    {
        "title": "Pachinko", "author": "Min Jin Lee", "year": 2017,
        "concepts": [
            "diaspora_and_marriage", "survival_over_romance",
            "shame_and_belonging", "mothers_sacrifices",
            "quiet_endurance_in_partnership", "identity_secrets_in_family",
        ]
    },
    {
        "title": "Convenience Store Woman", "author": "Sayaka Murata", "year": 2016,
        "concepts": [
            "refusing_normative_romance", "society_pressure_to_couple",
            "found_identity_outside_dating", "performative_normality",
            "boundaries_against_matchmaking",
        ]
    },
    {
        "title": "Breasts and Eggs", "author": "Mieko Kawakami", "year": 2019,
        "concepts": [
            "body_and_self_worth", "sisterhood_over_romance",
            "single_motherhood_choices", "economic_anxiety_and_intimacy",
        ]
    },
    {
        "title": "The Seven Husbands of Evelyn Hugo", "author": "Taylor Jenkins Reid", "year": 2017,
        "concepts": [
            "ambition_and_secret_love", "queer_love_hidden_in_public",
            "transactional_marriages", "true_partner_vs_headline_partner",
        ]
    },
    {
        "title": "Lessons in Chemistry", "author": "Bonnie Garmus", "year": 2022,
        "concepts": [
            "intellectual_equals_in_love", "sexism_at_work_and_home",
            "grief_after_sudden_loss", "single_parent_strength", "respect_as_attraction",
        ]
    },
    {
        "title": "Tomorrow, and Tomorrow, and Tomorrow", "author": "Gabrielle Zevin", "year": 2022,
        "concepts": [
            "creative_partnership_vs_romance", "friendship_heartbreak",
            "misaligned_expectations", "collaboration_after_rupture",
        ]
    },
    {
        "title": "The Midnight Library", "author": "Matt Haig", "year": 2020,
        "concepts": [
            "alternate_life_regrets_in_love", "what_if_with_ex",
            "choosing_present_relationship", "depression_and_connection",
        ]
    },
    {
        "title": "A Man Called Ove", "author": "Fredrik Backman", "year": 2012,
        "concepts": [
            "grief_after_spouse_death", "neighbors_as_found_family",
            "love_shown_through_acts", "softening_after_isolation",
        ]
    },
    {
        "title": "Eleanor Oliphant Is Completely Fine", "author": "Gail Honeyman", "year": 2017,
        "concepts": [
            "social_isolation_and_crush", "trauma_blocking_intimacy",
            "friendship_as_gateway_to_love", "healing_before_partnership",
        ]
    },
    {
        "title": "The Time Traveler's Wife", "author": "Audrey Niffenegger", "year": 2003,
        "concepts": [
            "uneven_timelines_in_love", "waiting_and_uncertainty",
            "commitment_without_control", "building_life_around_unpredictability",
        ]
    },
    {
        "title": "Atonement", "author": "Ian McEwan", "year": 2001,
        "concepts": [
            "false_accusation_destroying_lovers", "class_and_misreading",
            "guilt_lasting_lifetime", "art_as_attempted_repair", "war_separating_couples",
        ]
    },
    {
        "title": "On Chesil Beach", "author": "Ian McEwan", "year": 2007,
        "concepts": [
            "wedding_night_anxiety", "sexual_communication_failure",
            "pride_preventing_repair", "one_night_that_ends_marriage",
        ]
    },
    {
        "title": "The Sense of an Ending", "author": "Julian Barnes", "year": 2011,
        "concepts": [
            "unreliable_memory_of_exes", "late_life_reckoning",
            "jealousy_rewritten_as_story", "humility_about_the_past",
        ]
    },
    {
        "title": "Americanah", "author": "Chimamanda Ngozi Adichie", "year": 2013,
        "concepts": [
            "long_distance_migration_love", "race_and_dating_abroad",
            "rekindling_after_years", "returning_home_with_changed_self",
        ]
    },
    {
        "title": "The God of Small Things", "author": "Arundhati Roy", "year": 1997,
        "concepts": [
            "forbidden_caste_love", "family_violence_and_silence",
            "desire_against_law", "small_things_that_break_lives",
        ]
    },
    {
        "title": "The Namesake", "author": "Jhumpa Lahiri", "year": 2003,
        "concepts": [
            "immigrant_parents_marriage", "name_and_identity_in_dating",
            "cultural_hybrid_romance", "duty_to_family_vs_partner",
        ]
    },
    {
        "title": "Interpreter of Maladies", "author": "Jhumpa Lahiri", "year": 1999,
        "concepts": [
            "tourist_intimacy_and_confession", "marital_distance",
            "loneliness_in_couplehood", "small_maladies_of_love",
        ]
    },
    {
        "title": "Conversations with Friends", "author": "Sally Rooney", "year": 2017,
        "concepts": [
            "affair_with_married_person", "friendship_and_jealousy",
            "intellectualizing_feelings", "power_age_gaps", "ambiguous_commitment",
        ]
    },
    {
        "title": "Beautiful World, Where Are You", "author": "Sally Rooney", "year": 2021,
        "concepts": [
            "emails_as_intimacy", "anxiety_in_modern_dating",
            "friendship_sustaining_romance", "choosing_ordinary_love",
        ]
    },
    {
        "title": "Writers & Lovers", "author": "Lily King", "year": 2020,
        "concepts": [
            "grief_and_dating_again", "two_suitors_two_futures",
            "creative_block_and_romance", "choosing_kindness",
        ]
    },
    {
        "title": "The Marriage Plot", "author": "Jeffrey Eugenides", "year": 2011,
        "concepts": [
            "love_triangle_after_college", "mental_illness_and_caregiving",
            "theory_vs_lived_romance", "what_marriage_plot_means_now",
        ]
    },
    {
        "title": "South of the Border, West of the Sun", "author": "Haruki Murakami", "year": 1992,
        "concepts": [
            "childhood_friend_reappears", "marriage_and_temptation",
            "midlife_emptiness", "choosing_family_or_flame",
        ]
    },
    {
        "title": "Sputnik Sweetheart", "author": "Haruki Murakami", "year": 1999,
        "concepts": [
            "unrequited_love_triangle", "disappearance_and_longing",
            "one_sided_devotion", "mystery_of_the_other",
        ]
    },
    {
        "title": "Colorless Tsukuru Tazaki and His Years of Pilgrimage", "author": "Haruki Murakami", "year": 2013,
        "concepts": [
            "friend_group_exile_wound", "reconnecting_after_years",
            "travel_to_heal_bonds", "color_and_identity_in_love",
        ]
    },
    {
        "title": "Men Without Women", "author": "Haruki Murakami", "year": 2014,
        "concepts": [
            "men_after_breakups", "quiet_loneliness",
            "ex_wives_and_memory", "tenderness_in_short_encounters",
        ]
    },
    {
        "title": "The Wind-Up Bird Chronicle", "author": "Haruki Murakami", "year": 1994,
        "concepts": [
            "wife_disappearance", "marriage_mystery",
            "historical_trauma_in_present_bonds", "listening_as_love",
        ]
    },
    {
        "title": "1Q84", "author": "Haruki Murakami", "year": 2009,
        "concepts": [
            "two_worlds_seeking_each_other", "childhood_promise",
            "parallel_paths_converging", "trust_without_full_explanation",
        ]
    },
    {
        "title": "Like Water for Chocolate", "author": "Laura Esquivel", "year": 1989,
        "concepts": [
            "family_rules_blocking_love", "desire_expressed_through_care",
            "repressed_passion", "tradition_vs_choice",
        ]
    },
    {
        "title": "The House of the Spirits", "author": "Isabel Allende", "year": 1982,
        "concepts": [
            "generations_of_love_patterns", "passionate_patriarch_harm",
            "women_solidarity", "memory_keeping_relationships",
        ]
    },
    {
        "title": "A Suitable Boy", "author": "Vikram Seth", "year": 1993,
        "concepts": [
            "arranged_marriage_search", "mother_pressure",
            "choosing_among_suitors", "communal_tensions_and_love",
        ]
    },
    {
        "title": "Half of a Yellow Sun", "author": "Chimamanda Ngozi Adichie", "year": 2006,
        "concepts": [
            "love_in_civil_war", "infidelity_under_stress",
            "class_in_intimacy", "survival_bonding",
        ]
    },
    {
        "title": "Gone Girl", "author": "Gillian Flynn", "year": 2012,
        "concepts": [
            "performative_marriage", "revenge_fantasy_in_couples",
            "mutual_manipulation", "staying_for_control",
        ]
    },
    {
        "title": "Me Before You", "author": "Jojo Moyes", "year": 2012,
        "concepts": [
            "caregiving_and_romance", "autonomy_vs_saving_someone",
            "love_cannot_force_will_to_live", "grief_after_choice",
        ]
    },
    {
        "title": "The English Patient", "author": "Michael Ondaatje", "year": 1992,
        "concepts": [
            "affair_in_wartime", "bodies_and_memory",
            "loyalty_torn_by_nations", "healing_and_confession",
        ]
    },
    {
        "title": "Enduring Love", "author": "Ian McEwan", "year": 1997,
        "concepts": [
            "obsession_mistaken_for_love", "couple_under_external_threat",
            "trust_erosion", "shared_trauma_bond",
        ]
    },
    {
        "title": "White Teeth", "author": "Zadie Smith", "year": 2000,
        "concepts": [
            "immigrant_marriages", "friendship_across_cultures",
            "faith_and_romance_clash", "second_generation_identity_love",
        ]
    },
    {
        "title": "The Lowland", "author": "Jhumpa Lahiri", "year": 2013,
        "concepts": [
            "brothers_and_shared_love", "stepmother_bonds",
            "quiet_endurance", "return_to_homeland_memory",
        ]
    },
    {
        "title": "My Year of Rest and Relaxation", "author": "Ottessa Moshfegh", "year": 2018,
        "concepts": [
            "numbing_instead_of_relating", "toxic_friendship",
            "avoidant_withdrawal", "waking_up_to_connection",
        ]
    },
    {
        "title": "The Idiot (Batuman)", "author": "Elif Batuman", "year": 2017,
        "concepts": [
            "email_crush_era", "language_learning_and_misread_signals",
            "intellectual_infatuation", "slow_undefined_relationship",
        ]
    },
    {
        "title": "Either/Or", "author": "Elif Batuman", "year": 2022,
        "concepts": [
            "literature_as_relationship_map", "sex_and_meaning_search",
            "choosing_how_to_live_and_love",
        ]
    },
    {
        "title": "Middlesex", "author": "Jeffrey Eugenides", "year": 2002,
        "concepts": [
            "identity_and_desire", "family_secrets_shaping_love",
            "first_love_and_body", "becoming_oneself_in_relationship",
        ]
    },
    {
        "title": "The Handmaid's Tale", "author": "Margaret Atwood", "year": 1985,
        "concepts": [
            "control_over_bodies_and_bonds", "memory_of_former_partnership",
            "coerced_intimacy", "hope_as_relationship_fuel",
        ]
    },
    {
        "title": "Daniel Deronda", "author": "George Eliot", "year": 1876,
        "concepts": [
            "marriage_for_security_regretted", "moral_growth_through_empathy",
            "identity_search_and_love", "emotional_cruelty_in_marriage",
        ]
    },

]


def generate_classic_insight(book: dict, concept: str, index: int) -> str:
    """Story-first classic insight for user-facing 이야깃거리 storytelling."""
    if any("\uac00" <= ch <= "\ud7a3" for ch in concept):
        theme = concept.replace("_", " ")
    else:
        theme = concept.replace("_", " ").replace("-", " ").title()

    title = book["title"]
    author = book["author"]
    year = abs(int(book["year"]))
    era = (
        "고대" if year < 500 else
        "중세·전근대" if year < 1600 else
        "근대 초기" if year < 1800 else
        "19세기" if year < 1900 else
        "20세기" if year < 2000 else
        "동시대"
    )

    # Scene seeds keyed by theme keywords (Korean storytelling scaffold)
    scene_bank = {
        "이별": "헤어짐의 문 앞에서 마지막으로 돌아보는 장면",
        "재회": "오랜만에 다시 마주친 순간, 말이 먼저 나오지 않는 장면",
        "질투": "상대의 한 마디·한 시선에 마음이 요동치는 장면",
        "자존": "사랑받고 싶으면서도 자신을 굽히지 않으려는 장면",
        "기다림": "답이 오지 않는 시간을 견디는 장면",
        "오해": "사소한 침묵이 큰 단절로 커지는 장면",
        "결혼": "약속과 일상 사이에서 흔들리는 장면",
        "비밀": "말하지 못한 진실을 끌어안은 채 웃는 장면",
        "희생": "상대를 위해 자신을 뒤로 미루는 장면",
        "성장": "아픔 뒤에 서로를 다시 보게 되는 장면",
    }
    scene = "마음이 갈라지는 결정적 순간"
    cl = theme.lower() + " " + concept
    for k, v in scene_bank.items():
        if k in theme or k in concept or k in cl:
            scene = v
            break
    # English soft matches
    en_map = {
        "jealous": "질투가 관계를 흔드는 장면",
        "pride": "자존심 때문에 진심을 삼키는 장면",
        "break": "이별이 다가오는 장면",
        "wait": "기다림이 길어지는 장면",
        "trust": "신뢰가 시험받는 장면",
        "love": "사랑한다고 느끼지만 표현이 엇갈리는 장면",
        "marriage": "결혼·동반 약속 앞에서 망설이는 장면",
        "grief": "상실 뒤의 공허를 견디는 장면",
        "reunion": "재회의 설렘과 어색함이 겹치는 장면",
    }
    for k, v in en_map.items():
        if k in cl.lower():
            scene = v
            break

    return f"""## {theme}

**Story seed from 「{title}」 · {author} ({year}, {era})**

### 한 줄 이야기
{title}에서 우리는 **{scene}**을 만납니다. 테마는 **{theme}** — 오늘의 연애 고민과 같은 결의 감정입니다.

### 스토리 비트 (이야깃거리 본문용)
1. **시작**: 인물들은 서로를 원하지만, 상황·성격·사회 조건이 속도를 어긋나게 둡니다.
2. **충돌**: {theme}이(가) 표면으로 올라옵니다. 작은 선택(침묵, 자존심, 성급함, 회피)이 관계를 한 방향으로 밀어 붙입니다.
3. **대가**: 말이 늦거나 마음이 앞서면, 관계에는 되돌리기 어려운 금이 갑니다. 작품은 그 대가를 숨기지 않습니다.
4. **여운**: 완벽한 해피엔딩보다, “그때 나는 왜 그랬을까”를 남깁니다. 독자(사용자)가 자기 이야기로 옮겨 적게 만드는 여백이 핵심입니다.

### 유저에게 건네는 스토리텔링 톤
- “당신만 이상한 게 아닙니다. {title} 속 인물도 같은 자리에서 흔들렸습니다.”
- 교훈 나열 대신 **장면 → 감정 → 오늘의 한 수** 순서로 말합니다.
- 작품명·작가명을 자연스럽게 부르고, 현대 상황(문자, 읽씹, 장거리, 이별 후 충동)으로 한 번 더 번역합니다.

### 오늘의 연결 (상담 브릿지)
지금 사용자의 고민이 {theme}에 가깝다면, 고전은 정답이 아니라 **거울 스토리**입니다.
- 나는 이 이야기에서 누구의 자리에 서 있는가?
- 다음 장면에서 바꾸고 싶은 한 가지는 무엇인가?
- 침묵·복수·이상화 대신, 오늘 할 수 있는 짧은 대화는 무엇인가?

### Relationship Takeaway
「{title}」({author})의 {theme} 서사는 연애를 팁 목록이 아니라 **이어지는 이야기**로 보게 합니다. 사용자는 동질감과 함께, 다음 행동의 선택지를 이야기 속에서 고를 수 있습니다.
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
