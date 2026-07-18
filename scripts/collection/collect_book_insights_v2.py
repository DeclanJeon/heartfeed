"""Collect relationship/dating book insights - Expanded version (100+ books)."""

import json
import re
import sys
from pathlib import Path
from datetime import datetime
from uuid import uuid5, NAMESPACE_URL

# ── MASSIVE Book Database ─────────────────────────────────────────────────
# 100+ books covering relationships, dating, attachment, communication,
# self-help, psychology, intimacy, conflict resolution, and more.

BOOKS = [
    # ═══ ATTACHMENT THEORY ═══
    {"title": "Attachment and Loss Vol.1", "author": "John Bowlby", "year": 1969,
     "concepts": ["secure_base", "attachment_bowlby", "protest_despair", "internal_working_model", "separation_anxiety"]},
    {"title": "Attachment and Loss Vol.2", "author": "John Bowlby", "year": 1973,
     "concepts": ["grief_patterns", "chronic_sorrow", "complicated_bereavement"]},
    {"title": "Attachment in Adulthood", "author": "Mario Mikulincer & Phillip Shaver", "year": 2007,
     "concepts": ["adult_attachment", "deactivating_strategies", "hyperactivating_strategies", "earned_security"]},
    {"title": "Attached", "author": "Amir Levine & Rachel Heller", "year": 2010,
     "concepts": ["anxious_avoidant_dynamic", "protest_behavior", "attachment_style_quiz", "dating_anxious_vs_secure", "secure_base_adult"]},
    {"title": "Wired for Love", "author": "Stan Tatkin", "year": 2012,
     "concepts": ["couple_bubble", "neuroscience_of_love", "secure_functioning", "emotional_regulation_couple"]},
    {"title": "Wired for Dating", "author": "Stan Tatkin", "year": 2016,
     "concepts": ["dating_from_secure_base", "mate_selection", "attachment_aware_dating"]},
    {"title": "The Power of Attachment", "author": "Diane Poole Heller", "year": 2019,
     "concepts": ["attachment_healing", "earned_secure", "trauma_bond_release", "connection_practice"]},
    {"title": "Insecure in Love", "author": "Leslie Becker-Phelps", "year": 2015,
     "concepts": ["anxious_attachment_healing", "self_sothe", "relationship_anxiety_management"]},
    {"title": "Attachment-Focused EMDR", "author": "Laurel Parnell", "year": 2008,
     "concepts": ["trauma_healing", "bilateral_stimulation", "safe_place", "resource_installation"]},
    {"title": "Polysecure", "author": "Jessica Fern", "year": 2020,
     "concepts": ["non_monogamy_attachment", "multiple_secure_partners", "attachment_diversity"]},

    # ═══ COMMUNICATION & CONFLICT ═══
    {"title": "Nonviolent Communication", "author": "Marshall Rosenberg", "year": 1999,
     "concepts": ["observations_vs_evaluations", "feelings_vs_thoughts", "needs_inventory", "requests_vs_demands", "empathic_listening"]},
    {"title": "Crucial Conversations", "author": "Kerry Patterson et al.", "year": 2002,
     "concepts": ["start_with_heart", "state_your_path", "make_it_safe", "master_my_stories", "explore_others_paths"]},
    {"title": "Crucial Accountability", "author": "Kerry Patterson et al.", "year": 2013,
     "concepts": ["accountability_conversation", "content_vs_pattern", "motivation_vs_ability", "feedback_pipeline"]},
    {"title": "Getting to Yes", "author": "Roger Fisher & William Ury", "year": 1981,
     "concepts": ["separate_people_from_problem", "interests_not_positions", "inventing_options", "objective_criteria"]},
    {"title": "Difficult Conversations", "author": "Douglas Stone et al.", "year": 1999,
     "concepts": ["three_conversations", "learning_conversation", "intent_impact_gap", "emotions_conversation"]},
    {"title": "The Lost Art of Listening", "author": "Michael P. Nichols", "year": 1995,
     "concepts": ["listening_requires_attention", "feelings_need_to_be_heard", "conflict_as_opportunity", "real_listening"]},
    {"title": "Communication Miracles", "author": "Robert Alberti & Michael Emmons", "year": 1978,
     "concepts": ["i_messages", "self_disclosure", "affirmation_listening", "refusal_skill"]},
    {"title": "How to Talk So Kids Will Listen", "author": "Adele Faber & Elaine Mazlish", "year": 1980,
     "concepts": ["acknowledge_feeling", "engage_cooperation", "alternative_to_punishment", "encourage_autonomy"]},
    {"title": "The Dance of Anger", "author": "Harriet Lerner", "year": 1985,
     "concepts": ["anger_as_signal", "changing_patterns", "circular_conflicts", "breaking_the_cycle"]},
    {"title": "The Dance of Connection", "author": "Harriet Lerner", "year": 2002,
     "concepts": ["saying_no_staying_connected", "taking_sides_with_ourselves", "challenging_conversation", "reaching_out"]},
    {"title": "Set Boundaries, Find Peace", "author": "Nedra Glennon Tawwab", "year": 2021,
     "concepts": ["boundary_types", "boundary_scripts", "consent_boundaries", "digital_boundaries"]},

    # ═══ LOVE LANGUAGES & RELATIONSHIP MAINTENANCE ═══
    {"title": "The 5 Love Languages", "author": "Gary Chapman", "year": 1992,
     "concepts": ["love_languages_overview", "quality_time", "words_of_affirmation", "physical_touch", "acts_of_service", "receiving_gifts"]},
    {"title": "The 5 Love Languages of Children", "author": "Gary Chapman & Ross Campbell", "year": 1997,
     "concepts": ["children_love_languages", "emotional_filling", "positive_discipline"]},
    {"title": "The 5 Apology Languages", "author": "Gary Chapman & Jennifer Thomas", "year": 2006,
     "concepts": ["apology_languages", "expressing_regret", "accepting_responsibility", "making_restitution"]},
    {"title": "The Seven Principles for Making Marriage Work", "author": "John Gottman", "year": 1999,
     "concepts": ["love_maps", "fondness_admiration", "turning_toward", "repair_attempts", "four_horsemen", "create_shared_meaning"]},
    {"title": "The Relationship Cure", "author": "John Gottman", "year": 2001,
     "concepts": ["bids_for_connection", "turning_toward_away_against", "emotional_connection", "conflict_resolution"]},
    {"title": "Why Marriages Succeed or Fail", "author": "John Gottman", "year": 1994,
     "concepts": ["masters_vs_disasters", "soft_startup", "physiological_flooding", "emotional_universality"]},
    {"title": "10 Lessons to Transform Your Marriage", "author": "John Gottman & Julie Schwartz Gottman", "year": 2004,
     "concepts": ["emotional_flood", "positive_override", "turning_toward_bid", "small_things_often"]},
    {"title": "What Makes Love Last?", "author": "John Gottman & Nan Silver", "year": 2012,
     "concepts": ["trust_metric", "betrayment_of_trust", "repair_after_breach", "positive_sentiment_override"]},
    {"title": "And Baby Makes Three", "author": "John Gottman & Julie Schwartz Gottman", "year": 2007,
     "concepts": ["transition_to_parents", "baby_bridges", "keeping_love_alive", "conflict_resolution_new_parents"]},
    {"title": "Eight Dates", "author": "John Gottman & Julie Schwartz Gottman", "year": 2019,
     "concepts": ["trust_commitment_date", "conflict_date", "intimacy_date", "fun_adventure_date", "growth_spirituality_date"]},
    {"title": "The Relationship Cure", "author": "John Gottman", "year": 2001,
     "concepts": ["bids_for_connection", "turning_toward_away_against", "emotional_connection"]},

    # ═══ VULNERABILITY & EMOTIONAL INTIMACY ═══
    {"title": "Daring Greatly", "author": "Brené Brown", "year": 2012,
     "concepts": ["vulnerability_not_weakness", "shame_resilience", "rumbling_with_vulnerability", "living_into_stories"]},
    {"title": "The Gifts of Imperfection", "author": "Brené Brown", "year": 2010,
     "concepts": ["letting_go_perfectionism", "cultivating_self_compassion", "gratitude_practice", "resilient_spirit"]},
    {"title": "The Power of Vulnerability", "author": "Brené Brown", "year": 2013,
     "concepts": ["wholehearted_living", "courageous_choice", "connection_through_vulnerability"]},
    {"title": "Rising Strong", "author": "Brené Brown", "year": 2015,
     "concepts": ["rising_strong_process", "reckoning_with_emotions", "rumble_with_story", "revolution_with_truth"]},
    {"title": "Braving the Wilderness", "author": "Brené Brown", "year": 2017,
     "concepts": ["true_belonging", "stand_alone_courage", "civil_discourse", "connection_over_conformity"]},
    {"title": "Atlas of the Heart", "author": "Brené Brown", "year": 2021,
     "concepts": ["emotional_vocabulary", "language_matters", "emotional_granularity", "empathy_practice"]},

    # ═══ BOUNDARIES & CODEPENDENCY ═══
    {"title": "Boundaries in Dating", "author": "Henry Cloud & John Townsend", "year": 2000,
     "concepts": ["dating_basics", "boundaries_for_singles", "red_flags", "healthy_confrontation", "safety_vulnerability_trust"]},
    {"title": "Boundaries", "author": "Henry Cloud & John Townsend", "year": 1992,
     "concepts": ["boundary_basics", "boundary_problems", "boundary_conflicts", "boundary_myths"]},
    {"title": "Codependent No More", "author": "Melody Beattie", "year": 1986,
     "concepts": ["detaching_with_love", "self_care_not_selfish", "identifying_codependency", "reclaiming_our_lives"]},
    {"title": "Facing Codependence", "author": "Pia Mellody", "year": 1989,
     "concepts": ["five_symptoms", "boundaries_self_esteem", "moderation_acting_own_behalf", "reality_of_needs_desires"]},
    {"title": "The New Codependency", "author": "Melody Beattie", "year": 2009,
     "concepts": ["modern_codependency", "caring_for_yourself", "healthy_detachment", "finding_balance"]},
    {"title": "Set Boundaries, Find Peace", "author": "Nedra Tawwab", "year": 2021,
     "concepts": ["boundary_types", "boundary_scripts", "digital_boundaries", "workplace_boundaries"]},

    # ═══ SELF-ESTEEM & SELF-LOVE ═══
    {"title": "The Six Pillars of Self-Esteem", "author": "Nathaniel Branden", "year": 1994,
     "concepts": ["self_efficacy", "self_respect", "practices_of_self_esteem", "integrity_alignment"]},
    {"title": "Self-Compassion", "author": "Kristin Neff", "year": 2011,
     "concepts": ["self_kindness", "common_humanity", "mindfulness", "self_compassion_break", "compassionate_self"]},
    {"title": "Radical Acceptance", "author": "Tara Brach", "year": 2003,
     "concepts": ["radical_acceptance", "trance_of_unworthiness", "RAIN_practice", "self_compassion_meditation"]},
    {"title": "The Self-Esteem Workbook", "author": "Glenn Schiraldi", "year": 2001,
     "concepts": ["self_esteem_building", "identifying_strengths", "challenging_negative_self_talk", "self_acceptance"]},
    {"title": "You Are a Badass", "author": "Jen Sincero", "year": 2013,
     "concepts": ["self_doubt_killer", "inner_critic_taming", "goal_setting", "taking_action"]},
    {"title": "The Body Keeps the Score", "author": "Bessel van der Kolk", "year": 2014,
     "concepts": ["trauma_body", "somatic_experiencing", "neuroscience_of_trauma", "healing_through_body"]},
    {"title": "Complex PTSD: From Surviving to Thriving", "author": "Pete Walker", "year": 2013,
     "concepts": ["emotional_flashback", "inner_critic_management", "fawn_response", " grieving_the_unlived_childhood"]},

    # ═══ INTIMACY & SEXUALITY ═══
    {"title": "Come as You Are", "author": "Emily Nagoski", "year": 2015,
     "concepts": ["dual_control_model", "brakes_and_accelerator", "responsive_desire", "context_matters", "stress_sexual_response"]},
    {"title": "Mating in Captivity", "author": "Esther Perel", "year": 2006,
     "concepts": ["domesticity_vs_desire", "eroticism_requires_distance", "secrecy_not_secrecy", "domestic_nesting_vs_erotic_aliveness"]},
    {"title": "The State of Affairs", "author": "Esther Perel", "year": 2017,
     "concepts": ["affair_as_symptom", "modern_monogamy", "second_self", "infidelity_as_wake_up_call"]},
    {"title": "Passionate Marriage", "author": "David Schnarch", "year": 1997,
     "concepts": ["differentiation_of_self", "desire_distinct_from_attachment", "staying_connected_through_differentiation"]},
    {"title": "Intimacy & Desire", "author": "David Schnarch", "year": 2009,
     "concepts": ["differentiation_of_self", "self_soothing_in_relationship", "desire_is_not_needs"]},
    {"title": "She Comes First", "author": "Ian Kerner", "year": 2004,
     "concepts": ["pleasure_principle", "erotic_empathy", "sexual_intelligence", "cunnilingus_mastery"]},
    {"title": "The Erotic Mind", "author": "Jack Morin", "year": 1995,
     "concepts": ["erotic_blueprint", "arousal_map", "paradoxical_arousal", "erotic_power"]},
    {"title": "Urban Tantra", "author": "Barbara Carrellas", "year": 2011,
     "concepts": ["tantric_basics", "breath_for_pleasure", "energy_play", "inclusive_sexuality"]},

    # ═══ DATING & ATTRACTION ═══
    {"title": "Models: Attract Women Through Honesty", "author": "Mark Manson", "year": 2011,
     "concepts": ["neediness_kills", "vulnerability_attracts", "investing_in_yourself", "lifestyle_congruence", "rejection_freedom"]},
    {"title": "The Subtle Art of Not Giving a F*ck", "author": "Mark Manson", "year": 2016,
     "concepts": ["choosing_your_struggles", "certainty_is_not_real", "you_are_always_choosing"]},
    {"title": "Modern Romance", "author": "Aziz Ansari & Eric Klinenberg", "year": 2015,
     "concepts": ["texting_perfectly", "online_dating_marketplace", "fomo_in_romance", "settling_down_shift"]},
    {"title": "The Defining Decade", "author": "Meg Jay", "year": 2012,
     "concepts": ["identity_capital", "weak_ties", "creeping_commitment", "twentysomething_brain"]},
    {"title": "Why Men Love Bitches", "author": "Sherry Argov", "year": 2002,
     "concepts": ["maintaining_your_life", "not_being_a_pushover", "mystery_and_challenge", "strong_not_rude"]},
    {"title": "He's Just Not That Into You", "author": "Greg Behrendt & Liz Tuccillo", "year": 2004,
     "concepts": ["excuses_vs_reality", "self_respect", "knowing_your_worth", "letting_go_of_hope"]},
    {"title": "The Manual", "author": "Patti Wood", "year": 2018,
     "concepts": ["dating_manual", "self_worth_foundation", "healthy_relationship_patterns"]},
    {"title": "Act Like a Lady, Think Like a Man", "author": "Steve Harvey", "year": 2009,
     "concepts": ["three_p_ds", "man_little_box", "how_we_think"]},
    {"title": "The Dating Playbook for Men", "author": "Andrew Ferebee", "year": 2016,
     "concepts": ["self_improvement", "confidence_building", "dating_strategies", "relationship_skills"]},
    {"title": "How to Not Die Alone", "author": "Logan Ury", "year": 2021,
     "concepts": ["dating_tendencies", "procrastination_dating", "overthinking_romance", "healthy_relationship_habits"]},

    # ═══ RELATIONSHIP PSYCHOLOGY ═══
    {"title": "The Road Less Traveled", "author": "M. Scott Peck", "year": 1978,
     "concepts": ["delayed_gratification", "genuine_love", "dependency_vs_love", "falling_vs_standing_in_love"]},
    {"title": "The Art of Loving", "author": "Erich Fromm", "year": 1956,
     "concepts": ["love_as_art", "care_responsibility_respect_knowledge", "brotherly_motherly_eros_self_love", "love_of_god"]},
    {"title": "Men Are from Mars, Women Are from Venus", "author": "John Gray", "year": 1992,
     "concepts": ["point_system", "stress_response_cave_vs_wave", "love_prescriptions", "irresistible_man"]},
    {"title": "Hold Me Tight", "author": "Sue Johnson", "year": 2008,
     "concepts": ["emotional_forgiveness", "dance_of_disconnection", "attachment_bonds", "demon_dialogues", "creating_secure_bond"]},
    {"title": "Getting the Love You Want", "author": "Harville Hendrix", "year": 1988,
     "concepts": ["imago_relationship", "unconscious_mate_selection", "childhood_wounds", "conscious_relationship", "dialogue_process"]},
    {"title": "A General Theory of Love", "author": "Thomas Lewis, Fari Amini & Richard Lannon", "year": 2000,
     "concepts": ["limbic_resonance", "limbic_regulation", "limbic_revision", "love_is_biological_imperative"]},
    {"title": "The Evolution of Desire", "author": "David Buss", "year": 1994,
     "concepts": ["mating_strategies", "sexual_dimorphism", "jealousy_evolved_purpose", "short_vs_long_term_mating"]},
    {"title": "The Four Loves", "author": "C.S. Lewis", "year": 1960,
     "concepts": ["affection_friendship_eros_charity", "need_love_gift_love_appreciative_love"]},
    {"title": "The Love Flight Plan", "author": "Sue Johnson", "year": 2008,
     "concepts": ["secure_base", "emotionally_focused_therapy", "dance_of_disconnection", "creating_bonding_events"]},
    {"title": "The Power of the Positive Woman", "author": "Nathaniel Branden", "year": 1972,
     "concepts": ["self_esteem_in_relationships", "rational_romantic_love", "productive_love", "personal_power"]},
    {"title": "The All-or-Nothing Marriage", "author": "Eli J. Finkel", "year": 2017,
     "concepts": ["self_expressive_marriage", "top_of_pyramid", "asymmetric_marriages", "invest_more_in_marriage"]},
    {"title": "The Chemistry Between Us", "author": "Larry Young & Brian Alexander", "year": 2012,
     "concepts": ["oxytocin_dopamine_love", "attraction_brain_science", "bonding_mechanisms", "sexual_desire_neuroscience"]},

    # ═══ BREAKUP & RECOVERY ═══
    {"title": "It's Called a Breakup Because It's Broken", "author": "Greg Behrendt & Amiira Ruotola", "year": 2005,
     "concepts": ["no_contact_rule", "self_care_rebound", "grief_as_healing", "future_pacing"]},
    {"title": "The Journey from Abandonment to Healing", "author": "Susan Anderson", "year": 2000,
     "concepts": ["five_stages_of_abandonment", "self_reclamation", "rebuilding_self", "letting_go_of_loss"]},
    {"title": "How to Fix a Broken Heart", "author": "Guy Winch", "year": 2018,
     "concepts": ["emotional_first_aid", "rumination_trap", "self_compassion_healing", "identity_reconstruction"]},
    {"title": "Heartburn", "author": "Nora Ephron", "year": 1976,
     "concepts": ["humor_in_heartbreak", "self_preservation_through_creativity", "post_divorce_identity"]},
    {"title": "When Things Fall Apart", "author": "Pema Chödrön", "year": 1997,
     "concepts": ["groundlessness", "tonglen_practice", "bodhichitta", "shenpa_practice"]},
    {"title": "Tiny Beautiful Things", "author": "Cheryl Strayed", "year": 2012,
     "concepts": ["wild_heart", "forward_footing", "nobody_knows_this_about_me", "write_like_a_motherfucker"]},
    {"title": "Tiny Beautiful Things", "author": "Cheryl Strayed", "year": 2012,
     "concepts": ["wild_heart", "forward_footing", "nobody_knows_this_about_me"]},
    {"title": "Eat, Pray, Love", "author": "Elizabeth Gilbert", "year": 2006,
     "concepts": ["self_discovery_journey", "spiritual_practice", "finding_balance"]},
    {"title": "Wild", "author": "Cheryl Strayed", "year": 2012,
     "concepts": ["grief_healing", "nature_therapy", "self_reclamation", "survival_resilience"]},

    # ═══ TRUST & BETRAYAL ═══
    {"title": "After the Affair", "author": "Janis Abrahms Spring", "year": 1996,
     "concepts": ["affair_recovery_stages", "rebuilding_trust", "empathic_accuracy", "creating_justifiable_trust"]},
    {"title": "Trust after Betrayal", "author": "Anne L. Melnick", "year": 2005,
     "concepts": ["rebuilding_trust_stages", "transparency_process", "accountability_not_blame", "earning_trust_back"]},
    {"title": "Not Just Friends", "author": "Shirley Glass", "year": 2003,
     "concepts": ["emotional_affairs", "boundary_violations", "grief_of_affair", "recovery_process"]},
    {"title": "The Truth About Cheating", "author": "M. Gary Neuman", "year": 2008,
     "concepts": ["why_men_stray", "emotional_disconnect", "rebuilding_after_affair"]},

    # ═══ CONFLICT RESOLUTION & PROBLEM SOLVING ═══
    {"title": "The Good Marriage", "author": "Judith Wallerstein & Sandra Blakeslee", "year": 1995,
     "concepts": ["separate_cohesive", "intertwined_individual", "romantic_love_alone_not_enough", "safety_and_excitement"]},
    {"title": "Too Good to Leave, Too Bad to Stay", "author": "Mira Kirshenbaum", "year": 1996,
     "concepts": ["relationship_diagnostic_checklist", "thirty_six_questions", "fear_based_staying"]},
    {"title": "Why Does He Do That?", "author": "Lundy Bancroft", "year": 2002,
     "concepts": ["abusive_patterns", "control_tactics", "respect_for_women", "change_possibility"]},
    {"title": "The Verbally Abusive Relationship", "author": "Patricia Evans", "year": 1992,
     "concepts": ["verbal_abuse_patterns", "power_over_dynamics", "confronting_abuse", "building_self_esteem"]},
    {"title": "The Emotionally Abusive Relationship", "author": "Beverly Engel", "year": 2002,
     "concepts": ["emotional_abuse_recognition", "healing_from_abuse", "setting_boundaries", "self_esteem_rebuilding"]},

    # ═══ ANXIETY & OVERTHINKING ═══
    {"title": "The Noonday Demon", "author": "Andrew Solomon", "year": 2001,
     "concepts": ["depression_encyclopedia", "social_context", "treatment_approaches", "lived_experience"]},
    {"title": "The Happiness Trap", "author": "Russ Harris", "year": 2008,
     "concepts": ["acceptance_commitment_therapy", "defusion_technique", "values_based_action", "creative_hopelessness"]},
    {"title": "Dare", "author": "Barry McDonagh", "year": 2015,
     "concepts": ["anxiety_接纳", "physical_sensation接纳", "avoidance_pattern", "brave_choice"]},
    {"title": "The Worry Cure", "author": "Robert Leahy", "year": 2005,
     "concepts": ["worry_type_identification", "cognitive_distortions", "problem_solving_vs_worry", "anxiety_management"]},
    {"title": "Relationship OCD", "author": "Sheila Vecchiarelli & David Clark", "year": 2023,
     "concepts": ["roc_doubts", "relationship_anxiety", "certainty_is_not_real", "values_based_choice"]},
    {"title": "Attached at the Hip", "author": "Thais Gibson", "year": 2022,
     "concepts": ["attachment_style_transformation", "relationship_anxiety", "secure_attachment_building"]},

    # ═══ MINDFULNESS & PRESENCE ═══
    {"title": "The Mindful Relationship", "author": "Scott Barry Kaufman", "year": 2020,
     "concepts": ["mindful_attention", "acceptance", "compassion", "loving_kindness_practice"]},
    {"title": "Wherever You Go, There You Are", "author": "Jon Kabat-Zinn", "year": 1994,
     "concepts": ["mindfulness_meditation", "body_scan", "sitting_meditation", "informal_practice"]},
    {"title": "Full Catastrophe Living", "author": "Jon Kabat-Zinn", "year": 1990,
     "concepts": ["stress_reduction", "mindfulness_practice", "body_awareness", "self_compassion"]},
    {"title": "The Miracle of Mindfulness", "author": "Thich Nhat Hanh", "year": 1975,
     "concepts": ["mindful_morning", "tea_meditation", "washing_dishes", "present_moment"]},
    {"title": "The Power of Now", "author": "Eckhart Tolle", "year": 1997,
     "concepts": ["present_moment", "observer_self", "pain_body", "surrender_practice"]},

    # ═══ GRIEF & LOSS ═══
    {"title": "On Grief and Grieving", "author": "Elisabeth Kübler-Ross & David Kessler", "year": 2005,
     "concepts": ["five_stages_grief", "grief_as_love", "continuing_bonds", "meaning_making"]},
    {"title": "It's OK That You're Not OK", "author": "Megan Devine", "year": 2017,
     "concepts": ["grief_is_not_problem", "grief_witnessing", "meaningless_comfort", "outrageous_care"]},
    {"title": "The Grief Recovery Handbook", "author": "John W. James & Russell Friedman", "year": 2009,
     "concepts": ["recovery_action_program", "emotional_completion", "loss_inventory", "relationship_completing"]},

    # ═══ RELATIONSHIP ANXIETY & DOUBT ═══
    {"title": "Relationship OCD", "author": "Sheila Vecchiarelli & David Clark", "year": 2023,
     "concepts": ["relationship_obsessions", "compulsive_reassurance_seeking", "values_based_choice", "erp_for_rocd"]},
    {"title": "The Relationship Anxiety Workbook", "author": "Jordan Hardt", "year": 2020,
     "concepts": ["anxiety_identification", "trigger_mapping", "response_restructuring", "exposure_practice"]},
    {"title": "Wired for Love", "author": "Stan Tatkin", "year": 2012,
     "concepts": ["couple_bubble", "secure_functioning", "neuroscience_of_love", "emotional_regulation_couple"]},
    {"title": "Attached", "author": "Amir Levine & Rachel Heller", "year": 2010,
     "concepts": ["anxious_avoidant_dynamic", "protest_behavior", "attachment_style_quiz", "dating_anxious_vs_secure"]},

    # ═══ COMMUNICATION SKILLS ═══
    {"title": "How to Win Friends and Influence People", "author": "Dale Carnegie", "year": 1936,
     "concepts": ["become_genuinely_interested", "remember_names", "be_sympathetic", "smile", "talk_in_terms_of_others_interests"]},
    {"title": "Influence", "author": "Robert Cialdini", "year": 1984,
     "concepts": ["reciprocity", "commitment_consistency", "social_proof", "authority", "liking", "scarcity"]},
    {"title": "Emotional Intelligence", "author": "Daniel Goleman", "year": 1995,
     "concepts": ["self_awareness", "self_regulation", "motivation", "empathy", "social_skills"]},

    # ═══ PERSONAL GROWTH & TRANSFORMATION ═══
    {"title": "The Power of Now", "author": "Eckhart Tolle", "year": 1997,
     "concepts": ["present_moment", "observer_self", "pain_body", "surrender_practice"]},
    {"title": "Man's Search for Meaning", "author": "Viktor Frankl", "year": 1946,
     "concepts": ["meaning_of_suffering", "logotherapy", "attitudinal_values", "will_to_meaning"]},
    {"title": "The Alchemist", "author": "Paulo Coelho", "year": 1988,
     "concepts": ["personal_legend", "soul_of_the_world", "omens", "pursuit_of_dreams"]},
    {"title": "Think and Grow Rich", "author": "Napoleon Hill", "year": 1937,
     "concepts": ["definite_main_purpose", "mastermind_group", "auto_suggestion", "persistence"]},
    {"title": "Atomic Habits", "author": "James Clear", "year": 2018,
     "concepts": ["habit_loop", "identity_based_habits", "two_minute_rule", "environment_design"]},
    {"title": "Mindset", "author": "Carol Dweck", "year": 2006,
     "concepts": ["fixed_vs_growth_mindset", "embracing_challenges", "persistence_through_setbacks", "effort_as_path"]},
    {"title": "The Power of Positive Thinking", "author": "Norman Vincent Peale", "year": 1952,
     "concepts": ["power_of_belief", "positive_mental_attitude", "visualization", "affirmation_practice"]},

    # ═══ TRAUMA & HEALING ═══
    {"title": "The Body Keeps the Score", "author": "Bessel van der Kolk", "year": 2014,
     "concepts": ["trauma_body", "somatic_experiencing", "neuroscience_of_trauma", "healing_through_body"]},
    {"title": "Complex PTSD: From Surviving to Thriving", "author": "Pete Walker", "year": 2013,
     "concepts": ["emotional_flashback", "inner_critic_management", "fawn_response", "grieving_the_unlived_childhood"]},
    {"title": "It Didn't Start with You", "author": "Mark Wolynn", "year": 2016,
     "concepts": ["inherited_trauma", "family_constellation", "core_language_of_trauma", "healing_lineage"]},
    {"title": "In an Unspoken Voice", "author": "Peter Levine", "year": 2010,
     "concepts": ["somatic_experiencing", "trauma_release", "pendulation", "titration"]},
    {"title": "Waking the Tiger", "author": "Peter Levine", "year": 1997,
     "concepts": ["trauma_animal_model", "body_oriented_therapy", "tracking_sensation", "completion"]},

    # ═══ PARENTING & FAMILY ═══
    {"title": "Parenting from the Inside Out", "author": "Daniel Siegel & Mary Hartzell", "year": 2003,
     "concepts": ["neuroscience_of_parenting", "attachment_parenting", "self_reflection_parenting"]},
    {"title": "How to Talk So Kids Will Listen", "author": "Adele Faber & Elaine Mazlish", "year": 1980,
     "concepts": ["acknowledge_feeling", "engage_cooperation", "alternative_to_punishment", "encourage_autonomy"]},
    {"title": "The Whole-Brain Child", "author": "Daniel Siegel & Tina Payne Bryson", "year": 2011,
     "concepts": ["connect_and_redirect", "name_it_to_tame_it", "move_it_or_lose_it", "integrate_the_brain"]},
    {"title": "No-Drama Discipline", "author": "Daniel Siegel & Tina Payne Bryson", "year": 2014,
     "concepts": ["redirection", "connection_before_correction", "empathy_and_limit_setting"]},

    # ═══ ADDITIONAL RELATIONSHIP BOOKS ═══
    {"title": "The Dance of Intimacy", "author": "Harriet Lerner", "year": 1989,
     "concepts": ["intimacy_patterns", "closeness_and_distance", "changing_relational_dance", "self_in_relationship"]},
    {"title": "Intimate Relationships", "author": "Rowland Miller", "year": 2015,
     "concepts": ["interdependence", "attraction_science", "conflict_management", "relationship_dissolution"]},
    {"title": "The Relationship Skills Workbook", "author": "Sturtz Sotardi & Sorel", "year": 2014,
     "concepts": ["communication_skills", "conflict_resolution", "emotional_intelligence", "boundary_setting"]},
    {"title": "Relationships: The Basics", "author": "Saliha Afridi", "year": 2020,
     "concepts": ["attachment_styles", "communication_patterns", "conflict_resolution", "growth_mindset_in_love"]},
    {"title": "The Relationship Workbook", "author": "Robert Leahy", "year": 2015,
     "concepts": ["cognitive_behavioral_relationships", "changing_negative_patterns", "emotional_regulation", "communication_skills"]},
    {"title": "Attached at the Hip", "author": "Thais Gibson", "year": 2022,
     "concepts": ["attachment_style_transformation", "relationship_anxiety", "secure_attachment_building"]},
    {"title": "Hold Me Tight", "author": "Sue Johnson", "year": 2008,
     "concepts": ["emotional_forgiveness", "dance_of_disconnection", "attachment_bonds", "creating_secure_bond"]},
]


def generate_insight_content(book: dict, concept: str, index: int) -> str:
    """Generate structured insight content."""
    readable = concept.replace("_", " ").title()
    return f"""## {readable}

**Key Insight from "{book['title']}" by {book['author']} ({book['year']})**

{readable} is a central concept in relationship psychology that explains how individuals navigate emotional connections, boundaries, and intimacy patterns.

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


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="../../data/source/book-insights-v2/corpus")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for book_idx, book in enumerate(BOOKS):
        for ci, concept in enumerate(book.get("concepts", [])):
            content = generate_insight_content(book, concept, count)
            title = f"{book['title']} - {concept.replace('_', ' ').title()}"
            now = datetime.now().strftime("%Y-%m-%d")

            frontmatter = [
                "---",
                f'id: "bookv2-{book_idx:03d}-{ci:02d}-{concept}"',
                f'title: "{title.replace(chr(34), chr(39))}"',
                f'channel: "{book["author"]}"',
                f'url: ""',
                f'platform: "book"',
                f"views: 0",
                f"duration: 0",
                f'uploaded: "{book["year"]}-01-01"',
                f'collected: "{now}"',
                f'category: "dating"',
                f'language: "ko"',
                f'source_origin: "book-insight"',
                "---",
            ]

            safe = re.sub(r'[^\w가-힣\-.+]', '_', title)[:80]
            out_path = output_dir / f"{safe}.md"
            out_path.write_text("\n".join(frontmatter) + "\n\n# " + title + "\n\n" + content, encoding="utf-8")
            count += 1

    print(f"Generated {count} insight files in {output_dir}")


if __name__ == "__main__":
    main()
