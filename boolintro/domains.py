"""Object domains (each with five binary attributes) and character names.

Every character pairs a name with one domain, as in Atkinson et al.
("Prometheus choosing hotels").  Attribute names are chosen to be fairly
neutral, and rules use random negations, so common-sense priors cannot
predict a character's rule.
"""

from __future__ import annotations

DOMAINS: dict[str, dict] = {
    "hotel": {
        "question": "Which hotel would you prefer to stay at?",
        "plural": "hotels",
        "attributes": ["balcony", "pool", "noisy", "near_center", "pet_friendly"],
    },
    "restaurant": {
        "question": "Which restaurant would you prefer to eat at?",
        "plural": "restaurants",
        "attributes": ["outdoor_seating", "live_music", "open_late", "takes_reservations", "family_run"],
    },
    "car": {
        "question": "Which car would you prefer to buy?",
        "plural": "cars",
        "attributes": ["electric", "convertible", "four_wheel_drive", "manual_transmission", "sunroof"],
    },
    "apartment": {
        "question": "Which apartment would you prefer to rent?",
        "plural": "apartments",
        "attributes": ["furnished", "ground_floor", "has_parking", "has_elevator", "south_facing"],
    },
    "laptop": {
        "question": "Which laptop would you prefer to buy?",
        "plural": "laptops",
        "attributes": ["touchscreen", "backlit_keyboard", "dedicated_gpu", "lightweight", "has_ethernet_port"],
    },
    "vacation": {
        "question": "Which vacation package would you prefer to book?",
        "plural": "vacation packages",
        "attributes": ["beach", "mountains", "all_inclusive", "international", "guided_tours"],
    },
    "phone": {
        "question": "Which phone would you prefer to buy?",
        "plural": "phones",
        "attributes": ["foldable", "waterproof", "dual_sim", "headphone_jack", "expandable_storage"],
    },
    "book": {
        "question": "Which book would you prefer to read?",
        "plural": "books",
        "attributes": ["hardcover", "illustrated", "part_of_series", "nonfiction", "translated"],
    },
    "job": {
        "question": "Which job offer would you prefer to accept?",
        "plural": "job offers",
        "attributes": ["remote", "travel_required", "night_shifts", "has_bonus", "large_team"],
    },
    "movie": {
        "question": "Which movie would you prefer to watch?",
        "plural": "movies",
        "attributes": ["animated", "subtitled", "sequel", "over_two_hours", "black_and_white"],
    },
}

DOMAIN_ORDER: list[str] = list(DOMAINS)

NAMES: list[str] = [
    "Prometheus", "Gregor Samsa", "Elizabeth Bennet", "Captain Ahab", "Hermione Granger",
    "Odysseus", "Jane Eyre", "Sherlock Holmes", "Atticus Finch", "Holden Caulfield",
    "Hester Prynne", "Jay Gatsby", "Scarlett O'Hara", "Ebenezer Scrooge", "Dorian Gray",
    "Anna Karenina", "Raskolnikov", "Don Quixote", "Sancho Panza", "Frodo Baggins",
    "Bilbo Baggins", "Galadriel", "Aragorn", "Katniss Everdeen", "Jean Valjean",
    "Javert", "Madame Bovary", "Heathcliff", "Catherine Earnshaw", "Mr. Darcy",
    "Pip", "Miss Havisham", "Oliver Twist", "Huckleberry Finn", "Tom Sawyer",
    "Becky Thatcher", "Alice", "the Mad Hatter", "Dracula", "Victor Frankenstein",
    "Captain Nemo", "Phileas Fogg", "Robinson Crusoe", "Gulliver", "Moby Dick",
    "Achilles", "Hector", "Penelope", "Circe", "Medusa",
    "Athena", "Apollo", "Artemis", "Hermes", "Persephone",
    "Orpheus", "Eurydice", "Icarus", "Daedalus", "Pandora",
    "Gilgamesh", "Enkidu", "Beowulf", "Grendel", "King Arthur",
    "Merlin", "Guinevere", "Lancelot", "Morgan le Fay", "Robin Hood",
    "Maid Marian", "Little John", "Peter Pan", "Wendy Darling", "Captain Hook",
    "Pinocchio", "Geppetto", "Mowgli", "Baloo", "Bagheera",
    "Dorothy Gale", "the Tin Man", "the Scarecrow", "Willy Wonka", "Charlie Bucket",
    "Matilda Wormwood", "Mary Poppins", "Paddington", "Winnie the Pooh", "Eeyore",
    "Hamlet", "Ophelia", "Macbeth", "Lady Macbeth", "King Lear",
    "Cordelia", "Prospero", "Miranda", "Puck", "Falstaff",
]

assert len(NAMES) == len(set(NAMES)) == 100
for _d in DOMAINS.values():
    assert len(_d["attributes"]) == 5 and len(set(_d["attributes"])) == 5
