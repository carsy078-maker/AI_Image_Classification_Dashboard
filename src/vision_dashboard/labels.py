"""ImageNet-1k 라벨과 이모지 매핑.

라벨은 `assets/imagenet_classes.json` 에 캐시해 둔다. ONNX 백엔드만 쓰는
배포 환경에서 라벨 하나 얻자고 transformers 설정을 내려받게 만들 이유가 없다.
"""

from __future__ import annotations

import json
from functools import lru_cache

from .config import MODEL_NAME, ROOT

LABELS_PATH = ROOT / "assets" / "imagenet_classes.json"


@lru_cache(maxsize=1)
def id2label() -> list[str]:
    """클래스 인덱스 순서대로 정렬된 라벨 1000개."""
    if LABELS_PATH.exists():
        return json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    return fetch_and_cache_labels()


def fetch_and_cache_labels() -> list[str]:
    """원본 설정에서 라벨을 받아 캐시 파일로 저장한다."""
    from transformers import AutoConfig

    mapping = AutoConfig.from_pretrained(MODEL_NAME).id2label
    labels = [mapping[i] for i in range(len(mapping))]
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(labels, indent=0, ensure_ascii=False), encoding="utf-8")
    return labels


def label_of(index: int) -> str:
    labels = id2label()
    return labels[index] if 0 <= index < len(labels) else f"class_{index}"


# 라벨 문자열에 포함된 키워드로 이모지를 고른다. 위에서부터 순서대로 검사하므로
# 구체적인 규칙을 앞에 둔다 ("hot dog" 는 개가 아니라 음식이다).
#
# ImageNet-1k 는 개 품종만 120종이라 종별로 나열하면 끝이 없다. 품종 이름에
# 공통으로 나타나는 어간(terrier, retriever, spaniel, hound...)을 잡는 편이
# 규칙 수 대비 커버리지가 높다.
EMOJI_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    # 음식 - 동물 이름과 겹치는 항목이 있어 가장 먼저 검사한다
    (("hotdog", "hot dog", "cheeseburger", "bagel", "pretzel"), "🍔"),
    (("pizza",), "🍕"),
    (("ice cream", "ice lolly", "trifle"), "🍨"),
    (("espresso", "cup", "coffee"), "☕"),
    (("wine", "red wine", "beer", "cocktail"), "🍷"),
    (("banana", "orange", "lemon", "pineapple", "strawberry", "fig", "pomegranate"), "🍎"),
    (("broccoli", "cauliflower", "cucumber", "zucchini", "cabbage", "mushroom"), "🥦"),
    (("bread", "dough", "meat loaf", "burrito", "carbonara", "guacamole"), "🍽️"),
    # 개
    (
        (
            "terrier",
            "retriever",
            "spaniel",
            "hound",
            "poodle",
            "setter",
            "sheepdog",
            "shepherd",
            "corgi",
            "husky",
            "malamute",
            "collie",
            "pinscher",
            "mastiff",
            "bulldog",
            "chihuahua",
            "pug",
            "dalmatian",
            "pomeranian",
            "samoyed",
            "schnauzer",
            "dog",
            "puppy",
        ),
        "🐶",
    ),
    # 고양이·큰 고양잇과
    (("tabby", "siamese cat", "persian cat", "egyptian cat", "cat"), "🐱"),
    (("lion", "tiger", "leopard", "jaguar", "cheetah", "lynx", "cougar", "panther"), "🐯"),
    # 설치류·소형 포유류
    (("hamster", "guinea pig", "mouse", "rat", "porcupine", "beaver", "marmot"), "🐹"),
    (("rabbit", "hare", "wallaby"), "🐰"),
    (("squirrel", "chipmunk"), "🐿️"),
    # 대형 포유류
    (("bear", "panda"), "🐻"),
    (("elephant", "mammoth"), "🐘"),
    (("zebra",), "🦓"),
    (("horse", "pony", "sorrel"), "🐴"),
    (("cow", "ox", "bison", "buffalo", "bull"), "🐮"),
    (("sheep", "ram", "bighorn", "goat", "ibex"), "🐑"),
    (("pig", "hog", "boar", "warthog"), "🐷"),
    (("monkey", "ape", "gorilla", "chimpanzee", "orangutan", "baboon", "macaque", "lemur"), "🐵"),
    (("wolf", "coyote", "fox", "dingo", "jackal"), "🐺"),
    (("deer", "elk", "gazelle", "impala", "antelope"), "🦌"),
    (("camel", "llama", "giraffe", "hippopotamus", "rhinoceros"), "🦒"),
    # 조류
    (("penguin",), "🐧"),
    (("owl",), "🦉"),
    (("eagle", "hawk", "falcon", "vulture", "kite"), "🦅"),
    (("duck", "goose", "swan", "pelican", "flamingo", "stork", "crane bird"), "🦆"),
    (("cock", "hen", "chicken", "quail", "partridge", "peacock", "turkey"), "🐔"),
    (("parrot", "macaw", "cockatoo", "toucan", "hornbill", "lorikeet"), "🦜"),
    (("jay", "magpie", "robin", "finch", "sparrow", "warbler", "wren", "bulbul", "bird"), "🐦"),
    # 수생·파충류·곤충
    (("shark", "whale", "dolphin", "orca", "grampus"), "🐋"),
    (("fish", "goldfish", "eel", "ray", "sturgeon", "barracuda", "anemone fish"), "🐠"),
    (("crab", "lobster", "crayfish", "shrimp"), "🦀"),
    (("turtle", "tortoise", "terrapin"), "🐢"),
    (("snake", "serpent", "cobra", "viper", "python", "boa", "mamba"), "🐍"),
    (("lizard", "iguana", "chameleon", "gecko", "skink", "alligator", "crocodile"), "🦎"),
    (("frog", "toad", "salamander", "newt", "axolotl"), "🐸"),
    (("butterfly", "moth", "admiral", "monarch"), "🦋"),
    (("bee", "wasp", "ant", "beetle", "cricket", "grasshopper", "mantis", "dragonfly"), "🐝"),
    (("spider", "tarantula", "scorpion", "tick"), "🕷️"),
    (("snail", "slug", "jellyfish", "starfish", "urchin", "coral", "conch"), "🐚"),
    # 탈것
    (("airliner", "airship", "warplane", "aircraft", "plane"), "✈️"),
    (("ship", "boat", "canoe", "kayak", "catamaran", "yawl", "trimaran", "liner"), "🚢"),
    (("train", "locomotive", "streetcar", "subway"), "🚆"),
    (("truck", "van", "trailer", "lorry", "pickup"), "🚚"),
    (("bus", "trolleybus", "minibus"), "🚌"),
    (("motorcycle", "moped", "scooter", "motor scooter"), "🏍️"),
    (("bicycle", "bike", "unicycle", "tricycle"), "🚲"),
    (("convertible", "limousine", "jeep", "cab", "sports car", "wagon", "car"), "🚗"),
    (("rocket", "space shuttle", "missile"), "🚀"),
    # 사물
    (
        (
            "laptop",
            "notebook",
            "desktop computer",
            "monitor",
            "screen",
            "keyboard",
            "mouse computer",
        ),
        "💻",
    ),
    (("cellular telephone", "telephone", "dial telephone", "iphone", "phone"), "📱"),
    (("camera", "lens", "projector", "polaroid"), "📷"),
    (("clock", "watch", "sundial", "hourglass", "timer"), "⏰"),
    (
        ("guitar", "piano", "violin", "cello", "drum", "trumpet", "saxophone", "flute", "banjo"),
        "🎸",
    ),
    (("book", "bookcase", "library", "notebook", "binder", "envelope"), "📚"),
    (("chair", "sofa", "table", "desk", "bed", "wardrobe", "cabinet", "bench"), "🪑"),
    (("shoe", "boot", "sandal", "clog", "sneaker", "loafer"), "👟"),
    (("shirt", "jersey", "sweatshirt", "suit", "gown", "kimono", "coat", "jean"), "👕"),
    (("hat", "cap", "helmet", "bonnet", "sombrero", "turban"), "🎩"),
    (("bottle", "jug", "pitcher", "flask", "canteen"), "🍾"),
    (("umbrella", "parachute"), "☂️"),
    (("candle", "torch", "lamp", "lampshade", "spotlight", "candelabra"), "💡"),
    (("knife", "cleaver", "hatchet", "axe", "chain saw"), "🔪"),
    (("gun", "rifle", "revolver", "cannon", "holster"), "🔫"),
    # 자연·장소
    (("volcano", "geyser", "cliff", "valley", "alp", "promontory"), "🏔️"),
    (("seashore", "sandbar", "lakeside", "coral reef"), "🏖️"),
    (("daisy", "orchid", "lily", "rose", "sunflower", "flower"), "🌸"),
    (("tree", "oak", "pine", "maple", "willow", "cardoon", "yellow lady"), "🌳"),
    (("castle", "palace", "monastery", "church", "mosque", "dome", "stupa"), "🏰"),
    (("house", "home", "barn", "boathouse", "greenhouse", "residence", "mobile home"), "🏠"),
    (("bridge", "viaduct", "pier", "dam"), "🌉"),
)

FALLBACK_EMOJI = "🤖"


@lru_cache(maxsize=2048)
def emoji_for(label: str) -> str:
    """라벨에 어울리는 이모지. 매칭되는 규칙이 없으면 기본값을 돌려준다."""
    text = label.lower()
    for keywords, emoji in EMOJI_RULES:
        if any(k in text for k in keywords):
            return emoji
    return FALLBACK_EMOJI


def decorate(label: str, use_emoji: bool) -> str:
    return f"{emoji_for(label)} {label}" if use_emoji else label


def emoji_coverage() -> float:
    """규칙이 ImageNet 1000 클래스 중 몇 %를 덮는지. README 수치 검증용."""
    labels = id2label()
    hit = sum(1 for label in labels if emoji_for(label) != FALLBACK_EMOJI)
    return round(hit / len(labels) * 100, 1)
