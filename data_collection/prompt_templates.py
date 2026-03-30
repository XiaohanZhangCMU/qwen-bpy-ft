"""
System prompt, user-turn templates, and scene seed list for data collection.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Blender Python API (bpy) programmer. Your job is to write \
complete, self-contained Python scripts that run inside Blender to create 3D scenes.

Rules you must always follow:
1. Every script must start with `import bpy` and must work in Blender's headless \
background mode (`blender --background`).
2. Clear the default scene at the start: delete all default objects before adding yours.
3. Always add a camera and at least one light source.
4. Use only the standard Blender Python API (`bpy`). Do not import third-party packages.
5. The script must be self-contained — no external files, no relative imports.
6. All geometry should be roughly human-scale (metres). A standard door is ~2 m tall.
7. When placing multiple objects, avoid interpenetration: space objects apart realistically.
8. Wrap your script in a fenced code block: ```python\\n...\\n```
"""

# ---------------------------------------------------------------------------
# Turn templates
# ---------------------------------------------------------------------------

def format_initial_user_turn(scene_description: str) -> str:
    return (
        f"Create a Blender 3D scene: {scene_description}\n\n"
        "Write a complete bpy Python script that generates this scene. "
        "The script should create all objects, set up materials with basic colors, "
        "add a camera, add lighting, and produce a full scene layout."
    )


def format_repair_user_turn(stderr: str, exit_code: int, attempt: int) -> str:
    trimmed_stderr = stderr.strip()[-2000:] if stderr.strip() else "(no stderr output)"
    return (
        f"The script failed (attempt {attempt}, exit code {exit_code}).\n\n"
        f"Blender error output:\n```\n{trimmed_stderr}\n```\n\n"
        "Please fix all errors and return a corrected, complete bpy script."
    )


def format_layout_feedback_turn(issues: list[str]) -> str:
    issues_str = "\n".join(f"- {i}" for i in issues)
    return (
        "The script ran successfully but the scene has layout issues:\n"
        f"{issues_str}\n\n"
        "Please revise the script to fix these issues."
    )


def format_scene_check_turn(n_objects: int, n_mesh: int) -> str:
    return (
        f"The script ran successfully but the scene only has {n_objects} object(s) "
        f"({n_mesh} mesh object(s)). "
        "Please create a richer, more complete scene with more objects and details."
    )


# ---------------------------------------------------------------------------
# Scene seed list  (200+ diverse descriptions)
# ---------------------------------------------------------------------------

SCENE_SEEDS: list[str] = [
    # --- Bedrooms ---
    "a cozy master bedroom with a king-size bed, two nightstands, a wardrobe, and bedside lamps",
    "a minimalist bedroom with a low platform bed, a desk, and a single pendant light",
    "a children's bedroom with bunk beds, a toy chest, and a study table",
    "a teenage bedroom with a loft bed above a desk, a bookshelf, and a gaming chair",
    "a luxury hotel bedroom with a four-poster bed, two armchairs, and floor lamps",
    "a studio apartment bedroom corner with a Murphy bed, floating shelves, and a compact wardrobe",
    "a bohemian bedroom with a canopy bed, floor cushions, low shelves, and string lights",

    # --- Living Rooms ---
    "a modern living room with an L-shaped sofa, a coffee table, a TV stand, and a floor lamp",
    "a traditional living room with a three-seater sofa, two armchairs, a fireplace, and a rug",
    "a Scandinavian living room with a low sofa, a round coffee table, and a houseplant",
    "an open-plan living area with a sofa, bookshelf wall, and a dining table nearby",
    "a mid-century modern living room with tapered-leg furniture, a sideboard, and a record player",
    "a cozy reading nook with an armchair, a side table, a floor lamp, and wall-mounted bookshelves",
    "a minimalist Japanese living room with floor cushions, a low table, and a shoji screen",

    # --- Home Offices / Studies ---
    "a home office with an L-shaped desk, a monitor, an office chair, and a bookshelf",
    "a compact study nook with a wall-mounted desk, a chair, a corkboard, and shelves",
    "a creative studio with drafting tables, art supply storage, and track lighting",
    "a library study with floor-to-ceiling bookshelves, a reading chair, and a desk lamp",
    "a standing-desk office with dual monitors, a cable tray, and a whiteboard",

    # --- Kitchens ---
    "a modern kitchen with upper and lower cabinets, an island, a refrigerator, and pendant lights",
    "a farmhouse kitchen with open shelving, a large sink, a wooden dining table, and bar stools",
    "a galley kitchen with countertops on both sides, a stove, and overhead cabinets",
    "a compact studio kitchen with a two-burner stove, a mini-fridge, and a small counter",
    "a professional chef's kitchen with a commercial range, a prep island, and pot rack",

    # --- Dining Rooms ---
    "a formal dining room with a six-seat table, a sideboard, a chandelier, and dining chairs",
    "a casual dining area with a round table, four chairs, and a pendant light above",
    "a breakfast nook with a built-in bench, a small table, and a window",

    # --- Bathrooms ---
    "a master bathroom with a freestanding bathtub, a double vanity, and a walk-in shower",
    "a compact bathroom with a wall-mounted sink, a toilet, and a shower-over-bath",
    "a spa-style bathroom with a soaking tub, pebble flooring, and bamboo accessories",

    # --- Cafés / Coffee Shops ---
    "a cozy café with wooden tables, mismatched chairs, a counter with an espresso machine, and pendant lights",
    "a modern coffee shop with bar-height seating along the window, a minimalist counter, and exposed brick",
    "a vintage café with round marble-top tables, bentwood chairs, and a display case of pastries",
    "a rooftop café with outdoor tables, parasols, planters, and string lights",
    "an industrial-style café with metal pipe shelving, reclaimed wood tables, and Edison bulbs",

    # --- Restaurants ---
    "a fine-dining restaurant with linen-covered tables, upholstered chairs, and dim pendant lights",
    "a ramen shop with a counter bar, high stools, and hanging lanterns",
    "a pizza restaurant with checkered tablecloths, wooden chairs, and a stone oven visible in the back",
    "an outdoor terrace restaurant with wicker chairs, potted palms, and market lights",

    # --- Bars / Lounges ---
    "a cocktail bar with a long counter, bar stools, bottle shelving behind the bar, and dim lighting",
    "a speakeasy-style bar with low leather sofas, dark wood tables, and Edison bulb fixtures",
    "a rooftop bar with high-top tables, bar stools, a city-view railing, and string lights",

    # --- Offices / Workplaces ---
    "an open-plan office with rows of desks, monitors, divider panels, and overhead fluorescent lighting",
    "a meeting room with a long conference table, chairs on both sides, and a projector screen",
    "a reception area with a curved front desk, visitor chairs, and a logo on the wall",
    "a co-working space with hot-desks, phone booths, a communal kitchen, and lounge seating",

    # --- Retail / Shops ---
    "a boutique clothing shop with clothing racks, a mirror, a checkout counter, and track lighting",
    "a bookshop with floor-to-ceiling bookshelves, a reading corner, and rolling ladders",
    "a flower shop with display buckets of flowers, a wrapping counter, and hanging dried blooms",
    "a record store with LP bins, a listening station, and vintage music posters",
    "a bakery storefront with a glass display case, shelves of bread, a counter, and a menu board",

    # --- Parks / Outdoor Plazas ---
    "a city park corner with a bench, a lamp post, a trash bin, and surrounding trees",
    "a public plaza with a central fountain, surrounding benches, paving stones, and trees",
    "a playground with swings, a slide, a climbing frame, and park benches nearby",
    "a rooftop garden with raised planter beds, a pergola, outdoor furniture, and trellis",
    "a Japanese garden with a stone lantern, a koi pond, stepping stones, and bonsai trees",
    "a picnic area with wooden picnic tables, BBQ grills, and tall shade trees",

    # --- Street Markets / Night Markets ---
    "a night market lane with food stalls, hanging lanterns, folding tables, and tarp canopies",
    "a farmers market with vendor tents, wooden crates of produce, and a crowd path",
    "a street food corner with a noodle cart, plastic stools and tables, and a signboard",
    "an antique flea market with trestle tables covered in objects and parasols",
    "a flower market alley with rows of flower buckets and a small vendor booth",

    # --- Streets / Urban Exteriors ---
    "a city street corner with a fire hydrant, a mailbox, a bus stop shelter, and a lamp post",
    "a narrow cobblestone alley with potted plants, a doorway, and string lights overhead",
    "a suburban sidewalk with a picket fence, a mailbox, a tree, and a bus stop sign",
    "a parking lot entrance with a barrier gate, a booth, painted lines, and light poles",
    "a pedestrian bridge over a canal with railings, lamp posts, and benches",

    # --- Gyms / Fitness ---
    "a home gym with a barbell rack, a bench press, dumbbells, and rubber flooring",
    "a yoga studio with mats laid out, a wall mirror, and diffused overhead lighting",
    "a boxing gym with a ring, heavy bags hanging from the ceiling, and a speed bag",

    # --- Medical / Clinical ---
    "a doctor's examination room with an exam table, a doctor's stool, a desk, and a cabinet",
    "a dentist's office with a dental chair, overhead light arm, a tray of instruments, and a sink",
    "a hospital ward with two patient beds, privacy curtains, IV stands, and bedside tables",

    # --- Education ---
    "a primary school classroom with rows of small desks, a chalkboard, and a teacher's desk",
    "a lecture hall with tiered seating, folding desks, and a projector at the front",
    "a science lab with lab benches, stools, a fume hood, and overhead storage",
    "a kindergarten room with low tables, cubbies, colorful mats, and toy shelves",

    # --- Entertainment / Media ---
    "a home theater with a projector screen, recliner seats, a surround-sound rack, and dim lighting",
    "a recording studio control room with a mixing console, monitor speakers, and acoustic panels",
    "a photography studio with a backdrop, studio strobes on stands, a camera on a tripod",
    "an arcade with rows of arcade cabinets, neon signs, and carpet flooring",

    # --- Hotels / Hospitality ---
    "a hotel lobby with a reception desk, cluster seating, a luggage cart, and a chandelier",
    "a hotel room with a king bed, a writing desk, a TV unit, and an armchair by the window",
    "a hotel bathroom with a vanity, a glass shower, and towel rails",

    # --- Transport Hubs ---
    "a small airport gate area with rows of seats, a gate desk, and large windows",
    "a subway platform with benches, overhead signs, columns, and a track edge barrier",
    "a bus depot bay with a bus shelter, benches, a timetable board, and bins",

    # --- Nature / Rural ---
    "a farmyard with a barn, a tractor, hay bales, a fence, and a chicken coop",
    "a lakeside dock with wooden planks, mooring posts, a rowboat, and surrounding reeds",
    "a forest campsite with a tent, a campfire ring, log seats, and a hanging lantern",
    "a vineyard row with trellised vines, wooden posts, and a pruning cart",
    "a greenhouse with planting tables, overhead grow lights, shelves of seedlings, and a hose reel",

    # --- Industrial / Warehouse ---
    "a warehouse interior with pallet racking, a forklift, concrete floors, and overhead strip lighting",
    "a workshop bench area with tool pegboards, a workbench, a vise, and a stool",
    "a server room with rows of rack cabinets, cable trays, floor tiles, and cooling units",

    # --- Miscellaneous Interiors ---
    "a laundry room with a washer, a dryer, a folding table, shelving, and a hanging rail",
    "a mudroom with coat hooks, a bench with storage underneath, a shoe rack, and a mirror",
    "a walk-in closet with double hanging rails, shelf towers, a center island, and LED strip lighting",
    "a sauna room with tiered wooden benches, a stove, and stones, and a small window",
    "a wine cellar with floor-to-ceiling bottle racks, a tasting table, and pendant lights",
    "a meditation room with cushioned mats, a low altar table, candles, and soft diffused lighting",

    # --- Scenes with specific style prompts ---
    "a cyberpunk apartment corner with neon strip lights, holographic displays, and cluttered cables",
    "a Victorian parlor with a chesterfield sofa, a fireplace mantel, wall sconces, and heavy drapes",
    "a mid-century diner with a counter, swivel stools, booth seating, and a jukebox",
    "a brutalist concrete lobby with exposed concrete walls, a single bench, and a skylight",
    "an Art Deco hotel reception with a geometric floor, tall potted palms, and gilded fixtures",

    # --- Additional bedroom / living variations ---
    "a guest bedroom with a single bed, a nightstand, a small wardrobe, and neutral decor",
    "a nursery with a crib, a changing table, a rocking chair, and a mobile above the crib",
    "a sunroom with wicker furniture, potted plants, large windows, and a hanging chair",
    "a basement den with a sectional sofa, a pool table, a bar cart, and a dart board",
    "a penthouse terrace with a sun lounger, an outdoor sofa, planters, and a city skyline view",
    "a treehouse interior with wooden walls, a sleeping loft, a ladder, and small windows",

    # --- Additional outdoor / urban ---
    "a rooftop terrace with a pergola, outdoor dining set, string lights, and a grill",
    "a suburban backyard with a wooden deck, garden furniture, a BBQ, and a lawn",
    "a community garden plot with raised beds, a tool shed, a compost bin, and a watering can",
    "a street market entrance with a gate arch, a banner, vendor tables, and hanging signs",
    "an open-air amphitheater with tiered stone seating and a stage at the center",
    "a waterfront promenade with benches, lamp posts, a railing, and moored boats visible",
    "a desert courtyard with terracotta pots, a shaded pergola, mosaic tiles, and a water feature",
    "a zen rock garden with raked gravel, large stones, a wooden bridge, and bamboo",

    # --- Specialty / hobby ---
    "a model train layout room with a large table layout, track, miniature buildings, and shelving",
    "a ceramics studio with pottery wheels, a kiln, shelves of work, and a wedging table",
    "a florist's workroom with a long stem-cutting table, buckets of flowers, and a wall of tools",
    "a tattoo parlor with a reclining client chair, a stool, a tray of tools, and framed artwork",
    "a barber shop with two barber chairs, a mirror station, a waiting bench, and a pole",
]
