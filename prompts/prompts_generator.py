import random

# Lists of possible feature values
eye_shapes = ['almond-shaped', 'hooded', 'round', 'oval', 'downturned', 'upturned', 'monolid']
eye_colors = ['blue', 'green', 'brown', 'hazel', 'black', 'grey', 'amber']
nose_shapes = ['straight', 'snub', 'hooked', 'bulbous', 'narrow', 'wide']
lip_shapes = ['full', 'thin', 'heart-shaped', 'bow-shaped', 'downturned', 'upturned']
eyebrow_shapes = ['arched', 'straight', 'curved', 'angled', 'thin', 'thick']
skin_tones = ['fair', 'light', 'medium', 'olive', 'dark', 'deep', 'ebony']
hair_styles = ['straight', 'curly', 'wavy', 'frizzy', 'coiled', 'braided', 'slicked back']
hair_colors = ['blonde', 'brunette', 'black', 'red', 'ginger', 'auburn', 'silver']
face_shapes = ['oval', 'round', 'square', 'heart-shaped', 'diamond-shaped', 'triangle-shaped']
cheekbones = ['high', 'low', 'prominent', 'sunken', 'flat']
chin_shapes = ['pointed', 'round', 'cleft', 'double', 'square']
ear_shapes = ['small', 'large', 'attached', 'lobed', 'pointed']
expressions = ['smiling', 'serious', 'sad', 'surprised', 'angry', 'neutral']
hair_lengths = ['bald', 'shaved', 'buzz cut', 'short', 'medium', 'long']

# Function to generate a random elven princess portrait with given expression, hair length, and style
def generate_portrait(expression, hair_length, hair_style):
    # Randomize features that are not specified in the prompt
    eye_shape = random.choice(eye_shapes)
    eye_color = random.choice(eye_colors)
    nose_shape = random.choice(nose_shapes)
    lip_shape = random.choice(lip_shapes)
    eyebrow_shape = random.choice(eyebrow_shapes)
    skin_tone = random.choice(skin_tones)
    hair_color = random.choice(hair_colors)
    face_shape = random.choice(face_shapes)
    cheekbone = random.choice(cheekbones)
    chin_shape = random.choice(chin_shapes)
    ear_shape = random.choice(ear_shapes)

    # Generate a portrait with the specified expression, hair length, and style
    portrait = f"fantasy, fantasy portrait, realistic, game, portrait of a young with an {face_shape} face, {cheekbone} cheekbones, and a {chin_shape} chin, {eye_shape}, {eye_color} eyes, a {nose_shape} nose, {lip_shape} and {eyebrow_shape} eyebrows, {ear_shape} ears, {skin_tone}, clean skin, {hair_length} hair, {hair_style} hair, {hair_color} hair, realistic, realism, light, clean skin, from shoulders up, above shoulders, face, beauty"

    return portrait

# Generate portraits with different expressions, hair lengths, and styles
for expression in expressions:
    for hair_length in hair_lengths:
        for hair_style in hair_styles:
            portrait = generate_portrait(expression, hair_length, hair_style)
            print(portrait)
           
