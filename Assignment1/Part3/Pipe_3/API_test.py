from openai import OpenAI
import base64
from PIL import Image
import io

client = OpenAI(
    api_key="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",   # paste your key here
    base_url="https://openrouter.ai/api/v1"
)

# Load and upscale image to 224x224
img = Image.open(r"C:\Neural\Indian_Digits_Train\1.bmp")
img = img.resize((224, 224), Image.NEAREST)
img = img.convert("RGB")
buffer = io.BytesIO()
img.save(buffer, format="PNG")
image_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

prompt = """This image contains a single handwritten Eastern Arabic numeral
(also called Hindi-Arabic numerals, used in Arabic-speaking countries).
These are NOT Western digits. The shapes are completely different.
Here is how each Eastern Arabic digit looks:
- ٠ = 0 (small oval or dot)
- ١ = 1 (a vertical stroke, like a line)
- ٢ = 2 (looks like a reversed Z or hook)
- ٣ = 3 (looks like a reversed 3 or heart shape)
- ٤ = 4 (looks like a figure with a tail)
- ٥ = 5 (looks like a circle or oval)
- ٦ = 6 (looks like a backwards 7 or hook going down)
- ٧ = 7 (looks like a V or checkmark)
- ٨ = 8 (looks like a lambda or two humps)
- ٩ = 9 (looks like a loop with a tail going right)
Reply with ONLY the Western digit equivalent (0-9), nothing else."""

response = client.chat.completions.create(
    model="google/gemini-2.0-flash-exp:free",  # free vision model
    messages=[{
        "role": "user",
        "content": [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_data}"
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]
    }],
    max_tokens=5
)

print(f"Model says: {response.choices[0].message.content.strip()}")
