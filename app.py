import os
import json
import base64
from io import BytesIO
from typing import Optional, Tuple

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from PIL import Image, ImageDraw, ImageFont

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# ---------- Gemini client ----------

client = genai.Client()
GEMINI_MODEL = "gemini-2.5-flash"

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# ---------- Simple global session (single user) ----------
SESSION_STATE = {
    "step": "START",
    "product_name": None,
    "selected_image": None,
    "headline": None,
    "description": None,
    "banner_base64": None,
}


def reset_session():
    global SESSION_STATE
    SESSION_STATE = {
        "step": "START",
        "product_name": None,
        "selected_image": None,
        "headline": None,
        "description": None,
        "banner_base64": None,
    }


# ---------- Routes ----------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_msg = data.get("message", "").strip()

    reply, banner_b64 = await handle_chat(user_msg)
    return JSONResponse({"reply": reply, "banner_base64": banner_b64})


# ---------- State machine / brain ----------

async def handle_chat(user_msg: str) -> Tuple[str, Optional[str]]:
    global SESSION_STATE

    # Initial hidden message from frontend
    if user_msg == "__INIT__":
        reset_session()
        return (
            "Hi, I can help you design a marketing banner.\n"
            "What product are you designing?",
            None,
        )

    # Manual reset
    if user_msg.lower() in {"restart", "reset"}:
        reset_session()
        return ("Session reset. What product are you designing?", None)

    step = SESSION_STATE["step"]

    # STEP 1: Product name
    if step == "START":
        SESSION_STATE["product_name"] = user_msg
        SESSION_STATE["step"] = "ASK_IMAGE"

        try:
            files = [
                f for f in os.listdir("assets")
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
            ]
        except FileNotFoundError:
            return ("Assets folder not found. Please create an 'assets' folder.", None)

        if not files:
            return ("No images found in assets folder. Add images and retry.", None)

        image_list = [f for f in files if f.lower() != "logo.png"]
        files_str = ", ".join(image_list) or ", ".join(files)

        return (
            f"Great! You are designing a banner for **{user_msg}**.\n\n"
            f"Available images in assets: {files_str}\n"
            "Type the image filename you want to use (e.g. `shoes1.jpg`).",
            None,
        )

    # STEP 2: Ask user to choose image
    if step == "ASK_IMAGE":
        selected = user_msg
        path = os.path.join("assets", selected)

        if not os.path.isfile(path):
            return (f"`{selected}` not found in assets folder. Try again.", None)

        SESSION_STATE["selected_image"] = path
        SESSION_STATE["step"] = "ASK_HEADLINE"

        return ("Nice choice! Now provide a **headline** (e.g., `The new K-9000`).", None)

    # STEP 3: Headline
    if step == "ASK_HEADLINE":
        SESSION_STATE["headline"] = user_msg
        SESSION_STATE["step"] = "ASK_DESCRIPTION"

        return ("Got the headline ✅\nNow give a short **description**.", None)

    # STEP 4: Description → trigger Gemini + banner
    if step == "ASK_DESCRIPTION":
        SESSION_STATE["description"] = user_msg
        SESSION_STATE["step"] = "GENERATE_BANNER"

        try:
            banner_b64 = generate_banner_with_gemini()
        except Exception as e:
            SESSION_STATE["step"] = "DONE"
            return (f"❌ Error while generating banner: {e}", None)

        SESSION_STATE["banner_base64"] = banner_b64
        SESSION_STATE["step"] = "DONE"

        return ("Here is the banner I have designed for you:", banner_b64)

    # STEP DONE: already have banner
    if step == "DONE":
        return (
            "Banner already generated. Type `restart` to create a new one.",
            SESSION_STATE.get("banner_base64"),
        )

    # Fallback
    return ("Unknown state. Type `restart` to reset.", None)


# ---------- Gemini helpers ----------

def detect_mime(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".jpg", ".jpeg"]:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    return "image/jpeg"


def call_gemini_design(base_image_path: str, logo_path: str,
                       headline: str, description: str) -> dict:
    #return response from gemini with coordinates

    with open(base_image_path, "rb") as f:
        base_bytes = f.read()
    with open(logo_path, "rb") as f:
        logo_bytes = f.read()

    base_mime = detect_mime(base_image_path)
    logo_mime = detect_mime(logo_path)

    prompt = f"""
You are a senior graphic designer helping place text and a logo on a product marketing banner.

You will receive:
- A main product image (the banner background).
- A logo image.
- A headline: {headline!r}
- A description: {description!r}

Your task:
- Choose a good area on the image for the text (headline + description + product name) so it is clearly legible.
- Choose a good area for the logo.
- Pick a text color that has strong contrast with the background.
- give me reason for why the placement coordinates

Return ONLY JSON, no explanation, no markdown, exactly in this structure:

{{
  "text_placement": {{"x_rel": float, "y_rel": float}},
  "logo_placement": {{"x_rel": float, "y_rel": float}},
  "text_color": "#RRGGBB"
  "reason" : "why coordinates are there only?, what did you saw in the image??"
}}

Where:
- x_rel and y_rel are between 0.0 and 1.0 (relative coordinates within the image).
- text_color is a hex color string like "#FFFFFF".
"""
    print("\n\nPrompt to GEMINI : \n\n", prompt)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(data=base_bytes, mime_type=base_mime),
            types.Part.from_bytes(data=logo_bytes, mime_type=logo_mime),
            prompt,
        ],
    )
    print("\n\nGemini Response : \n", response)

    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)

    # basic sanity / clamping
    for key in ("text_placement", "logo_placement"):
        if key in data:
            for coord in ("x_rel", "y_rel"):
                val = float(data[key].get(coord, 0.1))
                data[key][coord] = max(0.0, min(1.0, val))
    print("\n DATA : ", data)
    return data


def call_gemini_qa(banner_bytes: bytes) -> dict:
# check eligibility
    prompt = """
You are a QA agent for banner legibility.

I will send you a marketing banner image that contains:
- A background product photo (may contain decorative or script text).
- A logo.
- A newly added headline and short description.

Important:
- IGNORE any decorative or script text that is part of the original photo/background.
- ONLY evaluate the newly added overlay text (headline, description, product name label).

Your job:
- Check if this overlay text is clearly readable (color contrast, size, placement).
- If it's readable enough for a typical user, mark is_legible as true.
- If it's hard to read, mark is_legible as false and explain why.

Return ONLY JSON exactly like:

{
  "is_legible": true or false,
  "critique": "short explanation"
}
"""
    print("\n\nPrompt to GEMINI : \n\n", prompt)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            types.Part.from_bytes(
                data=banner_bytes,
                mime_type="image/png",
            ),
            prompt,
        ],
    )
    print("\n\nGemini Response : \n", response)
    raw = response.text.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    data = json.loads(raw)
    print("\n DATA : ", data)
    return data


# ---------- Banner composition helpers ----------

def hex_to_rgba(hex_str: str, alpha: int = 255):
    hex_str = hex_str.strip().lstrip("#")
    if len(hex_str) != 6:
        return (255, 255, 255, alpha)
    r = int(hex_str[0:2], 16)
    g = int(hex_str[2:4], 16)
    b = int(hex_str[4:6], 16)
    return (r, g, b, alpha)

def wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.FreeTypeFont, max_width: int):
# splits text into multiple lines
    if not text:
        return []

    words = text.split()
    lines = []
    current = ""

    for word in words:
        # Try adding this word to the current line
        test = word if current == "" else current + " " + word

        bbox = draw.textbbox((0, 0), test, font=font)
        w = bbox[2] - bbox[0]

        if w <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines

# ---------- Main banner generation ----------

def generate_banner_with_gemini() -> str:

    # Uses Gemini to:
    # - Analyze image + logo + text
    # - Suggest placements and text color
    # - Composite banner with Pillow
    # - Run QA on final banner (non-blocking; logs warnings only)

    # Returns base64-encoded PNG.

    base_image_path = SESSION_STATE["selected_image"]
    headline = SESSION_STATE["headline"] or ""
    description = SESSION_STATE["description"] or ""
    product_name = SESSION_STATE["product_name"] or ""

    logo_path = os.path.join("assets", "logo.png")
    if not os.path.isfile(logo_path):
        raise Exception("logo.png missing in assets folder.")

    # 1. Ask Gemini for layout
    design = call_gemini_design(base_image_path, logo_path, headline, description)

    # 2. Compose with Pillow using that layout
    base_img = Image.open(base_image_path).convert("RGBA")
    logo_img = Image.open(logo_path).convert("RGBA")

    base_w, base_h = base_img.size

    # Optionally upscale very small images for better text
    if base_w < 800:
        scale = 800 / base_w
        base_img = base_img.resize((800, int(base_h * scale)), Image.LANCZOS)
        base_w, base_h = base_img.size
        logo_img = logo_img.resize(
            (int(logo_img.width * scale), int(logo_img.height * scale)),
            Image.LANCZOS,
        )

    # Resize logo (~15% width)
    target_w = int(base_w * 0.15)
    aspect = logo_img.height / logo_img.width
    logo_img = logo_img.resize((target_w, int(target_w * aspect)), Image.LANCZOS)

    draw = ImageDraw.Draw(base_img)

    # Coordinates from Gemini (relative → absolute)
    tx_rel = design.get("text_placement", {}).get("x_rel", 0.1)
    ty_rel = design.get("text_placement", {}).get("y_rel", 0.7)
    lx_rel = design.get("logo_placement", {}).get("x_rel", 0.8)
    ly_rel = design.get("logo_placement", {}).get("y_rel", 0.1)

    # Avoid placing text too close to the top; move towards lower third if needed
    if ty_rel < 0.3:
        ty_rel = 0.65

    text_x = int(tx_rel * base_w)
    text_y = int(ty_rel * base_h)
    logo_x = int(lx_rel * base_w)
    logo_y = int(ly_rel * base_h)

    # Clamp to stay inside image bounds
    text_x = max(20, min(base_w - 20, text_x))
    text_y = max(20, min(base_h - 20, text_y))
    logo_x = max(0, min(base_w - logo_img.width, logo_x))
    logo_y = max(0, min(base_h - logo_img.height, logo_y))

    # Paste logo with transparency
    base_img.paste(logo_img, (logo_x, logo_y), logo_img)

    # ---------- Fonts (TTF if available, with minimum sizes) ----------
    try:
        headline_size = max(int(base_h * 0.07), 36)
        desc_size = max(int(base_h * 0.04), 24)

        font_headline = ImageFont.truetype("assets/Roboto-Bold.ttf", size=headline_size)
        font_desc = ImageFont.truetype("assets/Roboto-Regular.ttf", size=desc_size)
    except Exception as e:
        print("⚠️ Could not load TTF fonts, using default:", e)
        font_headline = ImageFont.load_default()
        font_desc = ImageFont.load_default()

    # ---------- Prepare wrapped text lines ----------
    logical_lines = []
    if product_name:
        logical_lines.append((product_name.upper(), font_desc))
    if headline:
        logical_lines.append((headline, font_headline))
    if description:
        logical_lines.append((description, font_desc))

    if logical_lines:
        line_gap = 6
        padding = 12

        wrapped_lines = []  # list of (text, font)
        max_text_width = int(base_w * 0.6)  # text box at most 60% of width

        for text, font in logical_lines:
            for line in wrap_text(draw, text, font, max_text_width):
                wrapped_lines.append((line, font))

        # Measure block size
        max_w = 0
        total_h = 0
        for text, font in wrapped_lines:
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            max_w = max(max_w, w)
            total_h += h
        total_h += (len(wrapped_lines) - 1) * line_gap

        box_w = max_w + padding * 2
        box_h = total_h + padding * 2

        # Adjust position so box fits fully in image
        if text_x + box_w + 20 > base_w:
            text_x = base_w - box_w - 20
        if text_y + box_h + 20 > base_h:
            text_y = base_h - box_h - 20
        text_x = max(20, text_x)
        text_y = max(20, text_y)

        # Draw semi-transparent black box behind text
        box_img = Image.new("RGBA", (box_w, box_h), (0, 0, 0, 170))
        base_img.paste(box_img, (text_x, text_y), box_img)

        # White text (high contrast on dark box)
        text_color = (255, 255, 255, 255)

        cursor_y = text_y + padding
        for text, font in wrapped_lines:
            draw.text((text_x + padding, cursor_y), text, fill=text_color, font=font)
            bbox = draw.textbbox((0, 0), text, font=font)
            line_h = bbox[3] - bbox[1]
            cursor_y += line_h + line_gap

    # Save to buffer
    buffer = BytesIO()
    base_img.save(buffer, format="PNG")
    buffer.seek(0)
    banner_bytes = buffer.read()

    # 3. QA with Gemini (non-blocking: logs only)
    try:
        qa = call_gemini_qa(banner_bytes)
        is_legible = bool(qa.get("is_legible", True))
        critique = qa.get("critique", "")

        if not is_legible:
            print("QA warning – text may not be fully legible:", critique)
        else:
            print("QA passed – overlay text considered legible.")
    except Exception as e:
        print("QA call failed!:", e)

    # 4. Encode to base64 for frontend
    banner_b64 = base64.b64encode(banner_bytes).decode("utf-8")
    return banner_b64