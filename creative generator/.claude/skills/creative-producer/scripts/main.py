#!/usr/bin/env python3
"""Creative Producer — Generates Static Ads via Gemini 3.1 Flash Image Generation."""

import argparse
import atexit
import base64
import json
import os
import signal
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

try:
    import requests
except ImportError:
    print("Error: 'requests' not installed. Run: pip3 install requests")
    sys.exit(1)

try:
    from dotenv import load_dotenv
except ImportError:
    print("Error: 'python-dotenv' not installed. Run: pip3 install python-dotenv")
    sys.exit(1)

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))

GEMINI_MODEL = "gemini-3.1-flash-image-preview"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# Thread-safe lock for board_data.json writes
board_lock = threading.Lock()

# Process lock file to prevent concurrent runs
LOCK_FILE = os.path.join(PROJECT_ROOT, "creatives", ".generation.lock")


def acquire_process_lock():
    """Prevent multiple generation processes from running simultaneously."""
    if os.path.exists(LOCK_FILE):
        try:
            with open(LOCK_FILE) as f:
                lock_data = json.load(f)
            pid = lock_data.get("pid", 0)
            # Check if the process is still alive
            try:
                os.kill(pid, 0)
                print(f"ERROR: Another generation is already running (PID {pid}).")
                print(f"If this is wrong, delete {LOCK_FILE}")
                sys.exit(1)
            except OSError:
                # Process is dead — stale lock, clean it up
                print(f"Stale lock from dead process {pid} — cleaning up")
                cleanup_stuck_generating()
                os.unlink(LOCK_FILE)
        except (json.JSONDecodeError, KeyError):
            os.unlink(LOCK_FILE)

    with open(LOCK_FILE, "w") as f:
        json.dump({"pid": os.getpid(), "started": datetime.now().isoformat()}, f)


def release_process_lock():
    """Release the process lock and clean up any stuck placeholders."""
    try:
        os.unlink(LOCK_FILE)
    except FileNotFoundError:
        pass


def cleanup_stuck_generating():
    """Mark any stuck 'generating' entries as 'failed' and remove them if no image."""
    board_data_path = os.path.join(PROJECT_ROOT, "creatives", "board_data.json")
    if not os.path.exists(board_data_path):
        return
    with open(board_data_path) as f:
        data = json.load(f)
    changed = False
    # Remove stuck generating entries that have no image
    data["ads"] = [a for a in data["ads"] if not (a.get("status") == "generating" and not a.get("image_path"))]
    # Re-index
    for i, ad in enumerate(data["ads"]):
        if ad["index"] != i + 1:
            ad["index"] = i + 1
            changed = True
    changed = True  # Always write if we got here
    if changed:
        with open(board_data_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print("Cleaned up stuck 'generating' entries")


def load_config():
    """Load .env and return API key."""
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or api_key == "your_gemini_api_key_here":
        print("Error: GEMINI_API_KEY not set in .env")
        sys.exit(1)
    return api_key


def encode_image(image_path):
    """Read and base64-encode an image file."""
    if not os.path.exists(image_path):
        print(f"Warning: Image not found: {image_path}")
        return None, None

    ext = os.path.splitext(image_path)[1].lower()
    mime_map = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
    mime_type = mime_map.get(ext, "image/jpeg")

    with open(image_path, "rb") as f:
        data = base64.standard_b64encode(f.read()).decode("utf-8")
    return data, mime_type


def build_gemini_prompt(ad_prompt_json, product_image_path):
    """Build the Gemini API request payload from a JSON prompt + product image."""

    # Build the text prompt from the JSON structure
    meta = ad_prompt_json["meta"]
    canvas = ad_prompt_json["canvas"]
    layout = ad_prompt_json["layout"]
    product = ad_prompt_json["product"]
    text_overlays = ad_prompt_json["text_overlays"]
    visual_elements = ad_prompt_json.get("visual_elements", {})
    brand_elements = ad_prompt_json.get("brand_elements", {})
    gen_instructions = ad_prompt_json["generation_instructions"]

    # --- SCENE TYPE GUARD ---
    # Negative scenes (problem, pain, competitor criticism) must NEVER show
    # the actual product. This is enforced here — not just a suggestion.
    scene_type = meta.get("scene_type", "positive")
    is_negative_scene = scene_type == "negative"

    if is_negative_scene:
        # DO NOT encode the product image — it must not appear
        image_data = None
        mime_type = None
        print(f"  ⚠ NEGATIVE SCENE: Product image BLOCKED (scene_type=negative)")
    else:
        image_data, mime_type = encode_image(product_image_path)

    # Check if logo will be composited in post-processing
    logo_visible = brand_elements.get('logo', {}).get('visible', False)

    # Compose detailed text prompt
    prompt_text = f"""Generate a professional static advertisement image with the following exact specifications.

FORMAT: {meta['format']} aspect ratio, {meta['resolution']['width']}x{meta['resolution']['height']} pixels.

STYLE: {gen_instructions['style_reference']}

BACKGROUND:
- Type: {canvas['background']['type']}
- Primary color: {canvas['background']['primary_color']}"""

    if canvas['background'].get('secondary_color'):
        prompt_text += f"\n- Secondary color: {canvas['background']['secondary_color']}"
        prompt_text += f"\n- Gradient direction: {canvas['background'].get('gradient_direction', 'top_to_bottom')}"
    if canvas['background'].get('scene_description'):
        prompt_text += f"\n- Scene: {canvas['background']['scene_description']}"
    if canvas['background'].get('texture_description'):
        prompt_text += f"\n- Texture: {canvas['background']['texture_description']}"

    prompt_text += f"""

LIGHTING:
- Type: {canvas['lighting']['type']}
- Direction: {canvas['lighting']['direction']}
- Warmth: {canvas['lighting']['warmth']}
- Intensity: {canvas['lighting']['intensity']}
- Shadows: {canvas['lighting']['shadows']}

COLOR MOOD:
- Palette: {', '.join(canvas['color_mood']['palette'])}
- Mood: {canvas['color_mood']['mood']}
- Saturation: {canvas['color_mood']['saturation']}
- Contrast: {canvas['color_mood']['contrast']}

LAYOUT:
- Type: {layout['type']}
- Top zone ({layout['zones']['top']['height_percent']}%): {layout['zones']['top']['content']}"""

    if layout['zones']['top'].get('background'):
        prompt_text += f" on {layout['zones']['top']['background']} background"

    prompt_text += f"""
- Middle zone ({layout['zones']['middle']['height_percent']}%): {layout['zones']['middle']['content']}
- Bottom zone ({layout['zones']['bottom']['height_percent']}%): {layout['zones']['bottom']['content']}
- Alignment: {layout['alignment']}
- Outer margins: {layout['margins']['outer']}

PRODUCT PLACEMENT:"""

    if is_negative_scene:
        prompt_text += f"""
IMPORTANT: This is a NEGATIVE/PROBLEM scene. Do NOT show the advertised product.
- Show ONLY generic, unbranded, no-name products — no visible brand names, logos, or recognizable designs.
- The product shown should look cheap, generic, or worn out — representing the COMPETITOR/PROBLEM, not our brand.
- Position: {product['position']['x']} horizontally, {product['position']['y']} vertically
- Scale: {product['scale']} (relative to canvas)
- Perspective: {product['perspective']}
"""
    else:
        prompt_text += f"""
- Display the provided product image as: {product['display_mode']}
- Position: {product['position']['x']} horizontally, {product['position']['y']} vertically
- Scale: {product['scale']} (relative to canvas)
- Rotation: {product['rotation']} degrees
- Perspective: {product['perspective']}
- Shadow: {product['shadow']['type']}, {product['shadow']['intensity']} intensity
"""

    if not is_negative_scene and product.get('surface'):
        prompt_text += f"- Surface: {product['surface']}\n"
    if not is_negative_scene and product.get('decorative_elements'):
        elements = [e for e in product['decorative_elements'] if e]
        if elements:
            prompt_text += f"- Decorative elements: {', '.join(elements)}\n"

    # Text overlays — simplified to avoid Gemini rendering CSS specs as visible text
    prompt_text += "\nTEXT OVERLAYS (render ONLY the text content below — correct German spelling, clean typography):\n"
    prompt_text += "CRITICAL: Only render the actual text content. Do NOT render any font names, sizes, colors, CSS values, or technical specifications as visible text in the image.\n"
    for i, overlay in enumerate(text_overlays, 1):
        style = overlay['style']
        # Describe font style in natural language, not CSS values
        font_desc = style['font_family']
        if 'italic' in str(style.get('font_weight', '')):
            font_desc += " italic"
        if 'bold' in str(style.get('font_weight', '')):
            font_desc += " bold"

        size_desc = "large" if "4" in str(style.get('font_size', '')) else "medium" if "2" in str(style.get('font_size', '')) else "small"

        prompt_text += f"\n  Text {i} ({overlay['role']}):\n"
        prompt_text += f"  - Render this text: \"{overlay['content']}\"\n"
        prompt_text += f"  - Place at: {overlay['position']['x']}, {overlay['position']['y']}\n"
        prompt_text += f"  - Style: {font_desc}, {size_desc} size, {style['color']} color\n"
        prompt_text += f"  - Alignment: {style['text_align']}\n"

        if overlay.get('emphasis_words'):
            words = [w for w in overlay['emphasis_words'] if w]
            if words:
                emphasis_color = overlay.get('emphasis_style', {}).get('color', '')
                prompt_text += f"  - Make these words stand out: {', '.join(words)}"
                if emphasis_color:
                    prompt_text += f" (in {emphasis_color})"
                prompt_text += "\n"

    # Badges
    badges = visual_elements.get('badges', [])
    if badges:
        prompt_text += "\nBADGES/BUTTONS:\n"
        for badge in badges:
            prompt_text += f"  - {badge['type']}: \"{badge['text']}\" at {badge['position']['x']},{badge['position']['y']}"
            prompt_text += f" — {badge['style']['shape']} shape, bg:{badge['style']['background_color']}, text:{badge['style']['text_color']}\n"

    # Logo — tell Gemini to leave space, we composite the real logo later
    if logo_visible:
        logo = brand_elements['logo']
        prompt_text += f"\nLOGO SPACE: Leave clean empty space at {logo['position']} (approximately 15% width, 4% height) for logo placement in post-processing. Do NOT render any logo text or wordmark — the real logo will be composited later.\n"

    if brand_elements.get('trust_signals'):
        signals = [s for s in brand_elements['trust_signals'] if s]
        if signals:
            prompt_text += f"\nTRUST SIGNALS (small text at bottom): {' | '.join(signals)}\n"

    # Must include/avoid
    # Filter out logo-related must_include items since we handle logo in post-processing
    must_include = [m for m in gen_instructions['must_include'] if 'logo' not in m.lower() and 'wordmark' not in m.lower()]
    prompt_text += f"\nMUST INCLUDE: {', '.join(must_include)}\n"
    must_avoid = gen_instructions['must_avoid'] + [
        "Do NOT render any brand logo or 'JUNGLÜCK' wordmark text — it will be added in post-processing",
        "dirty or messy backgrounds, crumbs, dirty socks, cluttered surfaces, unwashed dishes — keep backgrounds clean and tidy even for casual/authentic styles",
        "inventing product colors or variants that don't exist — only show the exact product provided",
    ]
    if is_negative_scene:
        must_avoid += [
            "ANY branded product, ANY recognizable brand name or logo — this is a negative/problem scene, show ONLY generic unbranded items",
            "ANY reference to the advertised brand — the product shown must look like a cheap competitor, NOT the advertised product",
        ]
    prompt_text += f"MUST AVOID: {', '.join(must_avoid)}\n"
    prompt_text += f"\nQUALITY: {gen_instructions['quality_notes']}\n"
    prompt_text += f"TEXT RENDERING: {gen_instructions['text_rendering_notes']}\n"

    # Build API payload
    parts = []

    # Add product image as reference — ONLY for positive/neutral scenes
    if image_data and not is_negative_scene:
        parts.append({
            "inline_data": {
                "mime_type": mime_type,
                "data": image_data
            }
        })
        parts.append({
            "text": "Above is the product image to incorporate into the ad. Use this exact product (bottle/packaging design, label, colors) in the generated image.\n\n"
        })
    elif is_negative_scene:
        parts.append({
            "text": "IMPORTANT: This is a NEGATIVE scene showing a PROBLEM. Do NOT use any branded or recognizable product. Show only generic, unbranded items. No brand names, no logos, no recognizable product designs.\n\n"
        })

    parts.append({"text": prompt_text})

    # Map format to aspect ratio
    fmt = ad_prompt_json["meta"]["format"]
    aspect_map = {"4:5": "4:5", "9:16": "9:16", "1:1": "1:1", "16:9": "16:9"}
    aspect_ratio = aspect_map.get(fmt, "4:5")

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "responseModalities": ["TEXT", "IMAGE"],
            "temperature": 0.8,
            "imageConfig": {
                "aspectRatio": aspect_ratio,
                "imageSize": "4K"
            }
        },
    }

    return payload


def call_gemini(api_key, payload, max_retries=3):
    """Call Gemini API and return the generated image data."""
    url = GEMINI_URL.format(model=GEMINI_MODEL)

    for attempt in range(max_retries):
        try:
            print(f"  Calling Gemini API (attempt {attempt + 1})...")
            resp = requests.post(
                url,
                params={"key": api_key},
                json=payload,
                timeout=180,
            )
            resp.raise_for_status()

            data = resp.json()
            candidates = data.get("candidates", [])
            if not candidates:
                print(f"  Warning: No candidates in response")
                continue

            parts = candidates[0].get("content", {}).get("parts", [])

            # Extract image from response (API returns camelCase "inlineData")
            for part in parts:
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    return inline["data"], inline.get("mimeType") or inline.get("mime_type", "image/png")

            print(f"  Warning: No image in response parts")
            for part in parts:
                if "text" in part:
                    print(f"  Gemini text response: {part['text'][:500]}")
            feedback = data.get("promptFeedback", {})
            if feedback:
                print(f"  Prompt feedback: {feedback}")
            finish = candidates[0].get("finishReason", "")
            if finish:
                print(f"  Finish reason: {finish}")

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:
                wait = 10 * (attempt + 1)
                print(f"  Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  HTTP Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise
        except Exception as e:
            print(f"  Error: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
                continue
            raise

    return None, None


def get_region_brightness(img, x, y, w, h):
    """Calculate average brightness of a region in the image."""
    region = img.crop((x, y, x + w, y + h)).convert("RGB")
    pixels = list(region.getdata())
    if not pixels:
        return 128
    avg_r = sum(p[0] for p in pixels) / len(pixels)
    avg_g = sum(p[1] for p in pixels) / len(pixels)
    avg_b = sum(p[2] for p in pixels) / len(pixels)
    # Perceived brightness formula
    return 0.299 * avg_r + 0.587 * avg_g + 0.114 * avg_b


def composite_logo(image_path, logo_position, logo_color_mode, logo_size="small"):
    """Composite the real Junglück logo onto the generated image.
    Auto-detects whether to use dark or light logo based on background brightness."""
    if not HAS_PIL:
        print("  Warning: PIL not installed, skipping logo compositing")
        return

    logo_dark_path = os.path.join(PROJECT_ROOT, "branding", "logo_dark.png")
    logo_white_path = os.path.join(PROJECT_ROOT, "branding", "logo_white.png")

    if not os.path.exists(logo_dark_path):
        print(f"  Warning: Logo not found: {logo_dark_path}")
        return

    try:
        img = Image.open(image_path).convert("RGBA")
        img_w, img_h = img.size

        # Always use medium size minimum — logo should be clearly visible
        size_map = {"small": 0.22, "medium": 0.25, "large": 0.30}
        logo_scale = size_map.get(logo_size, 0.22)

        # Calculate preliminary position to measure background brightness
        margin_x = int(img_w * 0.04)
        margin_y = int(img_h * 0.03)

        # Temp logo size for position calculation
        logo_tmp = Image.open(logo_dark_path).convert("RGBA")
        logo_ratio = logo_tmp.height / logo_tmp.width
        new_logo_w = int(img_w * logo_scale)
        new_logo_h = int(new_logo_w * logo_ratio)

        pos_map = {
            "top_left": (margin_x, margin_y),
            "top_center": ((img_w - new_logo_w) // 2, margin_y),
            "top_right": (img_w - new_logo_w - margin_x, margin_y),
            "bottom_center": ((img_w - new_logo_w) // 2, img_h - new_logo_h - margin_y),
            "bottom_right": (img_w - new_logo_w - margin_x, img_h - new_logo_h - margin_y),
        }
        pos = pos_map.get(logo_position, pos_map["top_center"])

        # Auto-detect: measure brightness of the region where logo will be placed
        brightness = get_region_brightness(img, pos[0], pos[1], new_logo_w, new_logo_h)

        # Choose logo version based on actual background brightness
        # Bright background (>160) → dark logo, dark background (<160) → white logo
        if brightness > 160:
            chosen_logo_path = logo_dark_path
            chosen_mode = "dark"
        else:
            chosen_logo_path = logo_white_path if os.path.exists(logo_white_path) else logo_dark_path
            chosen_mode = "light" if os.path.exists(logo_white_path) else "dark"

        logo = Image.open(chosen_logo_path).convert("RGBA")
        logo = logo.resize((new_logo_w, new_logo_h), Image.LANCZOS)

        # Composite
        img.paste(logo, pos, logo)

        # Save back (as RGB for JPEG, RGBA for PNG)
        if image_path.lower().endswith(".jpg") or image_path.lower().endswith(".jpeg"):
            img = img.convert("RGB")
        img.save(image_path)
        print(f"  Logo composited ({chosen_mode} on brightness={brightness:.0f}, {logo_position})")

    except Exception as e:
        print(f"  Warning: Logo compositing failed: {e}")


def save_ad(image_data, mime_type, output_dir, ad_prompt_json, index):
    """Save generated ad image and metadata."""
    meta = ad_prompt_json["meta"]
    angle_slug = meta["angle"].lower().replace("/", "_").replace(" ", "_")
    sub_angle_slug = meta["sub_angle"].lower().replace(" ", "_").replace("/", "_")
    variant = meta["variant"]
    fmt = meta["format"].replace(":", "x")

    ext_map = {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp"}
    ext = ext_map.get(mime_type, "png")

    filename = f"{index:03d}_{angle_slug}_{sub_angle_slug}_v{variant}_{fmt}.{ext}"
    filepath = os.path.join(output_dir, filename)

    image_bytes = base64.standard_b64decode(image_data)
    with open(filepath, "wb") as f:
        f.write(image_bytes)

    print(f"  Saved: {filename} ({len(image_bytes) / 1024:.0f} KB)")

    # Composite real logo if specified
    brand_elements = ad_prompt_json.get("brand_elements", {})
    logo_config = brand_elements.get("logo", {})
    if logo_config.get("visible"):
        composite_logo(
            filepath,
            logo_config.get("position", "top_center"),
            logo_config.get("color_mode", "dark"),
            logo_config.get("size", "small")
        )

    return filename


def update_board_data(board_data_path, ads):
    """Thread-safe write of ads list to board_data.json."""
    with board_lock:
        with open(board_data_path, "w") as f:
            json.dump({"ads": ads}, f, indent=2, ensure_ascii=False)


def generate_single_ad(api_key, prompt_data, index, output_dir, rel_dir, board_data_path, existing_ads, placeholder_idx):
    """Generate a single ad. Called from thread pool."""
    ad_prompt = prompt_data["prompt"]
    product_image = prompt_data["product_image"]
    meta = ad_prompt["meta"]
    hook_text = next(
        (t["content"] for t in ad_prompt.get("text_overlays", []) if t["role"] == "headline"),
        ""
    )

    print(f"\n[{index}] {meta['angle']} > {meta['sub_angle']} (v{meta['variant']}, {meta['format']})")

    # Resolve product image path
    img_path = os.path.join(PROJECT_ROOT, product_image)
    if not os.path.exists(img_path):
        print(f"  Error: Product image not found: {img_path}")
        with board_lock:
            existing_ads[placeholder_idx]["status"] = "failed"
        update_board_data(board_data_path, existing_ads)
        return None

    # Build payload
    payload = build_gemini_prompt(ad_prompt, img_path)

    # Call Gemini
    image_data, mime_type = call_gemini(api_key, payload)

    if image_data:
        filename = save_ad(image_data, mime_type, output_dir, ad_prompt, index)

        # Update placeholder → done, with version tracking
        with board_lock:
            ad = existing_ads[placeholder_idx]
            ad["status"] = "done"
            ad["filename"] = f"{rel_dir}/{filename}"
            ad["image_path"] = f"{rel_dir}/{filename}"

            # Initialize or append to versions array
            if "versions" not in ad:
                ad["versions"] = []
            ad["versions"].append({
                "filename": f"{rel_dir}/{filename}",
                "image_path": f"{rel_dir}/{filename}"
            })
            ad["active_version"] = len(ad["versions"]) - 1

        update_board_data(board_data_path, existing_ads)
        print(f"  Board updated: image live!")

        return {
            "index": index,
            "filename": filename,
            "angle": meta["angle"],
            "sub_angle": meta["sub_angle"],
            "variant": meta["variant"],
            "format": meta["format"],
            "product_image": product_image,
            "hook_text": hook_text
        }
    else:
        print(f"  FAILED: No image generated")
        with board_lock:
            existing_ads[placeholder_idx]["status"] = "failed"
        update_board_data(board_data_path, existing_ads)
        return None


def generate_ads(api_key, prompts, output_dir):
    """Generate all ads in parallel."""
    board_data_path = os.path.join(PROJECT_ROOT, "creatives", "board_data.json")
    rel_dir = os.path.relpath(output_dir, os.path.join(PROJECT_ROOT, "creatives"))

    # Load existing board ads
    existing_ads = []
    if os.path.exists(board_data_path):
        with open(board_data_path) as f:
            existing_ads = json.load(f).get("ads", [])

    # Step 1: Create ALL placeholders at once
    placeholder_indices = []
    for i, prompt_data in enumerate(prompts, 1):
        ad_prompt = prompt_data["prompt"]
        meta = ad_prompt["meta"]
        hook_text = next(
            (t["content"] for t in ad_prompt.get("text_overlays", []) if t["role"] == "headline"),
            ""
        )

        placeholder = {
            "index": len(existing_ads) + 1,
            "filename": "",
            "image_path": "",
            "angle": meta["angle"],
            "sub_angle": meta["sub_angle"],
            "variant": meta["variant"],
            "format": meta["format"],
            "hook_text": hook_text,
            "product_image": prompt_data["product_image"],
            "status": "generating",
            "versions": [],
            "active_version": 0
        }
        placeholder_idx = len(existing_ads)
        existing_ads.append(placeholder)
        placeholder_indices.append(placeholder_idx)

    # Write all placeholders at once — board shows them immediately
    update_board_data(board_data_path, existing_ads)
    print(f"All {len(prompts)} placeholders added to board")

    # Step 2: Generate all ads in parallel
    manifest = {
        "generated_at": datetime.now().isoformat(),
        "total_prompts": len(prompts),
        "successful": 0,
        "failed": 0,
        "ads": []
    }

    # Run all ads in parallel — Gemini handles rate limits via 429 retries
    max_workers = min(5, len(prompts))

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, (prompt_data, ph_idx) in enumerate(zip(prompts, placeholder_indices), 1):
            future = executor.submit(
                generate_single_ad,
                api_key, prompt_data, i, output_dir, rel_dir,
                board_data_path, existing_ads, ph_idx
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            try:
                result = future.result()
                if result:
                    manifest["successful"] += 1
                    manifest["ads"].append(result)
                else:
                    manifest["failed"] += 1
            except Exception as e:
                print(f"  Error in ad {idx}: {e}")
                manifest["failed"] += 1

    return manifest


def main():
    parser = argparse.ArgumentParser(description="Creative Producer — Static Ad Generation")
    parser.add_argument("--prompts-file", required=True, help="Path to JSON file with ad prompts")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: creatives/<timestamp>)")
    args = parser.parse_args()

    # Prevent concurrent runs and clean up stale locks
    acquire_process_lock()
    atexit.register(release_process_lock)
    atexit.register(cleanup_stuck_generating)

    # Also clean up on SIGTERM/SIGINT
    def handle_signal(sig, frame):
        print(f"\nInterrupted (signal {sig}) — cleaning up...")
        cleanup_stuck_generating()
        release_process_lock()
        sys.exit(1)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    api_key = load_config()

    # Load prompts
    with open(args.prompts_file) as f:
        prompts = json.load(f)

    if not isinstance(prompts, list):
        prompts = [prompts]

    print(f"Loaded {len(prompts)} ad prompts")

    # Setup output directory
    if args.output_dir:
        output_dir = os.path.join(PROJECT_ROOT, args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(PROJECT_ROOT, "creatives", timestamp)

    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Generate
    manifest = generate_ads(api_key, prompts, output_dir)

    # Save manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    board_data_path = os.path.join(PROJECT_ROOT, "creatives", "board_data.json")
    print(f"\nBoard live-updated during generation: {board_data_path}")

    # Summary
    print(f"\n{'='*60}")
    print(f"DONE: {manifest['successful']} generated, {manifest['failed']} failed")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")
    print(f"Board: file://{board_data_path.replace(' ', '%20')}/../board.html")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
