"""
Asset generator for AlgoLens:
Generates favicon.svg, favicon.ico (16, 32, 48), favicon-16x16.png, favicon-32x32.png,
apple-touch-icon.png (180x180), icon-192.png, icon-512.png, icon-512-maskable.png,
site.webmanifest, and og-image.png (1200x630).
"""
import io
import json
import os
import cairosvg
from PIL import Image, ImageDraw, ImageFont

PUBLIC_DIR = "/home/nitin/algolens-ml/algo-lens-viz/public"

# 1. Base SVG
SVG_CONTENT = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="brandGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3b82f6" />
      <stop offset="100%" stop-color="#a855f7" />
    </linearGradient>
  </defs>
  <rect width="512" height="512" rx="112" fill="url(#brandGrad)" />
  <g transform="translate(96, 96) scale(13.3333)" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="m10.065 12.493-6.18 1.318a.934.934 0 0 1-1.108-.702l-.537-2.15a1.07 1.07 0 0 1 .691-1.265l13.504-4.44" />
    <path d="m13.56 11.747 4.332-.924" />
    <path d="m16 21-3.105-6.21" />
    <path d="M16.485 5.94a2 2 0 0 1 1.455-2.425l1.09-.272a1 1 0 0 1 1.212.727l1.515 6.06a1 1 0 0 1-.727 1.213l-1.09.272a2 2 0 0 1-2.425-1.455z" />
    <path d="m6.158 8.633 1.114 4.456" />
    <path d="m8 21 3.105-6.21" />
    <circle cx="12" cy="13" r="2" />
  </g>
</svg>"""

# Maskable SVG with safe padding (full bleed background, scaled icon within 80% safe zone)
MASKABLE_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="brandGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#3b82f6" />
      <stop offset="100%" stop-color="#a855f7" />
    </linearGradient>
  </defs>
  <rect width="512" height="512" fill="url(#brandGrad)" />
  <g transform="translate(128, 128) scale(10.6666)" stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" fill="none">
    <path d="m10.065 12.493-6.18 1.318a.934.934 0 0 1-1.108-.702l-.537-2.15a1.07 1.07 0 0 1 .691-1.265l13.504-4.44" />
    <path d="m13.56 11.747 4.332-.924" />
    <path d="m16 21-3.105-6.21" />
    <path d="M16.485 5.94a2 2 0 0 1 1.455-2.425l1.09-.272a1 1 0 0 1 1.212.727l1.515 6.06a1 1 0 0 1-.727 1.213l-1.09.272a2 2 0 0 1-2.425-1.455z" />
    <path d="m6.158 8.633 1.114 4.456" />
    <path d="m8 21 3.105-6.21" />
    <circle cx="12" cy="13" r="2" />
  </g>
</svg>"""

def render_png(svg_str, width, height):
    png_bytes = cairosvg.svg2png(bytestring=svg_str.encode("utf-8"), output_width=width, output_height=height)
    return Image.open(io.BytesIO(png_bytes))

def main():
    os.makedirs(PUBLIC_DIR, exist_ok=True)
    
    # Save favicon.svg
    with open(os.path.join(PUBLIC_DIR, "favicon.svg"), "w") as f:
        f.write(SVG_CONTENT)
    print("Created favicon.svg")

    # Generate PNGs
    img_16 = render_png(SVG_CONTENT, 16, 16)
    img_16.save(os.path.join(PUBLIC_DIR, "favicon-16x16.png"))
    print("Created favicon-16x16.png")

    img_32 = render_png(SVG_CONTENT, 32, 32)
    img_32.save(os.path.join(PUBLIC_DIR, "favicon-32x32.png"))
    print("Created favicon-32x32.png")

    img_48 = render_png(SVG_CONTENT, 48, 48)
    
    # Save favicon.ico with 16, 32, 48 layers
    ico_path = os.path.join(PUBLIC_DIR, "favicon.ico")
    img_48.save(ico_path, format="ICO", sizes=[(16, 16), (32, 32), (48, 48)])
    print("Created multi-resolution favicon.ico")

    img_180 = render_png(SVG_CONTENT, 180, 180)
    img_180.save(os.path.join(PUBLIC_DIR, "apple-touch-icon.png"))
    print("Created apple-touch-icon.png")

    img_192 = render_png(SVG_CONTENT, 192, 192)
    img_192.save(os.path.join(PUBLIC_DIR, "icon-192.png"))
    print("Created icon-192.png")

    img_512 = render_png(SVG_CONTENT, 512, 512)
    img_512.save(os.path.join(PUBLIC_DIR, "icon-512.png"))
    print("Created icon-512.png")

    img_maskable = render_png(MASKABLE_SVG, 512, 512)
    img_maskable.save(os.path.join(PUBLIC_DIR, "icon-512-maskable.png"))
    print("Created icon-512-maskable.png")

    # Generate site.webmanifest
    manifest = {
        "name": "AlgoLens — Algorithm Visualizer & Complexity Analyzer",
        "short_name": "AlgoLens",
        "description": "Interactive algorithm visualizer and AST-powered Big-O complexity analyzer.",
        "start_url": "/",
        "display": "standalone",
        "background_color": "#0c0f1d",
        "theme_color": "#0c0f1d",
        "icons": [
            {
                "src": "/icon-192.png",
                "sizes": "192x192",
                "type": "image/png"
            },
            {
                "src": "/icon-512.png",
                "sizes": "512x512",
                "type": "image/png"
            },
            {
                "src": "/icon-512-maskable.png",
                "sizes": "512x512",
                "type": "image/png",
                "purpose": "maskable"
            }
        ]
    }
    with open(os.path.join(PUBLIC_DIR, "site.webmanifest"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("Created site.webmanifest")

    # Generate 1200x630 OpenGraph card
    card_width, card_height = 1200, 630
    og = Image.new("RGBA", (card_width, card_height), (12, 15, 29, 255)) # #0c0f1d
    draw = ImageDraw.Draw(og)

    # Ambient radial gradient glows
    # Top-left blue glow
    for r in range(400, 0, -10):
        alpha = int(45 * (1 - r / 400))
        draw.ellipse([-100 - r, -100 - r, 300 + r, 300 + r], fill=(59, 130, 246, alpha))
    # Bottom-right purple glow
    for r in range(450, 0, -10):
        alpha = int(45 * (1 - r / 450))
        draw.ellipse([800 - r, 250 - r, 1300 + r, 750 + r], fill=(168, 85, 247, alpha))

    # Add telescope logo icon in center-left (e.g. 180x180 at x=140, y=225)
    logo_img = img_180.convert("RGBA")
    og.paste(logo_img, (140, 225), logo_img)

    # Render Typography using Pillow
    # Try finding system fonts (DejaVuSans, liberation, etc.)
    font_title = None
    font_sub = None
    font_tag = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]:
        if os.path.exists(font_path):
            font_title = ImageFont.truetype(font_path, 84)
            break
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
    ]:
        if os.path.exists(font_path):
            font_sub = ImageFont.truetype(font_path, 34)
            font_tag = ImageFont.truetype(font_path, 22)
            break

    if not font_title:
        font_title = ImageFont.load_default()
        font_sub = ImageFont.load_default()
        font_tag = ImageFont.load_default()

    # Title: AlgoLens
    draw.text((360, 215), "AlgoLens", fill=(255, 255, 255, 255), font=font_title)

    # Subtitle: Algorithm Visualizer & Complexity Analyzer
    draw.text((365, 320), "Algorithm Visualizer & Complexity Analyzer", fill=(203, 213, 225, 255), font=font_sub)

    # Description badge
    draw.rounded_rectangle([365, 385, 890, 428], radius=8, fill=(30, 41, 59, 200), outline=(51, 65, 85, 255))
    draw.text((385, 395), "ML-Powered Classification • AST Feature Analysis • Big-O Bounds", fill=(148, 163, 184, 255), font=font_tag)

    og_path = os.path.join(PUBLIC_DIR, "og-image.png")
    og.convert("RGB").save(og_path, quality=95)
    print("Created og-image.png (1200x630)")

if __name__ == "__main__":
    main()
