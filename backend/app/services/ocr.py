import json
import re

from PIL import Image, ImageEnhance, ImageFilter

from app.config import settings

PROMPT = """You are a receipt OCR parser. Your job is to read EXACTLY what is printed on the receipt — do not guess, invent, or fill in items that are not clearly visible.

Return ONLY valid JSON with no markdown fences, no explanation — just the raw JSON object.

Use exactly this shape:
{
  "merchant_name": "string or null",
  "total": number,
  "subtotal": number,
  "tax_total": number,
  "discount_total": number,
  "items": [
    {
      "name": "string",
      "quantity": number,
      "unit_price": number,
      "discount_amount": number,
      "total_price": number,
      "is_taxable": boolean,
      "tax_rate": number,
      "is_tip_line": false
    }
  ]
}

STRICT RULES — follow exactly:

1. ITEMS: Only include items that are explicitly printed on the receipt as purchased products. Do NOT include tax lines, subtotal lines, total lines, or store header info as items.

2. PRICES: Read the price exactly as printed. "unit_price" is the price per single unit before any discount or tax. "total_price" = (unit_price × quantity) - discount_amount. All numbers are plain decimals, no $ signs.

3. QUANTITY: Default to 1 unless the receipt clearly shows a quantity (e.g. "2 @" or "3 x").

4. TAX: Do NOT add a separate tax line item. Instead distribute tax onto each item:
   - If an item has a "T" or "*" marker it is taxable
   - tax_rate = (tax_total / sum_of_taxable_item_prices) × 100
   - Non-taxable items get is_taxable: false, tax_rate: 0

5. DISCOUNTS: If a discount line appears directly below an item (e.g. "INSTANT SAVINGS -$1.50"), set that item's discount_amount to 1.50. If it's a general discount not tied to any item, distribute proportionally. Items with no discount get discount_amount: 0.

6. TIP: If a tip is shown, include it as one item with is_tip_line: true, is_taxable: false, tax_rate: 0, discount_amount: 0.

7. TOTALS: "total" is the final charged amount. "subtotal" is pre-tax pre-discount sum. "tax_total" and "discount_total" are as printed (0 if not shown).

8. ACCURACY: If any value is unclear or unreadable, use 0 rather than guessing.
"""


def _preprocess_image(image_path: str) -> Image.Image:
    """Sharpen and enhance contrast to improve OCR accuracy."""
    img = Image.open(image_path).convert("RGB")

    # Upscale small images
    w, h = img.size
    if max(w, h) < 1500:
        scale = 1500 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    # Sharpen + boost contrast
    img = img.filter(ImageFilter.SHARPEN)
    img = ImageEnhance.Contrast(img).enhance(1.4)
    img = ImageEnhance.Sharpness(img).enhance(2.0)

    return img


def parse_receipt(image_path: str) -> dict:
    import google.generativeai as genai

    genai.configure(api_key=settings.GEMINI_API_KEY)

    img = _preprocess_image(image_path)

    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content([img, PROMPT])

    raw = response.text.strip()
    print(f"[OCR RAW]\n{raw}\n")  # log so we can debug

    # Strip markdown fences if model added them
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)

    data = json.loads(raw)

    # Safety: if the AI still outputs a tax line item, strip it and redistribute
    cleaned_items = []
    tax_line_total = 0.0
    for item in data.get("items", []):
        if item.get("is_tax_line"):
            tax_line_total += float(item.get("total_price", 0))
        else:
            cleaned_items.append(item)

    if tax_line_total > 0 and cleaned_items:
        taxable_subtotal = sum(
            float(i.get("total_price", 0))
            for i in cleaned_items
            if not i.get("is_tip_line") and i.get("is_taxable", True)
        )
        if taxable_subtotal > 0:
            rate = (tax_line_total / taxable_subtotal) * 100
            for item in cleaned_items:
                if not item.get("is_tip_line"):
                    item["is_taxable"] = True
                    item["tax_rate"] = round(rate, 4)

    data["items"] = cleaned_items
    return data
