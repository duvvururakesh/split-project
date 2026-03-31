import io
import os
import uuid
from datetime import date as date_type
from decimal import Decimal

import pillow_heif
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from PIL import Image
from sqlalchemy.orm import Session

pillow_heif.register_heif_opener()

from app.database import get_db
from app.middleware.auth import get_current_user
from app.models.expense import Expense, ExpenseParticipant
from app.models.receipt import Receipt, ReceiptItem, ReceiptItemAssignment
from app.models.user import User
from app.schemas.expenses import ExpenseResponse
from app.schemas.receipts import (
    AssignItemRequest,
    CreateExpenseFromReceiptRequest,
    ItemAssignmentResponse,
    ReceiptItemResponse,
    ReceiptResponse,
    UpdateReceiptItemRequest,
)
router = APIRouter(prefix="/receipts", tags=["receipts"])

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "heic", "heif", "pdf"}


def _parse_uuid(value: str, label: str = "id") -> uuid.UUID:
    """Convert a string to UUID, raising 422 on invalid format."""
    try:
        return uuid.UUID(value)
    except (ValueError, AttributeError):
        raise HTTPException(status_code=422, detail=f"Invalid {label}: {value}")


def _receipt_to_response(receipt: Receipt) -> ReceiptResponse:
    return ReceiptResponse(
        id=str(receipt.id),
        ocr_status=receipt.ocr_status,
        merchant_name=receipt.merchant_name,
        ocr_total=float(receipt.ocr_total) if receipt.ocr_total else None,
        image_url=f"/uploads/{os.path.basename(receipt.image_path)}",
        items=[
            ReceiptItemResponse(
                id=str(item.id),
                name=item.name,
                quantity=float(item.quantity),
                unit_price=float(item.unit_price),
                total_price=float(item.total_price),
                display_order=item.display_order,
                is_tax_line=item.is_tax_line,
                is_tip_line=item.is_tip_line,
                is_taxable=bool(item.is_taxable),
                tax_rate=float(item.tax_rate or 0),
                discount_amount=float(item.discount_amount or 0),
                assignments=[
                    ItemAssignmentResponse(
                        user_id=str(a.user_id),
                        assigned_amount=float(a.assigned_amount),
                    )
                    for a in item.assignments
                ],
            )
            for item in sorted(receipt.items, key=lambda x: x.display_order)
        ],
    )


@router.post("/upload", response_model=ReceiptResponse, status_code=201)
async def upload_receipt(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    ext = (file.filename or "").split(".")[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Only JPG, PNG, and WEBP images are allowed")

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    # Convert HEIC/HEIF to JPEG so Vision AI can read it
    if ext in {"heic", "heif"}:
        try:
            img = Image.open(io.BytesIO(contents))
            buf = io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=92)
            contents = buf.getvalue()
            ext = "jpg"
        except Exception:
            raise HTTPException(status_code=400, detail="Could not convert HEIC image — try exporting as JPEG")

    # Convert PDF first page to JPEG so Vision AI can read it
    if ext == "pdf":
        try:
            import fitz  # pymupdf
            pdf = fitz.open(stream=contents, filetype="pdf")
            if pdf.page_count == 0:
                raise HTTPException(status_code=400, detail="PDF has no pages")
            page = pdf[0]
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom → ~144 dpi, sharp enough for OCR
            pix = page.get_pixmap(matrix=mat, alpha=False)
            buf = io.BytesIO(pix.tobytes("jpeg", jpg_quality=92))
            contents = buf.getvalue()
            ext = "jpg"
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Could not read PDF — {e}")

    filename = f"{uuid.uuid4()}.{ext}"
    file_path = os.path.join(UPLOAD_DIR, filename)

    with open(file_path, "wb") as f:
        f.write(contents)

    receipt = Receipt(uploader_id=current_user.id, image_path=file_path, ocr_status="pending")
    db.add(receipt)
    db.commit()
    db.refresh(receipt)
    return _receipt_to_response(receipt)


# NOTE: /create-expense must be declared BEFORE /{receipt_id} routes to prevent
# FastAPI from greedily matching "create-expense" as a receipt_id path param.
@router.post("/create-expense", response_model=ExpenseResponse, status_code=201)
def create_expense_from_receipt(
    body: CreateExpenseFromReceiptRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from app.routes.expenses import _expense_to_response

    receipt_uuid = _parse_uuid(body.receipt_id, "receipt_id")
    receipt = db.query(Receipt).filter(Receipt.id == receipt_uuid).first()
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")

    # Compute per-person totals from assignments
    person_totals: dict[str, Decimal] = {}
    for assignment in body.assignments:
        item_uuid = _parse_uuid(assignment.receipt_item_id, "receipt_item_id")
        item = db.query(ReceiptItem).filter(
            ReceiptItem.id == item_uuid,
            ReceiptItem.receipt_id == receipt_uuid,
        ).first()
        if not item:
            continue
        share = item.total_price / len(assignment.user_ids)
        for uid in assignment.user_ids:
            person_totals[uid] = person_totals.get(uid, Decimal("0")) + share

    if not person_totals:
        raise HTTPException(status_code=400, detail="No valid item assignments found")

    total = sum(person_totals.values())

    expense = Expense(
        group_id=body.group_id,
        receipt_id=receipt.id,
        paid_by=body.paid_by,
        description=body.description,
        total_amount=total,
        split_method="itemized",
        date=date_type.today(),
        created_by=current_user.id,
    )
    db.add(expense)
    db.flush()

    for uid, amount in person_totals.items():
        db.add(ExpenseParticipant(expense_id=expense.id, user_id=uid, owed_amount=amount))

    # Save assignments for audit trail
    for assignment in body.assignments:
        item_uuid = _parse_uuid(assignment.receipt_item_id, "receipt_item_id")
        item = db.query(ReceiptItem).filter(ReceiptItem.id == item_uuid).first()
        if not item:
            continue
        share = item.total_price / len(assignment.user_ids)
        for uid in assignment.user_ids:
            db.add(ReceiptItemAssignment(receipt_item_id=item.id, user_id=uid, assigned_amount=share))

    db.commit()
    db.refresh(expense)
    return _expense_to_response(expense)


@router.post("/{receipt_id}/scan", response_model=ReceiptResponse)
def scan_receipt(
    receipt_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    receipt_uuid = _parse_uuid(receipt_id)
    receipt = db.query(Receipt).filter(Receipt.id == receipt_uuid).first()
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")

    receipt.ocr_status = "processing"
    db.commit()

    try:
        from app.services.ocr import parse_receipt
        result = parse_receipt(receipt.image_path)

        receipt.merchant_name = result.get("merchant_name")
        if result.get("total"):
            receipt.ocr_total = Decimal(str(result["total"]))

        for i, item in enumerate(result.get("items", [])):
            db.add(ReceiptItem(
                receipt_id=receipt.id,
                name=item["name"],
                quantity=Decimal(str(item.get("quantity", 1))),
                unit_price=Decimal(str(item["unit_price"])),
                discount_amount=Decimal(str(item.get("discount_amount", 0))),
                total_price=Decimal(str(item["total_price"])),
                display_order=i,
                is_tax_line=item.get("is_tax_line", False),
                is_tip_line=item.get("is_tip_line", False),
                is_taxable=item.get("is_taxable", False),
                tax_rate=Decimal(str(item.get("tax_rate", 0))),
            ))

        receipt.ocr_status = "completed"
        db.commit()
        db.refresh(receipt)
    except Exception as e:
        receipt.ocr_status = "failed"
        db.commit()
        err = str(e)
        if "api_key_invalid" in err or "API_KEY_INVALID" in err or "invalid api key" in err.lower():
            detail = "OCR unavailable: Invalid Gemini API key. Check GEMINI_API_KEY in .env."
        elif "quota" in err.lower() or "resource_exhausted" in err.lower():
            detail = "OCR unavailable: Gemini API quota exceeded. Try again later."
        else:
            detail = f"OCR failed — {err}"
        raise HTTPException(status_code=500, detail=detail)

    return _receipt_to_response(receipt)


@router.get("/{receipt_id}", response_model=ReceiptResponse)
def get_receipt(
    receipt_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    receipt_uuid = _parse_uuid(receipt_id)
    receipt = db.query(Receipt).filter(Receipt.id == receipt_uuid).first()
    if not receipt:
        raise HTTPException(status_code=404, detail="Receipt not found")
    return _receipt_to_response(receipt)


@router.patch("/{receipt_id}/items/{item_id}", response_model=ReceiptItemResponse)
def update_item(
    receipt_id: str,
    item_id: str,
    body: UpdateReceiptItemRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    receipt_uuid = _parse_uuid(receipt_id)
    item_uuid = _parse_uuid(item_id, "item_id")
    item = db.query(ReceiptItem).filter(
        ReceiptItem.id == item_uuid,
        ReceiptItem.receipt_id == receipt_uuid,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    if body.name is not None:
        item.name = body.name
    if body.quantity is not None:
        item.quantity = Decimal(str(body.quantity))
    if body.unit_price is not None:
        item.unit_price = Decimal(str(body.unit_price))
    if body.is_taxable is not None:
        item.is_taxable = body.is_taxable
    if body.tax_rate is not None:
        item.tax_rate = Decimal(str(body.tax_rate))
    if body.discount_amount is not None:
        item.discount_amount = Decimal(str(body.discount_amount))

    # If total_price explicitly provided, use it; otherwise recompute from parts
    if body.total_price is not None:
        item.total_price = Decimal(str(body.total_price))
    else:
        base = item.unit_price * item.quantity
        tax = base * (item.tax_rate / 100) if item.is_taxable else Decimal("0")
        discount = item.discount_amount or Decimal("0")
        item.total_price = base + tax - discount

    db.commit()
    db.refresh(item)
    return ReceiptItemResponse(
        id=str(item.id),
        name=item.name,
        quantity=float(item.quantity),
        unit_price=float(item.unit_price),
        total_price=float(item.total_price),
        display_order=item.display_order,
        is_tax_line=item.is_tax_line,
        is_tip_line=item.is_tip_line,
        is_taxable=bool(item.is_taxable),
        tax_rate=float(item.tax_rate or 0),
        discount_amount=float(item.discount_amount or 0),
        assignments=[
            ItemAssignmentResponse(user_id=str(a.user_id), assigned_amount=float(a.assigned_amount))
            for a in item.assignments
        ],
    )


@router.put("/{receipt_id}/items/{item_id}/assignments")
def update_item_assignments(
    receipt_id: str,
    item_id: str,
    body: AssignItemRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Replace all assignments for an item with new user_ids (split equally)."""
    receipt_uuid = _parse_uuid(receipt_id)
    item_uuid = _parse_uuid(item_id, "item_id")
    item = db.query(ReceiptItem).filter(
        ReceiptItem.id == item_uuid,
        ReceiptItem.receipt_id == receipt_uuid,
    ).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    # Delete existing assignments
    for a in item.assignments:
        db.delete(a)
    db.flush()

    # Create new assignments split equally
    if body.user_ids:
        share = item.total_price / len(body.user_ids)
        for uid in body.user_ids:
            user_uuid = _parse_uuid(uid, "user_id")
            db.add(ReceiptItemAssignment(
                receipt_item_id=item.id,
                user_id=user_uuid,
                assigned_amount=share,
            ))

    db.commit()
    db.refresh(item)
    return {"ok": True}
