import os
import json
import re
import base64
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

# API_KEY = os.getenv("llm_key") or os.getenv("GOOGLE_API_KEY")

API_KEY = os.environ.get("llm_key")

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=API_KEY,
    timeout=60,
    max_retries=2,
)

PROMPT = """Extract from these medical documents: Patient name, Diagnosis, Bill amount,
Admit date, Discharge date. Return ONLY valid JSON with keys:
"Patient_name", "Diagnosis", "Bill_amount", "admit_date", "discharge_date".
Set missing fields to null. If no readable text, return exactly: "invalid" """


def _resize_image(raw: bytes, max_dim: int = 1024) -> bytes:
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(raw))
    w, h = img.size
    if w > max_dim or h > max_dim:
        ratio = max_dim / max(w, h)
        img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format=img.format or "JPEG", quality=85)
    return buf.getvalue()


def _pdf_text(file_bytes: bytes) -> str:
    import fitz
    try:
        doc = fitz.open(stream=file_bytes, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
        return text.strip()
    except Exception:
        return ""


def _extract_text(resp) -> str:
    meta = resp.response_metadata if hasattr(resp, "response_metadata") else {}
    c = resp.content
    if isinstance(c, str):
        if c.strip():
            return c
    elif isinstance(c, list):
        parts = []
        for p in c:
            if isinstance(p, dict):
                txt = p.get("text") or p.get("content") or ""
                if txt:
                    parts.append(txt)
            elif hasattr(p, "text") and p.text:
                parts.append(p.text)
        if parts:
            return "".join(parts)
    return f"__NO_TEXT__ meta={meta}"


def _call_gemini(text_prompt: str, image_b64: str = None, image_mime: str = None) -> str:
    parts = [{"type": "text", "text": text_prompt}]
    if image_b64:
        parts.append({"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{image_b64}"}})
    msg = HumanMessage(content=parts)
    return _extract_text(llm.invoke([msg]))


def _img_to_b64(raw: bytes) -> str:
    return base64.b64encode(raw).decode("utf-8")


def scan_documents(prescription, bills, lab_reports):
    all_text = []
    all_images = []

    for doc_name, doc_file in [
        ("Prescription", prescription),
        ("Bills", bills),
        ("Lab Reports", lab_reports),
    ]:
        if doc_file is None:
            continue

        raw = doc_file.getvalue()
        ext = doc_file.name.split(".")[-1].lower()

        if ext == "pdf":
            text = _pdf_text(raw)
            if text:
                all_text.append(f"--- {doc_name} ---\n{text}")
            else:
                import fitz
                try:
                    doc = fitz.open(stream=raw, filetype="pdf")
                    pix = doc[0].get_pixmap(dpi=150)
                    img_bytes = pix.tobytes("png")
                    doc.close()
                    all_images.append((f"--- {doc_name} ---", img_bytes, "image/png"))
                except Exception as e:
                    return {"error": f"Could not read {doc_name}: {str(e)}"}

        elif ext in ("jpg", "jpeg"):
            try:
                raw = _resize_image(raw)
            except Exception:
                pass
            all_images.append((f"--- {doc_name} ---", raw, "image/jpeg"))
        else:
            return {"error": f"Unsupported file format: {ext}"}

    if not all_text and not all_images:
        return {"error": "No document content could be extracted."}

    try:
        full_prompt = PROMPT
        if all_text:
            full_prompt += "\n\n" + "\n\n".join(all_text)

        if all_images:
            parts = [{"type": "text", "text": full_prompt}]
            for label, img_bytes, mime in all_images:
                b64 = _img_to_b64(img_bytes)
                parts.append({"type": "text", "text": label})
                parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}})
            msg = HumanMessage(content=parts)
            combined = _extract_text(llm.invoke([msg]))
        else:
            msg = HumanMessage(content=[{"type": "text", "text": full_prompt}])
            combined = _extract_text(llm.invoke([msg]))

        clean = combined.strip().strip('"').strip("'").strip("`")
        if clean.lower() == "invalid":
            return {"error": "Gemini could not read the documents. They may be blank or unclear."}

        json_match = re.search(r'\{\s*"[^"]+":\s*.*\}', clean, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
            except json.JSONDecodeError as e:
                return {"error": f"Gemini returned malformed JSON: {e}. Raw: {json_match.group()[:200]}"}
            return {
                "Patient_name": data.get("Patient_name"),
                "Diagnosis": data.get("Diagnosis"),
                "Bill_amount": data.get("Bill_amount"),
                "admit_date": data.get("admit_date"),
                "discharge_date": data.get("discharge_date"),
            }

        return {"error": f"Gemini returned unexpected response: {clean[:200]}"}

    except Exception as e:
        err_str = str(e)
        if "10053" in err_str:
            return {"error": "Connection aborted — the payload may be too large. Try uploading smaller files or lower resolution images."}
        return {"error": f"Error: {err_str}"}
