# cad-extractor/main.py
# ---------------------------------------------------------------
# FastAPI service that:
#   • loads the eDOCr2 OCR models once on start-up
#   • exposes /extract  →  JSON   (old behaviour)
#   • exposes /extract?with_overlay=true  →  JSON + annotated PNG
# ---------------------------------------------------------------
import base64
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, List

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 

# ---- your repo -------------------------------------------------
from edocr2 import tools
from edocr2.ocr_it import ocr_drawing
from edocr2.keras_ocr.recognition import Recognizer
from edocr2.keras_ocr.detection import Detector
from postprocess import build_measurements          # <- gives DrawingResponse
from schemas import DrawingResponse, Measurement    # updated schema.py
# ---------------------------------------------------------------


app = FastAPI(
    title="CAD Measurement Extractor",
    description=("Extract dimensions, balloon numbers, types & sections "
                 "from PDF / raster mechanical drawings"),
    version="0.2.0"
)

# ------------------------------------------------------------------ #
#               ───  Model loading only once at start-up  ───         #
# ------------------------------------------------------------------ #
@app.on_event("startup")
def _load_models() -> None:
    """
    Heavy OCR components are initialised exactly once at process start
    so every request re-uses them.
    """
    global INFER_OPTS

    # — recognise geometric-tolerance symbols ---------------------------
    gdt_wts = "edocr2/models/recognizer_gdts.keras"
    recog_gdt = Recognizer(alphabet=tools.ocr_pipelines.read_alphabet(gdt_wts))
    recog_gdt.model.load_weights(gdt_wts)

    # — recognise numeric dimensions ------------------------------------
    dim_wts = "edocr2/models/recognizer_dimensions_2.keras"
    alphabet_dim = tools.ocr_pipelines.read_alphabet(dim_wts)
    recog_dim = Recognizer(alphabet=alphabet_dim)
    recog_dim.model.load_weights(dim_wts)

    # — detector ---------------------------------------------------------
    detector = Detector()                       # default CRAFT weights

    # — centralise all run-time kwargs so the handler can re-use -------
    INFER_OPTS = dict(
        binary_thres=127,
        language="eng",
        autoframe=False,
        frame_thres=0.95,
        recognizer_gdt=recog_gdt,
        GDT_thres=0.02,
        dimension_tuple=(detector, recog_dim, alphabet_dim),
        cluster_thres=20,
        max_char=15,
        max_img_size=2048,
        backg_save=False,
        output_path=".",
        save_mask=False,
        save_raw_output=False
    )


# ------------------------------------------------------------------ #
#                    ───  Balloon overlay helper  ───                 #
# ------------------------------------------------------------------ #
def _overlay_balloons(
    img_path: str | Path,
    measures: List[Measurement],
    radius: int = 18,
    colour: Tuple[int, int, int] = (0, 190, 0),
) -> str:
    """
    Paint solid coloured circles (balloons) at the centroid of every
    `Measurement.bbox` and write a PNG next to `img_path`.

    Returns
    -------
    out_path : str
        Path of the newly written PNG file.
    """
    im = cv2.imread(str(img_path))
    if im is None:
        raise RuntimeError(f"cv2 failed to read {img_path}")

    for m in measures:
        if not m.bbox:                       # safety
            continue
        # bbox is 4-point polygon -> centroid
        xs, ys = zip(*m.bbox)
        cx, cy = int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

        cv2.circle(im, (cx, cy), radius, colour, thickness=-1)
        cv2.putText(
            im,
            str(m.balloon_number),
            (cx - radius // 2, cy + radius // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA,
        )

    out_path = Path(img_path).with_stem(Path(img_path).stem + "_ballooned")
    cv2.imwrite(str(out_path), im)
    return str(out_path)


def _encode_base64(png_path: str | Path) -> str:
    with open(png_path, "rb") as fh:
        return "data:image/png;base64," + base64.b64encode(fh.read()).decode()


# ------------------------------------------------------------------ #
#                            End-point                                #
# ------------------------------------------------------------------ #
@app.post("/extract")
async def extract(
    file: UploadFile = File(...),
    with_overlay: bool = False,
):
    """
    Parameters
    ----------
    file : multipart/form-data
        PDF or raster image of a mechanical drawing.
    with_overlay : bool, optional
        If **true** and the upload is an image (PNG/JPEG),
        the response includes a base-64 PNG with green numbered
        balloons drawn on top of every detected dimension.
    """
    name_lower = file.filename.lower()
    if not (name_lower.endswith(".pdf")
            or name_lower.endswith((".png", ".jpg", ".jpeg"))):
        raise HTTPException(415, "Only PDF and raster images are supported")

    # ---- persist to disk so OCR can open it ----------------------------
    suffix = ".pdf" if name_lower.endswith(".pdf") else Path(name_lower).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        raw_dict, *_ = ocr_drawing(file_path=tmp_path, **INFER_OPTS)
        print(raw_dict)
        drawing = build_measurements(file.filename, raw_dict)
    except Exception as exc:
        raise HTTPException(500, f"OCR failed: {exc}") from exc
    finally:
        file.file.close()

    # ------------------------------------------------------------------
    # If the caller wants an overlay *and* provided a raster image,
    # paint the balloons and embed the resulting PNG.
    # ------------------------------------------------------------------
    if with_overlay and suffix.lower() in (".png", ".jpg", ".jpeg"):
        try:
            out_png = _overlay_balloons(tmp_path, drawing.measurements)
            encoded = _encode_base64(out_png)
        except Exception as exc:
            # Do not fail the request if overlaying bombs out; just log.
            print(f"[overlay] Failed: {exc}")
            encoded = None
    else:
        encoded = None

    # response: either plain old JSON or JSON + image -------------------
    if encoded:
        return {"drawing": drawing, "annotated_png": encoded}
    return drawing

# ------------------------------------------------------------------
#               ───  Serve the front-end assets  ───
# ------------------------------------------------------------------
from fastapi.staticfiles import StaticFiles

app.mount("/static", StaticFiles(directory="static", html=True), name="static")  # 2️⃣ NEW

@app.get("/")                                # 3️⃣ NEW
async def root():
    """Redirect / to the SPA entry-point."""
    return FileResponse("static/index.html")


# ------------------------------------------------------------------ #
#                           Launcher (dev)                            #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

