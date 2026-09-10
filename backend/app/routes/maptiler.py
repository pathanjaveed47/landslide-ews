import urllib.request
import urllib.error
from fastapi import APIRouter, HTTPException, Response

router = APIRouter(prefix="/maptiler", tags=["MapTiler"])

MAPTILER_KEY = "vHlJWoPioEAxFO8dswuF"

# In-memory tile cache to avoid repeat network requests
TILE_CACHE = {}
MAX_CACHE_SIZE = 5000

@router.get("/tiles/{style}/{z}/{x}/{y}.{ext}")
def get_maptiler_tile(style: str, z: int, x: int, y: int, ext: str):
    cache_key = f"{style}/{z}/{x}/{y}.{ext}"
    if cache_key in TILE_CACHE:
        content_type = "image/jpeg" if ext in ["jpg", "jpeg"] else "image/png"
        return Response(
            content=TILE_CACHE[cache_key],
            media_type=content_type,
            headers={"Cache-Control": "public, max-age=86400"}
        )

    # Validate style
    allowed_styles = ["hybrid", "outdoor-v2", "streets-v2-dark", "topo-v2", "satellite", "streets-v2", "backdrop-dark"]
    if style not in allowed_styles:
        style = "hybrid"

    ext_norm = "jpg" if ext in ["jpg", "jpeg"] else "png"
    target_url = f"https://api.maptiler.com/maps/{style}/256/{z}/{x}/{y}.{ext_norm}?key={MAPTILER_KEY}"

    headers = {
        "Referer": "https://cloud.maptiler.com/",
        "User-Agent": "LandSlideSentinel/1.0"
    }
    req = urllib.request.Request(target_url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", f"image/{ext_norm}")
            
            # Simple LRU-style cache prune
            if len(TILE_CACHE) > MAX_CACHE_SIZE:
                TILE_CACHE.clear()
            TILE_CACHE[cache_key] = data

            return Response(
                content=data,
                media_type=content_type,
                headers={"Cache-Control": "public, max-age=86400"}
            )
    except urllib.error.HTTPError as e:
        raise HTTPException(status_code=e.code, detail=f"MapTiler upstream error: {e.reason}")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"MapTiler tile gateway error: {str(e)}")
