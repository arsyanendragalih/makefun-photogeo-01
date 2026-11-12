#!/usr/bin/env python3
# photo-geolocator-01 entry: a vision agent that places 25 street-level photos.
# For each photo it calls claude-opus-5-5 (revision 2; the house doc confirms opus
# fits the $1 budget at <=1800 output tokens a photo) once through the metered
# gateway, asking it to reason from visible clues before giving a latitude and
# longitude, then posts every answer to the env at the end. Opus 5's published
# median geolocation error (181 km on Flickr street-level, no tools) is well under
# Sonnet 5's (384 km), so opus earns a meaningfully better score per photo.
#
# Deterministic given the model's answers: requests are a function of each photo's
# bytes and a fixed prompt, and answers are posted in photo order, so a replay makes
# the same calls and leaves the same state. A photo whose answer holds no usable
# coordinates is left unanswered (it counts the maximum error), and a photo whose
# model call the gateway refuses is left unanswered too (a refusal is final).
#
# Runs as the grader runs it: python3 -P -s /bot/main.py (Python 3.13, stdlib only).
import base64, json, os, re, sys, time, urllib.request

MODEL = "claude-opus-5-5"
MAX_TOKENS = 1200  # opus output is a one-line JSON; keeps worst-case budget well inside $1
PROMPT = (
    "Look carefully at this street-level photograph. There is no metadata of any kind.\n"
    "Think step by step about the specific geographic clues it shows before you answer:\n"
    "- Language and script on any signs, banners, shop fronts, billboards, or vehicles.\n"
    "- Road markings, lane direction, and whether traffic drives on the left or the right.\n"
    "- Architecture, building materials, rooflines, roofs, walls, and building styles.\n"
    "- Vegetation, dominant climate, terrain, elevation, and the season.\n"
    "- Landmarks, distinctive structures, or recognizable features such as mountains, coast, or desert.\n"
    "- Vehicle makes and models and the license-plate style, if any are visible.\n"
    "- Any text, logos, or place names you can read, however faint.\n"
    "Weigh those clues into a confident location: be precise about the likely country and "
    "region or city, and do not give a generic answer.\n"
    "Reply with exactly one line of JSON and nothing else — no prose, no markdown: "
    '{"lat": <latitude between -90 and 90>, "lon": <longitude between -180 and 180>}'
)


def need(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} is not set")
    return v


env_url = need("MAKEFUN_ENV_URL").rstrip("/")
llm_url = need("ANTHROPIC_BASE_URL").rstrip("/")
api_key = need("ANTHROPIC_API_KEY")

RETRY = (408, 429, 500, 502, 503, 504, 529)
MAX_ATTEMPTS = 3


def http_json(url: str, payload, headers, method: str):
    req = urllib.request.Request(url, data=payload, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=180) as r:
        return r.getcode(), r.read()


def ask(jpeg: bytes):
    body = json.dumps({
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": base64.b64encode(jpeg).decode()}},
                {"type": "text", "text": PROMPT},
            ],
        }],
    }).encode()
    hdrs = {"content-type": "application/json", "x-api-key": api_key, "anthropic-version": "2023-06-01"}
    for attempt in range(MAX_ATTEMPTS):
        try:
            code, raw = http_json(f"{llm_url}/v1/messages", body, hdrs, "POST")
        except urllib.error.HTTPError as e:
            body_txt = e.read().decode(errors="replace")
            if e.code in RETRY and attempt + 1 < MAX_ATTEMPTS:
                time.sleep(2 * (attempt + 1))
                continue
            sys.stderr.write(f"model refused HTTP {e.code}: {body_txt[:300]}\n")
            return None
        if code != 200:
            sys.stderr.write(f"model HTTP {code}\n")
            return None
        msg = json.loads(raw)
        return "".join(b.get("text", "") for b in msg.get("content", []) if b.get("type") == "text")


def is_lat(v):
    return isinstance(v, (int, float)) and -90 <= v <= 90


def is_lon(v):
    return isinstance(v, (int, float)) and -180 <= v <= 180


# Pull a bare signed decimal out of tokens that may carry a hemisphere letter/arrow.
_NUM = r"(-?\d+(?:\.\d+)?)\s*(N|S|E|W|°|º|north|south|east|west)?"


def coordinates(text: str):
    if not text:
        return None
    # First: any JSON object with lat/lon keys.
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            o = json.loads(m.group(0))
            lat = o.get("lat", o.get("latitude"))
            lon = o.get("lon", o.get("lng", o.get("longitude")))
            if is_lat(lat) and is_lon(lon):
                return float(lat), float(lon)
        except (ValueError, TypeError):
            pass
    # Second: an explicit lat,lon pair in a single JSON array.
    for m in re.finditer(r"\[[^\[\]]*\]", text):
        try:
            o = json.loads(m.group(0))
            if len(o) == 2 and is_lat(o[0]) and is_lon(o[1]):
                return float(o[0]), float(o[1])
        except (ValueError, TypeError):
            pass
    # Third: hemisphere-annotated lat/lon anywhere, e.g. "lat 34.05 N lon 118.24 W".
    latm = re.search(r"(-?\d+(?:\.\d+)?)\s*([NSnso]{1,})", text)
    lonm = re.search(r"(-?\d+(?:\.\d+)?)\s*([EWewo]{1,})", text)
    if latm and lonm:
        lat, lon = float(latm.group(1)), float(lonm.group(1))
        if latm.group(2).upper() in ("S", "SOUTH"):
            lat = -lat
        if lonm.group(2).upper() in ("W", "WEST"):
            lon = -lon
        if is_lat(lat) and is_lon(lon):
            return lat, lon
    # Fourth: "lat, lon" / "45.5 S, 73.7 W" adjacent pairs.
    pairs = re.findall(_NUM + r"\s*[,;\s]+" + _NUM, text, re.I)
    for la, lah, lo, loh in pairs:
        lat, lon = float(la), float(lo)
        if lah and lah.upper() in ("S", "SOUTH"):
            lat = -lat
        if loh and loh.upper() in ("W", "WEST"):
            lon = -lon
        if is_lat(lat) and is_lon(lon):
            return lat, lon
    # Fourth: the first two bare signed numbers are a usable lat,lon.
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
    for i in range(len(nums) - 1):
        if is_lat(nums[i]) and is_lon(nums[i + 1]):
            return nums[i], nums[i + 1]
    return None


def main():
    _, raw = http_json(f"{env_url}/photos", None, {}, "GET")
    photos = [p for p in json.loads(raw).get("photos", []) if not p.get("answered")]
    answers = []
    # One order: fetch + call + record per photo in listed order (env requests one at a time).
    for p in photos:
        pid = p["id"]
        _, jpeg = http_json(f"{env_url}{p['url']}", None, {}, "GET")
        text = ask(jpeg)
        if text is None:
            continue
        c = coordinates(text)
        if c is None:
            sys.stderr.write(f"no coords for {pid}: {text[:200]}\n")
            continue
        answers.append({"id": pid, "lat": c[0], "lon": c[1]})
    if not answers:
        sys.exit(0)
    code, resp = http_json(f"{env_url}/answers", json.dumps({"answers": answers}).encode(),
                           {"content-type": "application/json"}, "POST")
    sys.stdout.write(f"posted {len(answers)} of {len(photos)}: HTTP {code} {resp.decode()}\n")


if __name__ == "__main__":
    main()