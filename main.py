#!/usr/bin/env python3
# photo-geolocator-01 entry: a vision agent that places 25 street-level photos.
# For each photo it calls claude-sonnet-5-5 once through the metered gateway,
# asking it to reason from visible clues before giving a latitude/longitude, then
# posts every answer to the env at the end. Deterministic given the model's
# answers: requests are a function of each photo's bytes and a fixed prompt, and
# answers are posted in photo order, so a replay makes the same calls and leaves
# the same state. A photo whose answer holds no coordinates is left unanswered.
#
# Runs as the grader runs it: python3 -P -s /bot/main.py (Python 3.13, stdlib only).
import base64, json, os, re, sys, urllib.request

MODEL = "claude-sonnet-5-5"
MAX_TOKENS = 1024
PROMPT = (
    "Look carefully at this street-level photograph. There is no metadata of any kind.\n"
    "Think step by step about the specific geographic clues it shows before you answer:\n"
    "- Language and script on any signs, banners, shop fronts, billboards, or vehicles.\n"
    "- Road markings, lane direction, and whether traffic drives left or right.\n"
    "- Architecture, building materials, rooflines, and building styles.\n"
    "- Vegetation, dominant climate, terrain, and season.\n"
    "- Landmarks, distinctive structures, or recognizable features.\n"
    "- Vehicle makes/models and license-plate style, if any are visible.\n"
    "Using all of those clues, estimate where on Earth the photo was taken.\n"
    "Reply with exactly one line of JSON, nothing else: "
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
    with urllib.request.urlopen(req, timeout=120) as r:
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
                import time
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


def coordinates(text: str):
    if not text:
        return None
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            o = json.loads(m.group(0))
            lat = o.get("lat", o.get("latitude"))
            lon = o.get("lon", o.get("lng", o.get("longitude")))
            if is_lat(lat) and is_lon(lon):
                return float(lat), float(lon)
        except (ValueError, TypeError):
            pass
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